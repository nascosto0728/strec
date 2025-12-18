import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import math

import torch

def _debug_check(name: str, x, verbose: bool = False, raise_on_error: bool = False):
    """
    簡易的 NaN / Inf 檢查器。
    - name: 要印出的變數名稱（方便 log）
    - x: 要檢查的 tensor（或其他型別）
    - verbose: 若 True，無問題時也會印出 shape/dtype
    - raise_on_error: 若發現 NaN/Inf，則拋例外以中斷程式（方便追蹤 stack）
    """
    if not torch.is_tensor(x):
        if verbose:
            print(f"[DEBUG] {name}: not a tensor (type={type(x)})")
        return

    try:
        isnan = torch.isnan(x).any().item()
        isinf = torch.isinf(x).any().item()
    except Exception as e:
        print(f"[DEBUG] {name}: error checking isnan/isinf: {e}")
        return

    if isnan or isinf:
        info = f"[DEBUG] {name} INVALID -> isnan={isnan}, isinf={isinf}, shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}"
        print(info)
        # 嘗試印出前幾個元素作為 sample（避免大量輸出）
        try:
            sample = x.detach().cpu().view(-1)[:16].tolist()
            print("  sample:", sample)
        except Exception:
            pass
        if raise_on_error:
            raise RuntimeError(f"{name} contains NaN/Inf")
    else:
        if verbose:
            print(f"[DEBUG] {name}: ok (shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device})")

# ===================================================================
#  Core Component 1: User Memory Bank (CPU Offload)
# ===================================================================
class UserMemoryBank:
    """
    全域使用者記憶體庫 (Global User Memory Bank)
    支援 enable_cpu_offload 選項
    - True: 儲存在 CPU (節省顯存，適合大維度)
    - False: 儲存在 GPU (速度最快，適合小維度)
    """
    def __init__(self, n_users: int, d_model: int,  enable_cpu_offload: bool = False):
        self.n_users = n_users
        self.d_model = d_model
        self.enable_cpu_offload = enable_cpu_offload
        
        # 計算記憶體大小
        mem_size_gb = (n_users * d_model * d_model * 4) / (1024**3)
        location = "CPU" if enable_cpu_offload else "GPU"
        
        # 安全檢查 device
        if not enable_cpu_offload and torch.cuda.is_available():
             location += f" (device={torch.cuda.current_device()})"
        
        print(f"--- [MemoryBank] Allocating {n_users}x{d_model}x{d_model} state matrix on {location} ---")
        print(f"--- [MemoryBank] Estimated Size: {mem_size_gb:.2f} GB ---")
        
        if enable_cpu_offload:
            # CPU Mode: 使用 pin_memory 加速傳輸
            self.states = torch.zeros(n_users, d_model, d_model, dtype=torch.float32, pin_memory=True)
        else:
            # GPU Mode: 直接在 GPU 上分配
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.states = torch.zeros(n_users, d_model, d_model, dtype=torch.float32, device=device)
        
    def get_batch_states(self, user_ids: torch.Tensor, target_device: torch.device = None) -> torch.Tensor:
        """讀取舊狀態 (Detached from history because it's from storage)"""
        # index_select 的 index 必須與 tensor 在同一設備
        if self.enable_cpu_offload:
            # CPU -> GPU
            ids_cpu = user_ids.cpu()
            batch_states = self.states.index_select(0, ids_cpu)
            return batch_states.to(target_device, non_blocking=True)
        else:
            # GPU -> GPU (Zero Copy)
            return self.states.index_select(0, user_ids)
    
    def update_batch_states(self, user_ids: torch.Tensor, new_states: torch.Tensor):
        """寫入新狀態"""
        # Detach 截斷梯度流，避免顯存洩漏
        new_states_detached = new_states.detach()
        
        if self.enable_cpu_offload:
            # GPU -> CPU
            ids_cpu = user_ids.cpu()
            states_cpu = new_states_detached.cpu()
            self.states.index_copy_(0, ids_cpu, states_cpu)
        else:
            # GPU -> GPU
            self.states.index_copy_(0, user_ids, new_states_detached)

# ===================================================================
#  Core Component 2: Robust Titans Layer
# ===================================================================
class MetaGatedTitansLayer(nn.Module):
    """
    Meta-Gated Titans: Self-Referential Memory Update
    結合了 Feature Modulation (方案A) 與 Adaptive Gating (方案C)
    """
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # --- 1. Projections (Titans Core) ---
        self.proj_q = nn.Linear(d_model, d_model, bias=False)
        self.proj_k = nn.Linear(d_model, d_model, bias=False)
        self.proj_v = nn.Linear(d_model, d_model, bias=False)
        
        # Base Gates (預設的學習率/遺忘率生成器)
        self.proj_alpha = nn.Linear(d_model, 1) 
        self.proj_eta = nn.Linear(d_model, 1)   
        
        # --- 2. Meta-Controller (The "Brain") ---
        # Input: User Static (D) + Memory Context (D) = 2D
        # Output: 
        #   - Gamma (D): Feature Scaling
        #   - Beta (D): Feature Shifting
        #   - Alpha_bias (1): Forget Gate Bias
        #   - Eta_bias (1): Input Gate Bias
        self.meta_controller = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model), # Normalize for stability
            nn.ReLU(),
            nn.Linear(d_model, d_model * 2 + 2) # D(gamma) + D(beta) + 1(alpha) + 1(eta)
        )
        
        # --- 3. Output components ---
        self.proj_out = nn.Linear(d_model, d_model)
        
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu
        
        self.scale = d_model ** -0.5

    def _read_memory(self, query_emb, state):
        """Helper: 用 Query 去讀取 Memory"""
        q = self.proj_q(query_emb)
        q = F.normalize(q, p=2, dim=-1)
        q_vec = q.unsqueeze(2) # [B, D, 1]
        read_content = torch.bmm(state, q_vec).squeeze(2) # [B, D]
        return read_content

    def predict(self, user_static_emb, old_state):
        """
        Step 1: Predict (Read Only)
        與之前相同，使用 User Static + Memory 產生預測特徵
        """
        # Pre-Norm
        x_norm = self.norm1(user_static_emb)
        
        # Read from memory
        read_content = self._read_memory(x_norm, old_state)
        
        # Pass through Output Projection & FFN
        attn_output = self.proj_out(read_content)
        x = user_static_emb + self.dropout1(attn_output)
        
        x2 = self.norm2(x)
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        user_dynamic_emb = x + self.dropout2(ffn_out)
        
        return user_dynamic_emb

    def update(self, item_emb, old_state, user_static_emb):
        """
        Step 2: Meta-Gated Update (Write)
        [Change] 新增 user_static_emb 作為 Context 來源
        """
        B, D = item_emb.shape
        
        # 1. 準備 Meta-Context
        #    Context = User Static + Current Memory State
        #    這代表：「我是誰」以及「我現在的狀態」
        x_static_norm = self.norm1(user_static_emb)
        memory_context = self._read_memory(x_static_norm, old_state) # [B, D]
        
        meta_input = torch.cat([x_static_norm, memory_context], dim=-1) # [B, 2D]
        
        # 2. Meta-Controller 生成控制參數
        meta_out = self.meta_controller(meta_input) # [B, 2D + 2]
        
        # Split outputs
        gamma = meta_out[:, :D]      # [B, D] scaling
        beta = meta_out[:, D:2*D]    # [B, D] shifting
        alpha_bias = meta_out[:, -2:-1] # [B, 1]
        eta_bias = meta_out[:, -1:]     # [B, 1]
        
        # 3. Feature Modulation (戴上有色眼鏡)
        #    item_emb 是原始輸入，我們根據 context 對其進行變形
        #    使用 1 + tanh(gamma) 確保 scaling 接近 1 且受控，避免爆炸
        i_norm = self.norm1(item_emb)
        
        modulation_scale = 1.0 + torch.tanh(gamma) 
        modulated_item = i_norm * modulation_scale + beta
        
        # 4. Standard Titans Logic with Modulated Input
        #    注意：K, V 的生成現在基於 "Modulated Item"
        k = self.proj_k(modulated_item)
        v = self.proj_v(modulated_item)
        k = F.normalize(k, p=2, dim=-1)
        
        # 5. Adaptive Gating (動態閘門)
        #    原本的 Gate 是 Linear(item)，現在加上 Meta Bias
        #    這讓模型可以說：「雖然這個 Item 很普通，但我現在學習慾望很強 (Eta Bias High)」
        alpha_logits = self.proj_alpha(modulated_item) + alpha_bias
        eta_logits = self.proj_eta(modulated_item) + eta_bias
        
        alpha = torch.sigmoid(alpha_logits) # [B, 1]
        eta = torch.sigmoid(eta_logits) * self.scale
        
        # 6. Memory Update (Delta Rule)
        k_vec = k.unsqueeze(2)
        v_vec = v.unsqueeze(2)
        
        pred = torch.bmm(old_state, k_vec)
        error = v_vec - pred
        
        alpha_bc = alpha.unsqueeze(2)
        update_term = torch.bmm(error, k_vec.transpose(1, 2))
        
        new_state = (1 - alpha_bc) * old_state + eta.unsqueeze(2) * update_term
        
        return new_state

class TitansLayer(nn.Module):
    """
    Titans: Linear Memory with Delta Rule (Robust Version)
    """
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Projections
        # Q 用於 Predict (來源是 User Static)
        self.proj_q = nn.Linear(d_model, d_model, bias=False)
        
        # K, V, Gates 用於 Update (來源是 Item)
        self.proj_k = nn.Linear(d_model, d_model, bias=False)
        self.proj_v = nn.Linear(d_model, d_model, bias=False)
        self.proj_alpha = nn.Linear(d_model, 1) 
        self.proj_eta = nn.Linear(d_model, 1)   
        
        # Output components
        self.proj_out = nn.Linear(d_model, d_model)
        
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu
        
        self.scale = d_model ** -0.5

    def predict(self, user_static_emb, old_state):
        """
        Step 1: 預測 (Read). 使用 User Static Feature 去查詢 Memory
        user_static_emb: [B, D]
        old_state: [B, D, D]
        Return: user_dynamic_emb [B, D]
        """
        # Pre-Norm
        x_norm = self.norm1(user_static_emb)
        
        # Generate Query
        q = self.proj_q(x_norm) # [B, D]
        q = F.normalize(q, p=2, dim=-1)
        q_vec = q.unsqueeze(2) # [B, D, 1]
        
        # Retrieval: y = S_{t-1} * q
        # 這是 User 結合了歷史記憶後的當下狀態
        read_content = torch.bmm(old_state, q_vec).squeeze(2) # [B, D]
        
        # Pass through Output Projection & FFN
        attn_output = self.proj_out(read_content)
        x = user_static_emb + self.dropout1(attn_output) # Residual with input
        
        x2 = self.norm2(x)
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        user_dynamic_emb = x + self.dropout2(ffn_out)
        
        return user_dynamic_emb

    def update(self, item_emb, old_state):
        """
        Step 2: 更新 (Write). 使用 Item 去更新 Memory
        item_emb: [B, D] - 當前互動的 Item (Target)
        old_state: [B, D, D]
        Return: new_state [B, D, D]
        """
        # Pre-Norm Item
        i_norm = self.norm1(item_emb)
        
        # Generate K, V, Gates from Item
        k = self.proj_k(i_norm)
        v = self.proj_v(i_norm)
        k = F.normalize(k, p=2, dim=-1)
        
        alpha = torch.sigmoid(self.proj_alpha(i_norm)) # [B, 1]
        eta = torch.sigmoid(self.proj_eta(i_norm)) * self.scale
        
        # Reshape
        k_vec = k.unsqueeze(2) # [B, D, 1]
        v_vec = v.unsqueeze(2) # [B, D, 1]
        
        # Memory Update (Delta Rule)
        # 1. Reconstruction: v_hat = S * k
        pred = torch.bmm(old_state, k_vec)
        
        # 2. Surprise: e = v - v_hat
        error = v_vec - pred
        
        # 3. Update: S_new = (1-a)S + eta * (e * k^T)
        alpha_bc = alpha.unsqueeze(2)
        update_term = torch.bmm(error, k_vec.transpose(1, 2))
        
        new_state = (1 - alpha_bc) * old_state + eta.unsqueeze(2) * update_term
        
        return new_state
    
class CMS_ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        
        # [Fast Lane] 快速通道
        # 特點：淺、神經元少、反應快
        # 負責：適應 User 最近幾次的點擊 (Short-term context)
        self.fast_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, input_dim)
        )
        
        # [Slow Lane] 慢速通道
        # 特點：深、寬、參數多
        # 負責：儲存 User 的長期畫像 (Long-term profile)
        self.slow_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), # 多一層，增加容量
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # 融合閘門
        # Input: 原始特徵 x
        # Output: 0~1 的權重 alpha
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 融合層 (簡單相加或透過 Gate 融合)
        self.out_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        fast_out = self.fast_net(x)
        slow_out = self.slow_net(x)
        
        # 殘差連接：原始特徵 + 快速修正 + 長期偏好
        alpha = self.gate(x) # 決定依賴 Fast 的程度
        
        # 軟性切換
        fused = alpha * fast_out + (1 - alpha) * slow_out
        
        return self.out_norm(x + fused)

# ===================================================================
#  Core Component 3: Standard Transformer (Item Side)
# ===================================================================

class StandardTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.norm1(src)
        attn_output, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_output)
        src2 = self.norm2(src)
        ffn_out = self.linear1(src2)
        ffn_out = self.activation(ffn_out)
        ffn_out = self.dropout(ffn_out)
        ffn_out = self.linear2(ffn_out)
        src = src + self.dropout2(ffn_out)
        return src

# ===================================================================
#  Helper: Average Pooling
# ===================================================================
def average_pooling(embeddings: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    if embeddings.dim() == 4: 
        max_len = embeddings.size(2)
        mask = torch.arange(max_len, device=embeddings.device)[None, None, :] < seq_lens.unsqueeze(-1)
        mask = mask.float().unsqueeze(-1) 
        masked_sum = torch.sum(embeddings * mask, dim=2)
        count = mask.sum(dim=2) + 1e-9
        return masked_sum / count
    elif embeddings.dim() == 3: 
        max_len = embeddings.size(1)
        mask = torch.arange(max_len, device=embeddings.device)[None, :] < seq_lens[:, None]
        mask = mask.float().unsqueeze(-1) 
        masked_sum = torch.sum(embeddings * mask, dim=1)
        count = mask.sum(dim=1) + 1e-9
        return masked_sum / count
    raise ValueError(f"Unsupported dim: {embeddings.dim()}")

# ===================================================================
#  Main Model: DualTowerTitans
# ===================================================================

class DualTowerTitans(nn.Module):
    def __init__(self, global_meta: Dict[str, Any], cfg: Dict[str, Any]):
        super().__init__()
        self.hparams = cfg['model']
        self.temperature = 0.07 
        self.maxlen = self.hparams.get('max_seq_len', 30)
        
        # Buffers
        self.register_buffer('cates', torch.from_numpy(global_meta['cate_matrix']))
        self.register_buffer('cate_lens', torch.tensor(global_meta['cate_lens'], dtype=torch.int32))
        
        # Dimensions
        self.embed_dim = self.hparams['item_embed_dim']
        
        self.user_emb_dim = self.embed_dim
        self.item_emb_dim = self.embed_dim
        self.cate_emb_dim = self.embed_dim 
        
        # Tower Input Dimensions
        self.user_tower_input_dim = self.embed_dim 
        self.item_tower_input_dim = self.embed_dim
        
        # 1. Embeddings
        self._build_embeddings()
        
        # 2. Titans Memory Bank 
        enable_cpu_offload = self.hparams.get('titans_cpu_offload', False)
        self.user_memory_bank = UserMemoryBank(
            n_users=self.hparams['n_users'], 
            d_model=self.user_tower_input_dim,
            enable_cpu_offload=enable_cpu_offload
        )
        
        # 3. Encoders (Titans & Transformer)
        self._build_encoders()
        
        # 4. Heads
        self.user_mlp = self._create_mlp(self.user_emb_dim*2) # Static + Dynamic -> 2x
        self.item_mlp = self._create_mlp(self.item_emb_dim*2) # Static + History -> 2x

        # 5. Negatives Buffer
        self.user_history_buffer = nn.Embedding(self.hparams['n_items'], self.item_emb_dim)
        self.user_history_buffer.weight.requires_grad = False
        
        self._init_weights()

    def _build_embeddings(self):
        self.user_emb_w = nn.Embedding(self.hparams['n_users'], self.user_emb_dim, padding_idx=0)
        self.item_emb_w = nn.Embedding(self.hparams['n_items'], self.item_emb_dim, padding_idx=0) 
        self.cate_emb_w = nn.Embedding(self.hparams['n_cates'], self.cate_emb_dim, padding_idx=0)

        # Positional Embedding (只給 Item Tower 的 Transformer 用)
        self.item_tower_pos_emb = nn.Embedding(self.maxlen, self.item_tower_input_dim)

        # [新增] Learnable BOS Token (Beginning of Sequence)
        # 當 User 沒有任何歷史 (interacted_len == 0) 時，用這個向量來更新初始狀態
        self.bos_item_emb = nn.Parameter(torch.randn(1, self.item_emb_dim))
    
        # 初始化權重
        nn.init.normal_(self.bos_item_emb, mean=0, std=0.02)

    def _build_encoders(self):
        # --- User Tower: One Step Titans ---
        self.user_titans = MetaGatedTitansLayer(
            d_model=self.user_tower_input_dim,
            dim_feedforward=self.user_tower_input_dim * 4,
            dropout=self.hparams.get('transformer_dropout', 0.1)
        )
        
        # --- Item Tower: Transformer (Standard SASRec) ---
        self.item_transformer = nn.ModuleList([
            StandardTransformerLayer(
                d_model=self.item_tower_input_dim, 
                nhead=4,
                dim_feedforward=self.item_tower_input_dim * 4,
                dropout=0.1
            ) for _ in range(self.hparams.get('transformer_n_layers', 2))
        ])

    def _create_mlp(self, input_dim):
        # return nn.Sequential(
        #     nn.LayerNorm(input_dim),
        #     nn.Linear(input_dim, input_dim * 2),
        #     nn.ReLU(),
        #     nn.Linear(input_dim * 2, input_dim) 
        # )
        
        # 使用 CMS 替換原本的 MLP
        return CMS_ProjectionHead(
            input_dim=input_dim,
            hidden_dim=input_dim * 2, # 擴展維度
            dropout=0.1
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    # ===================================================================
    #  Forward Logic
    # ===================================================================

    def _lookup_item_features(self, item_ids: torch.Tensor) -> torch.Tensor:
        # 1. Item ID Embedding
        static_emb = self.item_emb_w(item_ids) # [B, D]
        
        # 2. Category Embedding (Average Pooling)
        item_cates = self.cates[item_ids]
        item_cates_emb = self.cate_emb_w(item_cates)
        avg_cate_emb = average_pooling(item_cates_emb, self.cate_lens[item_ids]) # [B, D]
        
        # 相加模式 (Additive)
        return static_emb + avg_cate_emb

    def _build_feature_representations(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # ---------------------------------------------------
        # 1. User Tower (Titans - Stateful)
        # ---------------------------------------------------
        user_static = self.user_emb_w(batch['user_id']) # [B, D]
        
        # A. 從 Memory Bank 讀取 Old State (Detached)
        state_before_prev = self.user_memory_bank.get_batch_states(batch['user_id'], user_static.device)
        
        # B. [動態提取] 準備 Prev Item Features (用於更新狀態)
        # 1. 取得歷史序列與長度
        hist_seq = batch['user_interacted_items'] # [B, MaxLen]
        hist_len = batch['user_interacted_len']   # [B]
        B = hist_len.size(0)

        # 2. 計算最後一個有效 Item 的 Index (Clamp to 0 to avoid error, masked later)
        last_item_idx = (hist_len - 1).clamp(min=0).unsqueeze(1)
        
        # 3. Gather Prev Item ID
        prev_item_ids = hist_seq.gather(1, last_item_idx).squeeze(1)
        
        # 4. Lookup Features (No Detach! We want gradient to flow here)
        prev_item_features = self._lookup_item_features(prev_item_ids) # [B, D]
        
        # 5. Cold Start Mask (Len == 0) -> Use BOS
        is_cold_start = (hist_len == 0).unsqueeze(1) # [B, 1]
        bos_expanded = self.bos_item_emb.expand(B, -1)
        
        # 混合 Input: 冷啟動用 BOS，否則用上一個 Item
        update_input_emb = torch.where(is_cold_start, bos_expanded, prev_item_features)
        
        # C. [Update Step] 用 Prev Item / BOS 更新狀態，得到 Current State (With Grad)
        current_state = self.user_titans.update(
            item_emb=update_input_emb, 
            old_state=state_before_prev,
            user_static_emb=user_static 
        )
        
        # D. [Predict Step] 使用 Current State 預測 User Feature
        titans_out = self.user_titans.predict(user_static, current_state)
        
        # Concat [B, 2D]
        user_features = torch.cat([user_static, titans_out], dim=-1)

        # [State Update] 只在訓練時更新 Memory Bank (Inference 時不更新，避免污染)
        if self.training:
            self.user_memory_bank.update_batch_states(batch['user_id'], current_state)
            
        # ---------------------------------------------------
        # 2. Item Tower (Transformer - Stateless)
        # ---------------------------------------------------
        item_composite_static = self._lookup_item_features(batch['item_id']) # [B, D]
        B_item, _ = item_composite_static.shape
        
        # Input: Item Sequence
        item_hist_user_ids = batch['item_interacted_users']
        item_hist_emb = self.user_emb_w(item_hist_user_ids).detach() # [B, T, D]
        seq_len_i = batch['item_interacted_len']
        
        # Add Positional Embedding
        pos_ids = torch.arange(item_hist_emb.size(1), device=item_hist_emb.device)
        item_seq_input = item_hist_emb + self.item_tower_pos_emb(pos_ids).unsqueeze(0)
        
        # Masks (Causal + Padding)
        src_mask = nn.Transformer.generate_square_subsequent_mask(item_hist_emb.size(1), device=item_hist_emb.device).bool()
        safe_lens_i = torch.clamp(seq_len_i, min=1)
        padding_mask_i = (torch.arange(item_hist_emb.size(1), device=item_hist_emb.device)[None, :] >= safe_lens_i[:, None])
        
        # Transformer Forward
        output = item_seq_input
        for layer in self.item_transformer:
            output = layer(output, src_mask=src_mask, src_key_padding_mask=padding_mask_i)
            
        # Pooling (Last)
        target_indices_i = (seq_len_i - 1).clamp(min=0).view(B_item, 1, 1).expand(-1, 1, self.item_tower_input_dim)
        item_history_emb = torch.gather(output, 1, target_indices_i).squeeze(1)
        
        # Update Negatives Buffer
        if self.training:
            self.user_history_buffer.weight[batch['item_id']] = item_history_emb.detach()
            
        item_features = torch.cat([item_composite_static, item_history_emb], dim=-1)

        return user_features, item_features

    def forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        user_features, item_features = self._build_feature_representations(batch)
        
        user_embedding = F.normalize(self.user_mlp(user_features), p=2, dim=-1, eps=1e-6)
        item_embedding = F.normalize(self.item_mlp(item_features), p=2, dim=-1, eps=1e-6)
        
        return user_embedding, item_embedding

    def _calculate_infonce_loss(self, user_embedding, item_embedding, labels):
        all_inner_product = torch.matmul(user_embedding, item_embedding.t())
        logits = all_inner_product / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits_stabilized = logits - logits_max
        exp_logits_den = torch.exp(logits_stabilized)
        denominator = exp_logits_den.sum(dim=1, keepdim=True)

        pred_scores = (user_embedding * item_embedding).sum(dim=1, keepdim=True)
        pred_logits = pred_scores / self.temperature
        pred_logits_stabilized = pred_logits - logits_max
        numerator = torch.exp(pred_logits_stabilized)

        infonce_pred = (numerator / (denominator + 1e-9)).squeeze(dim=1)
        infonce_pred = torch.clamp(infonce_pred, min=1e-9, max=1.0 - 1e-9)

        return F.binary_cross_entropy(infonce_pred, labels.float(), reduction='none')
    
    def calculate_loss(self, batch: Dict) -> torch.Tensor:
        student_session_emb, student_item_emb = self.forward(batch)
        loss_infonce = self._calculate_infonce_loss(student_session_emb, student_item_emb, batch['labels'])
        return loss_infonce.mean()

    def inference(self, batch: Dict[str, torch.Tensor], neg_item_ids_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. 取得 User Features (training=False, 不會更新 Memory)
        user_features, item_features = self._build_feature_representations(batch)
        
        # 2. Positive Scores
        pos_user_emb = F.normalize(self.user_mlp(user_features), p=2, dim=-1)
        pos_item_emb = F.normalize(self.item_mlp(item_features), p=2, dim=-1)
        
        pos_logits = torch.sum(pos_user_emb * pos_item_emb, dim=1)
        per_sample_loss = F.binary_cross_entropy_with_logits(pos_logits, batch['labels'].float(), reduction='none')

        # 3. Negative Samples Handling
        num_neg_samples = neg_item_ids_batch.shape[1]
        
        # A. 負樣本 Static (ID + Cate)
        neg_item_static_emb = self.item_emb_w(neg_item_ids_batch) 
        neg_item_cate_emb = self.cate_emb_w(self.cates[neg_item_ids_batch])
        neg_item_cates_len = self.cate_lens[neg_item_ids_batch]
        avg_cate_emb_for_neg_item = average_pooling(neg_item_cate_emb, neg_item_cates_len) 
        
        # 相加模式 (Additive)
        neg_item_emb_with_cate = neg_item_static_emb + avg_cate_emb_for_neg_item 
        
        # B. 負樣本 History (從 Buffer 讀取)
        item_history_emb_expanded = self.user_history_buffer(neg_item_ids_batch)
        
        # C. 組合 (Concat)
        neg_item_features = torch.cat([neg_item_emb_with_cate, item_history_emb_expanded.detach()], dim=2)
        
        # D. User Feature 擴展
        user_features_expanded = user_features.unsqueeze(1).expand(-1, num_neg_samples, -1)
        
        # E. MLP Projection
        neg_user_emb = self.user_mlp(user_features_expanded)
        neg_item_emb = self.item_mlp(neg_item_features)
        
        # [修正] 補上 eps=1e-6
        neg_user_emb_final = F.normalize(neg_user_emb, p=2, dim=-1, eps=1e-6)
        neg_item_emb_final = F.normalize(neg_item_emb, p=2, dim=-1, eps=1e-6)
        
        neg_logits = torch.sum(neg_user_emb_final * neg_item_emb_final, dim=2)
        
        return pos_logits, neg_logits, per_sample_loss
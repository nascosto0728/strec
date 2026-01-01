import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import math

# ===================================================================
#  Core Component 1: User Memory Bank (CPU Offload Support)
# ===================================================================
class UserMemoryBank:
    """
    全域使用者記憶體庫 (Global User Memory Bank)
    儲存每個 User 的 Titans 狀態矩陣 S。
    Shape: [N_USERS, N_HEADS, HEAD_DIM, HEAD_DIM]
    """
    def __init__(self, n_users: int, state_shape: Tuple[int, ...], enable_cpu_offload: bool = False):
        self.n_users = n_users
        self.state_shape = state_shape # e.g., (4, 8, 8) for H=4, Dh=8
        self.enable_cpu_offload = enable_cpu_offload
        
        # 計算記憶體佔用量並印出資訊
        state_numel = math.prod(state_shape)
        mem_size_gb = (n_users * state_numel * 4) / (1024**3)
        
        location = "CPU" if enable_cpu_offload else "GPU"
        if not enable_cpu_offload and torch.cuda.is_available():
             location += f" (device={torch.cuda.current_device()})"
             
        print(f"--- [MemoryBank] Allocating {n_users}x{state_shape} state ---")
        print(f"--- [MemoryBank] Estimated Size: {mem_size_gb:.2f} GB ---")
        
        full_shape = (n_users,) + state_shape
        
        # 初始化全零狀態
        if enable_cpu_offload:
            self.states = torch.zeros(full_shape, dtype=torch.float32, pin_memory=True)
        else:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.states = torch.zeros(full_shape, dtype=torch.float32, device=device)
        
    def get_batch_states(self, user_ids: torch.Tensor, target_device: torch.device = None) -> torch.Tensor:
        """從 Bank 讀取狀態 (Read)"""
        if self.enable_cpu_offload:
            ids_cpu = user_ids.cpu()
            batch_states = self.states.index_select(0, ids_cpu)
            return batch_states.to(target_device, non_blocking=True)
        else:
            return self.states.index_select(0, user_ids)
    
    def update_batch_states(self, user_ids: torch.Tensor, new_states: torch.Tensor):
        """將新狀態寫回 Bank (Write)，注意這裡會 Detach 截斷梯度"""
        new_states_detached = new_states.detach()
        if self.enable_cpu_offload:
            ids_cpu = user_ids.cpu()
            states_cpu = new_states_detached.cpu()
            self.states.index_copy_(0, ids_cpu, states_cpu)
        else:
            self.states.index_copy_(0, user_ids, new_states_detached)

# ===================================================================
#  Core Component 2: Multi-Head Meta-Gated Titans Layer
# ===================================================================
class MultiHeadMetaGatedTitansLayer(nn.Module):
    """
    Multi-Head Meta-Gated Titans (The Core Engine)
    特點：
    1. Multi-Head: 平行處理多個子空間的記憶。
    2. Meta-Gated: 使用 Meta-Controller 動態生成 scaling/shifting/gating 參數。
    3. Self-Referential: 更新規則取決於 (User Static + Current Memory)。
    """
    def __init__(self, d_model, n_heads=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # --- 1. Projections (Q, K, V) ---
        self.proj_q = nn.Linear(d_model, d_model, bias=False)
        self.proj_k = nn.Linear(d_model, d_model, bias=False)
        self.proj_v = nn.Linear(d_model, d_model, bias=False)
        
        # Base Gate Biases (Per Head)
        self.proj_alpha = nn.Linear(d_model, n_heads) 
        self.proj_eta = nn.Linear(d_model, n_heads)   
        
        # --- 2. Meta-Controller (小腦) ---
        # Input: User Static (D) + Memory Context (D) = 2D
        # Output: Gamma(D) + Beta(D) + Alpha_bias(H) + Eta_bias(H)
        self.meta_controller = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model * 2 + n_heads * 2) 
        )
        
        # --- 3. Output Fusion ---
        self.proj_out = nn.Linear(d_model, d_model)
        
        # FFN Block
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu
        
        self.scale = self.head_dim ** -0.5

    def _read_memory(self, query_emb, state):
        """Helper: 用 Query 讀取多頭記憶體"""
        B = query_emb.size(0)
        
        # Project & Reshape Q -> [B, H, Dh]
        q = self.proj_q(query_emb)
        q = q.view(B, self.n_heads, self.head_dim)
        q = F.normalize(q, p=2, dim=-1)
        
        # Read: y = S * q
        # [B, H, Dh, Dh] @ [B, H, Dh, 1] -> [B, H, Dh, 1]
        q_vec = q.unsqueeze(-1)
        read_content = torch.matmul(state, q_vec).squeeze(-1) 
        
        # Flatten back to [B, D]
        return read_content.view(B, self.d_model)

    def predict(self, user_static_emb, old_state, last_item_emb):
        # 1. 使用 "Last Item" 作為 Query 來激活記憶 (Associative Recall)
        #    這代表： "Based on what I just saw, what do I remember?"
        #    如果 last_item_emb 是 BOS (冷啟動)，則代表 "Based on nothing/start, what do I remember?"
        q_norm = self.norm1(last_item_emb) 
        read_content = self._read_memory(q_norm, old_state)
        
        # 2. 融合 User Static (人設) + Memory Output (聯想結果)
        #    這裡把 User Static 加回來，作為 Residual 或 Base
        #    這樣模型既知道 "我是誰" (Static)，也知道 "我剛看了什麼引發的聯想" (Memory)
        
        # Fusion: Linear Projection
        attn_output = self.proj_out(read_content)
        
        # Residual Connection
        x = user_static_emb + self.dropout1(attn_output)
        
        x2 = self.norm2(x)
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        user_dynamic_emb = x + self.dropout2(ffn_out)
        
        return user_dynamic_emb

    def update(self, item_emb, old_state, user_static_emb):
        """Step 2: Update (Write) with Meta-Gating"""
        B, D = item_emb.shape
        H, Dh = self.n_heads, self.head_dim
        
        # 1. Meta-Context Preparation
        # 使用 Current Item 作為 Query 去讀記憶
        i_norm = self.norm1(item_emb)
        memory_context = self._read_memory(i_norm, old_state) # [B, D]
        
        # Meta Input: User Static (Who) + Context (Memory match)
        user_static_norm = self.norm1(user_static_emb)
        meta_input = torch.cat([user_static_norm,  memory_context], dim=-1) # [B, 2D]

        # 2. Meta-Controller 生成參數
        meta_out = self.meta_controller(meta_input)
        
        gamma = meta_out[:, :D]        # Feature Scaling
        beta = meta_out[:, D:2*D]      # Feature Shifting
        alpha_bias = meta_out[:, 2*D:2*D+H] # Forget Gate Bias
        eta_bias = meta_out[:, -H:]         # Input Gate Bias
        
        # 3. Feature Modulation (特徵調變)
        i_norm = self.norm1(item_emb)
        modulation_scale = 1.0 + torch.tanh(gamma)
        modulated_item = i_norm * modulation_scale + beta
        
        # 4. Projections
        k = self.proj_k(modulated_item).view(B, H, Dh)
        v = self.proj_v(modulated_item).view(B, H, Dh)
        k = F.normalize(k, p=2, dim=-1)
        
        # 5. Adaptive Gating (Per Head)
        alpha_logits = self.proj_alpha(modulated_item) + alpha_bias
        eta_logits = self.proj_eta(modulated_item) + eta_bias
        
        alpha = torch.sigmoid(alpha_logits).view(B, H, 1, 1)
        eta = (torch.sigmoid(eta_logits) * self.scale).view(B, H, 1, 1)
        
        # 6. Memory Update (Delta Rule)
        k_vec = k.unsqueeze(-1)
        v_vec = v.unsqueeze(-1)
        
        # Reconstruction: v_hat = S * k
        pred = torch.matmul(old_state, k_vec)
        error = v_vec - pred
        
        # Update Term: error * k.T
        update_term = torch.matmul(error, k_vec.transpose(-1, -2))
        
        # New State
        new_state = (1 - alpha) * old_state + eta * update_term
        
        return new_state

# ===================================================================
#  Core Component 3: CMS Projection Head (For User Tower)
# ===================================================================
# class CMS_ProjectionHead(nn.Module):
#     """
#     Continuum Memory System (Gated)
#     包含 Fast Lane (短期適應) 與 Slow Lane (長期記憶)。
#     """
#     def __init__(self, input_dim, hidden_dim, dropout=0.1):
#         super().__init__()
        
#         # [Fast Lane] 淺層、快速適應
#         self.fast_net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim // 2),
#             nn.LayerNorm(hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 2, input_dim)
#         )
        
#         # [Slow Lane] 深層、長期記憶 (需配合 Optimizer 低 LR)
#         self.slow_net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim), 
#             nn.ReLU(),
#             nn.Linear(hidden_dim, input_dim)
#         )

#         # Gating 機制：決定依賴短期還是長期
#         self.gate = nn.Sequential(
#             nn.Linear(input_dim, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1),
#             nn.Sigmoid()
#         )
        
#         self.out_norm = nn.LayerNorm(input_dim)

#     def forward(self, x):
#         fast_out = self.fast_net(x)
#         slow_out = self.slow_net(x)
        
#         alpha = self.gate(x)
#         fused = alpha * fast_out + (1 - alpha) * slow_out
        
#         return self.out_norm(x + fused)

# ===================================================================
#  Core Component 4: Standard Transformer (For Item Tower)
# ===================================================================
class StandardTransformerLayer(nn.Module):
    """
    標準 Transformer Encoder Layer (用於捕捉 Item History 的共現關係)
    """
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
        
        self.user_tower_input_dim = self.embed_dim 
        self.item_tower_input_dim = self.embed_dim

        # [Config] 設定 Head 數量 (建議 4)
        self.n_heads = self.hparams.get('titans_n_heads', 4)
        self.head_dim = self.user_tower_input_dim // self.n_heads
        
        assert self.user_tower_input_dim % self.n_heads == 0, \
            f"Embed dim {self.user_tower_input_dim} not divisible by n_heads {self.n_heads}"
        
        # 1. Embeddings (加上 padding_idx=0 安全網)
        self._build_embeddings()
        
        # 2. Titans Memory Bank (Multi-Head Shape)
        state_shape = (self.n_heads, self.head_dim, self.head_dim)
        enable_cpu_offload = self.hparams.get('titans_cpu_offload', False)
        self.user_memory_bank = UserMemoryBank(
            n_users=self.hparams['n_users'], 
            state_shape=state_shape,
            enable_cpu_offload=enable_cpu_offload
        )
        
        # 3. Encoders
        self._build_encoders()
        
        # 4. Heads (分離設計：User 用 CMS, Item 用 Standard MLP)
        self.user_mlp = self._create_mlp(self.user_emb_dim*2)
        self.item_mlp = self._create_mlp(self.item_emb_dim*2)

        # 5. Negatives Buffer
        self.user_history_buffer = nn.Embedding(self.hparams['n_items'], self.item_emb_dim)
        self.user_history_buffer.weight.requires_grad = False
        
        self._init_weights()

    def _build_embeddings(self):
        # 統一使用 padding_idx=0，前處理需對應 shift
        self.user_emb_w = nn.Embedding(self.hparams['n_users'], self.user_emb_dim, padding_idx=0)
        self.item_emb_w = nn.Embedding(self.hparams['n_items'], self.item_emb_dim, padding_idx=0) 
        self.cate_emb_w = nn.Embedding(self.hparams['n_cates'], self.cate_emb_dim, padding_idx=0)

        # Positional Embedding (只給 Item Tower 的 Transformer 用)
        self.item_tower_pos_emb = nn.Embedding(self.maxlen, self.item_tower_input_dim)

        # Learnable BOS Token (冷啟動專用)
        self.bos_item_emb = nn.Parameter(torch.randn(1, self.item_emb_dim))
        nn.init.normal_(self.bos_item_emb, mean=0, std=0.02)

    def _build_encoders(self):
        # --- User Tower: Multi-Head Meta-Gated Titans ---
        self.user_titans = MultiHeadMetaGatedTitansLayer(
            d_model=self.user_tower_input_dim,
            n_heads=self.n_heads,
            dim_feedforward=self.user_tower_input_dim * 4,
            dropout=self.hparams.get('transformer_dropout', 0.1)
        )
        
        # --- Item Tower: 1-Layer Transformer ---
        self.item_transformer = nn.ModuleList([
            StandardTransformerLayer(
                d_model=self.item_tower_input_dim, 
                nhead=4,
                dim_feedforward=self.item_tower_input_dim * 4,
                dropout=0.1
            ) for _ in range(self.hparams.get('transformer_n_layers', 1)) # Default 1 layer
        ])

    def _create_mlp(self, input_dim):
        expansion_factor = 2
        inner_dim = int(input_dim * expansion_factor)
        dropout = 0.0 
        
        # PreNorm Residual MLP Block
        class PreNormResidualBlock(nn.Module):
            def __init__(self, dim, fn):
                super().__init__()
                self.fn = fn
                self.norm = nn.LayerNorm(dim)
            def forward(self, x):
                return self.fn(self.norm(x)) + x          
        def _block():
            return nn.Sequential(
                nn.Linear(input_dim, inner_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, input_dim),
                nn.Dropout(dropout)
            )
            
        return nn.Sequential(
            PreNormResidualBlock(input_dim, _block())
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
        # Additive Feature Fusion: ID + Category
        static_emb = self.item_emb_w(item_ids) 
        item_cates = self.cates[item_ids]
        item_cates_emb = self.cate_emb_w(item_cates)
        avg_cate_emb = average_pooling(item_cates_emb, self.cate_lens[item_ids]) 
        
        return static_emb + avg_cate_emb

    def _build_feature_representations(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # ---------------------------------------------------
        # 1. User Tower (Titans - Stateful)
        # ---------------------------------------------------
        user_static = self.user_emb_w(batch['user_id']) 
        
        # A. Read Old State
        state_before_prev = self.user_memory_bank.get_batch_states(batch['user_id'], user_static.device)
        
        # B. Prepare Prev Item / Cold Start Handling
        hist_seq = batch['user_interacted_items'] 
        hist_len = batch['user_interacted_len']   
        B = hist_len.size(0)

        last_item_idx = (hist_len - 1).clamp(min=0).unsqueeze(1)
        prev_item_ids = hist_seq.gather(1, last_item_idx).squeeze(1)
        prev_item_features = self._lookup_item_features(prev_item_ids) 
        
        # Cold Start: If len=0, use BOS token
        is_cold_start = (hist_len == 0).unsqueeze(1) 
        bos_expanded = self.bos_item_emb.expand(B, -1)
        update_input_emb = torch.where(is_cold_start, bos_expanded, prev_item_features)
        
        # C. Update State (Gradient flows through user_titans)
        current_state = self.user_titans.update(
            item_emb=update_input_emb, 
            old_state=state_before_prev,
            user_static_emb=user_static 
        )
        
        # D. Predict
        titans_out = self.user_titans.predict(
            user_static_emb=user_static, 
            old_state=current_state,
            last_item_emb=update_input_emb 
        )
        user_features = torch.cat([user_static, titans_out], dim=-1)

        # [State Update] Only during training
        if self.training:
            self.user_memory_bank.update_batch_states(batch['user_id'], current_state)
            
        # ---------------------------------------------------
        # 2. Item Tower (Transformer - Stateless)
        # ---------------------------------------------------
        item_composite_static = self._lookup_item_features(batch['item_id'])
        B_item, _ = item_composite_static.shape
        
        item_hist_user_ids = batch['item_interacted_users']
        item_hist_emb = self.user_emb_w(item_hist_user_ids).detach()
        seq_len_i = batch['item_interacted_len']
        
        # Positional Embedding (Optional but kept for Transformer)
        pos_ids = torch.arange(item_hist_emb.size(1), device=item_hist_emb.device)
        item_seq_input = item_hist_emb + self.item_tower_pos_emb(pos_ids).unsqueeze(0)
        
        # Padding Mask
        safe_lens_i = torch.clamp(seq_len_i, min=1)
        padding_mask_i = (torch.arange(item_hist_emb.size(1), device=item_hist_emb.device)[None, :] >= safe_lens_i[:, None])
        
        output = item_seq_input
        for layer in self.item_transformer:
            output = layer(output, src_mask=None, src_key_padding_mask=padding_mask_i)
            
        # Pooling: Last Valid Token
        target_indices_i = (seq_len_i - 1).clamp(min=0).view(B_item, 1, 1).expand(-1, 1, self.item_tower_input_dim)
        item_history_emb = torch.gather(output, 1, target_indices_i).squeeze(1)
        
        # Update Negatives Buffer
        if self.training:
            self.user_history_buffer.weight[batch['item_id']] = item_history_emb.detach()
            
        item_features = torch.cat([item_composite_static, item_history_emb], dim=-1)

        return user_features, item_features

    def forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        user_features, item_features = self._build_feature_representations(batch)
        
        # Normalize with eps for stability
        user_embedding = F.normalize(self.user_mlp(user_features), p=2, dim=-1, eps=1e-6)
        item_embedding = F.normalize(self.item_mlp(item_features), p=2, dim=-1, eps=1e-6)
        
        return user_embedding, item_embedding

    def _calculate_infonce_loss(self, user_embedding, item_embedding, labels):
        all_inner_product = torch.matmul(user_embedding, item_embedding.t())
        logits = all_inner_product / self.temperature
        
        # Stable Softmax
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
        user_features, item_features = self._build_feature_representations(batch)
        
        pos_user_emb = F.normalize(self.user_mlp(user_features), p=2, dim=-1, eps=1e-6)
        pos_item_emb = F.normalize(self.item_mlp(item_features), p=2, dim=-1, eps=1e-6)
        
        pos_logits = torch.sum(pos_user_emb * pos_item_emb, dim=1)
        per_sample_loss = F.binary_cross_entropy_with_logits(pos_logits, batch['labels'].float(), reduction='none')

        # Negative Samples
        num_neg_samples = neg_item_ids_batch.shape[1]
        
        neg_item_static_emb = self.item_emb_w(neg_item_ids_batch) 
        neg_item_cate_emb = self.cate_emb_w(self.cates[neg_item_ids_batch])
        neg_item_cates_len = self.cate_lens[neg_item_ids_batch]
        avg_cate_emb_for_neg_item = average_pooling(neg_item_cate_emb, neg_item_cates_len) 
        
        neg_item_emb_with_cate = neg_item_static_emb + avg_cate_emb_for_neg_item 
        
        item_history_emb_expanded = self.user_history_buffer(neg_item_ids_batch)
        neg_item_features = torch.cat([neg_item_emb_with_cate, item_history_emb_expanded.detach()], dim=2)
        
        user_features_expanded = user_features.unsqueeze(1).expand(-1, num_neg_samples, -1)
        
        # Use user_mlp and item_mlp accordingly
        neg_user_emb = self.user_mlp(user_features_expanded)
        neg_item_emb = self.item_mlp(neg_item_features)
        
        neg_user_emb_final = F.normalize(neg_user_emb, p=2, dim=-1, eps=1e-6)
        neg_item_emb_final = F.normalize(neg_item_emb, p=2, dim=-1, eps=1e-6)
        
        neg_logits = torch.sum(neg_user_emb_final * neg_item_emb_final, dim=2)
        
        return pos_logits, neg_logits, per_sample_loss
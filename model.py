
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from typing import Tuple, Optional, Dict, List
import random


class UserMemoryBank(nn.Module):
    """
    Titans Global User Memory Bank
    
    職責：
    1. 持久化儲存 Titans 的 Memory State (M) 和 Momentum State (S)。
    2. 實作「梯度防火牆」，確保 update 時切斷計算圖 (Detach)。
    3. 支援 CPU Offload (Pinned Memory) 以應對大規模 User。
    
    形狀定義：
    - State: [N_USERS, N_HEADS, HEAD_DIM, HEAD_DIM]
    - Momentum: 同上
    """
    def __init__(self, 
                 n_users: int, 
                 n_heads: int, 
                 head_dim: int, 
                 enable_cpu_offload: bool = True,
                 device: torch.device = None):
        super().__init__()
        self.n_users = n_users
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.enable_cpu_offload = enable_cpu_offload
        
        # 決定儲存裝置
        if self.enable_cpu_offload:
            self.store_device = torch.device('cpu')
            self.pin_memory_flag = True # 加速 CPU -> GPU 傳輸
        else:
            self.store_device = device if device else torch.device('cuda')
            self.pin_memory_flag = False
            
        # Titans 使用矩陣記憶體: [Head, Dim, Dim]
        self.state_shape = (n_heads, head_dim, head_dim)
        
        # 記憶體估算 (Float32 = 4 bytes, x2 for State+Momentum)
        total_elements = n_users * math.prod(self.state_shape) * 2 
        size_gb = (total_elements * 4) / (1024**3)
        
        print(f"--- [UserMemoryBank] Initialized ---")
        print(f"  - Capacity : {n_users} Users")
        print(f"  - Shape    : {self.state_shape} per user")
        print(f"  - Storage  : {'CPU RAM (Pinned)' if self.pin_memory_flag else 'GPU VRAM'}")
        print(f"  - Est. Size: {size_gb:.2f} GB")
        
        # 初始化 Buffers (全 0 初始化)
        # 關鍵：persistent=False 避免被自動存入 model checkpoint
        _states = torch.zeros(
            (n_users,) + self.state_shape, 
            dtype=torch.float32, 
            device=self.store_device
        )
        
        _momentums = torch.zeros(
            (n_users,) + self.state_shape, 
            dtype=torch.float32, 
            device=self.store_device
        )

        if self.pin_memory_flag:
            _states = _states.pin_memory()
            _momentums = _momentums.pin_memory()

        # [FIX] 根據 Offload 策略決定是否註冊為 Buffer
        if self.enable_cpu_offload:
            # Case A: CPU Offload
            # 直接賦值，不註冊。這樣執行 model.to('cuda') 時，這些 Tensor 不會被搬走。
            self.states = _states
            self.momentums = _momentums
        else:
            # Case B: GPU Mode
            # 註冊為 Buffer (persistent=False)，隨模型一起移動到 GPU。
            self.register_buffer('states', _states, persistent=False)
            self.register_buffer('momentums', _momentums, persistent=False)

    def get_batch(self, user_ids: torch.Tensor, target_device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        讀取狀態 (Read)
        
        Args:
            user_ids: [B] 使用者 ID
            target_device: 計算發生的裝置 (GPU)
        """
        # 如果 Bank 在 CPU，ID 也需要在 CPU 才能 index
        if self.enable_cpu_offload:
            ids_lookup = user_ids.cpu()
        else:
            ids_lookup = user_ids
            
        # 讀取 (Copy)
        # index_select 是一個複製操作，不會影響原始 Bank
        batch_states = torch.index_select(self.states, 0, ids_lookup)
        batch_moments = torch.index_select(self.momentums, 0, ids_lookup)
        
        # 傳輸到 GPU (non_blocking=True 利用 CUDA Stream 加速，如果 memory pinned)
        return (
            batch_states.to(target_device, non_blocking=True),
            batch_moments.to(target_device, non_blocking=True)
        )

    def update_batch(self, 
                     user_ids: torch.Tensor, 
                     new_states: torch.Tensor, 
                     new_momentums: torch.Tensor):
        """
        寫入狀態 (Write) - 梯度防火牆
        
        Args:
            user_ids: [B]
            new_states: [B, H, Dh, Dh]
            new_momentums: [B, H, Dh, Dh]
        """
        # 1. 梯度截斷 (Gradient Detachment) - CRITICAL!
        #    這確保了 Stage 1 不會 OOM，Stage 2 的 BPTT 邊界清晰
        states_detached = new_states.detach()
        moments_detached = new_momentums.detach()
        
        # 2. 準備寫入數據 (移至儲存裝置)
        if self.enable_cpu_offload:
            ids_target = user_ids.cpu()
            states_target = states_detached.cpu()
            moments_target = moments_detached.cpu()
        else:
            ids_target = user_ids
            states_target = states_detached
            moments_target = moments_detached
            
        # 3. In-place Update
        #    注意：我們依賴 UniqueUserBatchSampler 來保證 user_ids 不重複。
        #    如果重複，index_copy_ 的行為是 Last Write Wins (通常符合時序)。
        self.states.index_copy_(0, ids_target, states_target)
        self.momentums.index_copy_(0, ids_target, moments_target)
        
    def restore_snapshot(self, snapshot: Tuple[torch.Tensor, torch.Tensor]):
        """
        從 Snapshot 還原整個 Bank 的狀態。
        用於 Intra-Period Training 的每個 Epoch 重置。
        """
        s_snap, m_snap = snapshot
        
        # 確保 snapshot 在 CPU (通常 create_snapshot 已經放 CPU 了)
        # 使用 copy_ 將數據寫回 self.states (無論它在 CPU 還是 GPU)
        self.states.copy_(s_snap)
        self.momentums.copy_(m_snap)

    def save_bank(self, dir_path: str, name: str = "memory_bank.pt"):
        """
        手動儲存 Bank (因為它不隨 model.state_dict 自動存)
        建議只在每個 Period 結束或極重要的 Checkpoint 儲存。
        """
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, name)
        print(f"--- [UserMemoryBank] Saving to {path} ...")
        torch.save({
            'states': self.states, # 這裡存的是 CPU Tensor
            'momentums': self.momentums
        }, path)
        print("--- [UserMemoryBank] Saved. ---")
        
    def load_bank(self, path: str):
        """手動載入 Bank"""
        if not os.path.exists(path):
            print(f"--- [UserMemoryBank] No checkpoint found at {path}. Skipping load. ---")
            return
            
        print(f"--- [UserMemoryBank] Loading from {path} ...")
        # map_location 指定到 CPU，避免直接載入到 GPU 爆顯存
        data = torch.load(path, map_location=self.store_device)
        self.states.copy_(data['states'])
        self.momentums.copy_(data['momentums'])
        print("--- [UserMemoryBank] Loaded. ---")

    # [新增 1] 為了 Stage 2 準備的 Snapshot (Offload to CPU)
    def create_snapshot(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """建立當前狀態的 CPU 快照，供 Stage 2 使用"""
        # clone() 是必須的，否則會參照到同一塊記憶體
        # detach() 確保斷開計算圖
        s_snap = self.states.detach().clone().cpu()
        m_snap = self.momentums.detach().clone().cpu()
        return (s_snap, m_snap)

    # [新增 2] 為了 Validation Update 準備的暫存備份
    def backup_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """備份當前狀態 (保持在原裝置，不轉 CPU 以求快)"""
        return (self.states.detach().clone(), self.momentums.detach().clone())

    # [新增 3] 還原備份
    def restore_state(self, backup: Tuple[torch.Tensor, torch.Tensor]):
        """還原狀態"""
        s_bk, m_bk = backup
        # 使用 copy_ 確保寫回原本的 buffer 位置
        self.states.copy_(s_bk)
        self.momentums.copy_(m_bk)



# ==============================================================================
#  Part A: JIT-Compiled Sequential Scan (The Engine)
#  這是數學核心，負責處理時序依賴 (Recurrence)
# ==============================================================================
@torch.jit.script
def titans_scan_jit(
    q_seq: torch.Tensor,      # [B, L, H, D]
    k_seq: torch.Tensor,      # [B, L, H, D]
    v_seq: torch.Tensor,      # [B, L, H, D]
    alpha_seq: torch.Tensor,  # [B, L, H, 1]
    eta_seq: torch.Tensor,    # [B, L, H, 1]
    theta_seq: torch.Tensor,  # [B, L, H, 1]
    init_state: torch.Tensor, # [B, H, D, D]
    init_momentum: torch.Tensor # [B, H, D, D]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Titans Memory Update Rule (JIT Optimized)
    
    Returns:
        y_seq: [B, L, H, D] (Memory Readout)
        state_seq: [B, L, H, D, D] (History of States)
        moment_seq: [B, L, H, D, D] (History of Momentums)
    """
    batch_size, seq_len, n_heads, head_dim = q_seq.shape
    
    # 容器：用來存每一步的結果
    y_list: List[torch.Tensor] = []
    state_list: List[torch.Tensor] = []
    moment_list: List[torch.Tensor] = []
    
    # 當前狀態 (Carry)
    curr_state = init_state
    curr_momentum = init_momentum
    
    # Sequential Loop (這在 Python 中很慢，但在 JIT 中會被優化)
    for t in range(seq_len):
        # 1. Slice Inputs (保持維度 [B, H, D])
        q_t = q_seq[:, t]
        k_t = k_seq[:, t]
        v_t = v_seq[:, t]
        alpha_t = alpha_seq[:, t].unsqueeze(-1) # [B, H, 1, 1]
        eta_t = eta_seq[:, t].unsqueeze(-1)     # [B, H, 1, 1]
        theta_t = theta_seq[:, t].unsqueeze(-1) # [B, H, 1, 1]
        
        # 2. Memory Read (Recall)
        #    y_t = State_{t-1} * q_t
        #    [B, H, D, D] @ [B, H, D, 1] -> [B, H, D, 1]
        q_vec = q_t.unsqueeze(-1)
        y_t = torch.matmul(curr_state, q_vec).squeeze(-1)
        y_list.append(y_t)
        
        # 3. Calculate Surprise (Gradient)
        #    Reconstruction: v_hat = State_{t-1} * k_t
        k_vec = k_t.unsqueeze(-1) # [B, H, D, 1]
        v_vec = v_t.unsqueeze(-1) # [B, H, D, 1]
        
        v_hat = torch.matmul(curr_state, k_vec)
        diff = v_hat - v_vec # Error signal
        
        #    Surprise = diff * k_t^T (Outer Product)
        #    [B, H, D, 1] @ [B, H, 1, D] -> [B, H, D, D]
        grad_t = torch.matmul(diff, k_vec.transpose(-1, -2))
        
        # 4. Update Momentum (Momentum-based Gradient Descent)
        #    S_t = eta * S_{t-1} - theta * grad_t
        #    注意：這裡是負梯度方向更新 (Gradient Descent)
        curr_momentum = eta_t * curr_momentum - theta_t * grad_t
        
        # 5. Update State
        #    M_t = (1 - alpha) * M_{t-1} + S_t
        curr_state = (1.0 - alpha_t) * curr_state + curr_momentum
        
        # 6. Store History
        state_list.append(curr_state)
        moment_list.append(curr_momentum)
        
    # Stack results along time dimension
    y_seq = torch.stack(y_list, dim=1)           # [B, L, H, D]
    state_seq = torch.stack(state_list, dim=1)   # [B, L, H, D, D]
    moment_seq = torch.stack(moment_list, dim=1) # [B, L, H, D, D]
    
    return y_seq, state_seq, moment_seq

# ==============================================================================
#  Part B: Neural Memory Cell (The Wrapper)
# ==============================================================================

class NeuralMemoryCell(nn.Module):
    """
    Titans Neural Memory Cell
    
    負責投影、參數生成、以及調用 JIT 核心。
    能夠自動處理 Step Mode (L=1) 與 Sequence Mode (L>1)。
    """
    def __init__(self, input_dim: int, n_heads: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        
        # Projections: Input -> Q, K, V
        # 為了效率，可以合併成一個大的 Linear，但分開寫比較清晰
        self.proj_q = nn.Linear(input_dim, n_heads * head_dim, bias=False)
        self.proj_k = nn.Linear(input_dim, n_heads * head_dim, bias=False)
        self.proj_v = nn.Linear(input_dim, n_heads * head_dim, bias=False)
        
        # Meta-Parameters Projections: Input -> Alpha, Eta, Theta (Per Head)
        # Output dim = 3 * n_heads (每個 Head 都有獨立的一組控制參數)
        self.proj_params = nn.Linear(input_dim, 3 * n_heads) 
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()

    def _init_weights(self):
        # Xavier Init for projections
        nn.init.xavier_uniform_(self.proj_q.weight)
        nn.init.xavier_uniform_(self.proj_k.weight)
        nn.init.xavier_uniform_(self.proj_v.weight)
        
        # Param Projection Initialization (Critical Heuristics)
        nn.init.xavier_uniform_(self.proj_params.weight)
        nn.init.zeros_(self.proj_params.bias)
        
        # 手動調整 Bias 以獲得穩定的初始行為
        # Params layout in dim -1: [alpha_head_0..N, eta_head_0..N, theta_head_0..N]
        with torch.no_grad():
            H = self.n_heads
            # Alpha (Forget): bias -2.0 => sigmoid(-2.0) ~= 0.12 (Low forget, High retention)
            self.proj_params.bias[0:H].fill_(-2.0)
            
            # Eta (Momentum): bias +2.0 => sigmoid(2.0) ~= 0.88 (High momentum)
            self.proj_params.bias[H:2*H].fill_(2.0)
            
            # Theta (LR): bias -3.0 => softplus(-3.0) ~= 0.05 (Conservative update)
            self.proj_params.bias[2*H:].fill_(-3.0)

    def forward(self, 
                inputs: torch.Tensor, 
                prev_state: torch.Tensor, 
                prev_momentum: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: [B, L, D] (Seq Mode) or [B, D] (Step Mode)
            prev_state: [B, H, Dh, Dh]
            prev_momentum: [B, H, Dh, Dh]
            
        Returns:
            readout: [B, L, D] or [B, D] - Flattened output
            final_states: [B, L, H, Dh, Dh] or [B, H, Dh, Dh]
            final_momentums: 同上
        """
        # 1. 維度標準化 (Step Mode -> Seq Mode L=1)
        is_step_mode = (inputs.dim() == 2)
        if is_step_mode:
            inputs = inputs.unsqueeze(1) # [B, 1, D]
            
        B, L, _ = inputs.shape
        H, Dh = self.n_heads, self.head_dim
        
        # 2. Projections
        #    [B, L, D] -> [B, L, H*Dh] -> [B, L, H, Dh]
        q = self.proj_q(inputs).view(B, L, H, Dh)
        k = self.proj_k(inputs).view(B, L, H, Dh)
        v = self.proj_v(inputs).view(B, L, H, Dh)
        
        #    Normalize Keys (重要：矩陣記憶體的數值穩定性)
        #    L2 Norm along the head_dim
        k = F.normalize(k, p=2, dim=-1)
        
        # 3. Meta-Parameters Generation
        params = self.proj_params(inputs) # [B, L, 3*H]
        #    Split into alpha, eta, theta
        alpha_raw, eta_raw, theta_raw = torch.split(params, H, dim=-1)
        
        #    Activations
        alpha = torch.sigmoid(alpha_raw).unsqueeze(-1)  # [B, L, H, 1]
        eta = torch.sigmoid(eta_raw).unsqueeze(-1)      # [B, L, H, 1]
        theta = F.softplus(theta_raw).unsqueeze(-1)     # [B, L, H, 1]
        
        # 4. Execution (Call JIT Kernel)
        #    無論是 L=1 還是 L=50，統一走 JIT 路徑
        y_seq, state_seq, moment_seq = titans_scan_jit(
            q, k, v, alpha, eta, theta, prev_state, prev_momentum
        )
        
        # 5. Output Handling
        #    Flatten Heads: [B, L, H, Dh] -> [B, L, H*Dh]
        readout = y_seq.view(B, L, H * Dh)
        readout = self.dropout(readout)
        
        if is_step_mode:
            # Step Mode: Remove sequence dimension (Return Step Tensors)
            return readout.squeeze(1), state_seq.squeeze(1), moment_seq.squeeze(1)
        else:
            # Seq Mode: Return Sequence Tensors
            return readout, state_seq, moment_seq

    
class FeatureWiseGatedFusion(nn.Module):
    """
    Titans Gating Mechanism (Feature-wise)
    
    融合三個來源的資訊：
    1. User Static (Identity / Persistent Memory)
    2. Short-term Context (SW-Attn Output)
    3. Long-term Memory (Neural Memory Output)
    
    公式：
    Gate = Sigmoid( Linear( Cat(Norm(User), Norm(Short), Norm(Long)) ) )
    Result = Gate * Norm(Short) + (1-Gate) * Norm(Long)
    Output = User + Dropout(Result)
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        # 1. Independent LayerNorms
        #    確保三路來源的數值分佈在同一水平線上，這對 Gate 的訓練穩定性極重要
        self.norm_static = nn.LayerNorm(d_model)
        self.norm_short = nn.LayerNorm(d_model)
        self.norm_long = nn.LayerNorm(d_model)
        
        # 2. Gate Generator
        #    Input: 3 * D (Concat) -> Output: D (Feature-wise Gate)
        #    使用 Feature-wise 而非 Scalar，允許模型在某些維度聽短期的，某些維度聽長期的
        self.gate_proj = nn.Linear(d_model * 3, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                user_static: torch.Tensor, 
                y_short: torch.Tensor, 
                y_long: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_static: [B, D] (User ID Embedding)
            y_short:     [B, L, D] (Seq Mode) or [B, D] (Step Mode)
            y_long:      [B, L, D] (Seq Mode) or [B, D] (Step Mode)
            
        Returns:
            user_dynamic: [B, L, D] or [B, D]
        """
        # 1. 自動維度偵測與廣播 (Broadcasting)
        #    判斷是否為 Sequence Mode (3維)
        is_sequence = (y_short.dim() == 3)
        
        if is_sequence:
            # User Static: [B, D] -> [B, 1, D] -> [B, L, D]
            B, L, _ = y_short.shape
            user_static_exp = user_static.unsqueeze(1).expand(-1, L, -1)
        else:
            # Step Mode: 保持 [B, D]
            user_static_exp = user_static

        # 2. Apply Normalization
        u_norm = self.norm_static(user_static_exp)
        s_norm = self.norm_short(y_short)
        l_norm = self.norm_long(y_long)
        
        # 3. Calculate Gate
        #    Concat along feature dim
        concat_feat = torch.cat([u_norm, s_norm, l_norm], dim=-1)
        
        #    Sigmoid: [0, 1]
        #    gate -> 1: Prefer Short-term
        #    gate -> 0: Prefer Long-term
        gate = torch.sigmoid(self.gate_proj(concat_feat))
        
        # 4. Weighted Fusion
        #    Feature-wise interpolation
        y_fused = gate * s_norm + (1.0 - gate) * l_norm
        
        # 5. Residual Connection
        #    將動態特徵 (Fused) 疊加在靜態特徵 (User ID) 之上
        return user_static_exp + self.dropout(y_fused)


class StandardItemTower(nn.Module):
    """
    Standard Item Tower: Transformer + MLP
    
    負責將 Item 及其 Context (歷史購買者序列) 編碼成向量。
    
    架構：
    1. Static Branch: ItemID + CateID -> Embedding
    2. Context Branch: User History -> Transformer -> Last Valid Pooling
    3. Fusion: Concat(Static, Context) -> MLP -> Output
    """
    def __init__(self, 
                 user_emb_layer: nn.Embedding, # 共用的 User Embedding (Reference)
                 item_emb_layer: nn.Embedding, # 共用的 Item Embedding (Reference)
                 cate_emb_layer: nn.Embedding, # 共用的 Category Embedding (Reference)
                 d_model: int, 
                 n_heads: int, 
                 max_hist_len: int,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # 1. 引用共用的 Embeddings
        #    注意：直接使用外部傳入的 layer，確保梯度同步
        self.user_emb = user_emb_layer
        self.item_emb = item_emb_layer
        self.cate_emb = cate_emb_layer
        
        # 2. Context Encoder Specifics
        #    Item Context 專用的 Positional Embedding
        self.pos_emb = nn.Embedding(max_hist_len, d_model)
        
        #    Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True # Pre-Norm 訓練更穩定
        )
        #    一層通常足夠捕捉共現關係，這對推論速度也很重要
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # 3. Fusion & MLP Projection
        #    Input: Static (D) + Context (D) = 2D
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model) # 壓縮回 D 維度以計算點積
        )
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.pos_emb.weight)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, 
                item_ids: torch.Tensor,        # [B]
                cate_ids: torch.Tensor,        # [B]
                hist_user_ids: torch.Tensor,   # [B, L_hist]
                hist_lens: torch.Tensor        # [B]
               ) -> torch.Tensor:
        """
        Args:
            hist_lens: 每個 Item 的歷史互動者數量，用於 Masking 和 Pooling
        """
        B, L_hist = hist_user_ids.shape
        device = item_ids.device
        
        # =================================================
        # Part A: Context Encoding (Transformer)
        # =================================================
        # 1. Embeddings & Position
        hist_emb = self.user_emb(hist_user_ids) # [B, L, D]
        
        positions = torch.arange(L_hist, device=device).unsqueeze(0)
        seq_input = hist_emb + self.pos_emb(positions)
        
        # 2. Masking
        #    產生 src_key_padding_mask (True 為 padding)
        #    [B, L_hist]
        seq_mask = torch.arange(L_hist, device=device).unsqueeze(0) >= hist_lens.unsqueeze(1)
        
        # 3. Encode
        ctx_output = self.transformer(seq_input, src_key_padding_mask=seq_mask)
        
        # 4. Last Valid Token Pooling (The Tricky Part)
        #    目標：取出 index = len - 1 的向量
        #    Handle Empty History: 若 len=0, index=-1 -> clamp(0) 取第0個 (防止 index error)
        last_idx = (hist_lens - 1).clamp(min=0)
        
        #    Gather 需要 index 維度匹配: [B, 1, D]
        gather_idx = last_idx.view(B, 1, 1).expand(-1, -1, self.d_model)
        ctx_pooled = torch.gather(ctx_output, 1, gather_idx).squeeze(1) # [B, D]
        
        #    Zero-Masking for Cold Items (Critical!)
        #    若 len == 0，將 context vector 強制設為 0，避免 transformer 的 padding noise 污染
        valid_mask = (hist_lens > 0).float().unsqueeze(-1) # [B, 1]
        ctx_pooled = ctx_pooled * valid_mask
        
        # =================================================
        # Part B: Static Encoding
        # =================================================
        i_static = self.item_emb(item_ids)
        c_static = self.cate_emb(cate_ids)
        static_emb = i_static + c_static
        
        # =================================================
        # Part C: Fusion & MLP
        # =================================================
        # [B, D] + [B, D] -> [B, 2D]
        combined = torch.cat([static_emb, ctx_pooled], dim=-1)
        
        # [B, 2D] -> [B, D]
        final_emb = self.mlp(combined)
        
        return final_emb



class TitansUserTower(nn.Module):
    """
    Titans User Tower (MAG Architecture)
    
    整合 SW-Attn (短期) 與 Neural Memory (長期) 的 User 模型。
    
    架構：
    1. Input: User ID, Item Seq (Static ID+Cate)
    2. Branch 1: SW-Attn (Transformer Encoder)
    3. Branch 2: Neural Memory (Titans Cell)
    4. Fusion: Gated Fusion -> User Dynamic Embedding
    """
    def __init__(self, 
                 user_emb_layer: nn.Embedding, # Shared
                 item_emb_layer: nn.Embedding, # Shared
                 cate_emb_layer: nn.Embedding, # Shared
                 d_model: int, 
                 n_heads: int, 
                 max_seq_len: int,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # 1. Shared Embeddings
        self.user_emb = user_emb_layer
        self.item_emb = item_emb_layer
        self.cate_emb = cate_emb_layer
        
        # 2. Short-term Specifics
        #    SW-Attn 需要 Positional Embedding
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        #    Transformer Encoder (SW-Attn)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4, 
            dropout=dropout,
            batch_first=True,
            norm_first=True # Pre-Norm
        )
        #    2層通常足夠捕捉短期 Session 的轉移規律
        self.sw_attn_block = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 3. Long-term Specifics
        #    Neural Memory Cell
        head_dim = d_model // n_heads
        self.memory_cell = NeuralMemoryCell(d_model, n_heads, head_dim, dropout)
        
        # 4. Fusion
        self.fusion_layer = FeatureWiseGatedFusion(d_model, dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.pos_emb.weight)
        # 其他層 (Transformer, Memory, Fusion) 已經在各自的 __init__ 裡初始化了

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """生成上三角 Mask (防止偷看未來)"""
        # triu: 上三角保留，其他為 0。我們需要 mask 掉上三角(未來)
        # PyTorch Transformer mask: float('-inf') 為遮蔽, 0.0 為保留
        mask = torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
        return mask

    def forward(self, 
                user_id: torch.Tensor, 
                item_seq: torch.Tensor, 
                cate_seq: torch.Tensor,
                state_dict: Dict[str, torch.Tensor], # {'state': ..., 'momentum': ...}
                padding_mask: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward Pass (支援 Step Mode 與 Seq Mode)
        
        Args:
            item_seq: [B, L] or [B, 1]
            state_dict: 包含 detach 過的 prev_state, prev_momentum
            padding_mask: [B, L] boolean mask (True = Padding)
            
        Returns:
            user_dynamic_emb: [B, L, D] (用於計算 Loss)
            new_states: [B, L, H, Dh, Dh] (Memory History)
            new_momentums: [B, L, H, Dh, Dh]
        """
        B, L = item_seq.shape
        device = item_seq.device
        
        # A. Feature Lookup (Static Embeddings)
        u_static = self.user_emb(user_id) # [B, D]
        i_emb = self.item_emb(item_seq)   # [B, L, D]
        c_emb = self.cate_emb(cate_seq)   # [B, L, D]
        
        # Composite Input Embedding
        # 這是 Memory 和 SW-Attn 共用的基底輸入
        base_input_emb = i_emb + c_emb 
        
        # =================================================
        # Branch 1: Short-term (SW-Attn)
        # =================================================
        # 1. Add Positional Embedding
        #    Step Mode (L=1) 時，positions 為 [0]，這合理 (代表 relative position 0)
        #    Seq Mode 時，positions 為 [0, 1, ..., L-1]
        #    注意：如果要嚴格做 Streaming Pos Emb，需要傳入當前 step index，
        #    但在 SW-Attn 中，主要關注 Window 內的相對位置，重置 Pos Emb 通常無害。
        positions = torch.arange(L, device=device).unsqueeze(0)
        sw_input = base_input_emb + self.pos_emb(positions)
        
        # 2. Causal Masking
        #    只有在 L > 1 時才需要 Causal Mask
        causal_mask = self._generate_causal_mask(L, device) if L > 1 else None
        
        # 3. Encoding
        #    src_key_padding_mask: True 為 padding
        y_short = self.sw_attn_block(
            sw_input, 
            mask=causal_mask, 
            src_key_padding_mask=padding_mask
        )
        
        # =================================================
        # Branch 2: Long-term (Neural Memory)
        # =================================================
        # 1. Input: 純 Base Input (No Pos Emb)
        mem_input = base_input_emb
        
        # 2. Call Cell (Auto-detects Step vs Sequence inside)
        y_long, new_states, new_momentums = self.memory_cell(
            mem_input, 
            state_dict['state'], 
            state_dict['momentum']
        )
        
        # =================================================
        # Fusion
        # =================================================
        # 結合 Static, Short, Long
        user_dynamic_emb = self.fusion_layer(u_static, y_short, y_long)
        
        return user_dynamic_emb, new_states, new_momentums


class DualTowerTitans(nn.Module):
    """
    Dual Tower Titans (The Orchestrator)
    
    整合 User Tower (Titans), Item Tower (Standard), Memory Bank。
    負責兩階段訓練策略的邏輯調度與 Loss 計算。
    """
    def __init__(self, global_meta: dict, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.n_users = global_meta['n_users']
        self.n_items = global_meta['n_items']
        self.n_cates = global_meta['n_cates']
        
        # Hyperparams
        model_cfg = cfg['model']
        d_model = model_cfg['item_embed_dim']
        n_heads = model_cfg.get('titans_n_heads', 4)
        max_seq_len = model_cfg.get('max_seq_len', 50)
        dropout = model_cfg.get('dropout', 0.1)
        self.context_len = model_cfg.get('context_len', 50) # K: SW-Attn Window
        self.temperature = model_cfg.get('temperature', 0.07)
        
        # 1. Shared Embeddings (The Foundation)
        self.user_emb_w = nn.Embedding(self.n_users, d_model, padding_idx=0)
        self.item_emb_w = nn.Embedding(self.n_items, d_model, padding_idx=0)
        self.cate_emb_w = nn.Embedding(self.n_cates, d_model, padding_idx=0)
        
        # Buffer for Category Mapping (Fast lookup)
        # 假設 global_meta['cate_matrix'] 是 numpy array [N_items, 1]
        self.register_buffer('cates', torch.from_numpy(global_meta['cate_matrix'][:, 0]).long())
        
        # 2. Item Cache Buffer (For Stage 2)
        #    persistent=False: 不存入 checkpoint
        self.register_buffer('item_cache', torch.zeros(self.n_items, d_model), persistent=False)

        # [FIX] User Tower 需要足夠大的 Positional Embedding
        # Stage 2 的最大長度是 K + L_max
        total_seq_capacity = self.context_len + max_seq_len
        
        # 3. Sub-Modules
        #    A. User Tower
        self.user_tower = TitansUserTower(
            user_emb_layer=self.user_emb_w,
            item_emb_layer=self.item_emb_w,
            cate_emb_layer=self.cate_emb_w,
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=total_seq_capacity,
            dropout=dropout
        )
        
        #    B. Item Tower
        self.item_tower = StandardItemTower(
            user_emb_layer=self.user_emb_w,
            item_emb_layer=self.item_emb_w,
            cate_emb_layer=self.cate_emb_w,
            d_model=d_model,
            n_heads=n_heads,
            max_hist_len=max_seq_len,
            dropout=dropout
        )
        
        #    C. Memory Bank
        self.memory_bank = UserMemoryBank(
            n_users=self.n_users, 
            n_heads=n_heads, 
            head_dim=d_model // n_heads,
            enable_cpu_offload=model_cfg.get('titans_cpu_offload', True)
        )
        
        self._init_weights()
        self.reset_item_cache_to_static()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_emb_w.weight)
        nn.init.xavier_uniform_(self.item_emb_w.weight)
        nn.init.xavier_uniform_(self.cate_emb_w.weight)

    # ==========================================================================
    #  Cache Management
    # ==========================================================================
    
    def reset_item_cache_to_static(self):
        """用 Static Embedding (ID + Cate) 初始化 Cache"""
        print("--- [DualTowerTitans] Initializing Item Cache with Static Embeddings... ---")
        with torch.no_grad():
            all_items = torch.arange(self.n_items, device=self.item_cache.device)
            all_cates = self._get_cate_ids(all_items)
            
            i_static = self.item_emb_w(all_items)
            c_static = self.cate_emb_w(all_cates)
            static_emb = i_static + c_static
    
    def _update_cache_batch(self, item_ids: torch.Tensor, embeddings: torch.Tensor):
        """
        Incremental Cache Update (During Stage 1)
        將當前 Batch 計算出的 Contextual Embedding 寫入 Cache
        """
        # Detach gradient to prevent memory leak in cache
        emb_detached = embeddings.detach()
        # Unique updates to save bandwidth (Last write wins)
        # 其實直接 index_copy 也可以，PyTorch handle duplicates
        self.item_cache.index_copy_(0, item_ids, emb_detached)

    # ==========================================================================
    #  Helper Methods
    # ==========================================================================

    def toggle_stage(self, stage: int):
        """切換訓練階段：控制參數凍結"""
        print(f"--- [DualTowerTitans] Switching to Stage {stage} ---")
        
        if stage == 1:
            # Unlock Everything
            for p in self.parameters():
                p.requires_grad = True
            self.train()
            
        elif stage == 2:
            # Freeze Representation, Train Memory Dynamics
            self.eval() # 先設為 eval (關閉 Dropout 以穩定特徵)
            
            for p in self.parameters():
                p.requires_grad = False
                
            # Unlock Meta-Controller & Fusion
            for p in self.user_tower.memory_cell.proj_params.parameters():
                p.requires_grad = True
            for p in self.user_tower.fusion_layer.parameters():
                p.requires_grad = True
                
            # 手動開啟這兩層的 train mode (若需要 Dropout)
            self.user_tower.memory_cell.train() 
            self.user_tower.fusion_layer.train()

    def _get_cate_ids(self, item_ids: torch.Tensor) -> torch.Tensor:
        safe_ids = item_ids.clamp(0, self.n_items - 1)
        return self.cates[safe_ids]

    def _generate_sliding_window_mask(self, sz: int, window_size: int, device: torch.device) -> torch.Tensor:
        """生成 Stage 2 SW-Attn 專用的帶狀遮罩"""
        mask = torch.zeros(sz, sz, device=device)
        future_mask = torch.triu(torch.ones_like(mask), diagonal=1)
        past_mask = torch.tril(torch.ones_like(mask), diagonal=-(window_size + 1))
        combined_mask = future_mask + past_mask
        return mask.masked_fill(combined_mask.bool(), float('-inf'))
    
    def _get_collision_mask(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Shared Collision Mask Logic: [B, B]"""
        # Mask out same item IDs, keep diagonal
        collision_mask = (item_ids.unsqueeze(1) == item_ids.unsqueeze(0))
        collision_mask.fill_diagonal_(False)
        return collision_mask

    def compute_loss_with_collision_mask(self, user_emb, item_emb, item_ids) -> torch.Tensor:
        logits = torch.matmul(user_emb, item_emb.t()) / self.temperature
        collision_mask = self._get_collision_mask(item_ids)
        logits.masked_fill_(collision_mask, -1e9)
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    
    # ==========================================================================
    #  In-Model Metrics (New!)
    # ==========================================================================
    def _get_collision_mask(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        計算 Collision Mask，但 [重要] 排除對角線 (Self-Collision)。
        對角線是正樣本，絕不能被 Mask 成 -inf。
        """
        # [B, 1] == [1, B] -> [B, B]
        collision_mask = (item_ids.unsqueeze(1) == item_ids.unsqueeze(0))
        
        # [FIX] 強制將對角線設為 False (保留正樣本)
        # fill_diagonal_ 是 in-place 操作
        collision_mask.fill_diagonal_(False)
        
        return collision_mask

    def compute_metrics(self, user_emb: torch.Tensor, item_emb: torch.Tensor, item_ids: torch.Tensor, k_list: List[int] = [10, 20]) -> Dict[str, float]:
        B = user_emb.size(0)
        
        # [Safety] Batch 太小無法計算有意義的指標，回傳空字典 (Trainer 會忽略)
        if B < 10:
            return {}

        # 1. Normalize
        user_emb = F.normalize(user_emb, p=2, dim=1, eps=1e-6)
        item_emb = F.normalize(item_emb, p=2, dim=1, eps=1e-6)
        
        # 2. Scores [B, B]
        scores = torch.matmul(user_emb, item_emb.t())
        
        # 3. Masking
        #    取得正確的 Mask (對角線為 False)
        collision_mask = self._get_collision_mask(item_ids)
        
        #    將衝突負樣本設為 -inf (排到最後)
        scores.masked_fill_(collision_mask, float('-inf'))
        
        # 4. AUC Logic
        #    對角線是 Target Score
        target_scores = torch.diag(scores).unsqueeze(1) # [B, 1]
        
        #    比較：誰的分數比 Target 高？
        #    (-inf 的負樣本這裡會是 False，正確)
        greater_mask = (scores > target_scores)
        ranks_0based = greater_mask.sum(dim=1).float()
        
        #    有效負樣本數 = (B - 1) - 衝突數
        num_collisions = collision_mask.sum(dim=1).float()
        num_negatives = (B - 1) - num_collisions
        num_negatives.clamp_(min=1.0) # 防除以零
        
        auc_per_user = 1.0 - (ranks_0based / num_negatives)
        mean_auc = auc_per_user.mean().item()
        
        # [FIX] 初始化 metrics 字典
        metrics = {'auc': mean_auc}

        # 5. Top-K Logic
        #    [Debug] 確保 k_list 不為空
        if not k_list:
            k_list = [10, 20]

        max_k = max(k_list)
        candidate_size = scores.size(1)
        safe_k = min(max_k, candidate_size)
        
        #    TopK
        _, topk_indices = torch.topk(scores, k=safe_k, dim=1)
        
        #    Hit Matrix
        targets = torch.arange(B, device=scores.device).unsqueeze(1)
        hit_matrix = (topk_indices == targets)
        
        for k in k_list:
            # 處理 k 比 batch size 大的情況
            actual_limit = min(k, safe_k)
            
            # Recall
            hits_at_k = hit_matrix[:, :actual_limit].any(dim=1).float()
            metrics[f'recall@{k}'] = hits_at_k.mean().item()
            
            # NDCG
            vals = hit_matrix[:, :actual_limit].float()
            rank_pos = torch.arange(1, actual_limit + 1, device=scores.device).unsqueeze(0)
            discounts = 1.0 / torch.log2(rank_pos + 1.0)
            dcg = (vals * discounts).sum(dim=1)
            metrics[f'ndcg@{k}'] = dcg.mean().item()
            
        return metrics

    # ==========================================================================
    #  Forward: Stage 1 (Real-time Step Learning)
    # ==========================================================================

    def forward_stage1(self, 
                       batch: Dict, 
                       training: bool = True, 
                       update_cache: bool = False,
                       compute_metrics: bool = False,
                       k_list: List[int] = [10, 20]
                       ) -> Dict:
        """
        Stage 1: Step-based Learning
        Args:
            training: Controls Memory Bank Update (True=Update, False=Read-Only)
            update_cache: Whether to write item embeddings to global cache (Incremental Update)
            compute_metrics: Whether to return Recall/NDCG
        """
        user_ids = batch['user_id']
        target_item_id = batch['item_id']
        user_seq_window = batch['user_interacted_items']
        
        # 1. Read Memory Bank
        prev_states, prev_moments = self.memory_bank.get_batch(user_ids, user_ids.device)
        state_dict = {'state': prev_states, 'momentum': prev_moments}
        
        # 2. User Tower Logic (Manually split steps)
        #    A. Embeddings
        u_static = self.user_emb_w(user_ids)
        i_emb = self.item_emb_w(user_seq_window)
        c_emb = self.cate_emb_w(self._get_cate_ids(user_seq_window))
        base_input = i_emb + c_emb
        
        #    B. Titans (Last Step)
        titans_input = base_input[:, -1:, :] 
        y_long, new_states, new_moments = self.user_tower.memory_cell(
            titans_input, state_dict['state'], state_dict['momentum']
        )
        if y_long.dim() == 3: y_long = y_long.squeeze(1)
        
        #    C. SW-Attn (Full Window)
        positions = torch.arange(user_seq_window.size(1), device=user_ids.device).unsqueeze(0)
        sw_input = base_input + self.user_tower.pos_emb(positions)
        padding_mask = (user_seq_window == 0)
        y_short_all = self.user_tower.sw_attn_block(sw_input, src_key_padding_mask=padding_mask)
        y_short = y_short_all[:, -1, :]
        
        #    D. Fusion
        user_emb = self.user_tower.fusion_layer(u_static, y_short, y_long)
        
        # 3. Item Tower
        target_cate = self._get_cate_ids(target_item_id)
        target_item_emb = self.item_tower(
            item_ids=target_item_id,
            cate_ids=target_cate,
            hist_user_ids=batch['item_interacted_users'],
            hist_lens=batch['item_interacted_len'],
        )
        
        # 4. Operations
        results = {}
        
        # Loss
        loss = self.compute_loss_with_collision_mask(user_emb, target_item_emb, target_item_id)
        results['loss'] = loss
        
        # Memory Update (Controlled by flag)
        if training:
            if new_states.dim() == 5:
                new_states = new_states.squeeze(1)
                new_moments = new_moments.squeeze(1)
            self.memory_bank.update_batch(user_ids, new_states, new_moments)
            
        # Cache Update (Incremental)
        if update_cache:
            self._update_cache_batch(target_item_id, target_item_emb)
            
        # Metrics
        if compute_metrics:
            metrics = self.compute_metrics(user_emb, target_item_emb, target_item_id, k_list)
            results.update(metrics)
            
        # Return Embeddings for potential external use
        results['user_emb'] = user_emb
        results['item_emb'] = target_item_emb
        
        return results

    # ==========================================================================
    #  Forward: Stage 2 (Dense Sequence Memory Learning)
    # ==========================================================================

    def forward_stage2(self, batch: Dict, initial_states_snapshot: tuple) -> Dict:
        """
        Stage 2: Sequence-based Memory Dynamics Learning
        Input Structure: [ Context (K) | Period (L_max) ]
        - Context: Pre-padded (靠右對齊)
        - Period: Post-padded (靠左對齊)
        """
        user_ids = batch['user_id']
        # full_seq: [B, K + L_max]
        full_seq = batch['full_seq_items']
        mask_label = batch['loss_mask']
        
        K = self.context_len # 例如 50
        
        # 1. 切分序列
        #    Titans 只負責跑 Period 部分 (從 K 開始到最後)
        #    seq_for_titans: [B, L_max]
        seq_for_titans = full_seq[:, K:] 
        
        # -------------------------------------------------------
        # Part A: Titans Path (從 Snapshot 開始)
        # -------------------------------------------------------
        full_states_cpu, full_moments_cpu = initial_states_snapshot
        
        # user_ids 在 GPU，需轉回 CPU 做 index
        ids_cpu = user_ids.cpu()
        
        # Gather & Move to GPU
        # 注意：Snapshot 是 tensor，直接 indexing 會 copy 資料
        batch_states = full_states_cpu[ids_cpu].to(user_ids.device)
        batch_moments = full_moments_cpu[ids_cpu].to(user_ids.device)
        
        state_dict = {'state': batch_states, 'momentum': batch_moments}
        
        i_emb = self.item_emb_w(seq_for_titans)
        c_emb = self.cate_emb_w(self._get_cate_ids(seq_for_titans))
        titans_input_emb = i_emb + c_emb
        
        # Run Memory Cell
        # 注意：這裡雖然會跑過 Post-Padding，但 JIT 運算是並行的。
        # 我們靠最後的 Loss Mask 來確保 Padding 不影響梯度。
        y_long, _, _ = self.user_tower.memory_cell(
            titans_input_emb, 
            state_dict['state'], 
            state_dict['momentum']
        )
        # y_long: [B, L_max, D]
        
        # -------------------------------------------------------
        # Part B: SW-Attn Path (看全部 Context + Period)
        # -------------------------------------------------------
        # SW-Attn 需要看到 Context 才能預測 Period 的第一個 token
        i_emb_full = self.item_emb_w(full_seq)
        c_emb_full = self.cate_emb_w(self._get_cate_ids(full_seq))
        base_input_full = i_emb_full + c_emb_full
        
        # Pos Emb
        positions = torch.arange(full_seq.size(1), device=full_seq.device).unsqueeze(0)
        sw_input = base_input_full + self.user_tower.pos_emb(positions)
        
        # Masks
        # 1. Sliding Window Mask (限制視野)
        sw_mask = self._generate_sliding_window_mask(full_seq.size(1), K, full_seq.device)
        # 2. Padding Mask (忽略 Pad) - 重要！這會忽略 Context 的 Pre-pad 和 Period 的 Post-pad
        padding_mask_full = (full_seq == 0)
        
        y_short_full = self.user_tower.sw_attn_block(
            sw_input, 
            mask=sw_mask, 
            src_key_padding_mask=padding_mask_full
        )
        
        # Slice: 取出與 Titans 對齊的部分 (即 Period 部分)
        # y_short: [B, L_max, D]
        y_short = y_short_full[:, K:, :]
        
        # -------------------------------------------------------
        # Part C: Fusion & Loss
        # -------------------------------------------------------
        u_static = self.user_emb_w(user_ids)
        
        # Fusion
        user_dynamic = self.user_tower.fusion_layer(u_static, y_short, y_long)
        
        # Prepare Targets (Next Item Prediction)
        # Input:  [p1, p2, p3, 0]
        # Target: [p2, p3, 0,  0]
        # 我們要預測 seq_for_titans 的下一個 token
        
        # 預測值: 砍掉最後一個 (因為沒有下一個可以對答案)
        u_pred = user_dynamic[:, :-1, :].reshape(-1, self.cfg['model']['item_embed_dim'])
        
        # 真實值: 往左移一格 (從第2個開始)
        target_ids = seq_for_titans[:, 1:].reshape(-1)
        
        # [FIX] 使用 Dataset 傳來的 Mask 進行 Loss 過濾
        # Mask shape: [B, K+L]. 我們只要 Period 部分 [B, K:]
        # 並且因為 Shift，mask 也要 shift: mask[:, K+1:]
        # 注意維度對齊：
        # seq_for_titans 是 [B, L]
        # target_ids 是 seq_for_titans[:, 1:] -> 長度 L-1
        # 所以 mask 應該取 mask_label[:, K+1:] -> 長度 L-1
        
        # Flatten Mask
        target_mask = mask_label[:, K+1:].reshape(-1)
        
        # Final Valid Mask: (Not Padding in Data) AND (Valid Period in Mask)
        # 其實 Dataset 的 Mask 已經處理了 Padding (設為0)，所以直接用 mask 即可
        # 但為了保險，加上 target_ids != 0
        valid_mask = (target_mask > 0.5) & (target_ids != 0)
        
        if valid_mask.sum() == 0:
             return {'loss': torch.tensor(0.0, device=full_seq.device, requires_grad=True)}

        # Filter
        u_valid = u_pred[valid_mask]
        ids_valid = target_ids[valid_mask]
        
        # Lookup Cache (Stage 2 使用 Cache 來加速 Target Embedding lookup)
        i_valid = F.embedding(ids_valid, self.item_cache)
        
        # Compute Flattened InfoNCE
        loss = self.compute_loss_with_collision_mask(u_valid, i_valid, ids_valid)
        
        return {'loss': loss}
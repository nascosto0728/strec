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
# 輔助函式 
# ===================================================================
    
def average_pooling(embeddings: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    """
    對 Embedding 進行帶遮罩的平均池化。
    """
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
    
    else:
        raise ValueError(f"Unsupported embedding dimension: {embeddings.dim()}")


class EmbMLP(nn.Module):
    """
    PAM 模型的基礎版本 (Base MLP)。
    [Cleaned]: 移除了 item_init_vectors, cate_init_vectors 參數
    """
    def __init__(self, cates: np.ndarray, cate_lens: np.ndarray, hyperparams: Dict, train_config: Dict): 
        super().__init__()
        self.hparams = hyperparams
        self.train_config = train_config
        self.temperature = 0.07
        
        self.register_buffer('cates', torch.from_numpy(cates))
        self.register_buffer('cate_lens', torch.tensor(cate_lens, dtype=torch.int32))

        # --- 建立 Embedding Table ---
        self.user_emb_w = nn.Embedding(self.hparams['n_users'], self.hparams['user_embed_dim'])
        self.item_emb_w = nn.Embedding(self.hparams['n_items'], self.hparams['item_embed_dim'])
        self.cate_emb_w = nn.Embedding(self.hparams['n_cates'], self.hparams['cate_embed_dim'])

        # Buffer: 這裡假設 dim = item_embed_dim，與原始程式碼保持一致
        history_embed_dim = self.hparams['item_embed_dim'] 
        self.user_history_buffer = nn.Embedding(self.hparams['n_items'], history_embed_dim)
        self.user_history_buffer.weight.requires_grad = False
        
        # --- 建立 MLP (保留原本的雙路徑結構以維持穩定性) ---
        concat_dim = self.hparams['user_embed_dim'] + self.hparams['item_embed_dim'] + self.hparams['cate_embed_dim'] 
        layer_num = 1
        
        self.user_mlp = nn.Sequential(*[self.PreNormResidual(concat_dim, self._build_mlp_layers(concat_dim))]*layer_num)
        self.item_mlp = nn.Sequential(*[self.PreNormResidual(concat_dim, self._build_mlp_layers(concat_dim))]*layer_num)
        self.user_mlp_2 = nn.Sequential(*[self.PreNormResidual(concat_dim, self._build_mlp_layers(concat_dim))]*layer_num)
        self.item_mlp_2 = nn.Sequential(*[self.PreNormResidual(concat_dim, self._build_mlp_layers(concat_dim))]*layer_num)

        self._init_weights()

    class PreNormResidual(nn.Module):
        def __init__(self, dim, fn):
            super().__init__()
            self.fn = fn
            self.norm = nn.LayerNorm(dim)

        def forward(self, x):
            return self.fn(self.norm(x)) + x
            
    def _build_mlp_layers(self, dim, expansion_factor = 2, dropout = 0., dense = nn.Linear):
        inner_dim = int(dim * expansion_factor)
        return nn.Sequential(
            dense(dim, inner_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            dense(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_emb_w.weight)
        nn.init.xavier_uniform_(self.item_emb_w.weight)
        nn.init.xavier_uniform_(self.cate_emb_w.weight)
        for mlp in [self.user_mlp, self.item_mlp, self.user_mlp_2, self.item_mlp_2]:
            for layer in mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def _build_feature_representations(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        # User Tower
        static_u_emb = self.user_emb_w(batch['users'])
        hist_item_emb = self.item_emb_w(batch['item_history_matrix'])
        hist_cates = self.cates[batch['item_history_matrix']]
        hist_cates_len = self.cate_lens[batch['item_history_matrix']]
        hist_cates_emb = self.cate_emb_w(hist_cates)
        avg_cate_emb_for_hist = average_pooling(hist_cates_emb, hist_cates_len)
        hist_item_emb_with_cate = torch.cat([hist_item_emb, avg_cate_emb_for_hist], dim=2)
        
        # [Stability] 保留 .detach()
        user_history_emb = average_pooling(hist_item_emb_with_cate, batch['item_history_len']).detach()
        user_features = torch.cat([static_u_emb, user_history_emb], dim=-1)

        # Item Tower
        static_item_emb = self.item_emb_w(batch['items'])
        item_cates = self.cates[batch['items']]
        item_cates_len = self.cate_lens[batch['items']]
        item_cates_emb = self.cate_emb_w(item_cates)
        avg_cate_emb_for_item = average_pooling(item_cates_emb, item_cates_len)
        item_emb_with_cate = torch.cat([static_item_emb, avg_cate_emb_for_item], dim=1)
        item_history_user_emb = self.user_emb_w(batch['user_history_matrix'])
        
        # [Stability] 保留 .detach()
        item_history_emb = average_pooling(item_history_user_emb, batch['user_history_len']).detach()
        item_features = torch.cat([item_emb_with_cate, item_history_emb], dim=-1)

        if self.training:
            item_ids = batch['items']
            self.user_history_buffer.weight[item_ids] = item_history_emb.detach()

        return user_features, item_features

    def _get_embeddings_from_features(self, user_features: torch.Tensor, item_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        user_embedding = self.user_mlp(user_features) + self.user_mlp_2(user_features)
        item_embedding = self.item_mlp(item_features) + self.item_mlp_2(item_features)
        user_embedding = F.normalize(user_embedding, p=2, dim=-1)
        item_embedding = F.normalize(item_embedding, p=2, dim=-1)

        return user_embedding, item_embedding

    def forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        user_features, item_features = self._build_feature_representations(batch)
        user_embedding, item_embedding = self._get_embeddings_from_features(user_features, item_features)
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
    
    def calculate_loss(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [Cleaned]: 移除了 teacher_model 和 kd_weight 參數
        """
        user_embedding, item_embedding = self.forward(batch)
        losses = self._calculate_infonce_loss(user_embedding, item_embedding, batch['labels'])
        final_loss = losses.mean()
        return final_loss
        
    def inference(self, batch: Dict[str, torch.Tensor], neg_item_ids_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_features, item_features = self._build_feature_representations(batch)
        pos_user_emb, pos_item_emb = self._get_embeddings_from_features(user_features, item_features)
        pos_logits = torch.sum(pos_user_emb * pos_item_emb, dim=1)
        per_sample_loss = F.binary_cross_entropy_with_logits(pos_logits, batch['labels'].float(), reduction='none')

        num_neg_samples = neg_item_ids_batch.shape[1]
        neg_item_static_emb = self.item_emb_w(neg_item_ids_batch)
        neg_item_cate_emb = self.cate_emb_w(self.cates[neg_item_ids_batch])
        neg_item_cates_len = self.cate_lens[neg_item_ids_batch]
        avg_cate_emb_for_neg_item = average_pooling(neg_item_cate_emb, neg_item_cates_len)
        neg_item_history_emb = self.user_history_buffer(neg_item_ids_batch)
        neg_item_features = torch.cat([neg_item_static_emb, avg_cate_emb_for_neg_item, neg_item_history_emb], dim=2)
        
        user_features_expanded = user_features.unsqueeze(1).expand(-1, num_neg_samples, -1)
        neg_user_emb_final, neg_item_emb_final = self._get_embeddings_from_features(user_features_expanded, neg_item_features)
        neg_logits = torch.sum(neg_user_emb_final * neg_item_emb_final, dim=2)
        
        return pos_logits, neg_logits, per_sample_loss


class SASRec_MLP(EmbMLP):
    """
    SASRec Upgrade: 使用 Transformer Encoder 替換 AvgPool。
    """
    def __init__(self, cates: np.ndarray, cate_lens: np.ndarray, hyperparams: Dict, train_config: Dict): 
        
        super().__init__(cates, cate_lens, hyperparams, train_config)

        self.item_dim = self.hparams['item_embed_dim']
        self.user_dim = self.hparams['user_embed_dim']
        self.cate_dim = self.hparams['cate_embed_dim']
        
        self.item_seq_input_dim = self.item_dim + self.cate_dim
        self.user_seq_input_dim = self.user_dim
        
        self.maxlen = 30
        n_heads = self.hparams.get('transformer_n_heads', 4)
        n_layers = self.hparams.get('transformer_n_layers', 2)
        dropout = self.hparams.get('transformer_dropout', 0.1)
        
        # --- 模組 A: itemSeq ---
        dim_A = self.item_seq_input_dim
        self.item_seq_pos_emb = nn.Embedding(self.maxlen, dim_A)
        if dim_A % n_heads != 0:
            n_heads_A = next(h for h in [4, 2, 1] if dim_A % h == 0)
        else:
            n_heads_A = n_heads
        transformer_layer_A = nn.TransformerEncoderLayer(d_model=dim_A, nhead=n_heads_A, dim_feedforward=dim_A * 4, dropout=dropout, batch_first=True)
        self.item_seq_transformer = nn.TransformerEncoder(transformer_layer_A, num_layers=n_layers)

        # --- 模組 B: userSeq ---
        dim_B = self.user_seq_input_dim
        self.user_seq_pos_emb = nn.Embedding(self.maxlen, dim_B)
        if dim_B % n_heads != 0:
            n_heads_B = next(h for h in [4, 2, 1] if dim_B % h == 0)
        else:
            n_heads_B = n_heads
        transformer_layer_B = nn.TransformerEncoderLayer(d_model=dim_B, nhead=n_heads_B, dim_feedforward=dim_B * 4, dropout=dropout, batch_first=True)
        self.user_seq_transformer = nn.TransformerEncoder(transformer_layer_B, num_layers=n_layers)


    def _run_transformer_encoder(self, seq_emb: torch.Tensor, seq_lens: torch.Tensor, transformer_module: nn.TransformerEncoder, pos_emb_module: nn.Embedding) -> torch.Tensor:
        B, T, D = seq_emb.shape
        device = seq_emb.device
        padding_mask = torch.arange(T, device=device)[None, :] >= seq_lens[:, None]
        pos_ids = torch.arange(T, dtype=torch.long, device=device)
        pos_embeddings = pos_emb_module(pos_ids).unsqueeze(0) 
        seq_emb_with_pos = seq_emb + pos_embeddings
        transformer_output = transformer_module(src=seq_emb_with_pos, src_key_padding_mask=padding_mask)
        pooled_output = average_pooling(transformer_output, seq_lens)
        return pooled_output

    def _build_feature_representations(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        # User Tower
        static_u_emb = self.user_emb_w(batch['users'])
        hist_item_emb = self.item_emb_w(batch['item_history_matrix'])
        hist_cates = self.cates[batch['item_history_matrix']]
        hist_cates_len = self.cate_lens[batch['item_history_matrix']]
        hist_cates_emb = self.cate_emb_w(hist_cates)
        avg_cate_emb_for_hist = average_pooling(hist_cates_emb, hist_cates_len) 
        hist_item_emb_with_cate = torch.cat([hist_item_emb, avg_cate_emb_for_hist], dim=2) 
        user_history_emb = self._run_transformer_encoder(hist_item_emb_with_cate, batch['item_history_len'], self.item_seq_transformer, self.item_seq_pos_emb)
        
        # [Stability] 保留 .detach()，這很重要！
        user_features = torch.cat([static_u_emb, user_history_emb.detach()], dim=-1)

        # Item Tower
        static_item_emb = self.item_emb_w(batch['items'])
        item_cates = self.cates[batch['items']]
        item_cates_len = self.cate_lens[batch['items']]
        item_cates_emb = self.cate_emb_w(item_cates)
        avg_cate_emb_for_item = average_pooling(item_cates_emb, item_cates_len)
        item_emb_with_cate = torch.cat([static_item_emb, avg_cate_emb_for_item], dim=1)
        item_history_user_emb = self.user_emb_w(batch['user_history_matrix'])
        item_history_emb = self._run_transformer_encoder(item_history_user_emb, batch['user_history_len'], self.user_seq_transformer, self.user_seq_pos_emb)
        
        # [Stability] 保留 .detach()
        item_features = torch.cat([item_emb_with_cate, item_history_emb.detach()], dim=-1)

        # Cache Update
        item_history_emb_for_cache = item_features[:, -self.user_seq_input_dim:] 
        if self.training:
            self.user_history_buffer.weight[batch['items']] = item_history_emb_for_cache.detach()
        
        return user_features, item_features
    
    def calculate_loss(self, batch: Dict) -> torch.Tensor:
        student_session_emb, student_item_emb = self.forward(batch)
        loss_infonce = self._calculate_infonce_loss(student_session_emb, student_item_emb, batch['labels'])
        loss_main = loss_infonce.mean()
        return loss_main

    def inference(self, batch: Dict[str, torch.Tensor], neg_item_ids_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_features, item_features = self._build_feature_representations(batch)
        pos_user_emb, pos_item_emb = self._get_embeddings_from_features(user_features, item_features)
        pos_logits = torch.sum(pos_user_emb * pos_item_emb, dim=1)
        per_sample_loss = F.binary_cross_entropy_with_logits(pos_logits, batch['labels'].float(), reduction='none')

        num_neg_samples = neg_item_ids_batch.shape[1]
        neg_item_static_emb = self.item_emb_w(neg_item_ids_batch) 
        neg_item_cate_emb = self.cate_emb_w(self.cates[neg_item_ids_batch])
        neg_item_cates_len = self.cate_lens[neg_item_ids_batch]
        avg_cate_emb_for_neg_item = average_pooling(neg_item_cate_emb, neg_item_cates_len) 
        neg_item_emb_with_cate = torch.cat([neg_item_static_emb, avg_cate_emb_for_neg_item], dim=2) 
        
        item_history_emb_expanded = self.user_history_buffer(neg_item_ids_batch)
        neg_item_features = torch.cat([neg_item_emb_with_cate, item_history_emb_expanded.detach()], dim=2)
        user_features_expanded = user_features.unsqueeze(1).expand(-1, num_neg_samples, -1)
        neg_user_emb_final, neg_item_emb_final = self._get_embeddings_from_features(user_features_expanded, neg_item_features)
        neg_logits = torch.sum(neg_user_emb_final * neg_item_emb_final, dim=2)
        
        return pos_logits, neg_logits, per_sample_loss

class HyperLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, dropout=0.0):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.lora_up = nn.Linear(r, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight) 
        
    def forward(self, x, context_gate):
        base_out = self.base(x)
        down_out = self.lora_down(x)
        if context_gate.dim() == 2:
            gate = context_gate.unsqueeze(1) 
        else:
            gate = context_gate
        gated_down = down_out * gate
        lora_out = self.lora_up(gated_down)
        return base_out + self.dropout(lora_out)
    

class HyperLoRATransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, lora_r=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = HyperLoRALinear(d_model, dim_feedforward, r=lora_r, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = HyperLoRALinear(dim_feedforward, d_model, r=lora_r, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src, context_gate, src_mask=None, src_key_padding_mask=None):
        src2 = self.norm1(src)
        attn_output, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_output)
        src2 = self.norm2(src)
        ffn_out = self.linear1(src2, context_gate) 
        ffn_out = self.activation(ffn_out)
        ffn_out = self.dropout(ffn_out)
        ffn_out = self.linear2(ffn_out, context_gate)
        src = src + self.dropout2(ffn_out)
        return src
    
class HyperLoRASASRec(SASRec_MLP):
    """
    Hyper-LoRA SASRec:
    [Cleaned]
    1. 移除未使用參數
    2. 清理繼承造成的雙重初始化 (刪除父類 transformer)
    3. 保持 .detach() 確保數值穩定性
    """
    def __init__(self, cates: np.ndarray, cate_lens: np.ndarray, hyperparams: Dict, train_config: Dict): 
        
        super().__init__(cates, cate_lens, hyperparams, train_config)
        print("[HyperLoRASASRec] Initialized. FFN weights are modulated by Context.")
        
        # [Optimization] 刪除父類初始化的標準 Transformer，避免記憶體浪費
        if hasattr(self, 'item_seq_transformer'): del self.item_seq_transformer
        if hasattr(self, 'user_seq_transformer'): del self.user_seq_transformer
        
        self.lora_r = self.hparams.get('lora_r', 16)
        self.context_dim = self.hparams['item_embed_dim']
        
        self.hyper_gate_net = nn.Sequential(
            nn.Linear(self.context_dim, self.lora_r),
            nn.Tanh(), 
            nn.Linear(self.lora_r, self.lora_r),
            nn.Tanh()
        )
        
        # 模組 A
        dim_A = self.item_seq_input_dim
        layers_A = nn.ModuleList([
            HyperLoRATransformerLayer(d_model=dim_A, nhead=self.hparams.get('transformer_n_heads', 4), dim_feedforward=dim_A * 4, dropout=self.hparams.get('transformer_dropout', 0.1), lora_r=self.lora_r)
            for _ in range(self.hparams.get('transformer_n_layers', 2))
        ])
        self.item_seq_transformer_hyper = layers_A 

        # 模組 B
        dim_B = self.user_seq_input_dim
        layers_B = nn.ModuleList([
            HyperLoRATransformerLayer(d_model=dim_B, nhead=self.hparams.get('transformer_n_heads', 4), dim_feedforward=dim_B * 4, dropout=self.hparams.get('transformer_dropout', 0.1), lora_r=self.lora_r)
            for _ in range(self.hparams.get('transformer_n_layers', 2))
        ])
        self.user_seq_transformer_hyper = layers_B
        
        self.register_buffer('global_context_ema', torch.zeros(self.context_dim))
        self.ema_alpha = 0.95


    def _update_context_ema(self, current_batch_context):
        self.global_context_ema.data.mul_(self.ema_alpha).add_(current_batch_context.detach(), alpha=(1.0 - self.ema_alpha))

    def _run_hyper_transformer(self, seq_emb, seq_lens, context_vector, layers_module_list, pos_emb_module):
        B, T, D = seq_emb.shape
        device = seq_emb.device
        
        # 1. Generate Gate
        gate = self.hyper_gate_net(context_vector.unsqueeze(0)).expand(B, -1) # [B, r]
        
        # 2. Positional Embedding
        pos_ids = torch.arange(T, dtype=torch.long, device=device)
        seq_input = seq_emb + pos_emb_module(pos_ids).unsqueeze(0)
        
        # 3. Masks
        # Causal Mask (阻止看到未來)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device).bool()
        
        # [FIX 2] 防止 seq_lens 為 0 導致 Attention NaN
        # 如果 seq_lens 為 0，我們強制讓它變成 1 (只為了計算 mask 不報錯)
        # 真正計算完後，我們再用 zero_len_mask 把結果蓋成 0
        safe_lens = torch.clamp(seq_lens, min=1) 
        padding_mask = (torch.arange(T, device=device)[None, :] >= safe_lens[:, None])
        
        # 4. Forward Layers
        output = seq_input
        for layer in layers_module_list:
            output = layer(output, context_gate=gate, src_mask=causal_mask, src_key_padding_mask=padding_mask)
            
        # 5. Gather Last Valid Item
        # 處理 seq_len 為 0 的情況
        target_indices = (seq_lens - 1).clamp(min=0).view(B, 1, 1).expand(-1, 1, D)
        pooled_output = torch.gather(output, 1, target_indices).squeeze(1)
        
        # 確保 seq_len=0 的輸出為 0 (這一步會切斷 NaN 的傳播)
        mask_zero = (seq_lens == 0).unsqueeze(-1).expand(-1, D)
        pooled_output = pooled_output.masked_fill(mask_zero, 0.0)
        
        return pooled_output

    def _build_feature_representations(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:

        current_items_static = self.item_emb_w(batch['items'])
        batch_context = current_items_static.mean(dim=0)
        if self.training:
            self._update_context_ema(batch_context)
            context_to_use = batch_context
        else:
            context_to_use = self.global_context_ema
            
        # User Tower
        static_u_emb = self.user_emb_w(batch['users'])
        hist_item_emb = self.item_emb_w(batch['item_history_matrix'])
        hist_cates = self.cates[batch['item_history_matrix']]
        hist_cates_emb = self.cate_emb_w(hist_cates)
        avg_cate_emb_for_hist = average_pooling(hist_cates_emb, self.cate_lens[batch['item_history_matrix']])
        hist_item_emb_with_cate = torch.cat([hist_item_emb, avg_cate_emb_for_hist], dim=2).detach()
        user_history_emb = self._run_hyper_transformer(hist_item_emb_with_cate, batch['item_history_len'], context_to_use, self.item_seq_transformer_hyper, self.item_seq_pos_emb)
        
        # [Stability] 保留 .detach()
        user_features = torch.cat([static_u_emb, user_history_emb], dim=-1)

        # Item Tower
        static_item_emb = self.item_emb_w(batch['items'])
        item_cates = self.cates[batch['items']] 
        item_cates_emb = self.cate_emb_w(item_cates)
        avg_cate_emb_for_item = average_pooling(item_cates_emb, self.cate_lens[batch['items']])
        item_emb_with_cate = torch.cat([static_item_emb, avg_cate_emb_for_item], dim=1)
        item_history_user_emb = self.user_emb_w(batch['user_history_matrix']).detach()
        item_history_emb = self._run_hyper_transformer(item_history_user_emb, batch['user_history_len'], context_to_use, self.user_seq_transformer_hyper, self.user_seq_pos_emb)
        
        # [Stability] 保留 .detach()
        item_features = torch.cat([item_emb_with_cate, item_history_emb], dim=-1)

        return user_features, item_features
    

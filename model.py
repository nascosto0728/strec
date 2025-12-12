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

# ===================================================================
#  標準 Transformer 組件
# ===================================================================

class StandardTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # 標準 FFN: Linear -> Activation -> Dropout -> Linear
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 1. Self-Attention Block (Pre-Norm)
        src2 = self.norm1(src)
        attn_output, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_output)
        
        # 2. FFN Block (Pre-Norm)
        src2 = self.norm2(src)
        ffn_out = self.linear1(src2)
        ffn_out = self.activation(ffn_out)
        ffn_out = self.dropout(ffn_out)
        ffn_out = self.linear2(ffn_out)
        src = src + self.dropout2(ffn_out)
        
        return src
    
# ===================================================================
#  主模型: Dual-Tower SASRec
# ===================================================================

class DualTowerSASRec(nn.Module):
    def __init__(self, global_meta: Dict[str, Any], cfg: Dict[str, Any]):
        super().__init__()
        
        # 1. 配置管理
        self.hparams = cfg['model']
        self.temperature = 0.07 
        self.maxlen = self.hparams.get('max_seq_len', 30)
        
        # 2. 註冊 Buffer (不可訓練的靜態資料)
        self.register_buffer('cates', torch.from_numpy(global_meta['cate_matrix']))
        self.register_buffer('cate_lens', torch.tensor(global_meta['cate_lens'], dtype=torch.int32))
        
        # 3. 模組構建
        self._build_embeddings()
        self._build_encoders()
        self._build_prediction_heads()
        
        # 4. 初始化
        self._init_weights()

    def _build_embeddings(self):
        # ID Embeddings
        self.user_emb_w = nn.Embedding(self.hparams['n_users'], self.hparams['user_embed_dim'])
        self.item_emb_w = nn.Embedding(self.hparams['n_items'], self.hparams['item_embed_dim'])
        self.cate_emb_w = nn.Embedding(self.hparams['n_cates'], self.hparams['cate_embed_dim'])

        # Positional Embeddings
        # Item Tower Input Dim = Item ID + Category
        self.item_seq_input_dim = self.hparams['item_embed_dim'] + self.hparams['cate_embed_dim']
        self.item_seq_pos_emb = nn.Embedding(self.maxlen, self.item_seq_input_dim)
        
        # User Tower Input Dim = User ID
        self.user_seq_input_dim = self.hparams['user_embed_dim']
        self.user_seq_pos_emb = nn.Embedding(self.maxlen, self.user_seq_input_dim)

        # Buffer for Negatives (History Lookup)
        history_embed_dim = self.hparams['item_embed_dim']
        self.user_history_buffer = nn.Embedding(self.hparams['n_items'], history_embed_dim)
        self.user_history_buffer.weight.requires_grad = False

    def _build_encoders(self):
        """構建標準 Transformer Encoders"""
        
        # Helper: 構建 Stack
        def _make_stack(input_dim):
            return nn.ModuleList([
                StandardTransformerLayer(
                    d_model=input_dim,
                    nhead=self.hparams.get('transformer_n_heads', 4),
                    dim_feedforward=input_dim * 4,
                    dropout=self.hparams.get('transformer_dropout', 0.1)
                )
                for _ in range(self.hparams.get('transformer_n_layers', 2))
            ])

        # User Tower (處理 Item 序列)
        self.item_seq_transformer = _make_stack(self.item_seq_input_dim)
        
        # Item Tower (處理 User 序列 - CF 訊號)
        # 這是我們在 Phase 1 會改成 CMS + MeanPool 的部分
        self.user_seq_transformer = _make_stack(self.user_seq_input_dim)

    def _create_mlp(self, input_dim: int) -> nn.Sequential:
        """標準 MLP Head"""
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

    def _build_prediction_heads(self):
        """初始化最後的 Projection Heads"""
        # Concat Dim = User Static + User Sequence (Item Features)
        # 注意：這裡假設 Static 和 Sequence 維度拼接邏輯與之前一致
        concat_dim = self.hparams['user_embed_dim'] + self.hparams['item_embed_dim'] + self.hparams['cate_embed_dim']
        
        self.user_mlp = self._create_mlp(concat_dim)
        self.item_mlp = self._create_mlp(concat_dim)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_emb_w.weight)
        nn.init.xavier_uniform_(self.item_emb_w.weight)
        nn.init.xavier_uniform_(self.cate_emb_w.weight)
        
        def _init_mlp(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
        for mlp in [self.user_mlp, self.item_mlp]:
            mlp.apply(_init_mlp)

    # ===================================================================
    #  Forward 邏輯
    # ===================================================================

    def _lookup_item_features(self, item_ids: torch.Tensor) -> torch.Tensor:
        """取得 Item Static + Category Features"""
        static_emb = self.item_emb_w(item_ids)
        item_cates = self.cates[item_ids]
        item_cates_emb = self.cate_emb_w(item_cates)
        avg_cate_emb = average_pooling(item_cates_emb, self.cate_lens[item_ids])
        return torch.cat([static_emb, avg_cate_emb], dim=-1)

    def _run_transformer(self, seq_emb, seq_lens, layers_module_list, pos_emb_module=None, pooling_mode='last') -> torch.Tensor:
        B, T, D = seq_emb.shape
        device = seq_emb.device
        
        # 1. Positional Encoding (Conditional)
        if pos_emb_module is not None:
            pos_ids = torch.arange(T, dtype=torch.long, device=device)
            seq_input = seq_emb + pos_emb_module(pos_ids).unsqueeze(0)
        else:
            # Set Transformer Mode: No Position Info
            seq_input = seq_emb
        
        # 2. Masks
        if pos_emb_module is not None:
            src_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device).bool()
        else:
            src_mask = None # Full Attention
        safe_lens = torch.clamp(seq_lens, min=1)
        padding_mask = (torch.arange(T, device=device)[None, :] >= safe_lens[:, None])
        
        # 3. Layers Forward
        output = seq_input
        for layer in layers_module_list:
            output = layer(output, src_mask=src_mask, src_key_padding_mask=padding_mask)
            
        # 4. Pooling Strategy
        if pooling_mode == 'last':
            # Sequence Mode: Take last item
            target_indices = (seq_lens - 1).clamp(min=0).view(B, 1, 1).expand(-1, 1, D)
            pooled_output = torch.gather(output, 1, target_indices).squeeze(1)
        elif pooling_mode == 'mean':
            # Set Mode: Average all valid items
            # 使用 average_pooling 函式 (記得 import)
            pooled_output = average_pooling(output, seq_lens)
        else:
            raise ValueError(f"Unknown pooling mode: {pooling_mode}")
        
        # Mask zero length sequences
        mask_zero = (seq_lens == 0).unsqueeze(-1).expand(-1, D)
        pooled_output = pooled_output.masked_fill(mask_zero, 0.0)
        
        return pooled_output

    def _build_feature_representations(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """構建 User 與 Item 的特徵向量"""
        
        # --- 1. User Tower ---
        # Input: User ID + User History (Items)
        user_static = self.user_emb_w(batch['user_id'])
        
        # Lookup history items
        hist_item_features = self._lookup_item_features(batch['user_interacted_items'])
        hist_item_features = hist_item_features.detach() # Stop gradient to item emb
        
        # Run Transformer
        user_history_emb = self._run_transformer(
            hist_item_features, 
            batch['user_interacted_len'], 
            self.item_seq_transformer, 
            # self.item_seq_pos_emb,
            pos_emb_module=None,  # <--- Set Transformer Mode
            pooling_mode='last'
        )
        
        user_features = torch.cat([user_static, user_history_emb], dim=-1)

        # --- 2. Item Tower ---
        # Input: Item ID/Cate + Item History (Users)
        item_composite_static = self._lookup_item_features(batch['item_id'])
        
        item_hist_user_ids = batch['item_interacted_users']
        item_hist_emb = self.user_emb_w(item_hist_user_ids).detach() # Stop gradient to user emb
        
        # Run Transformer
        item_history_emb = self._run_transformer(
            item_hist_emb, 
            batch['item_interacted_len'], 
            self.user_seq_transformer, 
            pos_emb_module=self.user_seq_pos_emb,
            # pos_emb_module=None,  # <--- Set Transformer Mode
            pooling_mode='last'
        )
        
        item_features = torch.cat([item_composite_static, item_history_emb], dim=-1)

        # --- Cache Update Mechanism ---
        if self.training:
            # 將當前 Batch 計算好的 Item History Embedding 存入 Buffer 供推論時的負樣本使用
            self.user_history_buffer.weight[batch['item_id']] = item_history_emb.detach()

        return user_features, item_features

    def forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        user_features, item_features = self._build_feature_representations(batch)
        
        # Projection Heads
        user_embedding = self.user_mlp(user_features)
        item_embedding = self.item_mlp(item_features)
        
        # Normalize
        user_embedding = F.normalize(user_embedding, p=2, dim=-1)
        item_embedding = F.normalize(item_embedding, p=2, dim=-1)
        
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
        user_features, item_features = self._build_feature_representations(batch)
        pos_user_emb, pos_item_emb = self.forward(batch) # Reuse forward to get normalized embs
        
        # 注意: 這裡為了效率，我們直接用 forward 的輸出 (正規化後的)
        # 上面 forward 內部再跑一次 _build_feature_representations 有點浪費，但在 Inference 時通常沒差
        # 嚴謹寫法是:
        # user_features, item_features = self._build_feature_representations(batch)
        # pos_user_emb = F.normalize(self.user_mlp(user_features), p=2, dim=-1)
        # pos_item_emb = F.normalize(self.item_mlp(item_features), p=2, dim=-1)
        
        pos_logits = torch.sum(pos_user_emb * pos_item_emb, dim=1)
        per_sample_loss = F.binary_cross_entropy_with_logits(pos_logits, batch['labels'].float(), reduction='none')

        # Negative Samples Handling
        num_neg_samples = neg_item_ids_batch.shape[1]
        neg_item_static_emb = self.item_emb_w(neg_item_ids_batch) 
        neg_item_cate_emb = self.cate_emb_w(self.cates[neg_item_ids_batch])
        neg_item_cates_len = self.cate_lens[neg_item_ids_batch]
        avg_cate_emb_for_neg_item = average_pooling(neg_item_cate_emb, neg_item_cates_len) 
        neg_item_emb_with_cate = torch.cat([neg_item_static_emb, avg_cate_emb_for_neg_item], dim=2) 
        
        item_history_emb_expanded = self.user_history_buffer(neg_item_ids_batch)
        neg_item_features = torch.cat([neg_item_emb_with_cate, item_history_emb_expanded.detach()], dim=2)
        
        user_features_expanded = user_features.unsqueeze(1).expand(-1, num_neg_samples, -1)
        
        # MLP Projection for Negatives
        neg_user_emb = self.user_mlp(user_features_expanded)
        neg_item_emb = self.item_mlp(neg_item_features)
        
        neg_user_emb_final = F.normalize(neg_user_emb, p=2, dim=-1)
        neg_item_emb_final = F.normalize(neg_item_emb, p=2, dim=-1)
        
        neg_logits = torch.sum(neg_user_emb_final * neg_item_emb_final, dim=2)
        
        return pos_logits, neg_logits, per_sample_loss
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

class GatingMechanism(nn.Module):
    """
    通用 Gating 模組，支援 Token-dependent 或 Context-dependent
    """
    def __init__(self, d_model, context_dim, source='token'):
        super().__init__()
        self.source = source
        self.d_model = d_model
        
        # 根據來源決定輸入維度
        in_dim = d_model if source == 'token' else context_dim
        
        # 簡單的投影層生成 Gate (Element-wise)
        # 論文建議使用 Sigmoid 來獲得稀疏性 (0~1)
        self.gate_net = nn.Linear(in_dim, d_model)
        
        # 初始化：偏置設為正值，讓訓練初期 Gate 接近 1 (全通)，避免梯度消失
        nn.init.constant_(self.gate_net.bias, 2.0)
        nn.init.xavier_uniform_(self.gate_net.weight)

    def forward(self, x, context=None):
        """
        x: [Batch, Seq, Dim] - 當前的 Token Embedding
        context: [Batch, Context_Dim] - Global Context (如果 source='context')
        """
        if self.source == 'token':
            # Input-Dependent (論文做法): Gate 由 x 自己決定
            gate_input = x
        else:
            # Context-Dependent (HyperNet做法): Gate 由 Context 決定
            # 需要擴展維度以匹配序列: [B, C_dim] -> [B, 1, C_dim]
            if context is None:
                raise ValueError("Context must be provided when gating_source='context'")
            gate_input = context.unsqueeze(1)
        
        # 生成 Gate: [B, S, D] 或 [B, 1, D] (自動廣播)
        gate = torch.sigmoid(self.gate_net(gate_input))
        
        return x * gate

class HyperLoRALinear(nn.Module):
    """
    保留原本的 LoRA 架構 (因為實驗證明有效)
    """
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
        
        # 處理 HyperNet Gate 維度
        if context_gate.dim() == 2:
            gate = context_gate.unsqueeze(1)
        else:
            gate = context_gate
            
        # LoRA 分支
        down_out = self.lora_down(x)
        gated_down = down_out * gate # HyperLoRA 的核心
        lora_out = self.lora_up(gated_down)
        
        return base_out + self.dropout(lora_out)

class HyperLoRATransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, lora_r=8, 
                 use_gated_attention=False, gating_source='token', context_dim=None):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # [新增] Attention Gating 機制
        self.use_gated_attention = use_gated_attention
        if self.use_gated_attention:
            self.attn_gate = GatingMechanism(d_model, context_dim, source=gating_source)
        
        # FFN (保持 HyperLoRA)
        self.linear1 = HyperLoRALinear(d_model, dim_feedforward, r=lora_r, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = HyperLoRALinear(dim_feedforward, d_model, r=lora_r, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src, hyper_gate, context_vector=None, src_mask=None, src_key_padding_mask=None):
        """
        hyper_gate: 用於控制 LoRA 的 Gate (來自 HyperNet)
        context_vector: 用於 GatedAttention 的原始 Context (如果 source='context')
        """
        src2 = self.norm1(src)
        attn_output, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        
        # [Gating Injection] Post-Attention Gating (G1)
        if self.use_gated_attention:
            attn_output = self.attn_gate(attn_output, context=context_vector)
            
        src = src + self.dropout1(attn_output)
        
        # FFN Block (HyperLoRA)
        src2 = self.norm2(src)
        ffn_out = self.linear1(src2, hyper_gate) 
        ffn_out = self.activation(ffn_out)
        ffn_out = self.dropout(ffn_out)
        ffn_out = self.linear2(ffn_out, hyper_gate)
        src = src + self.dropout2(ffn_out)
        return src
        
class HyperLoRASASRec(nn.Module):
    def __init__(self, global_meta: Dict[str, Any], cfg: Dict[str, Any]):
        super().__init__()
        
        # 1. 配置管理
        self.hparams = cfg['model']
        self.temperature = 0.07 
        self.maxlen = self.hparams.get('max_seq_len', 30) 
        self.lora_r = self.hparams.get('lora_r', 16)
        
        # [Gating 設定]
        self.use_gated_attention = self.hparams.get('use_gated_attention', False)
        self.gating_source = self.hparams.get('gating_source', 'token') # 'token' or 'context'
        
        # 2. 註冊 Buffer
        self.register_buffer('cates', torch.from_numpy(global_meta['cate_matrix']))
        self.register_buffer('cate_lens', torch.tensor(global_meta['cate_lens'], dtype=torch.int32))
        
        # 3. Context Dim
        self.context_dim = self.hparams['item_embed_dim'] # 預設 Context 是 Item Mean

        # 4. 模組構建
        self._build_embeddings()
        self._build_hyper_components()
        self._build_encoders()
        self._build_prediction_heads()
        
        self._init_weights()

    def _build_embeddings(self):
        # ID Embeddings
        self.user_emb_w = nn.Embedding(self.hparams['n_users'], self.hparams['user_embed_dim'])
        self.item_emb_w = nn.Embedding(self.hparams['n_items'], self.hparams['item_embed_dim'])
        self.cate_emb_w = nn.Embedding(self.hparams['n_cates'], self.hparams['cate_embed_dim'])

        # Positional Embeddings
        self.item_seq_input_dim = self.hparams['item_embed_dim'] + self.hparams['cate_embed_dim']
        self.item_seq_pos_emb = nn.Embedding(self.maxlen, self.item_seq_input_dim)
        
        self.user_seq_input_dim = self.hparams['user_embed_dim']
        self.user_seq_pos_emb = nn.Embedding(self.maxlen, self.user_seq_input_dim)

        # Buffer for Negatives
        history_embed_dim = self.hparams['item_embed_dim']
        self.user_history_buffer = nn.Embedding(self.hparams['n_items'], history_embed_dim)
        self.user_history_buffer.weight.requires_grad = False

    def _build_hyper_components(self):
        """
        HyperNetwork 用於生成控制 LoRA 的 Gate
        """
        self.hyper_gate_net = nn.Sequential(
            nn.Linear(self.context_dim, self.lora_r),
            nn.Tanh(), 
            nn.Linear(self.lora_r, self.lora_r),
            nn.Tanh()
        )

    def _build_transformer_block(self, input_dim: int) -> nn.ModuleList:
        """
        構建 Transformer 層，傳入 Gating 設定
        """
        return nn.ModuleList([
            HyperLoRATransformerLayer(
                d_model=input_dim,
                nhead=self.hparams.get('transformer_n_heads', 4),
                dim_feedforward=input_dim * 4,
                dropout=self.hparams.get('transformer_dropout', 0.1),
                lora_r=self.lora_r,
                # Gating Params
                use_gated_attention=self.use_gated_attention,
                gating_source=self.gating_source,
                context_dim=self.context_dim
            )
            for _ in range(self.hparams.get('transformer_n_layers', 2))
        ])

    def _build_encoders(self):
        self.item_seq_transformer_hyper = self._build_transformer_block(self.item_seq_input_dim)
        self.user_seq_transformer_hyper = self._build_transformer_block(self.user_seq_input_dim)

    def _create_mlp(self, input_dim: int) -> nn.Sequential:
        expansion_factor = 2
        inner_dim = int(input_dim * expansion_factor)
        dropout = 0.0 
        
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

    # --- Forward Helpers ---

    def _lookup_item_features(self, item_ids: torch.Tensor) -> torch.Tensor:
        static_emb = self.item_emb_w(item_ids)
        item_cates = self.cates[item_ids]
        item_cates_emb = self.cate_emb_w(item_cates)
        avg_cate_emb = average_pooling(item_cates_emb, self.cate_lens[item_ids])
        return torch.cat([static_emb, avg_cate_emb], dim=-1)

    def _run_hyper_transformer(self, seq_emb, seq_lens, context_vector, layers_module_list, pos_emb_module):
        B, T, D = seq_emb.shape
        device = seq_emb.device

        if context_vector.dim() == 1:
            context_vector = context_vector.unsqueeze(0) # [128] -> [1, 128]

        # 1. 生成控制 LoRA 的 Gate (來自 Context)
        hyper_gate = self.hyper_gate_net(context_vector).expand(B, -1)
        
        # 2. Positional Encoding
        pos_ids = torch.arange(T, dtype=torch.long, device=device)
        seq_input = seq_emb + pos_emb_module(pos_ids).unsqueeze(0)
        
        # 3. Masks
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device).bool()
        safe_lens = torch.clamp(seq_lens, min=1)
        padding_mask = (torch.arange(T, device=device)[None, :] >= safe_lens[:, None])
        
        # 4. Layers Forward
        output = seq_input
        for layer in layers_module_list:
            output = layer(
                output, 
                hyper_gate=hyper_gate,      # 控制 LoRA
                context_vector=context_vector, # 控制 GatedAttention (如果 source='context')
                src_mask=causal_mask, 
                src_key_padding_mask=padding_mask
            )
            
        # 5. Pooling (Take last item)
        target_indices = (seq_lens - 1).clamp(min=0).view(B, 1, 1).expand(-1, 1, D)
        pooled_output = torch.gather(output, 1, target_indices).squeeze(1)
        
        mask_zero = (seq_lens == 0).unsqueeze(-1).expand(-1, D)
        pooled_output = pooled_output.masked_fill(mask_zero, 0.0)
        
        return pooled_output

    def _build_feature_representations(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        簡化後的特徵構建：移除 EMA 與 Ablation 邏輯，只保留最有效的 Batch Mean
        """
        # --- 1. Context Management ---
        current_items_static = self.item_emb_w(batch['item_id'])
        batch_context = current_items_static.mean(dim=0)
        
        # [重要] 這裡我們統一使用 Batch Context
        # 因為實驗證明 EMA 沒用，且 Random 會掉分，所以直接用 Batch Mean
        context_to_use = batch_context

        # --- 2. User Tower ---
        user_static = self.user_emb_w(batch['user_id'])
        hist_item_features = self._lookup_item_features(batch['user_interacted_items'])
        hist_item_features = hist_item_features.detach()
        
        user_history_emb = self._run_hyper_transformer(
            hist_item_features, 
            batch['user_interacted_len'], 
            context_to_use, 
            self.item_seq_transformer_hyper, 
            self.item_seq_pos_emb
        )
        user_features = torch.cat([user_static, user_history_emb], dim=-1)

        # --- 3. Item Tower ---
        item_composite_static = self._lookup_item_features(batch['item_id'])
        item_hist_user_ids = batch['item_interacted_users']
        item_hist_emb = self.user_emb_w(item_hist_user_ids).detach()
        
        item_history_emb = self._run_hyper_transformer(
            item_hist_emb, 
            batch['item_interacted_len'], 
            context_to_use, 
            self.user_seq_transformer_hyper, 
            self.user_seq_pos_emb
        )
        item_features = torch.cat([item_composite_static, item_history_emb], dim=-1)

        return user_features, item_features
    
    def _get_embeddings_from_features(self, user_features: torch.Tensor, item_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        user_embedding = self.user_mlp(user_features) 
        item_embedding = self.item_mlp(item_features) 
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
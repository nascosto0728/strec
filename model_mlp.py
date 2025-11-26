# model.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import math

# ===================================================================
# 輔助函式 
# ===================================================================
    
def average_pooling(embeddings: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    """
    對 Embedding 進行帶遮罩的平均池化 (PyTorch 版本)。
    """
    if embeddings.dim() == 4: # e.g., [B, T, C_max, E]
        # seq_lens: [B, T] -> mask: [B, T, C_max]
        max_len = embeddings.size(2)
        mask = torch.arange(max_len, device=embeddings.device)[None, None, :] < seq_lens.unsqueeze(-1)
        mask = mask.float().unsqueeze(-1) # [B, T, C_max, 1]
        
        masked_sum = torch.sum(embeddings * mask, dim=2)
        count = mask.sum(dim=2) + 1e-9
        return masked_sum / count
    
    elif embeddings.dim() == 3: # e.g., [B, T, E]
        # seq_lens: [B] -> mask: [B, T]
        max_len = embeddings.size(1)
        mask = torch.arange(max_len, device=embeddings.device)[None, :] < seq_lens[:, None]
        mask = mask.float().unsqueeze(-1) # [B, T, 1]
        
        masked_sum = torch.sum(embeddings * mask, dim=1)
        count = mask.sum(dim=1) + 1e-9
        return masked_sum / count
    
    else:
        raise ValueError(f"Unsupported embedding dimension: {embeddings.dim()}")


class EmbMLP(nn.Module):
    """
    PAM 模型的 PyTorch 實現版本。
    """
    def __init__(self, cates: np.ndarray, cate_lens: np.ndarray, hyperparams: Dict, train_config: Dict,
                 item_init_vectors: torch.Tensor = None,  
                 cate_init_vectors: torch.Tensor = None): 
        super().__init__()
        self.hparams = hyperparams
        self.train_config = train_config
        self.temperature = 0.07
        
        # --- 註冊不會被優化器更新的 Buffer ---
        self.register_buffer('cates', torch.from_numpy(cates))
        # *** FIX: 使用 torch.tensor() 而不是 torch.from_numpy() 來處理 list ***
        self.register_buffer('cate_lens', torch.tensor(cate_lens, dtype=torch.int32))

        # --- (*** MODIFIED ***) ---
        # 刪除 self.item_init_vectors = ...
        # 我們直接把它註冊為 Buffer，這樣它就會自動 .to(device) 且不參與
        if item_init_vectors is not None:
            self.register_buffer('anchor_item_vectors', item_init_vectors)
        else:
            self.register_buffer('anchor_item_vectors', None)
            
        if cate_init_vectors is not None:
            self.register_buffer('anchor_cate_vectors', cate_init_vectors)
        else:
            self.register_buffer('anchor_cate_vectors', None)
        # --- (修改結束) ---

        # --- 建立 Embedding Table ---
        self.user_emb_w = nn.Embedding(self.hparams['num_users'], self.hparams['user_embed_dim'])
        self.item_emb_w = nn.Embedding(self.hparams['num_items'], self.hparams['item_embed_dim'])
        self.cate_emb_w = nn.Embedding(self.hparams['num_cates'], self.hparams['cate_embed_dim'])

        # 這個緩存將儲存每個用戶的「最新」平均池化歷史
        history_embed_dim = self.hparams['item_embed_dim'] 
        self.user_history_buffer = nn.Embedding(self.hparams['num_items'], history_embed_dim)
        # 將緩存設為非可訓練
        self.user_history_buffer.weight.requires_grad = False
       

        # --- 建立 MLP 權重 (使用 nn.Sequential) ---
        concat_dim = self.hparams['user_embed_dim'] + self.hparams['item_embed_dim'] + self.hparams['cate_embed_dim'] 

        layer_num = 1
        self.user_mlp = nn.Sequential(
                *[self.PreNormResidual(concat_dim, self._build_mlp_layers(concat_dim))]*layer_num,
            )
        self.item_mlp = nn.Sequential(
                *[self.PreNormResidual(concat_dim, self._build_mlp_layers(concat_dim))]*layer_num,
            )

        self.user_mlp_2 = nn.Sequential(
                *[self.PreNormResidual(concat_dim, self._build_mlp_layers(concat_dim))]*layer_num,
            )
        self.item_mlp_2 = nn.Sequential(
                *[self.PreNormResidual(concat_dim, self._build_mlp_layers(concat_dim))]*layer_num,
            )

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
        # 1. 初始化 User 
        nn.init.xavier_uniform_(self.user_emb_w.weight)
        nn.init.xavier_uniform_(self.item_emb_w.weight)
        nn.init.xavier_uniform_(self.cate_emb_w.weight)

        # 4. 初始化 MLPs 
        for mlp in [self.user_mlp, self.item_mlp, self.user_mlp_2, self.item_mlp_2]:
            for layer in mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
                    

    def _build_feature_representations(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """建立使用者和物品的特徵表示 (融合細粒度時間特徵)"""

        # === 使用者表示 ===
        # 獲取靜態身份 
        static_u_emb = self.user_emb_w(batch['users'])
        u_emb = static_u_emb 
        
        # 歷史序列
        hist_item_emb = self.item_emb_w(batch['item_history_matrix'])
        hist_cates = self.cates[batch['item_history_matrix']]
        hist_cates_len = self.cate_lens[batch['item_history_matrix']]
        hist_cates_emb = self.cate_emb_w(hist_cates)
        avg_cate_emb_for_hist = average_pooling(hist_cates_emb, hist_cates_len)
        hist_item_emb_with_cate = torch.cat([hist_item_emb, avg_cate_emb_for_hist], dim=2)

        user_history_emb = average_pooling(hist_item_emb_with_cate, batch['item_history_len']).detach()

        user_features = torch.cat([u_emb,user_history_emb], dim=-1)

        # === 物品表示 ===
        # 獲取靜態身份
        static_item_emb = self.item_emb_w(batch['items'])
        item_emb = static_item_emb

        # 歷史序列
        item_cates = self.cates[batch['items']]
        item_cates_len = self.cate_lens[batch['items']]
        item_cates_emb = self.cate_emb_w(item_cates)
        avg_cate_emb_for_item = average_pooling(item_cates_emb, item_cates_len)
        item_emb_with_cate = torch.cat([item_emb, avg_cate_emb_for_item], dim=1)
        item_history_user_emb = self.user_emb_w(batch['user_history_matrix'])

        item_history_emb = average_pooling(item_history_user_emb, batch['user_history_len']).detach()

        item_features = torch.cat([item_emb_with_cate, item_history_emb], dim=-1)

        # 在訓練時，更新緩存
        if self.training:
            item_ids = batch['items']
            # 我們必須 .detach()，因為緩存更新這個操作不應該參與反向傳播
            self.user_history_buffer.weight[item_ids] = item_history_emb.detach()

        return user_features, item_features


    def _get_embeddings_from_features(self, user_features: torch.Tensor, item_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        (新函式) 執行 MLP 和歸一化。
        """
        # --- User MLP Path ---
        user_embedding = self.user_mlp(user_features)
        user_embedding_2 = self.user_mlp_2(user_features)
        user_embedding =  user_embedding + user_embedding_2
    
        # --- Item MLP Path ---
        item_embedding = self.item_mlp(item_features)
        item_embedding_2 = self.item_mlp_2(item_features)
        item_embedding =  item_embedding + item_embedding_2
        
        user_embedding = F.normalize(user_embedding, p=2, dim=-1)
        item_embedding = F.normalize(item_embedding, p=2, dim=-1)
        
        return user_embedding, item_embedding
    def forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """定義基礎的前向傳播：從 raw features 到 embeddings"""
        user_features, item_features = self._build_feature_representations(batch)

        # *** 修正：調用新的輔助函式 ***
        user_embedding, item_embedding = self._get_embeddings_from_features(user_features, item_features)
        
        return user_embedding, item_embedding
    
    def _calculate_infonce_loss(self, user_embedding, item_embedding, labels):
        """計算 InfoNCE 損失 """
        
        # --- 分母 (Denominator) 計算 ---
        # 1. 計算 user-item 分數矩陣 [B, B]，M[i,j] = user_i @ item_j
        all_inner_product = torch.matmul(user_embedding, item_embedding.t())
        logits = all_inner_product / self.temperature
        
        # 2. 應用 Log-Sum-Exp 技巧穩定化
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits_stabilized = logits - logits_max
        exp_logits_den = torch.exp(logits_stabilized)
        
        # 3. 分母是穩定後 exp_logits 的行和 (對每個 user，匯總所有 item 的分數)
        denominator = exp_logits_den.sum(dim=1, keepdim=True)

        # --- 分子 (Numerator) 計算 ---
        # 1. 獨立計算正樣本對 (user_i, item_i) 的分數
        pred_scores = (user_embedding * item_embedding).sum(dim=1, keepdim=True)
        pred_logits = pred_scores / self.temperature
        
        # 2. 使用與分母相同的 logits_max 來穩定化分子
        #    確保分子和分母的縮放標準一致
        pred_logits_stabilized = pred_logits - logits_max
        numerator = torch.exp(pred_logits_stabilized)

        # --- 最終計算 ---
        infonce_pred = (numerator / (denominator + 1e-9)).squeeze(dim=1)
        infonce_pred = torch.clamp(infonce_pred, min=1e-9, max=1.0 - 1e-9)
        
        return F.binary_cross_entropy(infonce_pred, labels.float(), reduction='none')
    
    def calculate_loss(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        計算訓練時的總損失。
        """
        # 1. 執行一次標準的前向傳播
        user_embedding, item_embedding = self.forward(batch)
        
        # 2. 計算整個批次的 InfoNCE 損失
        losses = self._calculate_infonce_loss(user_embedding, item_embedding, batch['labels'])
        
        # 3. 返回損失的平均值
        final_loss = losses.mean()
        
        return final_loss
        
    
    def inference(
        self,
        batch: Dict[str, torch.Tensor],
        neg_item_ids_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        在 Baseline 模式下執行推論，計算正樣本和指定負樣本的分數。
        """
        # --- 1. 計算正樣本的 Embedding 和 Logits ---
        user_features, item_features = self._build_feature_representations(batch)
        pos_user_emb, pos_item_emb = self._get_embeddings_from_features(user_features, item_features)
        pos_logits = torch.sum(pos_user_emb * pos_item_emb, dim=1)
        per_sample_loss = F.binary_cross_entropy_with_logits(
            pos_logits, batch['labels'].float(), reduction='none'
        )

        # --- 2. 計算負樣本的 Logits ---
        num_neg_samples = neg_item_ids_batch.shape[1]
        
        neg_item_static_emb = self.item_emb_w(neg_item_ids_batch)
        neg_item_cate_emb = self.cate_emb_w(self.cates[neg_item_ids_batch])
        neg_item_cates_len = self.cate_lens[neg_item_ids_batch]
        avg_cate_emb_for_neg_item = average_pooling(neg_item_cate_emb, neg_item_cates_len)
        neg_item_history_emb = self.user_history_buffer(neg_item_ids_batch)
        neg_item_features = torch.cat([
            neg_item_static_emb, 
            avg_cate_emb_for_neg_item,
            neg_item_history_emb
        ], dim=2)
        
        user_features_expanded = user_features.unsqueeze(1).expand(-1, num_neg_samples, -1)

        neg_user_emb_final, neg_item_emb_final = self._get_embeddings_from_features(user_features_expanded, neg_item_features)

        neg_logits = torch.sum(neg_user_emb_final * neg_item_emb_final, dim=2)
        
        # --- 3. 返回結果 ---
        return pos_logits, neg_logits, per_sample_loss
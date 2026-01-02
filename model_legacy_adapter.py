import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Dict, List, Any

class LegacyTitansAdapter(nn.Module):
    """
    Adapter Pattern for Old-Titans Model.
    負責將新版 Config 轉譯給舊模型，並對齊新版 Trainer 的介面。
    """
    def __init__(self, global_meta: Dict[str, Any], cfg: Dict[str, Any], legacy_model_class):
        super().__init__()
        
        # -------------------------------------------------------
        # [Config Adaptation] 參數轉譯層
        # -------------------------------------------------------
        legacy_cfg = copy.deepcopy(cfg)
        model_cfg = legacy_cfg['model']
        
        if 'dropout' in model_cfg:
            model_cfg['transformer_dropout'] = model_cfg['dropout']
            
        model_cfg.setdefault('transformer_n_layers', 1)
        model_cfg.setdefault('titans_n_heads', 4)
        model_cfg.setdefault('max_seq_len', 50)
        model_cfg.setdefault('titans_cpu_offload', True)
        
        print(f"--- [Adapter] Config adapted for Legacy Model: {model_cfg.keys()} ---")

        # -------------------------------------------------------
        # [Data Adaptation] 資料修補層 (Fix KeyError: 'cate_lens')
        # -------------------------------------------------------
        # 舊模型預期 'cate_lens' 存在，用於計算多類別的 Average Pooling。
        # 如果目前資料流沒有這個 Key，我們假設每個 Item 只有 1 個類別，補上全為 1 的列表。
        if 'cate_lens' not in global_meta:
            print("--- [Adapter] 'cate_lens' missing. Generating default lens (ones)... ---")
            n_items = global_meta.get('n_items', model_cfg.get('n_items'))
            # 建立一個長度為 n_items 的列表，值全為 1
            global_meta['cate_lens'] = [1] * n_items

        # -------------------------------------------------------
        # Initialize Legacy Model
        # -------------------------------------------------------
        self.model = legacy_model_class(global_meta, legacy_cfg)
        
        # Mapping 屬性供 Trainer 存取
        self.memory_bank = self.model.user_memory_bank
        
        # 舊模型沒有 item_cache，建立一個 dummy buffer 防止 Trainer 報錯
        self.register_buffer('item_cache', torch.zeros(1))

    def toggle_stage(self, stage: int):
        """Old-Titans 沒有 Stage 概念，僅在 Stage 1 開啟訓練"""
        if stage == 1:
            self.model.train()
            for p in self.model.parameters():
                p.requires_grad = True
        elif stage == 2:
            # 舊模型不支援 Stage 2 序列訓練，將其設為 Eval
            self.model.eval()

    def reset_item_cache_to_static(self):
        """Dummy implementation"""
        pass

    def _get_collision_mask(self, item_ids: torch.Tensor) -> torch.Tensor:
        """計算 Collision Mask，排除對角線 (Self-Collision)"""
        collision_mask = (item_ids.unsqueeze(1) == item_ids.unsqueeze(0))
        collision_mask.fill_diagonal_(False)
        return collision_mask

    def compute_metrics(self, user_emb: torch.Tensor, item_emb: torch.Tensor, item_ids: torch.Tensor, k_list: List[int]) -> Dict[str, float]:
        """
        [Copied from New-Titans] 使用完全相同的邏輯計算指標
        """
        B = user_emb.size(0)
        if B < 10: return {}

        # Normalize
        user_emb = F.normalize(user_emb, p=2, dim=1, eps=1e-6)
        item_emb = F.normalize(item_emb, p=2, dim=1, eps=1e-6)
        
        # Scores
        scores = torch.matmul(user_emb, item_emb.t())
        
        # Masking
        collision_mask = self._get_collision_mask(item_ids)
        scores.masked_fill_(collision_mask, float('-inf'))
        
        # AUC
        target_scores = torch.diag(scores).unsqueeze(1)
        greater_mask = (scores > target_scores)
        ranks_0based = greater_mask.sum(dim=1).float()
        
        num_collisions = collision_mask.sum(dim=1).float()
        num_negatives = (B - 1) - num_collisions
        num_negatives.clamp_(min=1.0)
        
        auc_per_user = 1.0 - (ranks_0based / num_negatives)
        mean_auc = auc_per_user.mean().item()
        
        metrics = {'auc': mean_auc}

        # Top-K
        if not k_list: k_list = [10, 20]
        max_k = max(k_list)
        safe_k = min(max_k, scores.size(1))
        
        _, topk_indices = torch.topk(scores, k=safe_k, dim=1)
        targets = torch.arange(B, device=scores.device).unsqueeze(1)
        hit_matrix = (topk_indices == targets)
        
        for k in k_list:
            actual_limit = min(k, safe_k)
            hits_at_k = hit_matrix[:, :actual_limit].any(dim=1).float()
            metrics[f'recall@{k}'] = hits_at_k.mean().item()
            
            vals = hit_matrix[:, :actual_limit].float()
            rank_pos = torch.arange(1, actual_limit + 1, device=scores.device).unsqueeze(0)
            discounts = 1.0 / torch.log2(rank_pos + 1.0)
            dcg = (vals * discounts).sum(dim=1)
            metrics[f'ndcg@{k}'] = dcg.mean().item()
            
        return metrics

    def forward_stage1(self, 
                       batch: Dict, 
                       training: bool = True, 
                       update_cache: bool = False,
                       compute_metrics: bool = False,
                       k_list: List[int] = [10, 20]) -> Dict:
        """
        適配器核心：呼叫舊模型的 forward，並處理欄位映射與型態轉換
        """
        # 控制 Memory Update
        self.model.train(training)
        
        # -------------------------------------------------------
        # [Field Mapping & Tensor Conversion] 修復點
        # -------------------------------------------------------
        # 舊模型預期 'user_interacted_len' 為一個 Tensor。
        # 根據 utils_tmp.py，user 的長度資訊隱藏在 user_interacted_items 的非零元素中。
        
        if 'user_interacted_len' not in batch or batch['user_interacted_len'] is None:
            # 計算 user 歷史序列的實際長度 (排除 padding 0)
            # batch['user_interacted_items'] shape: [B, max_seq_len]
            u_items = batch['user_interacted_items']
            # 計算每列非零的數量作為長度
            u_lens = (u_items != 0).sum(dim=1)
            batch['user_interacted_len'] = u_lens

        # 確保 labels 鍵值存在 (舊模型 _calculate_infonce_loss 需要)
        # 如果新版 dataset 沒給標籤，在 Stage 1 預設正樣本標籤全是 1
        if 'labels' not in batch:
            batch['labels'] = torch.ones(batch['user_id'].size(0), device=batch['user_id'].device)

        # 1. 取得 Embeddings
        # 現在 batch 具備了 'user_interacted_len' (Tensor)，不會再報 AttributeError
        user_emb, item_emb = self.model(batch)
        
        # 2. 計算 Loss
        loss = self.model._calculate_infonce_loss(user_emb, item_emb, batch['labels']).mean()
        
        outputs = {'loss': loss}
        
        # 3. 計算 Metrics
        if compute_metrics:
            metrics = self.compute_metrics(user_emb, item_emb, batch['item_id'], k_list=k_list)
            outputs.update(metrics)
            
        return outputs

    def forward_stage2(self, batch: Dict, initial_states_snapshot: tuple) -> Dict:
        """Old-Titans 不支援 Stage 2，回傳 0 loss"""
        dummy_loss = torch.tensor(0.0, device=batch['user_id'].device, requires_grad=True)
        return {'loss': dummy_loss}
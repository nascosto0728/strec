import os
import yaml
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, List, Set, Optional
import copy

import torch
from torch.utils.data import DataLoader

# 引入重構後的模組
from model import DualTowerSASRec
from model_cms import CMSDualTowerSASRec

from utils import (
    prepare_data_pipeline,
    RecommendationDataset,
    sample_negatives_batch
)

def set_seed(seed: int):
    """固定隨機種子以確保實驗可重現性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 確保 CUDA 運算確定性 (會稍微影響效能但對除錯很重要)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class StreamingTrainer:
    """
    串流推薦實驗控制器。
    負責：資料載入、模型初始化、週期性訓練 (Incremental Learning)、評估。
    """
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.device = self._get_device()
        
        # 1. 準備全量資料與 Meta Info
        # df: 全量資料, meta: {'cate_matrix', ...}, maps: {'user_map', ...}
        self.full_df, self.global_meta, self.id_maps = prepare_data_pipeline(self.cfg)
        
        # 將 Meta Info 注入 config 供 Model 使用
        self.cfg['model']['n_users'] = self.global_meta['n_users']
        self.cfg['model']['n_items'] = self.global_meta['n_items']
        self.cfg['model']['n_cates'] = self.global_meta['n_cates']
        
        # 2. 狀態追蹤器 (State Trackers)
        # 用於負採樣：記錄每個用戶看過的所有物品 (跨 Period 累績)
        self.user_history_tracker: Dict[int, Set[int]] = {}
        # 用於負採樣：記錄所有出現過的物品池
        self.seen_items_pool: Set[int] = set()
        
        # 3. 實驗結果容器
        self.results_log = []

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            print("--- Device: CUDA ---")
            return torch.device("cuda")
        else:
            print("--- Device: CPU ---")
            return torch.device("cpu")

    def _init_model(self):
        """根據 Config 初始化模型"""
        model_type = self.cfg.get('model_type', 'hyper_lora_sasrec')
        print(f"--- Initializing Model: {model_type} ---")
        
        if model_type == 'dual_tower_sasrec':
            model = DualTowerSASRec(global_meta=self.global_meta, cfg=self.cfg).to(self.device)
        elif model_type == 'cms_dual_tower':
            # Phase 1: User Tower (Transfomer) + Item Tower (CMS + MeanPool)
            model = CMSDualTowerSASRec(global_meta=self.global_meta, cfg=self.cfg).to(self.device)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        return model

    def _load_prev_period_weights(self, model: torch.nn.Module, period_id: int, lr: float):
        """嘗試載入上一個 Period 的最佳權重 (模擬串流繼承)"""
        if period_id <= self.cfg['train_start_period']:
            return

        prev_dir_name = f"{self.cfg['dir_name']}_lr{lr}"
        prev_ckpt_path = os.path.join('./checkpoints', prev_dir_name, f'period_{period_id-1}', 'best_model.pth')
        
        if os.path.exists(prev_ckpt_path):
            try:
                state_dict = torch.load(prev_ckpt_path, map_location=self.device)
                model.load_state_dict(state_dict)
                print(f"--- [Init] Weights loaded from Period {period_id-1} ---")
            except Exception as e:
                print(f"--- [Warning] Failed to load weights: {e} ---")
        else:
            print(f"--- [Init] No checkpoint found for Period {period_id-1}. Training from scratch. ---")

    def _get_dataloader(self, df: pd.DataFrame, shuffle: bool) -> DataLoader:
        """建立標準 DataLoader"""
        dataset = RecommendationDataset(df, max_seq_len=30) # 硬編碼 30 或從 config 讀
        return DataLoader(
            dataset,
            batch_size=self.cfg['model']['batch_size'],
            shuffle=shuffle,
            num_workers=self.cfg.get('num_workers', 0)
        )

    def run(self):
        """執行主實驗流程"""
        # 針對每個 Learning Rate 跑一次完整的串流實驗
        for lr in self.cfg['learning_rates']:
            print(f"\n{'='*60}\n>>> Starting Stream with LR: {lr}\n{'='*60}")
            self._run_stream_for_lr(lr)

    def _run_stream_for_lr(self, lr: float):
        """針對單一 LR 的串流迴圈"""
        # 重置狀態
        self.user_history_tracker.clear()
        self.seen_items_pool.clear()
        self.results_log.clear()
        
        # 初始化模型 (每個 LR 重新開始)
        model = self._init_model()
        frozen_params_ids = set()
        
        trainable_params = filter(lambda p: id(p) not in frozen_params_ids, model.parameters())
        optimizer = torch.optim.Adam(
            trainable_params, 
            lr=lr, 
            weight_decay=self.cfg.get('weight_decay', 0.0)
        )

        # Period Loop
        start_p = self.cfg['train_start_period']
        end_p = self.cfg['num_periods']
        
        for p_id in range(start_p, end_p-1):
            print(f"\n--- Period {p_id} Start ---")
            
            # 1. 準備當期資料
            curr_df = self.full_df[self.full_df['period'] == p_id]
            if curr_df.empty:
                print(f"Period {p_id} is empty. Skipping.")
                continue
                
            
            # 2. 繼承權重
            self._load_prev_period_weights(model, p_id, lr)
            
            # 3. 準備 DataLoaders
            # Train / Val Split
            val_ratio = self.cfg.get('validation_split', 0.2)
            split_idx = int(len(curr_df) * (1 - val_ratio))
            train_df = curr_df.iloc[:split_idx]
            val_df = curr_df.iloc[split_idx:]
            
            train_loader = self._get_dataloader(train_df, shuffle=True)
            val_loader = self._get_dataloader(val_df, shuffle=False) if not val_df.empty else None
            
            # 4. 訓練 (Training)
            best_model_state = self._train_period(model, optimizer, train_loader, val_loader, p_id, lr)


            # 5. 更新全域狀態 (History Tracker) 
            self._update_history_tracker(curr_df)
            
            # 6. 載入當期最佳模型進行評估
            if best_model_state:
                model.load_state_dict(best_model_state)
                
            # 7. 評估 (Forward Transfer Evaluation on T+1)
            #    測試集是下一個 Period 的資料
            next_p_id = p_id + 1
            if next_p_id < end_p:
                test_df = self.full_df[self.full_df['period'] == next_p_id]
                # Optional: Downsample test set for speed
                # test_df = test_df.iloc[::5] 
                
                if next_p_id >= self.cfg.get('test_start_period', 8):
                    # 只評估有正樣本的資料
                    test_df = test_df[test_df['label'] == 1]
                    if not test_df.empty:
                        test_loader = self._get_dataloader(test_df, shuffle=False)
                        metrics = self._evaluate(model, test_loader, next_p_id)
                        self.results_log.append(metrics)
        
        # End of Stream Summary
        self._print_summary(lr)

    def _train_period(self, model, optimizer, train_loader, val_loader, p_id, lr):
        """單一 Period 的訓練迴圈 (含 Early Stopping)"""
        save_dir = f"./checkpoints/{self.cfg['dir_name']}_lr{lr}/period_{p_id}"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'best_model.pth')
        
        best_val_loss = float('inf')
        patience = self.cfg.get('patience', 3)
        counter = 0
        best_state = None
        
        print(f"Training: {len(train_loader.dataset)} samples. Validation: {len(val_loader.dataset) if val_loader else 0} samples.")

        for epoch in range(1, self.cfg.get('max_epochs', 10) + 1):
            # --- Train ---
            model.train()
            train_losses = []
            
            for batch in tqdm(train_loader, desc=f"Ep {epoch}", leave=False):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                optimizer.zero_grad()
                loss = model.calculate_loss(batch)
                
                if torch.isnan(loss):
                    print("[Warning] NaN loss detected. Skipping batch.")
                    continue
                
                loss.backward()
                
                # # Gradient Clipping to prevent NaN in Transformer/Embedding
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                
                optimizer.step()
                train_losses.append(loss.item())
                
            avg_train_loss = np.mean(train_losses) if train_losses else 0.0
            
            # --- Validate ---
            avg_val_loss = float('inf')
            if val_loader:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        loss = model.calculate_loss(batch)
                        val_losses.append(loss.item())
                avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
            else:
                # 若無 Val set，直接以 Train loss 為準 (不建議)
                avg_val_loss = avg_train_loss

            # --- Checkpoint & Early Stop ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, save_path)
                counter = 0
            else:
                counter += 1
                
            print(f"  Ep {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Patience: {counter}/{patience}")
            
            if counter >= patience:
                print("  Early stopping triggered.")
                break
                
        return best_state

    def _evaluate(self, model, test_loader, target_p_id) -> Dict[str, float]:
        """評估 Forward Transfer"""
        print(f"--- Evaluating on Period {target_p_id} (FWT) ---")
        model.eval()
        
        Ks = self.cfg['evaluation'].get('Ks', [5, 10, 20])
        n_neg = self.cfg['evaluation'].get('sampling_size', 99)
        
        hits = np.zeros(len(Ks))
        ndcgs = np.zeros(len(Ks))
        aucs = []
        n_pos = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing", leave=False):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                B = batch['user_id'].size(0)
                n_pos += B
                
                # 1. 負採樣
                #    需要 pos_item_id 和 user_id 來排除已看過的
                neg_ids = sample_negatives_batch(
                    self.seen_items_pool,
                    batch['item_id'],
                    batch['user_id'],
                    self.user_history_tracker,
                    n_neg,
                    self.device,
                    n_items=self.global_meta['n_items']
                )
                
                # 2. Inference
                #    pos_scores: [B], neg_scores: [B, n_neg]
                pos_scores, neg_scores, _ = model.inference(batch, neg_ids)
                
                # 3. Metrics Calculation
                #    Concat -> [B, 1 + n_neg] (Positive is at index 0)
                all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
                
                #    Rank: How many items have score > pos_score
                #    (Using broadcasting)
                #    Note: This is strictly >. If ties, this counts as better rank.
                #    If using >=, rank is conservative.
                ranks = (all_scores > pos_scores.unsqueeze(1)).sum(dim=1).cpu().numpy() # 0-based rank
                
                # AUC (Per Query)
                # pair-wise comparison: mean(pos > neg)
                auc = (pos_scores.unsqueeze(1) > neg_scores).float().mean(dim=1).cpu().numpy()
                aucs.extend(auc)
                
                # Top-K
                for i, k in enumerate(Ks):
                    # rank < k means rank is 0, 1, ..., k-1 (i.e., within top k)
                    hits[i] += (ranks < k).sum()
                    
                    # NDCG: 1 / log2(rank + 2)
                    # if hit: 1/log2(r+2), else 0
                    hit_mask = (ranks < k)
                    ndcgs[i] += (1.0 / np.log2(ranks[hit_mask] + 2)).sum()

        # Aggregate
        metrics = {
            'period': target_p_id,
            'gauc': np.mean(aucs) if aucs else 0.0,
            'auc': np.mean(aucs) if aucs else 0.0 # user-averaged AUC is conceptually GAUC here
        }
        for i, k in enumerate(Ks):
            metrics[f'recall@{k}'] = hits[i] / n_pos if n_pos > 0 else 0.0
            metrics[f'ndcg@{k}'] = ndcgs[i] / n_pos if n_pos > 0 else 0.0
            

        print(f"\nPeriod {target_p_id} (Test) Evaluation Finished ---")
        print(f"  - GAUC     : {metrics.get('gauc', 0.0):.4f}")
        print(f"  - AUC      : {metrics.get('auc', 0.0):.4f}")
        print("  -----------------------------------------------------")
        if metrics:
            for k in Ks:
                print(f"  - Recall@{k:<2} : {metrics.get(f'recall@{k}', 0.0):.4f}   |   NDCG@{k:<2} : {metrics.get(f'ndcg@{k}', 0.0):.4f}")
        else:
            print("  - No positive samples for Recall/NDCG in this period.")
        print("  -----------------------------------------------------")

             
        return metrics

    def _update_history_tracker(self, df: pd.DataFrame):
        """將當前 DataFrame 的互動紀錄加入全域 History"""
        # Group by user for efficiency
        interactions = df.groupby('userId')['itemId'].apply(set).to_dict()
        for u, items in interactions.items():
            if u not in self.user_history_tracker:
                self.user_history_tracker[u] = set()
            self.user_history_tracker[u].update(items)
            
        self.seen_items_pool.update(df['itemId'].unique())

    def _print_summary(self, lr):
        """列印整場實驗的平均指標"""
        if not self.results_log:
            return
        
        Ks = self.cfg['evaluation'].get('Ks', [5, 10, 20])
            
        print(f"\n{'='*20} [ Summary for LR {lr} ] {'='*20}")
        df_res = pd.DataFrame(self.results_log)
        mean_res = df_res.mean(numeric_only=True)

        print(f"  - GAUC     : {mean_res.get('gauc', 0.0):.4f}")
        print(f"  - AUC      : {mean_res.get('auc', 0.0):.4f}")
        print("  -----------------------------------------------------")
        if mean_res is not None and not mean_res.empty:
            for k in Ks:
                print(f"  - Recall@{k:<2} : {mean_res.get(f'recall@{k}', 0.0):.4f}   |   NDCG@{k:<2} : {mean_res.get(f'ndcg@{k}', 0.0):.4f}")
        else:
            print("  - No positive samples for Recall/NDCG in this period.")
        print("  -----------------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    
    # Load Config
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        exit(1)
        
    # Set Seed
    set_seed(config.get('seed', 2354))
    
    # Enforce Deterministic Behavior for Debugging (Optional)
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # Run
    trainer = StreamingTrainer(config)
    trainer.run()
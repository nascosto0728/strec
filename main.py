import os
import time
import yaml
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
import copy
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import HyperLoRASASRec, SASRec_MLP, EmbMLP

from utils import (
    prepare_data_from_dfs,
    RecommendationDataset, 
    sample_negative_items, 
)



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

# ######################################################################
# # 輔助函式：模型評估
# ######################################################################

def run_evaluation(
    model: torch.nn.Module, 
    test_loader: DataLoader, 
    item_history_dict: Dict[int, set], 
    seen_items_pool: set, 
    config: Dict[str, Any], 
    device: torch.device, 
    Ks: List[int], 
    sampling_size: int
) -> Dict[str, float]:
    """
    評估函式 (保持不變)
    """
    model.eval() 
    
    total_recalls = np.zeros(len(Ks))
    total_ndcgs = np.zeros(len(Ks))
    all_per_sample_aucs = []
    all_user_ids_for_gauc = []
    total_positive_items = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            batch_cpu = {k: v for k, v in batch.items()}
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_size = len(batch['users'])
            total_positive_items += batch_size

            # 1. 負採樣
            neg_item_ids_list = []
            for i in range(batch_size):
                item_j_id = batch['items'][i].item()
                user_i_id = batch_cpu['users'][i].item()
                seen_items = item_history_dict.get(user_i_id, set())
                
                neg_items = sample_negative_items(
                    item_pool=seen_items_pool,
                    seen_items_set=seen_items,
                    positive_item_id=item_j_id,
                    num_samples=sampling_size,
                    device=device
                )
                neg_item_ids_list.append(neg_items)
            
            neg_item_ids_batch = torch.stack(neg_item_ids_list) 

            # 2. 推論
            pos_logits, neg_logits, _ = model.inference(batch, neg_item_ids_batch)

            # 3. 排名
            pos_logits = pos_logits.unsqueeze(1) 
            all_logits = torch.cat([pos_logits, neg_logits], dim=1) 
            ranks = (all_logits > pos_logits).sum(dim=1).cpu().numpy()

            # 4. AUC
            auc_per_sample = (pos_logits > neg_logits).float().mean(dim=1)
            all_per_sample_aucs.extend(auc_per_sample.cpu().numpy())
            all_user_ids_for_gauc.extend(batch_cpu['users_raw'].numpy())

            # 5. Recall / NDCG
            for rank in ranks:
                for j, k in enumerate(Ks):
                    if rank < k:
                        total_recalls[j] += 1
                        total_ndcgs[j] += 1 / np.log2(rank + 2)
    
    metrics = {}
    if total_positive_items > 0:
        gauc_df = pd.DataFrame({'user': all_user_ids_for_gauc, 'auc': all_per_sample_aucs})
        metrics['auc'] = np.mean(all_per_sample_aucs)
        
        user_auc_mean = gauc_df.groupby('user')['auc'].mean()
        user_counts = gauc_df.groupby('user').size()
        metrics['gauc'] = (user_auc_mean * user_counts).sum() / user_counts.sum()
        
        final_recalls = total_recalls / total_positive_items
        final_ndcgs = total_ndcgs / total_positive_items
        for k, rec, ndcg in zip(Ks, final_recalls, final_ndcgs):
            metrics[f'recall@{k}'] = rec
            metrics[f'ndcg@{k}'] = ndcg
            
    return metrics


# ######################################################################
# # 主實驗函式 (Refactored: No ER, No KD, No GNN)
# ######################################################################

def run_experiment(config: Dict[str, Any]):
    print("--- 0. Setting up device ---")
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("--- Using CUDA. ---")
    else:
        print("--- Using CPU. ---")

    random.seed(config.get('seed', 42))
    torch.manual_seed(config.get('seed', 42))
    np.random.seed(config.get('seed', 42))

    # --- 1. 數據 I/O ---
    print("--- 1. Loading raw data files ---")
    full_data_df = pd.read_parquet(config['data_path'])
    if config.get('debug_sample', 0) > 0:
        sample_ratio = config['debug_sample']
        print(f"--- [DEBUG] Sampling 1/{sample_ratio} of data ---")
        full_data_df = full_data_df.iloc[::sample_ratio]  
    full_meta_df = pd.read_parquet(config['meta_path'])

    # --- 2. 數據轉換 ---
    print("--- 2. Processing and remapping data ---")
    remapped_full_df, cates, cate_lens, hyperparams_updates, item_map, full_cate_map = prepare_data_from_dfs(
        full_data_df, full_meta_df, config
    )
    hyperparams = {**config['model'], **hyperparams_updates}
    config['model'] = hyperparams 
    
    # 轉換為 NumPy
    cates_np = np.array(cates)
    cate_lens_np = np.array(cate_lens)
    
    # --- 驗證資料 ---
    if remapped_full_df.empty:
        print("!!! FATAL ERROR: remapped_full_df is empty! !!!")
        return
    print(f"Total rows: {len(remapped_full_df)}")

    # --- 全局評估資訊 ---
    sampling_size = config['evaluation'].get('sampling_size', 99)
    Ks = config['evaluation'].get('Ks', [5, 10, 20, 50])

    # --- 3. 進入學習率調參迴圈 ---
    for lr in config['learning_rates']:
        print(f"\n{'='*50}\nStarting run with Learning Rate: {lr}\n{'='*50}")
        dir_name_with_lr = f"{config['dir_name']}_lr{lr}"
        
        # 儲存過去的測試 DataLoader，用於回溯評估 (BWT)
        past_test_loaders = {} 
        results_over_periods = []
        
        item_history_dict = {} 
        seen_items_pool = set()
        
        # --- 4. 增量訓練主迴圈 ---
        for period_id in range(config['train_start_period'], config['num_periods']):
            print(f"\n{'='*25} Period {period_id} {'='*25}")
            
            # --- 4.1 資料準備 ---
            current_period_df = remapped_full_df[remapped_full_df['period'] == period_id]
            
            if current_period_df.empty:
                print(f"No training data for period {period_id}. Skipping.")
                continue
            
            positive_samples_count = (current_period_df['label'] == 1).sum()
            print(f"--- [Data] Period {period_id}. Rows: {len(current_period_df)}, Pos: {positive_samples_count}")
            
            if positive_samples_count == 0:
                print(f"--- [Warning] No positive samples. Updating history only.")
                seen_items_pool.update(current_period_df['itemId'].unique())
                continue
                
            # 準備測試集 (T+1)
            test_loader_current = None
            if period_id >= config['test_start_period'] and (period_id + 1) < config['num_periods']:
                test_set = remapped_full_df[remapped_full_df['period'] == (period_id + 1)].iloc[::5] 
                
                if not test_set.empty:
                    test_set_pos = test_set[test_set['label'] == 1].copy()
                    if not test_set_pos.empty:
                        test_dataset_pos = RecommendationDataset(test_set_pos, {})
                        test_loader_current = DataLoader(
                            test_dataset_pos,
                            batch_size=config['model']['batch_size'],
                            shuffle=False,
                            num_workers=config.get('num_workers', 0)
                        )
                        past_test_loaders[period_id + 1] = test_loader_current

            # 更新歷史資訊
            current_period_interactions_dict = current_period_df.groupby('userId')['itemId'].apply(set).to_dict()
            for user_id, item_set in current_period_interactions_dict.items():
                item_history_dict.setdefault(user_id, set()).update(item_set)
            seen_items_pool.update(current_period_df['itemId'].unique())
            
            # --- 4.2 訓練/驗證集切分 ---
            val_split_ratio = config.get('validation_split', 0.2)
            if val_split_ratio > 0:
                split_point = int(len(current_period_df) * (1.0 - val_split_ratio))
                train_set_final_df = current_period_df.iloc[:split_point]
                val_set_df = current_period_df.iloc[split_point:]
            else:
                train_set_final_df = current_period_df
                val_set_df = None
            
            train_dataset = RecommendationDataset(train_set_final_df, {}) 
            train_loader = DataLoader(
                train_dataset, 
                batch_size=config['model']['batch_size'], 
                shuffle=True, 
                num_workers=config.get('num_workers', 0)
            )
            
            val_loader = None
            if val_set_df is not None and not val_set_df.empty:
                val_dataset = RecommendationDataset(val_set_df, {})
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=config['model']['batch_size'],
                    shuffle=False, 
                    num_workers=config.get('num_workers', 0)
                )

            # --- 4.3 模型初始化與載入 ---
            model_type = config.get('model_type', 'hyper_lora_sasrec')
            print(f"--- Building model: {model_type} ---")
            
            if model_type == 'mlp':
                model = EmbMLP(cates_np, cate_lens_np, hyperparams, config).to(device)
            elif model_type == 'sasrec':
                model = SASRec_MLP(cates_np, cate_lens_np, hyperparams, config).to(device)
            elif model_type == 'hyper_lora_sasrec':
                model = HyperLoRASASRec(cates_np, cate_lens_np, hyperparams, config).to(device)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            
            transformer_params = list(map(id, model.item_seq_transformer_hyper.parameters())) + \
                                list(map(id, model.user_seq_transformer_hyper.parameters())) + \
                                list(map(id, model.hyper_gate_net.parameters()))
            base_params = filter(lambda p: id(p) not in transformer_params, model.parameters())

            optimizer = torch.optim.Adam([
                {'params': base_params, 'lr': lr},            # MLP 和 Embedding 正常學習
                {'params': model.item_seq_transformer_hyper.parameters(), 'lr': lr * 0.1}, # Transformer 慢速微調
                {'params': model.user_seq_transformer_hyper.parameters(), 'lr': lr * 0.1},
                {'params': model.hyper_gate_net.parameters(), 'lr': lr * 1}
            ], weight_decay=config.get('weight_decay', 0.0))
            # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.get('weight_decay', 0.0))

            # 載入上一期權重 (Streaming Simulation)
            if period_id > config['train_start_period']:
                prev_ckpt_dir = os.path.join('./checkpoints', dir_name_with_lr, f'period_{period_id-1}')
                ckpt_path = os.path.join(prev_ckpt_dir, 'best_model.pth')
                if os.path.exists(ckpt_path):
                    try:
                        print(f"--- Loading weights from: {ckpt_path} ---")
                        state_dict = torch.load(ckpt_path, map_location=device) 
                        model.load_state_dict(state_dict)
                    except Exception as e:
                        print(f"--- [Warning] Could not load weights: {e} ---")
                else:
                    print(f"--- [Warning] No checkpoint found. Training from scratch. ---")

            # --- 5. 訓練迴圈 ---
            max_epochs = config.get('max_epochs', 10)
            patience = config.get('patience', 3)
            patience_counter = 0
            best_val_loss = float('inf')
            
            period_ckpt_dir = os.path.join('./checkpoints', dir_name_with_lr, f'period_{period_id}')
            os.makedirs(period_ckpt_dir, exist_ok=True)
            best_model_path = os.path.join(period_ckpt_dir, 'best_model.pth')
            
            pbar_epochs = tqdm(range(1, max_epochs + 1), desc="Epochs", leave=True)
            for epoch_id in pbar_epochs:
                # (a) 訓練
                model.train()
                losses = []
                for batch in train_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    optimizer.zero_grad()
                    loss = model.calculate_loss(batch) # 這裡不再需要 teacher 參數
                    
                    if torch.isnan(loss):
                        continue
                                             
                    loss.backward() 

                    optimizer.step()
                    losses.append(loss.item())

                if not losses:
                    break 
                else:
                    avg_train_loss = np.mean(losses)
                
                # (b) 驗證
                if val_loader is None:
                    torch.save(model.state_dict(), best_model_path)
                    continue

                model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        loss = model.calculate_loss(batch)
                        if not torch.isnan(loss):
                            val_losses.append(loss.item())
                        
                if not val_losses:
                    avg_val_loss = float('inf')
                else:
                    avg_val_loss = np.mean(val_losses)

                # (c) Early Stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), best_model_path)
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break 

                pbar_epochs.set_postfix(train_loss=f"{avg_train_loss:.4f}", val_loss=f"{avg_val_loss:.4f}")
            
            print(f"--- Training finished for period {period_id}. ---")

            # --- 6. 評估階段 ---
            if test_loader_current is not None:
                if os.path.exists(best_model_path):
                    model.load_state_dict(torch.load(best_model_path, map_location=device))
                
                # 6.1 BWT (Backward Transfer)
                print(f"--- Running Backward Transfer (BWT) Evaluation ---")
                backward_metrics_list = []
                for past_period_id, past_loader in past_test_loaders.items():
                    if past_period_id == (period_id + 1): continue
                    
                    past_metrics = run_evaluation(
                        model, past_loader, item_history_dict, seen_items_pool, 
                        config, device, Ks, sampling_size
                    )
                    backward_metrics_list.append(past_metrics)
                    print(f"    - Period {past_period_id} GAUC: {past_metrics.get('gauc', 0.0):.4f}")
                
                if backward_metrics_list:
                    avg_bwt_gauc = np.mean([m.get('gauc', 0.0) for m in backward_metrics_list])
                    print(f"--- Average BWT GAUC: {avg_bwt_gauc:.4f} ---")

                # 6.2 FWT (Forward Transfer)
                print(f"--- Running Forward Transfer (FWT) Evaluation (Period {period_id + 1}) ---")
                metrics_forward = run_evaluation(
                    model, test_loader_current, item_history_dict, seen_items_pool, 
                    config, device, Ks, sampling_size
                )
                results_over_periods.append(metrics_forward)
                
                
                print(f"\n--- Period {period_id} (Model) -> Period {period_id + 1} (Test) Evaluation Finished ---")
                print(f"  - GAUC     : {metrics_forward.get('gauc', 0.0):.4f}")
                print(f"  - AUC      : {metrics_forward.get('auc', 0.0):.4f}")
                print("  -----------------------------------------------------")
                if metrics_forward:
                    for k in Ks:
                        print(f"  - Recall@{k:<2} : {metrics_forward.get(f'recall@{k}', 0.0):.4f}   |   NDCG@{k:<2} : {metrics_forward.get(f'ndcg@{k}', 0.0):.4f}")
                else:
                    print("  - No positive samples for Recall/NDCG in this period.")
                print("  -----------------------------------------------------")

        # --- 7. 總結報告 ---
        if results_over_periods:
            print(f"\n{'='*20} [ Learning Rate {lr} Summary ] {'='*20}")
            avg_metrics = {}
            report_metric_keys = ['gauc', 'auc'] + [f'recall@{k}' for k in Ks] + [f'ndcg@{k}' for k in Ks]
            
            for key in report_metric_keys:
                valid_values = [m.get(key) for m in results_over_periods if m.get(key) is not None and np.isfinite(m.get(key))]
                avg_metrics[key] = np.mean(valid_values) if valid_values else 0.0
                
            print("\n--- Average Forward Transfer Metrics ---")
            print(f"  - GAUC     : {avg_metrics.get('gauc', 0.0):.4f}")
            print(f"  - AUC      : {avg_metrics.get('auc', 0.0):.4f}")
            print("-------------------------------------------------------")
            for k in Ks:
                print(f"  - Recall@{k:<2} : {avg_metrics.get(f'recall@{k}', 0.0):.4f}   |   NDCG@{k:<2} : {avg_metrics.get(f'ndcg@{k}', 0.0):.4f}")
            print("-------------------------------------------------------")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming Recommendation (Cleaned)")
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config not found at {args.config}")
        exit()

    os.environ['TORCH_USE_CUDA_DSA'] = '1'  # 啟用 CUDA DSA
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 啟用 CUDA Launch Blocking
    
    run_experiment(config)
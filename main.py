import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import random
import os
import gc
from collections import defaultdict
from torch.utils.data import DataLoader

from model import DualTowerTitans
from trainer import StreamingTrainer
from utils import (
    prepare_data_pipeline, 
    RecommendationDataset, 
    UniqueUserBatchSampler
)

def set_seed(seed):
    """固定隨機種子以確保實驗可重現性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_metrics(period_id, metrics, prefix="Eval"):
    """美化打印指標 (還原詳細格式)"""
    if not metrics:
        return
    
    # 提取 Loss 與 AUC
    loss_val = metrics.get('loss', 0.0)
    auc_val = metrics.get('auc', 0.0)
    
    print(f"  [{prefix} Period {period_id}] Results:")
    print(f"  - Loss     : {loss_val:.4f}")
    print(f"  - GAUC     : {auc_val:.4f}") # In-Batch User Average AUC = GAUC
    print("  " + "-"*45)
    
    # 處理 Top-K 指標
    # 找出所有的 k 值
    k_set = set()
    for key in metrics.keys():
        if key.startswith('recall@'):
            k_set.add(int(key.split('@')[1]))
    sorted_ks = sorted(list(k_set))
    
    for k in sorted_ks:
        r_k = metrics.get(f'recall@{k}', 0.0)
        n_k = metrics.get(f'ndcg@{k}', 0.0)
        print(f"  - Recall@{k:<2}: {r_k:.4f}   |   NDCG@{k:<2}: {n_k:.4f}")
        
    print("  " + "-"*45)

def main():
    # 1. Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    # 2. Load Config
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # 3. Setup
    set_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"🚀 Starting Streaming Experiment on {device}")
    print(f"{'='*60}\n")

    # 4. Data Pipeline
    #    這會觸發 Cache 讀取或重新建構
    full_df, global_meta, _ = prepare_data_pipeline(cfg)
    
    # 注入 Meta
    cfg['model']['n_users'] = global_meta['n_users']
    cfg['model']['n_items'] = global_meta['n_items']
    cfg['model']['n_cates'] = global_meta['n_cates']
    
    # 5. Model & Trainer Init
    model = DualTowerTitans(global_meta, cfg).to(device)

    # === [DEBUG Tool] Register NaN Hooks ===
    def check_nan_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any():
                print(f"!!! NaN detected in {module.__class__.__name__} !!!")
                # print(f"Output: {output}") # 印出數值會太多，先只印層名
                raise RuntimeError(f"NaN detected in layer: {module}")
        elif isinstance(output, tuple):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor) and torch.isnan(out).any():
                    print(f"!!! NaN detected in {module.__class__.__name__} output[{i}] !!!")
                    raise RuntimeError(f"NaN detected in layer: {module}")

    # 註冊 Hook 到所有子模組
    print("--- [Debug] Registering NaN hooks to all layers ---")
    for name, layer in model.named_modules():
        layer.register_forward_hook(check_nan_hook)
    # =======================================


    trainer = StreamingTrainer(model, cfg, device)

    # 6. Experiment Control Parameters
    exp_cfg = cfg.get('experiment', {})
    
    # 從資料中取得實際最大 Period
    max_data_period = int(full_df['period'].max())
    # 設定實驗終止點 (取 Config 與 Data 的最小值)
    target_num_periods = exp_cfg.get('num_periods', 9999)
    end_p = min(target_num_periods, max_data_period + 1)
    
    start_train_p = exp_cfg.get('train_start_period', 0)
    start_test_p = exp_cfg.get('test_start_period', 0)
    
    results_log = []

    print(f"--- Plan: Run Period 0 -> {end_p - 1} ---")
    print(f"--- Train Start (Stage 2): P{start_train_p} | Test Start: P{start_test_p} ---\n")

    # 7. Streaming Loop
    for p_id in range(end_p):
        print(f"\n{'='*40}\n>>> Period {p_id} Processing\n{'='*40}")
        
        # -------------------------------------------------
        # A. Data Slicing
        # -------------------------------------------------
        curr_df = full_df[full_df['period'] == p_id]
        
        # 下一個 Period (用於 Forward Transfer Eval)
        next_p_id = p_id + 1
        next_df = full_df[full_df['period'] == next_p_id] if next_p_id < (max_data_period + 1) else None

        if curr_df.empty:
            print(f"Period {p_id} is empty. Skipping.")
            continue

        # -------------------------------------------------
        # B. Stage 1: Step Mode (Always Run for Memory Update)
        # -------------------------------------------------
        # 即使 p_id < start_train_p，我們通常也要跑 Stage 1 來更新 Memory Bank，
        # 否則 Memory 會是空的。
        print(f"[Data] Step Loader: {len(curr_df)} interactions")
        
        ds_step = RecommendationDataset(
            curr_df, 
            mode='step', 
            max_seq_len=cfg['model']['max_seq_len'] # 這裡用作 context window padding
        )
        
        unique_sampler = UniqueUserBatchSampler(
            ds_step.user_ids, 
            batch_size=cfg['train']['batch_size_stage1']
        )
        
        loader_step = DataLoader(
            ds_step,
            batch_sampler=unique_sampler, 
            num_workers=0
        )

        # -------------------------------------------------
        # C. Stage 2: Seq Mode (Conditional)
        # -------------------------------------------------
        loader_seq = None
        # 條件：非 P0 (無歷史) 且 達到訓練起始期
        if p_id > 0 and p_id >= start_train_p:
            print(f"[Data] Seq Loader: Preparing sequences...")
            ds_seq = RecommendationDataset(
                curr_df, 
                mode='seq', 
                max_seq_len=cfg['model']['max_seq_len'], # L_max
                context_len=cfg['model']['context_len']  # K
            )
            
            if len(ds_seq) > 0:
                loader_seq = DataLoader(
                    ds_seq,
                    batch_size=cfg['train']['batch_size_stage2'],
                    shuffle=True, 
                    num_workers=0
                )
            else:
                print("[Warning] No valid sequences found for Stage 2.")
        else:
             print(f"[Control] Skipping Stage 2 (Before train_start_period {start_train_p})")

        # -------------------------------------------------
        # D. Eval Loader (Conditional)
        # -------------------------------------------------
        loader_eval = None
        if next_df is not None and not next_df.empty and next_p_id >= start_test_p:
            print(f"[Data] Eval Loader: Next Period {next_p_id} ({len(next_df)} samples)")
            ds_eval = RecommendationDataset(
                next_df,
                mode='step',
                max_seq_len=cfg['model']['max_seq_len']
            )
            
            loader_eval = DataLoader(
                ds_eval,
                batch_size=cfg['train']['batch_size_stage1'],
                shuffle=False,
                num_workers=0,
                drop_last=True
            )
        elif next_p_id < start_test_p:
            print(f"[Control] Skipping Evaluation (Next P{next_p_id} < test_start {start_test_p})")

        # -------------------------------------------------
        # E. Execute Period
        # -------------------------------------------------
        # run_period 內部會根據 loader 是否為 None 自動跳過相應階段
        period_metrics = trainer.run_period(
            p_id, 
            loader_step, 
            loader_seq, 
            loader_eval
        )
        
        # Log Results
        if period_metrics:
            # 補上 Period 資訊
            period_metrics['period'] = next_p_id 
            results_log.append(period_metrics)
            print_metrics(next_p_id, period_metrics, prefix="Final Eval")
        
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

    # -------------------------------------------------
    # F. Final Summary
    # -------------------------------------------------
    print("\n" + "="*60)
    print(" >>> Experiment Summary")
    print("="*60)
    
    if results_log:
        df_res = pd.DataFrame(results_log)
        mean_res = df_res.mean(numeric_only=True)
        
        print(f"Total Evaluated Periods: {len(df_res)}")
        
        # 1. 顯示平均 GAUC
        if 'auc' in mean_res:
            print(f"\n  [Average GAUC] : {mean_res['auc']:.4f}")
            print("  " + "-"*30)
        
        # 2. 顯示 Top-K
        # 解析所有的 K
        k_set = set()
        for key in mean_res.index:
            if key.startswith('recall@'):
                k_set.add(int(key.split('@')[1]))
        sorted_ks = sorted(list(k_set))
        
        for k in sorted_ks:
            r_val = mean_res.get(f'recall@{k}', 0.0)
            n_val = mean_res.get(f'ndcg@{k}', 0.0)
            print(f"  Recall@{k:<2} : {r_val:.4f}   |   NDCG@{k:<2} : {n_val:.4f}")
            
    else:
        print("No evaluation metrics collected.")
    print("="*60)

if __name__ == "__main__":
    main()
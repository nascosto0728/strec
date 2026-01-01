import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict
import logging
from tqdm import tqdm  # [New] Import tqdm

# 簡單的統計工具
class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class StreamingTrainer:
    def __init__(self, model, cfg, device, logger=None):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        # Hyperparams
        self.lr_stage1 = cfg['train'].get('lr_stage1', 1e-3)
        self.lr_stage2 = cfg['train'].get('lr_stage2', 1e-4)
        
        # 從 Config 讀取 K List
        self.k_list = cfg['evaluation'].get('Ks', [10, 20])
        
        # Gradient Clipping Threshold
        self.max_grad_norm = cfg['train'].get('max_grad_norm', 5.0)
        
        self.optimizer = None

    def _init_optimizer(self, stage: int):
        """根據 Stage 初始化 Optimizer"""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        lr = self.lr_stage1 if stage == 1 else self.lr_stage2
        
        # 強制將 weight_decay 轉為 float
        wd = float(self.cfg['train'].get('weight_decay', 1e-5))
        
        self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=wd)
        
        if self.logger:
            self.logger.info(f"Initialized Optimizer for Stage {stage}: {len(trainable_params)} tensor groups, LR={lr}, WD={wd}")

    def _update_meters(self, meters, outputs, batch_size):
        """通用 Metrics 更新函式"""
        meters['loss'].update(outputs['loss'].item(), batch_size)
        
        # 自動抓取所有 recall@k, ndcg@k, auc
        for k, v in outputs.items():
            if k.startswith('recall') or k.startswith('ndcg') or k == 'auc':
                meters[k].update(v, batch_size)

    def train_step_epoch(self, dataloader: DataLoader, period_id: int):
        """Stage 1: Step-based Training loop (with tqdm)"""
        self.model.train()
        meters = defaultdict(AverageMeter)
        
        # [New] Tqdm Progress Bar
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"P{period_id} [S1 Train]", leave=False)
        
        for batch_idx, batch in pbar:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model.forward_stage1(
                batch, 
                training=True, 
                update_cache=True, 
                compute_metrics=True,
                k_list=self.k_list
            )
            
            loss = outputs['loss']
            loss.backward()
            
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Update Meters
            batch_size = batch['user_id'].size(0)
            self._update_meters(meters, outputs, batch_size)
            
            # [New] Update Tqdm Postfix
            postfix_dict = {'loss': f"{meters['loss'].avg:.4f}"}
            if 'auc' in meters:
                postfix_dict['auc'] = f"{meters['auc'].avg:.4f}"
            if 'recall@10' in meters:
                postfix_dict['r10'] = f"{meters['recall@10'].avg:.4f}"
            
            pbar.set_postfix(postfix_dict)

        return {k: v.avg for k, v in meters.items()}

    def train_seq_epoch(self, dataloader: DataLoader, snapshot: tuple, period_id: int):
        """Stage 2: Sequence-based Training loop (with tqdm)"""
        self.model.eval() 
        meters = defaultdict(AverageMeter)
        
        # [New] Tqdm Progress Bar
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"P{period_id} [S2 Train]", leave=False)
        
        for batch_idx, batch in pbar:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model.forward_stage2(batch, snapshot)
            loss = outputs['loss']
            
            if loss.requires_grad:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            batch_size = batch['user_id'].size(0)
            meters['loss'].update(loss.item(), batch_size)
            
            # [New] Update Tqdm Postfix (Stage 2 目前只有 Loss)
            pbar.set_postfix({'loss': f"{meters['loss'].avg:.4f}"})

        return {k: v.avg for k, v in meters.items()}

    def evaluate(self, dataloader: DataLoader, update_memory: bool = False):
        """Evaluation Loop (with tqdm)"""
        self.model.eval()
        meters = defaultdict(AverageMeter)
        
        backup = None
        if update_memory:
            backup = self.model.memory_bank.backup_state()
        
        # [New] Tqdm Progress Bar
        desc = "Evaluation (Online)" if update_memory else "Evaluation"
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=desc, leave=False)
        
        with torch.no_grad():
            for batch_idx, batch in pbar:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                
                outputs = self.model.forward_stage1(
                    batch, 
                    training=update_memory,
                    update_cache=False,
                    compute_metrics=True,
                    k_list=self.k_list
                )
                
                batch_size = batch['user_id'].size(0)
                self._update_meters(meters, outputs, batch_size)
                
                # [New] Update Tqdm Postfix
                postfix_dict = {'loss': f"{meters['loss'].avg:.4f}"}
                if 'auc' in meters:
                    postfix_dict['auc'] = f"{meters['auc'].avg:.4f}"
                if 'recall@10' in meters:
                    postfix_dict['r10'] = f"{meters['recall@10'].avg:.4f}"
                
                pbar.set_postfix(postfix_dict)

        if update_memory and backup is not None:
            self.model.memory_bank.restore_state(backup)
            
        return {k: v.avg for k, v in meters.items()}

    def run_period(self, period_id, loader_step, loader_seq, loader_eval):
        """Orchestrate one full period"""
        self.logger.info(f"\n{'='*40}\nStarting Period {period_id}\n{'='*40}")
        
        # --- Stage 1 ---
        self.logger.info(f"--- Period {period_id}: Stage 1 (Greedy Step) ---")
        self.model.toggle_stage(1)
        self._init_optimizer(stage=1)
        
        metrics_s1 = self.train_step_epoch(loader_step, period_id)
        
        log_str = f"Loss: {metrics_s1['loss']:.4f}"
        if 'auc' in metrics_s1: log_str += f" | AUC: {metrics_s1['auc']:.4f}"
        self.logger.info(f"Period {period_id} Stage 1 Done. {log_str}")
        
        # --- Transition ---
        snapshot = None
        if period_id > 0 and loader_seq is not None:
            self.logger.info("Creating Memory Snapshot for Stage 2...")
            snapshot = self.model.memory_bank.create_snapshot()
        
        # --- Stage 2 ---
        if period_id > 0 and loader_seq is not None:
            self.logger.info(f"--- Period {period_id}: Stage 2 (Seq Memory) ---")
            self.model.toggle_stage(2)
            self._init_optimizer(stage=2)
            
            metrics_s2 = self.train_seq_epoch(loader_seq, snapshot, period_id)
            self.logger.info(f"Period {period_id} Stage 2 Done. Loss: {metrics_s2['loss']:.4f}")
            
            del snapshot
        else:
            self.logger.info(f"Skipping Stage 2 for Period {period_id}")

        # --- Evaluation ---
        metrics_eval = {}
        if loader_eval is not None:
            self.logger.info(f"--- Period {period_id}: Evaluation ---")
            metrics_eval = self.evaluate(loader_eval, update_memory=False)
            
            auc_val = metrics_eval.get('auc', 0.0)
            self.logger.info(f"Period {period_id} Eval Result: Loss={metrics_eval['loss']:.4f} | GAUC={auc_val:.4f}")
            
        return metrics_eval
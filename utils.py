import os
import hashlib
import json
import pickle
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler, BatchSampler
from collections import defaultdict, deque
from tqdm import tqdm
from typing import Dict, Any, List, Set, Tuple, Optional, Union

# ==============================================================================
#  1. Helper Functions (Padding & Hashing)
# ==============================================================================

def generate_cache_key(config: Dict) -> str:
    """根據影響資料生成的參數產生 Hash Key"""
    relevant_params = {
        'period_split_mode': config['data'].get('period_split_mode'),
        'period_interval_hours': config['data'].get('period_interval_hours'),
        'num_periods': config['data'].get('num_periods'),
        'history_mode': config['data'].get('history_mode'),
        'history_window_hours': config['data'].get('history_window_hours'),
        'max_seq_len': config['model'].get('max_seq_len', 50),
        'dataset_path': config['data_path']
    }
    param_str = json.dumps(relevant_params, sort_keys=True)
    return hashlib.md5(param_str.encode('utf-8')).hexdigest()

def pad_sequence_1d(
    seq: Union[List[int], np.ndarray], 
    max_len: int, 
    pad_val: int = 0, 
    padding: str = 'post'
) -> np.ndarray:
    """
    單一序列的 Padding 函式
    Args:
        padding: 'pre' (補在前面) 或 'post' (補在後面)
    """
    seq = np.array(seq, dtype=np.int64)
    length = len(seq)

    if length == 0:
        return np.full(max_len, pad_val, dtype=np.int64)
    if length == max_len:
        return seq
    
    if length > max_len:
        # 截斷：永遠保留最近的 (最後 max_len 個)
        return seq[-max_len:]
    
    # 需要 Padding
    padded = np.full(max_len, pad_val, dtype=np.int64)
    if padding == 'post':
        padded[:length] = seq
    elif padding == 'pre':
        padded[-length:] = seq
    else:
        raise ValueError(f"Unknown padding mode: {padding}")
        
    return padded

# ==============================================================================
#  2. Data Processing Core (Runtime Augmentation)
# ==============================================================================

def assign_periods(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """動態切分 Period"""
    mode = config['data'].get('period_split_mode', 'quantile')
    print(f"--- [Runtime] Splitting Periods (Mode: {mode}) ---")

    if mode == 'time':
        interval_hours = config['data'].get('period_interval_hours', 168)
        if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        start_time = df['timestamp'].min()
        time_delta = (df['timestamp'] - start_time).dt.total_seconds() / 3600.0
        df['period'] = (time_delta // interval_hours).astype(int)
        
    elif mode == 'quantile':
        n_periods = config['data'].get('num_periods', 30)
        ts_numeric = df['timestamp'].astype(np.int64)
        df['period'] = pd.qcut(ts_numeric, n_periods, labels=False)
        
    return df

def build_runtime_history(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """建立 user_interacted_items (Context) 與 item_interacted_users"""
    mode = config['data'].get('history_mode', 'count')
    max_len = config['model'].get('max_seq_len', 50)
    print(f"--- [Runtime] Building Interaction History (Mode: {mode}) ---")
    
    time_window_sec = 0
    if mode == 'time':
        hours = config['data'].get('history_window_hours', 24)
        time_window_sec = hours * 3600
        
    # 容器初始化
    n_rows = len(df)
    user_seqs = np.empty(n_rows, dtype=object)
    item_seqs = np.empty(n_rows, dtype=object)
    user_lens = np.zeros(n_rows, dtype=np.int32)
    item_lens = np.zeros(n_rows, dtype=np.int32)
    
    # 準備迭代數據
    timestamps = df['timestamp'].values.astype(np.int64) 
    if np.issubdtype(df['timestamp'].dtype, np.datetime64):
        timestamps = timestamps // 10**9
        
    user_ids = df['user_id'].values
    item_ids = df['item_id'].values
    
    u_hist = defaultdict(deque)
    i_hist = defaultdict(deque)
    
    for idx, (uid, iid, ts) in enumerate(tqdm(zip(user_ids, item_ids, timestamps), total=n_rows, desc="Seq Gen")):
        # Lazy Removal (Time window)
        if mode == 'time':
            cutoff = ts - time_window_sec
            while u_hist[uid] and u_hist[uid][0][1] < cutoff: u_hist[uid].popleft()
            while i_hist[iid] and i_hist[iid][0][1] < cutoff: i_hist[iid].popleft()
        
        # Max Len Constraint
        while len(u_hist[uid]) >= max_len: u_hist[uid].popleft()
        while len(i_hist[iid]) >= max_len: i_hist[iid].popleft()
            
        # Snapshot (取當前狀態作為 History)
        # 注意：這裡存的是 Python List，之後 Dataset 轉 Numpy
        user_seqs[idx] = [x[0] for x in u_hist[uid]]
        item_seqs[idx] = [x[0] for x in i_hist[iid]]
        user_lens[idx] = len(user_seqs[idx])
        item_lens[idx] = len(item_seqs[idx])
        
        # Update State (包含當前 item)
        u_hist[uid].append((iid, ts))
        i_hist[iid].append((uid, ts))
        
    df['user_interacted_items'] = user_seqs
    df['item_interacted_users'] = item_seqs
    df['user_interacted_len'] = user_lens
    df['item_interacted_len'] = item_lens
    return df

# ==============================================================================
#  3. Main Pipeline Function
# ==============================================================================

def prepare_data_pipeline(config: Dict) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    準備全量資料：
    1. 載入 Cache 或 重新建構 History
    2. 確保 ID Shift (從 1 開始)
    3. 強制 Index Reset (對齊 Numpy Array)
    """
    cache_dir = './cache_data'
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = generate_cache_key(config)
    cache_path = os.path.join(cache_dir, f"runtime_data_{cache_key}.parquet")
    
    print("--- [Data] Loading Static Meta ---")
    with open(config['meta_path'], 'rb') as f:
        static_meta = pickle.load(f)
        
    if os.path.exists(cache_path):
        print(f"--- [Cache] Found cached data: {cache_path} ---")
        df = pd.read_parquet(cache_path)
    else:
        print("--- [Cache] Building from scratch... ---")
        df = pd.read_parquet(config['data_path'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # [Crucial] ID Shift: Ensure 0 is padding
        # 假設原始 ID 從 0 開始，全部 +1
        print("--- [Data] Shifting IDs by +1 (0 reserved for padding) ---")
        df['user_id'] = df['user_id'] + 1
        df['item_id'] = df['item_id'] + 1
        
        df = assign_periods(df, config)
        df = build_runtime_history(df, config)
        
        print(f"--- [Cache] Saving to {cache_path}... ---")
        df.to_parquet(cache_path, index=False)

    # 強制重置 Index，確保 iloc 和 loc 一致，且對齊 Dataset 內部的 Numpy Array
    df = df.reset_index(drop=True)

    # 處理 Meta Matrix (也要 Shift IDs)
    item_to_cate = static_meta['item_to_cate']
    # 原始 meta 是 0-based，我們需要建立一個 shift 過的 matrix
    # n_items + 1 (for padding 0) + 1 (for shift) -> No, just n_items + 1 (0 is pad, 1..N is items)
    # 原始 n_items 是 max_id + 1。現在 max_id 變成了 original_max + 1。
    # 所以新的 n_items 應該是 static_meta['n_items'] + 1
    
    n_items_shifted = static_meta['n_items'] + 1
    n_cates_shifted = len(static_meta['cate_map']) + 1 # 假設 cate 也要 shift? 通常 cate 0 是 padding
    
    # 建立 Shifted Cate Matrix
    # Matrix Row 0 is Padding (all 0)
    # Row i對應 Item ID i
    max_cate_len = 5
    cate_matrix = np.zeros((n_items_shifted, max_cate_len), dtype=np.int64)
    
    for raw_iid, cates in item_to_cate.items():
        shifted_iid = raw_iid + 1
        # cates 也是 ID，假設 cate 也從 1 開始 (如果是 0-based)
        # 這裡假設 cate_map 已經處理好 ID，或者我們也統一 +1
        # 為了安全，假設 cates 裡面的值需要 +1
        shifted_cates = [c + 1 for c in cates][:max_cate_len]
        cate_matrix[shifted_iid, :len(shifted_cates)] = shifted_cates
        
    global_meta = {
        'n_users': static_meta['n_users'] + 1, # Shifted
        'n_items': n_items_shifted,
        'n_cates': n_cates_shifted + 1, # +1 for cate shift safety
        'cate_matrix': cate_matrix
    }
    
    id_maps = {
        'user_map': static_meta['user_map'], # 這是原始 ID -> 0-based ID 的映射
        'item_map': static_meta['item_map']
    }
    
    return df, global_meta, id_maps

# ==============================================================================
#  4. Recommendation Dataset (The Transformer)
# ==============================================================================

class RecommendationDataset(Dataset):
    def __init__(self, 
                 df: pd.DataFrame, 
                 mode: str = 'step', 
                 max_seq_len: int = 50,
                 context_len: int = 50): # K
        """
        Args:
            mode: 'step' (Row-based) or 'seq' (User-based Chunking)
            max_seq_len: Stage 1 的 history window 或 Stage 2 的 Period Length (L_max)
            context_len: Stage 2 的 Context Length (K)
        """
        # [Safety] 強制 reset index，並建立 Numpy Views
        self.data_df = df.reset_index(drop=True)
        self.mode = mode
        self.max_seq_len = max_seq_len # L_max (for Stage 2) or Window (for Stage 1)
        self.context_len = context_len # K (for Stage 2)
        
        # 轉為 Numpy Array 以加速存取 (Object reference for lists)
        self.user_ids = self.data_df['user_id'].values
        self.item_ids = self.data_df['item_id'].values
        self.user_hist_lists = self.data_df['user_interacted_items'].values
        self.item_hist_lists = self.data_df['item_interacted_users'].values
        
        # Pre-compute item/user lengths for item tower masking
        self.item_hist_lens = self.data_df['item_interacted_len'].values
        
        # Stage 2: Chunking Logic (User-based)
        if self.mode == 'seq':
            self.seq_tasks = self._build_seq_tasks()
            
    def _build_seq_tasks(self) -> List[Tuple[int, List[int]]]:
        """
        Stage 2 Chunking:
        將每個 User 的行為切分成多個 Chunk。
        Return: List of (anchor_row_idx, chunk_item_ids)
        """
        tasks = []
        # Group indices by user: {uid: [row_idx1, row_idx2...]}
        # 這裡假設 df 是時間排序的，所以 row_indices 也是時間排序的
        user_groups = self.data_df.groupby('user_id').indices
        
        for uid, row_indices in user_groups.items():
            # 該 User 在本期的所有 Item IDs
            # 直接用 Numpy indexing 取出
            period_items = self.item_ids[row_indices]
            total_items = len(period_items)
            
            # Chunking Strategy: Sliding Window or Fixed Chunks?
            # 根據討論：切分成多個 Chunk，保留斷層。
            # Chunk 1: 0~50, Anchor: row_indices[0]
            # Chunk 2: 50~100, Anchor: row_indices[50]
            
            step = self.max_seq_len # L_max
            for start_idx in range(0, total_items, step):
                end_idx = min(start_idx + step, total_items)
                
                # Anchor Reuse: 直接使用該 Chunk 第一個行為對應的 Row Index
                # 該 Row 的 user_interacted_items 就是這個 Chunk 所需的 Context
                anchor_row = row_indices[start_idx]
                
                # 取出 Chunk 的 Items
                chunk_items = period_items[start_idx:end_idx]
                
                tasks.append((anchor_row, chunk_items))
                
        return tasks

    def __len__(self) -> int:
        if self.mode == 'step':
            return len(self.data_df)
        else:
            return len(self.seq_tasks)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.mode == 'step':
            return self._get_step_item(idx)
        elif self.mode == 'seq':
            return self._get_seq_item(idx)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _get_step_item(self, idx: int) -> Dict[str, Any]:
        """Stage 1: Row-based Item"""
        uid = self.user_ids[idx]
        iid = self.item_ids[idx]
        
        # User Context (Padding)
        # Stage 1 通常使用 Post-padding (對齊 Item Tower) 或 Pre-padding? 
        # Standard Item Tower 需要 Post-padding (masking 方便)
        # User Tower (SW-Attn) 兩種皆可，這裡統一用 Post-padding
        u_hist = self.user_hist_lists[idx]
        u_hist_pad = pad_sequence_1d(u_hist, self.max_seq_len, padding='post')
        
        # Item Context (Padding)
        i_hist = self.item_hist_lists[idx]
        i_hist_pad = pad_sequence_1d(i_hist, self.max_seq_len, padding='post')
        i_len = min(len(i_hist), self.max_seq_len)
        
        return {
            'user_id': int(uid),
            'item_id': int(iid), # Target
            'user_interacted_items': u_hist_pad, # [K]
            # Item Tower Info
            'item_interacted_users': i_hist_pad,
            'item_interacted_len': int(i_len)
        }

    def _get_seq_item(self, idx: int) -> Dict[str, Any]:
        """Stage 2: Chunk-based Sequence [Context | Period]"""
        anchor_row, chunk_items = self.seq_tasks[idx]
        uid = self.user_ids[anchor_row]
        
        # 1. Context (From Anchor Row)
        #    Context 需要 Pre-Padding (靠右對齊)
        #    並且截斷到 context_len (K)
        context_raw = self.user_hist_lists[anchor_row]
        context_seq = pad_sequence_1d(context_raw, self.context_len, padding='pre')
        
        # 2. Period (From Chunk Items)
        #    Period 需要 Post-Padding (靠左對齊)
        #    截斷到 max_seq_len (L_max) -- 其實 chunking 已經保證了長度 <= L_max
        period_seq = pad_sequence_1d(chunk_items, self.max_seq_len, padding='post')
        
        # 3. Concatenate: [K + L_max]
        full_seq = np.concatenate([context_seq, period_seq])
        
        # 4. Mask Creation (Input-Aligned)
        #    0: Ignore (Context & Padding)
        #    1: Valid Period
        mask = np.zeros(len(full_seq), dtype=np.float32)
        
        # Mask 範圍: Context 結束後 (K) ~ Valid Chunk 結束前
        chunk_len = len(chunk_items)
        start_valid = self.context_len # K
        end_valid = start_valid + chunk_len
        mask[start_valid:end_valid] = 1.0
        
        return {
            'user_id': int(uid),
            'full_seq_items': full_seq.astype(np.int64), # [K+L]
            # Loss Mask (由 Model 負責 shift)
            'loss_mask': mask # [K+L]
        }

# ==============================================================================
#  5. Unique User Batch Sampler (Stage 1)
# ==============================================================================

class UniqueUserBatchSampler(BatchSampler):
    """
    長度優先 + 強制截斷 Sampler (Length-Sorted Stratified Sampler)
    
    策略：
    1. 依照 User 的資料長度由大到小排序。
    2. 每一輪 (Layer) 優先選取資料最長的前 N 個 User 組成 Batch。
    3. [重要] 當活躍 User 數量少於 min_batch_size 時，停止產出 (Drop Last)，
       防止 Batch 過小導致 In-Batch Negative Sampling 失效。
    """
    def __init__(self, user_ids: np.ndarray, batch_size: int, min_batch_size: int = 16):
        self.batch_size = batch_size
        self.min_batch_size = min_batch_size
        self.user_ids = user_ids
        
        # 預處理：建立 User -> Indices 的映射
        self.user_group_indices = defaultdict(list)
        for idx, uid in enumerate(user_ids):
            self.user_group_indices[uid].append(idx)
            
        # 計算總 Batch 數 (僅供參考，因為有 Drop 機制，實際會少一點)
        self.num_batches = len(user_ids) // batch_size

    def __iter__(self):
        # 1. 重建狀態
        local_user_indices = defaultdict(deque)
        # 同時維護一個「剩餘長度表」用來排序
        user_remaining_len = {}
        
        for uid, indices in self.user_group_indices.items():
            dq = deque(indices)
            local_user_indices[uid] = dq
            user_remaining_len[uid] = len(dq)
            
        # 初始活躍 User 池
        active_users = list(local_user_indices.keys())
        
        while True:
            # [Core Strategy 1] 依照剩餘長度排序 (由大到小)
            # 這樣可以保證長序列的 User 盡可能湊在一起被訓練
            # (雖然這會稍微增加一點排序開銷，但 User 數十萬級別還算快)
            active_users.sort(key=lambda u: user_remaining_len[u], reverse=True)
            
            # [Core Strategy 2] Min Batch Size Check (Drop Last)
            # 如果剩餘的活躍 User 不足以湊出最小安全 Batch，就結束 Epoch
            if len(active_users) < self.min_batch_size:
                break
                
            # 開始分層取樣
            # 因為已經排序過，我們直接依序切分 active_users
            # 這一層可以產出的 Batch 數量
            n_batches_in_layer = len(active_users) // self.batch_size
            
            # 如果這一層連一個完整 Batch 都湊不滿 (但大於 min)，就只產出一個 partial batch
            if n_batches_in_layer == 0:
                # 這裡 len(active_users) 介於 min ~ batch_size 之間
                # 直接把所有人打包成一個 Batch
                current_batch = []
                users_to_process = active_users # 全部
            else:
                # 正常情況，可以切出幾個 Full Batch
                # 我們只處理前 N * B 個 User (剩下的暫時不動，留給下一輪排序)
                n_take = n_batches_in_layer * self.batch_size
                users_to_process = active_users[:n_take]
            
            # 產生 Batch
            # 這裡我們利用 chunking
            for i in range(0, len(users_to_process), self.batch_size):
                batch_users = users_to_process[i : i + self.batch_size]
                
                batch_indices = []
                for uid in batch_users:
                    idx = local_user_indices[uid].popleft()
                    batch_indices.append(idx)
                    
                    # 更新剩餘長度
                    user_remaining_len[uid] -= 1
                    
                yield batch_indices
            
            # [Update Active Users]
            # 剔除已經沒資料的 User
            # 注意：我們只檢查剛才處理過的那些 users_to_process
            # 未處理的 users (tail of the sort) 當然還在
            new_active = []
            for uid in active_users:
                if user_remaining_len[uid] > 0:
                    new_active.append(uid)
            active_users = new_active
            
            # 如果沒有人了，結束
            if not active_users:
                break

    def __len__(self):
        return self.num_batches
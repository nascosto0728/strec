import os
import hashlib
import json
import pickle
import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from collections import defaultdict, deque
from tqdm import tqdm
from typing import Dict, Any, List, Set, Tuple

# ==============================================================================
#  1. 輔助函數：Padding 與 Hashing
# ==============================================================================

def pad_sequences_pure_python(sequences: np.ndarray, maxlen: int, value: int = 0) -> np.ndarray:
    """
    高效 Padding 函數
    sequences: 包含 list 的 numpy array (object dtype)
    """
    # 預先分配記憶體
    n = len(sequences)
    padded_matrix = np.full((n, maxlen), value, dtype=np.int32)
    
    for i, seq in enumerate(sequences):
        # seq 可能是 list 或 np.array
        length = len(seq)
        if length == 0:
            continue
        if length > maxlen:
            # Truncate (保留最近的)
            padded_matrix[i, :] = seq[-maxlen:]
        else:
            # Pad (Pre-pad 還是 Post-pad? 這裡使用 Post-pad 以對齊原始邏輯)
            # 若為 SASRec/Transformer，通常把 padding 放在前面或後面皆可，但在這裡填入有效值即可
            padded_matrix[i, :length] = seq
            
    return padded_matrix

def generate_cache_key(config: Dict) -> str:
    """
    根據影響資料生成的參數產生唯一的 Hash Key。
    如果改了 period 設定或 history 視窗，Hash 就會變，強迫重新生成。
    """
    # 提取相關參數
    relevant_params = {
        'period_split_mode': config['data'].get('period_split_mode'),
        'period_interval_hours': config['data'].get('period_interval_hours'),
        'num_periods': config['data'].get('num_periods'),
        'history_mode': config['data'].get('history_mode'),
        'history_window_hours': config['data'].get('history_window_hours'),
        'max_seq_len': config['model'].get('max_seq_len', 50),
        'dataset_path': config['data_path'] # 原始檔路徑
    }
    
    # 轉成 JSON string 並排序 key 以確保一致性
    param_str = json.dumps(relevant_params, sort_keys=True)
    return hashlib.md5(param_str.encode('utf-8')).hexdigest()

# ==============================================================================
#  2. 核心邏輯：動態資料生成 (Runtime Augmentation)
# ==============================================================================

def assign_periods(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """根據配置動態切分 Period"""
    mode = config['data'].get('period_split_mode', 'quantile')
    print(f"--- [Runtime] Splitting Periods (Mode: {mode}) ---")

    if mode == 'time':
        interval_hours = config['data'].get('period_interval_hours', 168) # 預設 1 週
        # 確保是 timestamp 格式
        if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
        start_time = df['timestamp'].min()
        time_delta = (df['timestamp'] - start_time).dt.total_seconds() / 3600.0
        df['period'] = (time_delta // interval_hours).astype(int)
        print(f"    Interval: {interval_hours} hrs. Total Periods: {df['period'].max() + 1}")

    elif mode == 'quantile':
        n_periods = config['data'].get('num_periods', 30)
        # qcut 需要數值型 timestamp
        ts_numeric = df['timestamp'].astype(np.int64)
        df['period'] = pd.qcut(ts_numeric, n_periods, labels=False)
        print(f"    Quantile Split. Total Periods: {n_periods}")
    else:
        raise ValueError(f"Unknown period_split_mode: {mode}")

    return df

def build_runtime_history(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    在記憶體中建立歷史序列。
    支援 'time' (時間視窗) 與 'count' (固定數量) 模式。
    """
    mode = config['data'].get('history_mode', 'count')
    max_len = config['model'].get('max_seq_len', 50)
    print(f"--- [Runtime] Building Interaction History (Mode: {mode}) ---")
    
    # 時間視窗設定
    time_window_sec = 0
    if mode == 'time':
        hours = config['data'].get('history_window_hours', 24)
        time_window_sec = hours * 3600
        print(f"    Time Window: {hours} hours")
    
    n_rows = len(df)
    
    # 初始化 Numpy Object Array (用來存 list)
    user_seqs = np.empty(n_rows, dtype=object)
    item_seqs = np.empty(n_rows, dtype=object)
    user_lens = np.zeros(n_rows, dtype=np.int32)
    item_lens = np.zeros(n_rows, dtype=np.int32)
    
    # 準備資料迭代器 (確保已排序)
    # 使用 Unix Timestamp (int) 進行比較最快
    if np.issubdtype(df['timestamp'].dtype, np.datetime64):
        timestamps = df['timestamp'].astype(np.int64) // 10**9
    else:
        timestamps = df['timestamp'].values.astype(np.int64)
        
    user_ids = df['user_id'].values
    item_ids = df['item_id'].values
    
    # 狀態容器: {id: deque([(id, timestamp), ...])}
    u_hist = defaultdict(deque)
    i_hist = defaultdict(deque)
    
    # Loop
    for idx, (uid, iid, ts) in enumerate(tqdm(zip(user_ids, item_ids, timestamps), total=n_rows, desc="Seq Gen")):
        
        # --- A. 清理過期歷史 (Lazy Removal) ---
        if mode == 'time':
            cutoff = ts - time_window_sec
            # 檢查 deque 頭部 (最舊的)，如果過期就 pop
            while u_hist[uid] and u_hist[uid][0][1] < cutoff:
                u_hist[uid].popleft()
            while i_hist[iid] and i_hist[iid][0][1] < cutoff:
                i_hist[iid].popleft()
        
        # --- B. 限制長度 (Max Len) ---
        # 即使是時間模式，也必須限制最大長度以防 OOM
        while len(u_hist[uid]) >= max_len:
            u_hist[uid].popleft()
        while len(i_hist[iid]) >= max_len:
            i_hist[iid].popleft()
            
        # --- C. 記錄當前狀態 (Snapshot) ---
        # 注意：取出 deque 中的 ID 部分
        u_snapshot = [x[0] for x in u_hist[uid]]
        i_snapshot = [x[0] for x in i_hist[iid]]
        
        user_seqs[idx] = u_snapshot
        item_seqs[idx] = i_snapshot
        user_lens[idx] = len(u_snapshot)
        item_lens[idx] = len(i_snapshot)
        
        # --- D. 更新狀態 ---
        u_hist[uid].append((iid, ts))
        i_hist[iid].append((uid, ts))
        
    # Assign back to DataFrame
    df['user_interacted_items'] = user_seqs
    df['item_interacted_users'] = item_seqs
    df['user_interacted_len'] = user_lens
    df['item_interacted_len'] = item_lens
    
    return df

def process_meta_matrix(meta_dict: Dict, config: Dict) -> Tuple[np.ndarray, np.ndarray, int]:
    """將 Meta Dict (從 pkl 讀入) 轉換為 Tensor 矩陣"""
    print("--- [Data] converting Meta Dict to Matrix ---")
    item_map = meta_dict['item_map']
    cate_map = meta_dict['cate_map']
    item_to_cate = meta_dict['item_to_cate']
    
    n_items = len(item_map) + 1
    n_cates = len(cate_map) + 1
    
    # 設定最大類別長度 (通常 5 夠用，也可寫入 config)
    max_cate_len = 5 
    
    cate_matrix = np.zeros((n_items, max_cate_len), dtype=np.int32)
    cate_lens = np.zeros(n_items, dtype=np.int32)
    
    for internal_iid, cates in item_to_cate.items():
        # cates 已經是 internal IDs list
        if not cates: continue
        
        trunc_cates = cates[:max_cate_len]
        cate_matrix[internal_iid, :len(trunc_cates)] = trunc_cates
        cate_lens[internal_iid] = len(trunc_cates)
        
    return cate_matrix, cate_lens, n_cates

# ==============================================================================
#  3. 主要 Pipeline (含 Cache)
# ==============================================================================

def prepare_data_pipeline(config: Dict) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    主資料處理管道。
    1. 檢查 Cache
    2. 若無 Cache，載入最小 log -> 切分 Period -> 建立 History -> 存 Cache
    3. 回傳 DataFrame 與 Meta
    """
    
    # 1. 計算 Cache Key 與 路徑
    cache_key = generate_cache_key(config)
    cache_dir = './cache_data'
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"runtime_data_{cache_key}.parquet")
    
    # 2. 嘗試載入 Meta (這是必須的)
    print("--- [Data] Loading Static Meta ---")
    with open(config['meta_path'], 'rb') as f:
        static_meta = pickle.load(f)
        
    # 3. 嘗試載入 Cache
    if os.path.exists(cache_path):
        print(f"--- [Cache] Found cached runtime data: {cache_path} ---")
        print("--- [Cache] Loading directly (Skipping history build)... ---")
        df = pd.read_parquet(cache_path)
    else:
        print("--- [Cache] No matching cache found. Building from scratch... ---")
        # A. 載入原始輕量 Log
        df = pd.read_parquet(config['data_path'])
        
        # B. 確保排序 (Double check)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # C. 切分 Period
        df = assign_periods(df, config)
        
        # D. 建立序列
        df = build_runtime_history(df, config)
        
        # E. 儲存 Cache
        print(f"--- [Cache] Saving to {cache_path} for future runs... ---")
        # Parquet 支援儲存 list column (PyArrow)
        df.to_parquet(cache_path, index=False)

    # 4. 處理 Meta Matrix
    cate_matrix, cate_lens, n_cates = process_meta_matrix(static_meta, config)
    
    # 5. 包裝全域 Meta
    global_meta = {
        'n_users': static_meta['n_users'],
        'n_items': static_meta['n_items'],
        'n_cates': n_cates,
        'cate_matrix': cate_matrix,
        'cate_lens': cate_lens
    }
    
    # 6. ID Maps (Evaluation 用)
    maps = {
        'user_map': static_meta['user_map'],
        'item_map': static_meta['item_map']
    }
    
    return df, global_meta, maps

# ==============================================================================
#  4. Dataset Class
# ==============================================================================

class RecommendationDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame, max_seq_len: int = 50):
        # 因為 df 來自 parquet read，記憶體應該是連續的
        self.data_df = data_df.reset_index(drop=True)
        self.max_seq_len = max_seq_len
        
        # 預先提取 Numpy Arrays 以加速 __getitem__
        # 注意: 這裡取出的是 Object array (contain lists)
        self.user_seqs = self.data_df['user_interacted_items'].values
        self.item_seqs = self.data_df['item_interacted_users'].values
        self.user_ids = self.data_df['user_id'].values
        self.item_ids = self.data_df['item_id'].values
        self.labels = self.data_df['label'].values
        
        # 這些在 Parquet 讀回來可能是 numpy int
        self.u_lens = self.data_df['user_interacted_len'].values
        self.i_lens = self.data_df['item_interacted_len'].values

    def __len__(self) -> int:
        return len(self.data_df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 1. 取得序列 (List or Array)
        u_seq = self.user_seqs[idx]
        i_seq = self.item_seqs[idx]
        
        # Parquet 讀回來有時會是 None (如果空的)
        if u_seq is None: u_seq = []
        if i_seq is None: i_seq = []
        
        # 2. Padding (On-the-fly)
        # 雖然 Pad 函式支援 Batch，但這裡做單筆也很快
        # 因為已經限制過 max_len，這裡主要是補 0
        u_seq_padded = np.zeros(self.max_seq_len, dtype=np.int32)
        i_seq_padded = np.zeros(self.max_seq_len, dtype=np.int32)
        
        u_len = min(len(u_seq), self.max_seq_len)
        i_len = min(len(i_seq), self.max_seq_len)
        
        if u_len > 0:
            # 取後段 (因為是 Post-Padding 且要保留最近行為)
            # 但我們在 build history 時已經確保了順序是 old -> new
            # 所以直接取 seq 即可 (已經截斷過了嗎? build history 有截斷，但這裡再保險一次)
             u_seq_padded[:u_len] = u_seq[-u_len:] # 取最後 u_len 個填入前段 (Post-Pad 補0在後??)
             # 修正：通常是 Post-Padding (數據在左，0在右)
             u_seq_padded[:u_len] = u_seq[-u_len:]
             
        if i_len > 0:
            i_seq_padded[:i_len] = i_seq[-i_len:]

        return {
            'user_id': int(self.user_ids[idx]),
            'item_id': int(self.item_ids[idx]),
            'labels': float(self.labels[idx]),
            
            # User Tower
            'user_interacted_items': u_seq_padded,
            'user_interacted_len': u_len,
            
            # Item Tower
            'item_interacted_users': i_seq_padded,
            'item_interacted_len': i_len,
        }

# ==============================================================================
#  5. Negative Sampling (維持原樣)
# ==============================================================================

def sample_negatives_batch(
    pos_item_ids: torch.Tensor,
    pos_user_ids: torch.Tensor,
    user_history_dict: Dict[int, Set[int]],
    n_samples: int,
    device: torch.device,
    n_items: int,
    candidate_pool: torch.Tensor = None 
) -> torch.Tensor:
    """
    高效版負採樣：採用拒絕採樣 (Rejection Sampling)
    """
    batch_size = pos_item_ids.size(0)
    
    # 1. 採樣
    if candidate_pool is not None:
        num_candidates = candidate_pool.size(0)
        rand_indices = torch.randint(0, num_candidates, (batch_size, n_samples), device=device)
        neg_samples = candidate_pool[rand_indices]
    else:
        neg_samples = torch.randint(0, n_items, (batch_size, n_samples), device=device)
    
    # 2. 衝突檢查 (CPU Fallback)
    pos_item_ids_cpu = pos_item_ids.cpu().numpy()
    pos_user_ids_cpu = pos_user_ids.cpu().numpy()
    neg_samples_cpu = neg_samples.cpu().numpy()
    
    if candidate_pool is not None:
        candidate_pool_cpu = candidate_pool.cpu().numpy()
        num_candidates = len(candidate_pool_cpu)
    else:
        candidate_pool_cpu = None
        
    for i in range(batch_size):
        uid = pos_user_ids_cpu[i]
        pos_item = pos_item_ids_cpu[i]
        seen_set = user_history_dict.get(uid, set())
        
        for j in range(n_samples):
            sampled_item = neg_samples_cpu[i, j]
            
            # 衝突條件：(已看過) OR (等於當前正樣本)
            while sampled_item in seen_set or sampled_item == pos_item:
                if candidate_pool_cpu is not None:
                    rand_idx = random.randint(0, num_candidates - 1)
                    sampled_item = candidate_pool_cpu[rand_idx]
                else:
                    sampled_item = random.randint(0, n_items - 1)
                neg_samples_cpu[i, j] = sampled_item
                
    return torch.from_numpy(neg_samples_cpu).to(device)
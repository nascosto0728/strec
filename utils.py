import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Tuple, List, Set
import random

def pad_sequences_pure_python(
    sequences: List[list], 
    maxlen: int, 
    value: int = 0
) -> np.ndarray:
    """
    簡易版 Padding 函式 (Post Padding, Post Truncating)。
    """
    padded_matrix = np.full((len(sequences), maxlen), value, dtype=np.int32)
    for i, seq in enumerate(sequences):
        # Truncate
        if len(seq) > maxlen:
            seq = seq[:maxlen]
        # Pad (Post) - 其實矩陣初始化已經做了，這裡只要填入有效值
        padded_matrix[i, :len(seq)] = seq
    return padded_matrix

class RecommendationDataset(Dataset):
    """
    Dataset 負責將 DataFrame 轉換為 Model 所需的 Tensor 字典。
    採用新的變數命名規範。
    """
    def __init__(self, data_df: pd.DataFrame, max_seq_len: int = 30):
        self.data_df = data_df.reset_index(drop=True)
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.data_df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data_df.iloc[idx]
        
        # 1. 解析序列 (需確保 DataFrame 中已經是 List 格式)
        # user_interacted_items: 用戶看過的 items
        u_history = row['user_interacted_items'] 
        # item_interacted_users: 看過該 item 的 users
        i_history = row['item_interacted_users']
        # i_history = i_history[:5]  

        # i_history = list(i_history) 
        # random.shuffle(i_history)
        # 2. Padding
        u_seq_padded = pad_sequences_pure_python([u_history], self.max_seq_len)[0]
        i_seq_padded = pad_sequences_pure_python([i_history], self.max_seq_len)[0]
        

        # 3. 建構回傳字典 (對齊新命名)
        return {
            # IDs
            'user_id': int(row['userId']),
            'item_id': int(row['itemId']),
            'labels': float(row['label']),
            
            # User Tower Input (User's History)
            'user_interacted_items': u_seq_padded,
            'user_interacted_len': min(len(u_history), self.max_seq_len),
            
            # Item Tower Input (Item's History)
            'item_interacted_users': i_seq_padded,
            'item_interacted_len': min(len(i_history), self.max_seq_len),
            
            # Meta info for evaluation
            'user_id_raw': row['userId_raw'],
            'item_id_raw': row['itemId_raw'],
        }

def process_category_data(meta_df: pd.DataFrame, item_map: Dict, max_cate_len: int = 5) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    處理物品類別資料，轉為矩陣格式。
    """
    # 建立 cate_map
    all_cates = set()
    for c_list in meta_df['cateId']:
        all_cates.update(c_list)

    cate_map = {c: i for i, c in enumerate(sorted(list(all_cates)), 1)}
    n_cates = len(cate_map) + 1 # 含 padding
    
    # 建立 item -> cate_indices 矩陣
    n_items_with_pad = len(item_map) + 1 # 這是 prepare_data_pipeline 傳進來的 map 長度 + 1
    # 分配矩陣：大小為 [Max_Item_ID + 1, Max_Cate_Len]
    # 預設值 0 即為 Padding
    cate_matrix = np.zeros((n_items_with_pad, max_cate_len), dtype=np.int32)
    cate_lens = np.zeros(n_items_with_pad, dtype=np.int32)
    
    # 建立查詢表以加速
    item_to_cates = dict(zip(meta_df['itemId'], meta_df['cateId']))
    
    # 填充矩陣 (依照 item_map 的順序 0 ~ N-1)
    # 反轉 item_map: internal_id -> raw_id
    id_to_raw = {v: k for k, v in item_map.items()}
    
    for internal_id in range(1, n_items_with_pad):
        raw_id = id_to_raw.get(internal_id)
        if raw_id in item_to_cates:
            raw_cates = item_to_cates[raw_id]
            # 轉換為 shifted cate id
            mapped_cates = [cate_map[c] for c in raw_cates if c in cate_map][:max_cate_len]
            cate_matrix[internal_id, :len(mapped_cates)] = mapped_cates
            cate_lens[internal_id] = len(mapped_cates)
            
    return cate_matrix, cate_lens, n_cates

def prepare_data_pipeline(config: Dict) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    主資料處理管道。
    回傳:
        - remapped_df: 處理好的全量 DataFrame
        - global_meta: 包含 cates, n_users 等全域資訊的字典
        - maps: ID 映射表
    """
    print("--- [Data] Loading Raw Data ---")
    df = pd.read_parquet(config['data_path'])
    meta = pd.read_parquet(config['meta_path'])
    
    if config.get('debug_sample', 0) > 0:
        df = df.iloc[::config['debug_sample']]
        print(f"--- [DEBUG] Sampling enabled. Rows: {len(df)}")
    df = df[(config['train_start_period'] <= df['period']) &
            (df['period'] < config['train_start_period'] + config['num_periods'])
            ]

    # 1. 基礎清洗 (字串 -> List)
    print("--- [Data] Parsing Sequences ---")
    # 這裡直接對應原始欄位名稱
    df['user_interacted_items'] = df['itemSeq'].fillna('').astype(str).apply(lambda x: [int(i) for i in x.split('#') if i])
    df['item_interacted_users'] = df['userSeq'].fillna('').astype(str).apply(lambda x: [int(i) for i in x.split('#') if i])
    meta['cateId'] = meta['cateId'].fillna('').astype(str).apply(lambda x: [c for c in x.split('#') if c])
    
    # 2. 建立全域 ID Map
    print("--- [Data] Building ID Maps ---")
    all_users = set(df['userId'].unique())
    # 也要包含歷史序列中的 ID
    for seq in df['item_interacted_users']: all_users.update(seq)
    
    all_items = set(df['itemId'].unique())
    for seq in df['user_interacted_items']: all_items.update(seq)
    
    user_map = {u: i for i, u in enumerate(sorted(list(all_users)), 1)}
    item_map = {i: k for k, i in enumerate(sorted(list(all_items)), 1)}
    
    n_users = len(user_map) + 1
    n_items = len(item_map) + 1
    print(f"    Total Users: {n_users-1}, Total Items: {n_items-1}")
    
    # 3. Remap DataFrame
    print("--- [Data] Remapping IDs ---")
    # 為了評估保留 Raw ID
    df['userId_raw'] = df['userId']
    df['itemId_raw'] = df['itemId']
    
    df['userId'] = df['userId'].map(user_map)
    df['itemId'] = df['itemId'].map(item_map)
    
    # Remap Sequences (比較慢，但只需做一次)
    # 使用 Set 加速查找
    user_set = set(user_map.keys())
    item_set = set(item_map.keys())
    
    df['user_interacted_items'] = df['user_interacted_items'].apply(
        lambda seq: [item_map[i] for i in seq if i in item_set]
    )
    df['item_interacted_users'] = df['item_interacted_users'].apply(
        lambda seq: [user_map[u] for u in seq if u in user_set]
    )
    
    # 移除 Map 不到的髒資料
    df.dropna(subset=['userId', 'itemId'], inplace=True)
    df = df.astype({'userId': 'int32', 'itemId': 'int32'})
    
    # 4. 處理 Category Meta
    print("--- [Data] Processing Categories ---")
    cate_matrix, cate_lens, n_cates = process_category_data(meta, item_map)
    
    # 5. 包裝回傳
    global_meta = {
        'n_users': n_users,
        'n_items': n_items,
        'n_cates': n_cates,
        'cate_matrix': cate_matrix,  # np.array
        'cate_lens': cate_lens       # np.array
    }
    
    maps = {'user_map': user_map, 'item_map': item_map}
    
    return df, global_meta, maps

def sample_negatives_batch(
    pos_item_ids: torch.Tensor,
    pos_user_ids: torch.Tensor,
    user_history_dict: Dict[int, Set[int]],
    n_samples: int,
    device: torch.device,
    n_items: int,
    candidate_pool: torch.Tensor = None  # [通用接口] 傳入想要採樣的 ID 集合 Tensor
) -> torch.Tensor:
    """
    高效版負採樣：採用拒絕採樣 (Rejection Sampling)
    
    參數:
    - candidate_pool: 如果不是 None，只從這個 Tensor 裡的 ID 進行採樣。
                      如果是 None，則從 [0, n_items) 範圍採樣。
    """
    batch_size = pos_item_ids.size(0)
    
    # --- 1. 決定採樣來源 ---
    if candidate_pool is not None:
        # 從指定的 Pool (Active 或 Seen) 中採樣
        num_candidates = candidate_pool.size(0)
        # 隨機生成 index [0, num_candidates-1]
        rand_indices = torch.randint(0, num_candidates, (batch_size, n_samples), device=device)
        # 映射回真實 item id
        neg_samples = candidate_pool[rand_indices]
    else:
        # 從全域 [0, n_items) 採樣
        neg_samples = torch.randint(0, n_items, (batch_size, n_samples), device=device)
    
    # --- 2. 衝突檢查與修正 (拒絕採樣) ---
    pos_item_ids_cpu = pos_item_ids.cpu().numpy()
    pos_user_ids_cpu = pos_user_ids.cpu().numpy()
    neg_samples_cpu = neg_samples.cpu().numpy()
    
    # 準備 fallback 重抽用的 pool (轉 numpy 以加速單點操作)
    if candidate_pool is not None:
        candidate_pool_cpu = candidate_pool.cpu().numpy()
        num_candidates = len(candidate_pool_cpu)
    else:
        candidate_pool_cpu = None
        num_candidates = n_items

    for i in range(batch_size):
        uid = pos_user_ids_cpu[i]
        pos_item = pos_item_ids_cpu[i]
        # 取得該 User 所有的歷史 (用於過濾 False Negatives)
        seen_set = user_history_dict.get(uid, set())
        
        for j in range(n_samples):
            sampled_item = neg_samples_cpu[i, j]
            
            # 衝突條件：(已看過) OR (等於當前正樣本)
            while sampled_item in seen_set or sampled_item == pos_item:
                # 重新採樣
                if candidate_pool_cpu is not None:
                    rand_idx = random.randint(0, num_candidates - 1)
                    sampled_item = candidate_pool_cpu[rand_idx]
                else:
                    sampled_item = random.randint(0, n_items - 1)
                
                neg_samples_cpu[i, j] = sampled_item
                
    return torch.from_numpy(neg_samples_cpu).to(device)
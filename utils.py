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
    
    cate_map = {c: i for i, c in enumerate(sorted(list(all_cates)))}
    n_cates = len(cate_map)
    
    # 建立 item -> cate_indices 矩陣
    n_items = len(item_map)
    cate_matrix = np.zeros((n_items, max_cate_len), dtype=np.int32)
    cate_lens = np.zeros(n_items, dtype=np.int32)
    
    # 建立查詢表以加速
    item_to_cates = dict(zip(meta_df['itemId'], meta_df['cateId']))
    
    # 填充矩陣 (依照 item_map 的順序 0 ~ N-1)
    # 反轉 item_map: internal_id -> raw_id
    id_to_raw = {v: k for k, v in item_map.items()}
    
    for internal_id in range(n_items):
        raw_id = id_to_raw.get(internal_id)
        if raw_id in item_to_cates:
            raw_cates = item_to_cates[raw_id]
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
    
    user_map = {u: i for i, u in enumerate(sorted(list(all_users)))}
    item_map = {i: k for k, i in enumerate(sorted(list(all_items)))}
    
    n_users = len(user_map)
    n_items = len(item_map)
    print(f"    Total Users: {n_users}, Total Items: {n_items}")
    
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
    item_pool: Set[int], # 這裡為了相容介面保留，但在高效版中我們主要用 n_items
    pos_item_ids: torch.Tensor,
    pos_user_ids: torch.Tensor,
    user_history_dict: Dict[int, Set[int]],
    n_samples: int,
    device: torch.device,
    n_items: int # 需要知道物品總數
) -> torch.Tensor:
    """
    高效版負採樣：採用拒絕採樣 (Rejection Sampling)
    """
    batch_size = pos_item_ids.size(0)
    
    # 1. 先隨機生成 [B, n_samples] 的矩陣
    # 假設 item id 是從 0 到 n_items-1
    neg_samples = torch.randint(0, n_items, (batch_size, n_samples), device=device)
    
    # 2. 檢查衝突 (Collision Check)
    # 這裡我們需要將 history 轉為較好查詢的格式，或者逐個 check
    # 由於 history 長度不一，完全向量化檢查較難，我們採用 "樂觀採樣 + 修正"
    
    pos_item_ids_cpu = pos_item_ids.cpu().numpy()
    pos_user_ids_cpu = pos_user_ids.cpu().numpy()
    neg_samples_cpu = neg_samples.cpu().numpy() # 轉 numpy 處理比較快
    
    for i in range(batch_size):
        uid = pos_user_ids_cpu[i]
        pos_item = pos_item_ids_cpu[i]
        seen_set = user_history_dict.get(uid, set())
        
        for j in range(n_samples):
            sampled_item = neg_samples_cpu[i, j]
            
            # 如果採樣到 (看過的) 或 (當前正樣本)
            # 進行 Resample (拒絕採樣)
            # 通常 while 迴圈只會執行 0 或 1 次
            while sampled_item in seen_set or sampled_item == pos_item:
                sampled_item = random.randint(0, n_items - 1)
                neg_samples_cpu[i, j] = sampled_item
                
    return torch.from_numpy(neg_samples_cpu).to(device)
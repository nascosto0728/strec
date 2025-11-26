import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import random 
import torch
from torch.utils.data import Dataset

def pad_sequences_native(
    sequences: List[list], 
    maxlen: int, 
    dtype: str = 'int32', 
    padding: str = 'post', 
    truncating: str = 'post', 
    value: int = 0
) -> np.ndarray:
    """
    用純 NumPy 實現的 Keras pad_sequences 的替代品。
    """
    truncated_sequences = []
    for seq in sequences:
        if len(seq) > maxlen:
            if truncating == 'pre':
                truncated_sequences.append(seq[-maxlen:])
            elif truncating == 'post':
                truncated_sequences.append(seq[:maxlen])
            else:
                raise ValueError(f"Truncating type '{truncating}' not understood")
        else:
            truncated_sequences.append(seq)
            
    num_samples = len(truncated_sequences)
    padded_matrix = np.full((num_samples, maxlen), value, dtype=dtype)

    for i, seq in enumerate(truncated_sequences):
        seq_len = len(seq)
        if padding == 'pre':
            padded_matrix[i, -seq_len:] = seq
        elif padding == 'post':
            padded_matrix[i, :seq_len] = seq
        else:
            raise ValueError(f"Padding type '{padding}' not understood")
            
    return padded_matrix


class RecommendationDataset(Dataset):
    """
    PyTorch Dataset Class for loading recommendation data.
    """
    def __init__(self, data_df: pd.DataFrame, his_data: Dict):
        self.data_df = data_df.reset_index(drop=True)
        self.his = his_data

    def __len__(self) -> int:
        return len(self.data_df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_row = self.data_df.iloc[idx]

        item_history_seq = pad_sequences_native([sample_row['itemSeq']], maxlen=30, padding='post', truncating='post', dtype='int32')[0]
        user_history_seq = pad_sequences_native([sample_row['userSeq']], maxlen=30, padding='post', truncating='post', dtype='int32')[0]
        
        batch_data = {
            'users': sample_row['userId'].astype(np.int32),
            'items': sample_row['itemId'].astype(np.int32),
            'labels': sample_row['label'].astype(np.int32),
            'users_raw': sample_row['userId_raw'].astype(np.int64),
            'items_raw': sample_row['itemId_raw'].astype(np.int64),
            'item_history_matrix': item_history_seq,
            'item_history_len': np.minimum(len(sample_row['itemSeq']), 30).astype(np.int32),
            'user_history_matrix': user_history_seq,
            'user_history_len': np.minimum(len(sample_row['userSeq']), 30).astype(np.int32),
            'period': sample_row['period'].astype(np.int64),
        }
        
        return batch_data


def process_cate(cate_ls: list) -> Tuple[np.ndarray, list]:
    cate_lens = [len(cate) for cate in cate_ls]
    max_len = max(cate_lens) if cate_lens else 0
    cate_seqs_matrix = np.zeros([len(cate_ls), max_len], np.int32)
    for i, cate_seq in enumerate(cate_ls):
        cate_seqs_matrix[i, :len(cate_seq)] = cate_seq
    return cate_seqs_matrix, cate_lens

def prepare_data_from_dfs(full_data_df: pd.DataFrame, full_meta_df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, np.ndarray, list, Dict, Dict, Dict]:
    full_data_df['userId_raw'] = full_data_df['userId']
    full_data_df['itemId_raw'] = full_data_df['itemId']

    print("Cleaning sequence string format...")
    for col in ['itemSeq', 'userSeq']:
        full_data_df[col] = full_data_df[col].fillna('').apply(lambda x: [int(i) for i in x.split('#') if i])
    full_meta_df['cateId'] = full_meta_df['cateId'].fillna('').apply(lambda x: [c for c in x.split('#') if c])

    print("Building full category map...")
    all_cate_names = set()
    _ = [all_cate_names.update(seq) for seq in full_meta_df['cateId']]
    full_cate_map = {raw_name: i for i, raw_name in enumerate(sorted(list(all_cate_names)))}
    config['model']['num_cates'] = len(full_cate_map)

    print("Starting static global data processing...")
    all_user_ids = set(full_data_df['userId_raw'].unique())
    _ = [all_user_ids.update(seq) for seq in full_data_df['userSeq']]
    all_item_ids = set(full_data_df['itemId_raw'].unique())
    _ = [all_item_ids.update(seq) for seq in full_data_df['itemSeq']]
    user_map = {raw_id: i for i, raw_id in enumerate(sorted(list(all_user_ids)))}
    item_map = {raw_id: i for i, raw_id in enumerate(sorted(list(all_item_ids)))}
    print(f"Global maps created. Total users: {len(user_map)}, Total items: {len(item_map)}")

    def map_seq(seq, id_map):
        return [id_map.get(i) for i in seq if i in id_map]
    remapped_df = full_data_df.copy()
    remapped_df['userId'] = remapped_df['userId_raw'].map(user_map)
    remapped_df['itemId'] = remapped_df['itemId_raw'].map(item_map)
    remapped_df['userSeq'] = remapped_df['userSeq'].apply(lambda seq: map_seq(seq, user_map))
    remapped_df['itemSeq'] = remapped_df['itemSeq'].apply(lambda seq: map_seq(seq, item_map))
    remapped_df.dropna(subset=['userId', 'itemId'], inplace=True)
    remapped_df = remapped_df.astype({'userId': 'int32', 'itemId': 'int32'})
    print("Full DataFrame remapped.")

    meta_df = full_meta_df.copy()
    meta_df['itemId'] = meta_df['itemId'].map(item_map)
    meta_df.dropna(subset=['itemId'], inplace=True)
    meta_df['cateId'] = meta_df['cateId'].apply(lambda seq: map_seq(seq, full_cate_map))
    item_cate_map = pd.Series(meta_df['cateId'].values, index=meta_df['itemId']).to_dict()
    cate_ls = [item_cate_map.get(i, []) for i in range(len(item_map))]
    cates, cate_lens = process_cate(cate_ls)

    hyperparams_updates = {'num_users': len(user_map), 'num_items': len(item_map)}
    
    return remapped_df, cates, cate_lens, hyperparams_updates, item_map, full_cate_map


def sample_negative_items(
    item_pool: set,
    seen_items_set: set,
    positive_item_id: int,
    num_samples: int,
    device: torch.device
) -> torch.Tensor:
    """
    採樣負樣本。
    """
    candidate_negatives = item_pool - seen_items_set - {positive_item_id}
    n_candidates = len(candidate_negatives)

    if n_candidates == 0:
        candidate_negatives = item_pool - {positive_item_id}
        if not candidate_negatives: 
             return torch.tensor([positive_item_id] * num_samples, dtype=torch.long, device=device)
        sampled_ids = random.choices(list(candidate_negatives), k=num_samples)

    elif n_candidates < num_samples:
        sampled_ids = random.choices(list(candidate_negatives), k=num_samples)
    else:
        sampled_ids = random.sample(list(candidate_negatives), k=num_samples)

    return torch.tensor(sampled_ids, dtype=torch.long, device=device)
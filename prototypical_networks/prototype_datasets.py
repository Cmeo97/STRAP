import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any
import numpy as np
import random
import time
import h5py
from benchmarking.benchmark_utils import concat_obs_group

class RobustManeuverDataset(Dataset):
    def __init__(self, data_dict: Dict[int, List[np.ndarray]], n_way: int, k_shot: int, n_query: int, n_episodes: int):
        self.data_dict = data_dict
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        self.classes = list(data_dict.keys())

    def __len__(self):
        return self.n_episodes
    
    def __getitem__(self, idx):
        sampled_classes = random.sample(self.classes, self.n_way)
        support_set_data = []
        query_set_data = []
        
        for class_id in sampled_classes:
            all_examples = self.data_dict[class_id]
            samples = random.sample(all_examples, self.k_shot + self.n_query)
            
            # Apply Time-Warping to Support Examples
            for s in samples[:self.k_shot]:
                warped = time_warp_augmentation(s)
                support_set_data.append(warped)
            
            # Keep Query examples as they are (or warp them too for more difficulty)
            query_set_data.extend(samples[self.k_shot:])
            
        return support_set_data + query_set_data


def custom_collate_fn(batch_list: List[List[np.ndarray]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        padded_sequences: (Total_Samples, Max_Len, N_DIM)
        mask: (Total_Samples, Max_Len) - 1 for real data, 0 for padding
    """
    # Flatten the list of lists
    all_sequences = [seq for episode_data in batch_list for seq in episode_data]
    
    if not all_sequences:
        return torch.empty(0), torch.empty(0)

    max_len = max(s.shape[0] for s in all_sequences)
    num_samples = len(all_sequences)
    dim = all_sequences[0].shape[1]
    
    padded_sequences = torch.zeros(num_samples, max_len, dim, dtype=torch.float32)
    mask = torch.zeros(num_samples, max_len, dtype=torch.float32)
    
    for i, seq in enumerate(all_sequences):
        length = seq.shape[0]
        padded_sequences[i, :length, :] = torch.from_numpy(seq)
        mask[i, :length] = 1.0 # Mark real data
        
    return padded_sequences, mask


def time_warp_augmentation(sequence, warp_range=(0.8, 1.2)):
    """
    Randomly stretches or compresses a maneuver sequence.
    sequence: numpy array of shape (L, D)
    warp_range: (min_stretch, max_stretch)
    """
    L, D = sequence.shape
    # 1. Randomly pick a new length
    warp_factor = np.random.uniform(warp_range[0], warp_range[1])
    new_len = int(L * warp_factor)
    
    # 2. Convert to torch for fast interpolation
    # Shape needs to be (Batch, Channels, Length) for 1D interpolation
    seq_tensor = torch.from_numpy(sequence).float().permute(1, 0).unsqueeze(0)
    
    # 3. Resample
    warped_tensor = torch.nn.functional.interpolate(seq_tensor, size=new_len, mode='linear', align_corners=True)
    
    # 4. Return as (New_L, D)
    return warped_tensor.squeeze(0).permute(1, 0).numpy()

def process_target_data(target_data_file, num_episodes=None):
    """Simulates realistic, variable-length maneuver data."""
    maneuvers = {}
    class_names ={}
    with h5py.File(target_data_file, 'r') as f:
        for i, task in enumerate(f.keys()):
            obs = f[task]['obs']
            # for libero dataset structure
            if 'joint_states' in obs and 'gripper_states' in obs and 'ee_pos' in obs:
                all_data = concat_obs_group(obs, feature_keys=['joint_states', 'gripper_states', 'ee_pos'])
            # for nuscene dataset structure
            elif 'velocity' in obs and 'acceleration' in obs and 'yaw_rate' in obs:
                all_data = concat_obs_group(obs, feature_keys=['velocity', 'acceleration', 'yaw_rate'])
            # for droid dataset structure
            elif 'cartesian_positions' in obs and 'gripper_states' in obs and 'joint_states' in obs:
                all_data = concat_obs_group(obs, feature_keys=['cartesian_positions', 'gripper_states', 'joint_states'])
            else:
                raise ValueError("Unknown dataset structure or missing expected keys.")
            
            if num_episodes is not None:
                all_data = all_data[:num_episodes]
            # Convert to list of 2D arrays for sampling
            maneuvers[i] = [all_data[j] for j in range(all_data.shape[0])]
            class_names[i] = task

    return maneuvers, class_names

def process_offline_data(offline_data):
    target_segments_list = {}  
    with h5py.File(offline_data, 'r') as f:
        data = f['data']
        demo_list = list(data.keys())
        for demo in demo_list:
            obs = data[demo]['obs']
            # for libero dataset structure
            if 'joint_states' in obs and 'gripper_states' in obs and 'ee_pos' in obs:
                all_data = concat_obs_group(obs, feature_keys=['joint_states', 'gripper_states', 'ee_pos'])
            # for nuscene dataset structure
            elif 'velocity' in obs and 'acceleration' in obs and 'yaw_rate' in obs:
                all_data = concat_obs_group(obs, feature_keys=['velocity', 'acceleration', 'yaw_rate'])
            # for droid dataset structure
            elif 'cartesian_positions' in obs and 'gripper_states' in obs and 'joint_states' in obs:
                all_data = concat_obs_group(obs, feature_keys=['cartesian_positions', 'gripper_states', 'joint_states'])
            else:
                raise ValueError("Unknown dataset structure or missing expected keys.")
            target_segments_list[demo] = all_data.astype(np.float32)
    return target_segments_list
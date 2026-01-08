import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import json
from typing import List, Tuple, Dict, Any
import random
import time
from strap.utils.retrieval_utils import (
    get_distance_matrix,
    compute_accumulated_cost_matrix_subsequence_dtw_21,
    compute_optimal_warping_path_subsequence_dtw_21,
)
# --- Configuration Constants ---
N_DIM = 12              # Number of features
FEATURE_SIZE = 512      # Output size of the learned embedding
N_CLASSES = 9           # Number of maneuver types

# Training Parameters
N_WAY = 5               # Number of classes per episode
K_SHOT = 7              # Number of support examples per class
N_QUERY = 3             # Number of query examples per class
N_EPISODES = 400        # Total training episodes

# --- 1. Model Architecture ---
class RobustSequenceEncoder(nn.Module):
    def __init__(self, input_dim=N_DIM, output_dim=FEATURE_SIZE):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, output_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x, mask=None, return_temporal=False):
        # x: (B, L, D) -> (B, D, L)
        x = x.permute(0, 2, 1) 
        temporal_features = self.feature_extractor(x) # (B, Out_Dim, L)
        
        if return_temporal:
            return temporal_features.permute(0, 2, 1)
        
        # --- CRITICAL: MASKED MAX POOLING ---
        if mask is not None:
            # mask: (B, L) -> Reshape to (B, 1, L) to match features
            mask = mask.unsqueeze(1) 
            # Set padded areas to a very small number so max() ignores them
            temporal_features = temporal_features.masked_fill(mask == 0, -1e9)
        
        pooled = torch.max(temporal_features, dim=2)[0] 
        
        # L2 Normalization (Essential for Euclidean Distance stability)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)

def robust_prototypical_loss(embeddings, n_way, k_shot, n_query):
    """
    embeddings: (Batch_Size * Samples_Per_Episode, Feature_Dim)
    """
    device = embeddings.device
    samples_per_episode = n_way * (k_shot + n_query)
    batch_size = embeddings.size(0) // samples_per_episode
    
    # 1. Reshape to (Batch_Size, Samples_Per_Episode, Feature_Dim)
    embeddings = embeddings.view(batch_size, samples_per_episode, -1)
    
    # 2. Split into Support and Query
    # support: (Batch_Size, n_way * k_shot, Dim)
    # query: (Batch_Size, n_way * n_query, Dim)
    support = embeddings[:, :n_way * k_shot]
    query = embeddings[:, n_way * k_shot:]
    
    # 3. Calculate prototypes for each episode in the batch
    # prototypes shape: (Batch_Size, n_way, Dim)
    prototypes = support.reshape(batch_size, n_way, k_shot, -1).mean(dim=2)
    
    # 4. Compute Squared Euclidean Distance
    # Using torch.cdist per batch item (requires some broadcasting or loop)
    # query: (B, Q_total, Dim), prototypes: (B, Way, Dim)
    distances = torch.cdist(query, prototypes)**2  # (B, Q_total, Way)
    
    # 5. Calculate Cross Entropy Loss
    # Flatten across batch and query dims
    target_labels = torch.arange(n_way).repeat_interleave(n_query).to(device)
    target_labels = target_labels.repeat(batch_size) # Repeat for each episode in batch
    
    # Reshape distances to (B * Q_total, Way)
    logits = -distances.reshape(-1, n_way)
    
    loss = torch.nn.functional.cross_entropy(logits, target_labels)
    
    return loss

# --- 2. Dataset ---

def process_target_data(target_data_file):
    """Simulates realistic, variable-length maneuver data."""
    maneuvers = {}
    class_names ={}
    with h5py.File(target_data_file, 'r') as f:
        for i, task in enumerate(f.keys()):
            data = f[task]

            joint_states = data['joint_states'][:]
            gripper_states = data['gripper_states'][:]
            ee_pos = data['ee_pos'][:]
            all_data = np.concatenate([joint_states, gripper_states, ee_pos], axis=2)
            # Convert to list of 2D arrays for sampling
            maneuvers[i] = [all_data[j] for j in range(all_data.shape[0])]
            class_names[i] = task

    return maneuvers, class_names

def process_offline_data(offline_data):
    target_segments_list = {}  
    with h5py.File(offline_data, 'r') as f:
        data = f['data']
        demo_list = list(data.keys())
        maneuver = [[] for _ in range(N_CLASSES)]
        for demo in demo_list:
            joint_states = data[demo]['obs/joint_states'][:]
            gripper_states = data[demo]['obs/gripper_states'][:]
            ee_pos = data[demo]['obs/ee_pos'][:]
            states = data[demo]['states'][:]
            #states = reduce_features_pca(states, n_components=N_DIM)
            all_data = np.concatenate([joint_states, gripper_states, ee_pos], axis=1)
            target_segments_list[demo] = all_data.astype(np.float32)
    return target_segments_list

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
    
    padded_sequences = torch.zeros(num_samples, max_len, N_DIM, dtype=torch.float32)
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

# --- 3. Training Loop ---
def train_robust_network(encoder, maneuver_data):
    print("--- Starting Robust Euclidean Training ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    # Add a Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    # Use the Robust Dataset with Time-Warping Augmentation
    dataset = RobustManeuverDataset(maneuver_data, N_WAY, K_SHOT, N_QUERY, N_EPISODES)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
    
    encoder.train()
    total_loss = 0
    
    # 'batch_sequences' and 'mask' are generated by custom_collate_fn
    for i, (batch_sequences, mask) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Move to GPU
        batch_sequences = batch_sequences.to(device) # (B*Samples, L, Dim)
        mask = mask.to(device)                       # (B*Samples, L)
        
        # 3. Forward Pass with Masking
        # Mask ensures padding doesn't affect Global Max Pooling
        embeddings = encoder(batch_sequences, mask=mask)
        
        # 4. Compute Loss (Handles the Batch Dim automatically)
        loss = robust_prototypical_loss(embeddings, N_WAY, K_SHOT, N_QUERY)
        
        loss.backward()
        
        # Gradient clipping prevents spikes
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        if i % 5 == 0:
            avg_loss = total_loss / (i + 1)
            print(f"Batch {i} | Episodes {i*4}/{N_EPISODES} | Avg Loss: {avg_loss:.4f}")
            
    return encoder

# --- 4. Retrieval ---

def retrieve_maneuver_dtw(prototype_seq, scene_embeddings):
    """
    Finds the best matching subsequence in a scene for a given prototype.
    prototype_seq: (Len_P, Dim) Tensor
    scene_embeddings: (Len_S, Dim) Tensor
    """
    # 1. Calculate Pairwise Distances (PyTorch is faster for this matrix math)
    # We use Squared Euclidean Distance
    distance_matrix = get_distance_matrix(prototype_seq, scene_embeddings)
    accumulated_cost_matrix = compute_accumulated_cost_matrix_subsequence_dtw_21(
        distance_matrix
    )
    path = compute_optimal_warping_path_subsequence_dtw_21(accumulated_cost_matrix)
    start = path[0, 1]
    if start < 0:
        assert start == -1
        start = 0
    end = path[-1, 1]
    cost = accumulated_cost_matrix[-1, end]
    normalized_score = cost / len(prototype_seq)
    # Note that the actual end index is inclusive in this case so +1 to use python : based indexing
    end = end + 1
    
    # 4. Normalize the score by the length of the prototype
    # This makes scores comparable across different maneuver lengths
    #final_score = accumulated_cost_matrix[len(prototype_seq), end + 1] / len(prototype_seq)
    
    return {
        "start_idx": int(start),
        "end_idx": int(end),
        "score": float(normalized_score)
    }

def get_prototypes(encoder, reference_maneuvers):
    device = next(encoder.parameters()).device
    
    # --- 1. Calculate Temporal Prototypes (Variable Length) ---
    prototypes = {}
    print("Calculating Temporal Prototypes from variable-length data...")
    
    for class_id, sequences in reference_maneuvers.items():
        temporal_embeddings_list = []
        
        # Process each example individually to handle different lengths
        for seq_np in sequences[:K_SHOT]:
            with torch.no_grad():
                # (L, Dim) -> (1, L, Dim)
                seq_tensor = torch.from_numpy(seq_np).float().unsqueeze(0).to(device)
                # Get (1, L, Feature_Dim)
                features = encoder(seq_tensor, return_temporal=True).squeeze(0)
                temporal_embeddings_list.append(features)
        
        # To get the "Average Sequence" of different lengths, we interpolate 
        # them all to the MEDIAN length of the group before averaging.
        median_len = int(np.median([len(s) for s in temporal_embeddings_list]))
        
        resized_embeddings = []
        for feat in temporal_embeddings_list:
            # (L, Dim) -> (1, Dim, L) for interpolation
            feat_reshaped = feat.permute(1, 0).unsqueeze(0)
            resized = torch.nn.functional.interpolate(feat_reshaped, size=median_len, mode='linear', align_corners=True)
            resized_embeddings.append(resized.squeeze(0).permute(1, 0))
            
        # Now they are the same length, we can stack and mean
        prototypes[class_id] = torch.stack(resized_embeddings).mean(dim=0)
    return prototypes

def retrieve(encoder, prototypes, offline_data_list, class_names):
    device = next(encoder.parameters()).device
    all_results = dict.fromkeys(class_names.values(), [])
    task_results = [[] for _ in range(N_CLASSES)]

    for offline_data in offline_data_list:
        task_name = os.path.splitext(offline_data)[0]
        offline_data = os.path.join(offline_data_dir, offline_data)
        test_scenes = process_offline_data(offline_data)
        print(f"Retrieving maneuvers from task: {task_name}")

        for class_id, proto_seq in prototypes.items():
            print(f"Prototype for {class_names[class_id]} ready for DTW retrieval.")
            demo_results = []
            for scene_name, scene_data in test_scenes.items():                
                with torch.no_grad():
                    scene_tensor = torch.from_numpy(scene_data).float().unsqueeze(0).to(device)
                    # Scenes are usually large, so this returns (Scene_L, Feature_Dim)
                    scene_features = encoder(scene_tensor, return_temporal=True).squeeze(0)
                
                # Numba DTW handles the different lengths of proto vs scene automatically
                match = retrieve_maneuver_dtw(proto_seq.cpu().numpy(), scene_features.cpu().numpy())
                match['scene'] = scene_name
                match['task'] = task_name
                # TODO: Tune this threshold based on validation set or target set
                if match['score'] < 10:  # Arbitrary threshold to filter bad matches
                    demo_results.append(match)
            # Filter overlaps
            task_results[class_id].extend(demo_results)
    
    for class_id in range(N_CLASSES):
        task_results[class_id] = sorted(task_results[class_id], key=lambda x: x['score'])
        all_results[class_names[class_id]] = task_results[class_id]

    return all_results
# --- Main Execution ---

if __name__ == '__main__':
    # A. Simulate Data
    # 50 examples per class for robust prototype calculation
    reference_data = "data/retrieval_results/target_dataset.hdf5"
    reference_maneuver_data, class_names = process_target_data(reference_data)
    pretrained = True  # Set to False to train from scratch

    encoder = RobustSequenceEncoder(input_dim=N_DIM, output_dim=FEATURE_SIZE)

    if not pretrained:
        # B. Train Model
        start_time = time.time()
        encoder = train_robust_network(encoder, reference_maneuver_data)
        training_time = time.time() - start_time
        print(f"Total training time: {training_time:.2f} seconds.")
    
        # Save the trained model
        torch.save(encoder.state_dict(), f'prototype_libero_model_{N_DIM}.pth')

    else:
        # load the trained model (for inference)
        encoder.load_state_dict(torch.load(f'prototype_libero_model_{N_DIM}.pth'))
    
    encoder.eval()
    prototypes = get_prototypes(encoder, reference_maneuver_data)

    # C. Retrieval
    final_results = {}
    offline_data_dir = 'data/LIBERO/libero_90/*demo.hdf5'
    offline_data_list = glob.glob(offline_data_dir)
    
    final_results = retrieve(encoder, prototypes, offline_data_list, class_names)

    with open(f'prototype_results_{N_DIM}.json', 'w') as f:
        json.dump(final_results, f, indent=4)
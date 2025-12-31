import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import numba as nb
import h5py
import json
from typing import List, Tuple, Dict, Any
import random
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from scipy.stats import wasserstein_distance_nd
from strap.utils.retrieval_utils import (
    get_distance_matrix,
    compute_accumulated_cost_matrix_subsequence_dtw_21,
    compute_optimal_warping_path_subsequence_dtw_21,
)
# --- Configuration Constants ---
N_DIM = 12              # Number of features: [Linear Accel, Velocity, Yaw Rate]
REF_FPS = 50            # Motion data frequency (Hz)
FEATURE_SIZE = 256      # Output size of the learned embedding
N_CLASSES = 3           # Number of maneuver types (e.g., LT, RT, Stop, Brake)

# Training Parameters
N_WAY = N_CLASSES       # Number of classes per episode
K_SHOT = 10              # Number of support examples per class
N_QUERY = 5             # Number of query examples per class
N_EPISODES = 400        # Total training episodes

# Retrieval Parameters
WINDOW_SECONDS = 8      # Fixed window size for retrieval (400 frames)
RETRIEVAL_STRIDE = 100   # Frames to slide the window per step
IOU_THRESHOLD = 0.5 # Max time-overlap allowed for NMS (0.0 to 1.0)
TARGET_SIZE = 400

def parse_episode(episode_dset):
    # The [()] syntax reads the single value from a scalar dataset.
    episodes_data_raw = episode_dset[()]
    
    # 3. Handle data encoding based on how it was saved:

    if isinstance(episodes_data_raw, bytes):
        # If saved as a variable-length byte string (most common for h5py.string_dtype)
        episodes_json = episodes_data_raw.decode('utf-8')
    elif isinstance(episodes_data_raw, str):
        # If saved as a fixed-length string array (less common for JSON)
        episodes_json = episodes_data_raw
    else:
        # Should not happen if saved correctly
        print(f"Error: Unexpected data type read: {type(episodes_data_raw)}")
        return None
    
    # 4. Deserialize the JSON string back into a Python object
    return json.loads(episodes_json)

# --- 1. Model Architecture ---

import torch.nn.functional as F

class RobustSequenceEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=128):
        super().__init__()
        # We keep the layers accessible so we can use them for DTW later
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, output_dim, kernel_size=5, padding=2),
            nn.ReLU()
        )

    def forward(self, x, return_temporal=False):
        # x shape: (B, L, D) -> Permute to (B, D, L)
        x = x.permute(0, 2, 1) 
        temporal_features = self.feature_extractor(x) # (B, FEATURE_SIZE, L)
        
        if return_temporal:
            # Return (B, L, FEATURE_SIZE) for DTW retrieval
            return temporal_features.permute(0, 2, 1)
        
        # Global Max Pooling for robust classification training
        pooled = torch.max(temporal_features, dim=2)[0] 
        return pooled

def robust_prototypical_loss(embeddings, n_way, k_shot, n_query):
    """
    Standard Euclidean Prototypical Loss.
    Fast and effective when combined with Max Pooling and Augmentation.
    """
    support = embeddings[:n_way * k_shot]
    query = embeddings[n_way * k_shot:]
    
    # Calculate prototypes (Mean of Max-Pooled features)
    prototypes = support.reshape(n_way, k_shot, -1).mean(dim=1)
    
    # Squared Euclidean Distance
    # (N_query_total, 1, Dim) - (1, N_way, Dim)
    distances = torch.cdist(query.unsqueeze(0), prototypes.unsqueeze(0)).squeeze(0)**2
    
    target_labels = torch.arange(n_way).repeat_interleave(n_query).to(embeddings.device)
    loss = F.cross_entropy(-distances, target_labels)
    
    return loss

# --- 2. Data Simulation and Dataset ---

def process_target_data(reference_data):
    """Simulates realistic, variable-length maneuver data."""
    maneuvers = {}
    with h5py.File(reference_data, 'r') as f:
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
            maneuver[0].append(all_data[0:110].astype(np.float32))
            maneuver[1].append(all_data[120:190].astype(np.float32))
            maneuver[2].append(all_data[190:-1].astype(np.float32))
        maneuvers = {i: maneuver[i] for i in range(N_CLASSES)}

    return maneuvers

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

def reduce_features_pca(time_series_2d, n_components=10):
    """
    Reduces the feature dimension of a single 2D time series sample 
    while keeping the number of timestamps exactly the same.
    
    Parameters:
    time_series_2d: np.array of shape (Time_steps, Features)
    n_components: Target number of features (default 10)
    
    Returns:
    reduced_series: np.array of shape (Time_steps, n_components)
    """
    # 1. Check if the requested components are valid
    n_features = time_series_2d.shape[1]
    if n_components > n_features:
        print(f"Warning: n_components ({n_components}) is greater than "
              f"original features ({n_features}). Returning original data.")
        return time_series_2d

    # 2. Standardize features (Scale along the Time axis)
    # This ensures each feature has mean=0 and variance=1 across time
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(time_series_2d)
    
    # 3. Initialize and fit PCA on the feature dimension
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(scaled_data)

    return reduced_data

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


def custom_collate_fn(batch_list: List[List[np.ndarray]]) -> torch.Tensor:
    """
    Custom collate function to handle the list-of-sequences output from the dataset.
    It flattens the episode data and applies padding to create a single tensor.
    """
    # Flattens the list of lists into a single list of maneuver sequences
    all_sequences = [seq for episode_data in batch_list for seq in episode_data]
    
    if not all_sequences:
        return torch.empty(0)

    # Padding logic: find max length
    max_len = max(s.shape[0] for s in all_sequences)
    
    padded_sequences = torch.zeros(len(all_sequences), max_len, N_DIM, dtype=torch.float32)
    
    for i, seq in enumerate(all_sequences):
        length = seq.shape[0]
        padded_sequences[i, :length, :] = torch.from_numpy(seq)
        
    return padded_sequences


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
    warped_tensor = F.interpolate(seq_tensor, size=new_len, mode='linear', align_corners=True)
    
    # 4. Return as (New_L, D)
    return warped_tensor.squeeze(0).permute(1, 0).numpy()

# --- 3. Training Loop ---
def train_robust_network(encoder, maneuver_data):
    print("--- Starting Robust Euclidean Training ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    
    # Use the Robust Dataset with Time-Warping Augmentation
    dataset = RobustManeuverDataset(maneuver_data, N_WAY, K_SHOT, N_QUERY, N_EPISODES)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    
    encoder.train()
    total_loss = 0
    
    for i, batch_sequences in enumerate(dataloader):
        optimizer.zero_grad()
        
        # batch_sequences: (Total_Samples, Time, N_DIM)
        batch_sequences = batch_sequences.squeeze(0).to(device)
        
        # Forward pass (returns Max-Pooled vectors)
        embeddings = encoder(batch_sequences)
        
        # Calculate loss
        loss = robust_prototypical_loss(embeddings, N_WAY, K_SHOT, N_QUERY)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if i % 10 == 0:
            print(f"Episode {i}/{N_EPISODES} | Loss: {total_loss/(i+1):.4f}")
            
    return encoder



# --- 4. Retrieval and NMS ---

# --- Method A: Using Scipy (Recommended) ---
def resize_scipy(data, new_size):
    # Create an index for current data (0 to 399)
    x_old = np.linspace(0, 1, data.shape[0])
    # Create an index for the new target size (0 to 299)
    x_new = np.linspace(0, 1, new_size)
    
    # Kind='linear' or 'cubic' for smoother interpolation
    f = interp1d(x_old, data, axis=0, kind='linear')
    return f(x_new)

def wasserstein_distance(support, test_data, results) -> float:
    """
    Calculates the Wasserstein distance between two embeddings.
    This is a placeholder function; actual implementation may vary.
    """
    support_data = [[] for _ in range(N_CLASSES)]
    for type in support.keys():
        for item in support[type]:
            if len(support_data[type]) <= 5:
                new_data = resize_scipy(item, TARGET_SIZE)
                support_data[type].append(new_data)

    output = [[] for _ in range(N_CLASSES)]
    for scene_key, scene_val in results.items():
        for item in scene_val:
            dynamics_data = test_data[scene_key][item['start_idx']:item['end_idx']]
            dynamics_data = resize_scipy(dynamics_data, TARGET_SIZE)
            output[item['class_id']].append(dynamics_data)
    
    class_wds = []
    for i in range(N_CLASSES):
        # Distance between support set distribution and retrieved test set distribution
        s_data = np.array(support_data[i])
        s_data = s_data.reshape(s_data.shape[0], -1)
        o_data = np.array(output[i])
        o_data = o_data.reshape(o_data.shape[0], -1)
        if s_data.shape[0] == 0 or o_data.shape[0] == 0:
            continue
        wd = wasserstein_distance_nd(s_data, o_data)
        class_wds.append(wd)
    mean_wasserstein = np.mean(class_wds)


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
    # Note that the actual end index is inclusive in this case so +1 to use python : based indexing
    end = end + 1
    
    # 4. Normalize the score by the length of the prototype
    # This makes scores comparable across different maneuver lengths
    #final_score = accumulated_cost_matrix[len(prototype_seq), end + 1] / len(prototype_seq)
    
    return {
        "start_idx": int(start),
        "end_idx": int(end),
        "score": float(cost)
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
                seq_tensor = torch.from_numpy(seq_np).unsqueeze(0).to(device)
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
            resized = F.interpolate(feat_reshaped, size=median_len, mode='linear', align_corners=True)
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
                    scene_tensor = torch.from_numpy(scene_data).unsqueeze(0).to(device)
                    # Scenes are usually large, so this returns (Scene_L, Feature_Dim)
                    scene_features = encoder(scene_tensor, return_temporal=True).squeeze(0)
                
                # Numba DTW handles the different lengths of proto vs scene automatically
                match = retrieve_maneuver_dtw(proto_seq.cpu().numpy(), scene_features.cpu().numpy())
                match['scene'] = scene_name
                match['task'] = task_name
                if match['score'] < 250:  # Arbitrary threshold to filter bad matches
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
    reference_data = "/home/zillur/programs/STRAP/data/LIBERO/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5"
    offline_data1 = 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it_demo.hdf5'
    class_names = {0: "TURN ON STOVE", 1: "PICK UP MOKA POT", 2: "PUT MOKA POT ON STOVE"}
    reference_maneuver_data = process_target_data(reference_data)
    pretrained = False

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
    offline_data_dir = '/home/zillur/programs/STRAP/data/LIBERO/libero_90'
    offline_data_list = [
        'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it_demo.hdf5',
        'KITCHEN_SCENE3_put_the_moka_pot_on_the_stove_demo.hdf5',
        'KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate_demo.hdf5'
    ]
    # C. Retrieval
    final_results = retrieve(encoder, prototypes, offline_data_list, class_names)

    with open(f'prototype_strap_results_{N_DIM}.json', 'w') as f:
        json.dump(final_results, f, indent=4)

    #data = wasserstein_distance(reference_maneuver_data, target_segments_list, retrieval_results)

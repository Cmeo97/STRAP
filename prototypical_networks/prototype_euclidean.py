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
import yaml
import argparse
from tqdm import tqdm

from prototypical_networks.prototype_datasets import (
    process_offline_data,
    process_target_data
)
from benchmarking.benchmark_utils import process_retrieval_results

# --- Configuration Constants ---
FEATURE_SIZE = 256      # Output size of the learned embedding

# Training Parameters
K_SHOT = 7              # Number of support examples per class
N_QUERY = 3             # Number of query examples per class
N_EPISODES = 100        # Total training episodes

# --- 1. Model Architecture ---

class SequenceEncoder(nn.Module):
    """
    1D CNN Encoder that processes variable-length time series data.
    Uses Global Average Pooling (GAP) to produce a fixed-size embedding.
    This handles the variable-length sequences.
    """
    def __init__(self, input_dim, output_dim=FEATURE_SIZE):
        super().__init__()
        # Input channel is N_DIM (3); Output is 64
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, output_dim, kernel_size=5, padding=2)
        
        self.relu = nn.ReLU()
        # No final projection layer, the output of conv3 is the embedding dimension

    def forward(self, x):
        # Input x shape: (B, L, D) -> Permute to (B, D, L) for Conv1d
        x = x.permute(0, 2, 1) 
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.conv3(x))
        
        # Output x shape: (B, FEATURE_SIZE, L)
        
        # Global Average Pooling over the sequence length (L)
        # This reduces the sequence dimension, making the output fixed-length (B, FEATURE_SIZE)
        x = torch.mean(x, dim=2) 
        
        # Final Embedding shape: (B, FEATURE_SIZE)
        return x

def prototypical_loss(input_embeddings: torch.Tensor, n_way: int, k_shot: int, n_query: int):
    """
    Calculates the Prototypical Loss based on Euclidean distance to prototypes.
    FIXED: Correctly generates target labels for the query set.
    """
    
    # 1. Separate Support and Query Embeddings
    support_embeddings = input_embeddings[:n_way * k_shot]
    query_embeddings = input_embeddings[n_way * k_shot:]
    
    # 2. Calculate Prototypes (Mean of Support Embeddings for each class)
    # Reshape: (N_WAY, K_SHOT, FEATURE_SIZE) -> Mean across K_SHOT
    prototypes = support_embeddings.reshape(n_way, k_shot, -1).mean(dim=1)
    
    # 3. Calculate Squared Euclidean Distances
    # query: (N_QUERY * N_WAY, 1, FEATURE_SIZE)
    query = query_embeddings.unsqueeze(1) 
    # prototypes: (1, N_WAY, FEATURE_SIZE)
    prototypes = prototypes.unsqueeze(0)
    
    # Distances: (N_QUERY * N_WAY, N_WAY)
    distances = torch.sum((query - prototypes) ** 2, dim=2) 
    
    # 4. Convert Distances to Log Probabilities
    # We want low distance to mean high probability, so we take the negative.
    log_probs = -distances
    
    # 5. Generate Correct Target Labels for Query Samples (Crucial Fix)
    # Target classes should be [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, ...]
    target_labels = torch.arange(n_way, dtype=torch.long)
    target_labels = target_labels.repeat_interleave(n_query) # FIX IS HERE
    
    # 6. Calculate Loss
    loss = nn.functional.cross_entropy(log_probs, target_labels.to(input_embeddings.device))
    
    # Calculate Accuracy
    _, predicted_classes = torch.max(log_probs.data, 1)
    accuracy = torch.sum(predicted_classes == target_labels.to(input_embeddings.device)).item() / target_labels.size(0)
    
    return loss, accuracy

# --- 2. Data Simulation and Dataset ---
class EpisodicManeuverDataset(Dataset):
    """
    Dataset to sample N-way K-shot episodes for Prototypical Network training.
    """
    def __init__(self, data_dict: Dict[int, List[np.ndarray]], n_way: int, k_shot: int, n_query: int, n_episodes: int, n_dim: int):
        self.data_dict = data_dict
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        self.classes = list(data_dict.keys())
        self.n_dim = n_dim

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Sample N_WAY classes for the episode
        sampled_classes = random.sample(self.classes, self.n_way)
        
        support_set_data = []
        query_set_data = []
        
        for class_id in sampled_classes:
            # 2. Sample K_SHOT + N_QUERY examples from the chosen class
            all_examples = self.data_dict[class_id]
            
            # Ensure enough examples are available (safety check)
            if len(all_examples) < self.k_shot + self.n_query:
                 raise ValueError(f"Class {class_id} requires {self.k_shot + self.n_query} examples, but only has {len(all_examples)}.")
                 
            samples = random.sample(all_examples, self.k_shot + self.n_query)
            
            # 3. Split into Support and Query sets
            support_set_data.extend(samples[:self.k_shot])
            query_set_data.extend(samples[self.k_shot:])
            
        # Combine all data: Support first, then Query
        all_data = support_set_data + query_set_data
        
        return all_data


def create_collate_fn(n_dim):
    """Create a collate function with the correct n_dim."""
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
        
        padded_sequences = torch.zeros(len(all_sequences), max_len, n_dim, dtype=torch.float32)
        
        for i, seq in enumerate(all_sequences):
            length = seq.shape[0]
            padded_sequences[i, :length, :] = torch.from_numpy(seq)
            
        return padded_sequences
    return custom_collate_fn


# --- 3. Training Loop ---

def train_prototypical_network(encoder, maneuver_data: Dict[int, List[np.ndarray]], n_way: int, n_dim: int):
    """
    Trains the SequenceEncoder using the Prototypical Network training paradigm.
    """
    print("--- Starting Prototypical Network Training ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    
    optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
    
    dataset = EpisodicManeuverDataset(maneuver_data, n_way, K_SHOT, N_QUERY, N_EPISODES, n_dim)
    # Batch size is 1 because one item = one episode
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=create_collate_fn(n_dim))
    
    encoder.train()
    total_loss = 0
    total_acc = 0
    
    for i, batch_sequences in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Move padded sequences to device
        batch_sequences = batch_sequences.squeeze(0).to(device) # Squeeze batch_size=1 dim
        
        # Forward pass: get embeddings for all Support + Query samples
        embeddings = encoder(batch_sequences)
        
        # Calculate Prototypical Loss (using the fixed function)
        loss, acc = prototypical_loss(embeddings, n_way, K_SHOT, N_QUERY)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += acc
    
        avg_loss = total_loss / (i+1)
        avg_acc = total_acc / (i+1)
        
        if i % 5 == 0:
            print(f"Episode {i}/{N_EPISODES} | Avg Loss: {avg_loss:.4f} | Avg Accuracy: {avg_acc*100:.2f}%")

    print("\n--- Training Complete ---")
    return encoder


# --- 4. Retrieval and NMS ---

def retrieve_maneuvers(encoder: SequenceEncoder, maneuver_data: Dict[int, List[np.ndarray]], 
                       offline_data_list: List[str], output_path: str, episode_names: Dict[int, str],
                       window_size: int = 50, stride: int = 5, TOP_K: int = 100):
    """
    Uses the trained encoder with sliding window and Euclidean distance to find maneuvers.
    Saves results in HDF5 format compatible with prototype_retrieval output.
    
    Args:
        encoder: Trained sequence encoder
        maneuver_data: Reference maneuver data {class_id: [examples]}
        offline_data_list: List of paths to offline HDF5 files
        output_path: Path to save retrieval results HDF5 file
        episode_names: Mapping from class_id to episode name
        window_size: Size of sliding window (in frames)
        stride: Stride for sliding window
        TOP_K: Number of top matches to keep per episode
    """
    print("\n--- Starting Maneuver Retrieval with Euclidean Distance ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.eval()
    
    # 1. Calculate the Class Prototypes (mean embedding per class)
    all_embeddings = []
    class_labels = []
    
    print("Calculating class prototypes...")
    for class_id, examples in maneuver_data.items():
        for ex in examples[:K_SHOT]:  # Use K_SHOT examples for prototype
            ex_tensor = torch.from_numpy(ex).unsqueeze(0).to(device)
            ex_tensor = ex_tensor.float()
            with torch.no_grad():
                embedding = encoder(ex_tensor).cpu().numpy().squeeze()
            all_embeddings.append(embedding)
            class_labels.append(class_id)
            
    all_embeddings = np.stack(all_embeddings)
    
    # Calculate prototypes (mean embedding per class)
    prototypes = {}
    for class_id in maneuver_data.keys():
        class_embeddings = all_embeddings[np.array(class_labels) == class_id]
        prototypes[class_id] = torch.from_numpy(class_embeddings.mean(axis=0)).to(device)
        print(f"Prototype for {episode_names[class_id]} calculated from {len(class_embeddings)} examples.")

    # 2. Sliding Window Search on Offline Data
    with h5py.File(output_path, 'w') as outfile:
        results = outfile.create_group('results')
        
        for class_id, prototype in prototypes.items():
            episode_key = episode_names[class_id]
            episode = results.create_group(episode_key)
            episode_results = []
            
            print(f"\nSearching for {episode_key}...")
            for offline_file in tqdm(offline_data_list, desc=f"Retrieving data for {episode_key}"):
                test_scenes = process_offline_data(offline_file)
                
                for demo_key, demo_data in test_scenes.items():
                    L, D = demo_data.shape
                    
                    # Skip if demo is shorter than window
                    if L < window_size:
                        continue
                    
                    window_starts = np.arange(0, L - window_size + 1, stride)
                    
                    # Process windows in batches for efficiency
                    batch_size = 16
                    for i in range(0, len(window_starts), batch_size):
                        batch_starts = window_starts[i:i + batch_size]
                        batch_windows = [demo_data[start:start+window_size, :] for start in batch_starts]
                        
                        # Convert batch to tensor and encode
                        batch_tensor = torch.stack([torch.from_numpy(w) for w in batch_windows]).to(device)
                        batch_tensor = batch_tensor.float()
                        with torch.no_grad():
                            window_embeddings = encoder(batch_tensor)  # (Batch_Size, FEATURE_SIZE)
                        
                        # Compare each window embedding to current prototype
                        for j, embed in enumerate(window_embeddings):
                            start_frame = int(batch_starts[j])
                            end_frame = int(start_frame + window_size)
                            
                            # Squared Euclidean distance in embedding space
                            dist = torch.sum((embed - prototype) ** 2).item()
                            
                            episode_results.append({
                                'cost': dist,
                                'start_idx': start_frame,
                                'end_idx': end_frame,
                                'demo_key': demo_key,
                                'offline_file': offline_file
                            })
            
            # 3. Process and save top K results
            processed_results = process_retrieval_results(episode_results, top_k=TOP_K)
            for match_key, data in processed_results.items():
                match_group = episode.create_group(match_key)
                match_group.attrs['ep_meta'] = json.dumps({
                    "lang": data['lang_instruction']
                })
                match_group.attrs['file_path'] = data['file_path']
                match_group.attrs['demo_key'] = data['demo_key']
                for data_key, value in data.items():
                    if data_key not in ['file_path', 'demo_key', 'lang_instruction']:
                        match_group.create_dataset(data_key, data=value)
    
    print(f"\nRetrieval results saved to {output_path}")

# --- Main Execution ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Euclidean Distance-based Maneuver Retrieval')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file.')
    parser.add_argument('--dataset_type', default='droid', choices=['libero', 'nuscene', 'droid'], 
                       help='Type of dataset to use.')
    parser.add_argument('--pretrained', default=False, action='store_true', help='Use pretrained model for retrieval.')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))

    if args.dataset_type == 'libero':
        target_data = config['dataset_paths']['libero_target']
        offline_data_dir = config['dataset_paths']['libero_offline']
        retrieved_output = os.path.join(config['retrieval_paths']['libero'], 'libero_retrieval_results_prototype.hdf5')
        checkpoint_name = config['prototype_ckpt_paths']['libero'].replace('.pth', '_euclidean.pth')
        TOP_K = 150
    elif args.dataset_type == 'nuscene':
        target_data = config['dataset_paths']['nuscene_target']
        offline_data_dir = config['dataset_paths']['nuscene_offline']
        retrieved_output = os.path.join(config['retrieval_paths']['nuscene'], 'nuscene_retrieval_results_prototype.hdf5')
        checkpoint_name = config['prototype_ckpt_paths']['nuscene'].replace('.pth', '_euclidean.pth')
        TOP_K = 50
    elif args.dataset_type == 'droid':
        target_data = config['dataset_paths']['droid_target']
        offline_data_dir = config['dataset_paths']['droid_offline']
        retrieved_output = os.path.join(config['retrieval_paths']['droid'], 'droid_retrieval_results_prototype.hdf5')
        checkpoint_name = config['prototype_ckpt_paths']['droid'].replace('.pth', '_euclidean.pth')
        TOP_K = 80
    else:
        raise ValueError("Unsupported dataset type!")

    os.makedirs(os.path.dirname(retrieved_output), exist_ok=True)
    
    # Load reference data using new schema
    print("Loading reference maneuver data...")
    reference_maneuver_data, episode_names = process_target_data(target_data)
    window_size = int(np.mean([x[0].shape[0] for x in reference_maneuver_data.values()]))
    stride = window_size // 4  # 25% overlap
    print(f"Using window size: {window_size}, stride: {stride}")
    N_DIM = reference_maneuver_data[0][0].shape[-1]
    N_CLASSES = len(reference_maneuver_data)
    N_WAY = N_CLASSES // 2  # Update N_WAY based on number of classes
    
    print(f"Dataset info: {N_CLASSES} classes, {N_DIM} dimensions")
    
    # Initialize encoder
    encoder = SequenceEncoder(input_dim=N_DIM, output_dim=FEATURE_SIZE)

    if not args.pretrained:
        # Train Model
        start_time = time.time()
        encoder = train_prototypical_network(encoder, reference_maneuver_data, N_WAY, N_DIM)
        training_time = time.time() - start_time
        print(f"Total training time: {training_time:.2f} seconds.")
        
        # Save the trained model
        torch.save(encoder.state_dict(), checkpoint_name)
        print(f"Model saved to {checkpoint_name}")
    else:
        # Load the trained model (for inference)
        if not os.path.exists(checkpoint_name):
            raise ValueError(f"Checkpoint file not found: {checkpoint_name}. Train the model first.")
        encoder.load_state_dict(torch.load(checkpoint_name))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder.to(device)
        print(f"Model loaded from {checkpoint_name}")

    encoder.eval()
    
    # Retrieval
    offline_data_list = glob.glob(offline_data_dir)
    print(f"Found {len(offline_data_list)} offline data files")
    
    retrieve_maneuvers(encoder, reference_maneuver_data, offline_data_list, retrieved_output,
                      episode_names, window_size=window_size, stride=stride, TOP_K=TOP_K)

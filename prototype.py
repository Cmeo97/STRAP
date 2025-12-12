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

# --- Configuration Constants ---
N_DIM = 3               # Number of features: [Linear Accel, Velocity, Yaw Rate]
REF_FPS = 50            # Motion data frequency (Hz)
FEATURE_SIZE = 128      # Output size of the learned embedding
N_CLASSES = 4           # Number of maneuver types (e.g., LT, RT, Stop, Brake)

# Training Parameters
N_WAY = N_CLASSES       # Number of classes per episode
K_SHOT = 4              # Number of support examples per class
N_QUERY = 2             # Number of query examples per class
N_EPISODES = 100        # Total training episodes

# Retrieval Parameters
WINDOW_SECONDS = 8      # Fixed window size for retrieval (400 frames)
RETRIEVAL_STRIDE = 100   # Frames to slide the window per step
NMS_OVERLAP_THRESHOLD = 0.5 # Max time-overlap allowed for NMS (0.0 to 1.0)


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

class SequenceEncoder(nn.Module):
    """
    1D CNN Encoder that processes variable-length time series data.
    Uses Global Average Pooling (GAP) to produce a fixed-size embedding.
    This handles the variable-length sequences.
    """
    def __init__(self, input_dim=N_DIM, output_dim=FEATURE_SIZE):
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

def simulate_maneuvers(reference_data):
    """Simulates realistic, variable-length maneuver data."""
    maneuvers = {}
    for i in range(N_CLASSES):
        maneuvers[i] = []
        # --- Maneuver Specific Logic (Simulates feature importance) ---
        
        # 0: Left Turn (High Yaw, medium velocity variation)
        if i == 0: 
            with h5py.File(reference_data, 'r') as f:
                scene_list = list(f.keys())
                for scene in scene_list:
                    dynamics = f[scene]['dynamics'][:]
                    episodes = parse_episode(f[scene]['episodes'])
                    for episode in episodes:
                        if episode['type'] == 'left_turn':
                            start = episode['start_idx']
                            end = episode['end_idx'] + 1
                            data = dynamics[start:end, :]
                            maneuvers[i].append(data.astype(np.float32))
        elif i == 1: # 1: Straight Driving (Low Yaw, steady velocity)
            with h5py.File(reference_data, 'r') as f:
                scene_list = list(f.keys())
                for scene in scene_list:
                    dynamics = f[scene]['dynamics'][:]
                    episodes = parse_episode(f[scene]['episodes'])
                    for episode in episodes:
                        if episode['type'] == 'straight_driving':
                            start = episode['start_idx']
                            end = episode['end_idx'] + 1
                            data = dynamics[start:end, :]
                            maneuvers[i].append(data.astype(np.float32))
        
        elif i == 2: # 2: Traffic Stop (Velocity to zero)
            with h5py.File(reference_data, 'r') as f:
                scene_list = list(f.keys())
                for scene in scene_list:
                    dynamics = f[scene]['dynamics'][:]
                    episodes = parse_episode(f[scene]['episodes'])
                    for episode in episodes:
                        if episode['type'] == 'stop':
                            start = episode['start_idx']
                            end = episode['end_idx'] + 1
                            data = dynamics[start:end, :]
                            maneuvers[i].append(data.astype(np.float32))
        
        else: # 3: Right Turn (High Yaw, medium velocity variation)
            with h5py.File(reference_data, 'r') as f:
                scene_list = list(f.keys())
                for scene in scene_list:
                    dynamics = f[scene]['dynamics'][:]
                    episodes = parse_episode(f[scene]['episodes'])
                    for episode in episodes:
                        if episode['type'] == 'right_turn':
                            start = episode['start_idx']
                            end = episode['end_idx'] + 1
                            data = dynamics[start:end, :]          
                            maneuvers[i].append(data.astype(np.float32))
            
    return maneuvers


class EpisodicManeuverDataset(Dataset):
    """
    Dataset to sample N-way K-shot episodes for Prototypical Network training.
    """
    def __init__(self, data_dict: Dict[int, List[np.ndarray]], n_way: int, k_shot: int, n_query: int, n_episodes: int):
        self.data_dict = data_dict
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        self.classes = list(data_dict.keys())

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


# --- 3. Training Loop ---

def train_prototypical_network(maneuver_data: Dict[int, List[np.ndarray]]):
    """
    Trains the SequenceEncoder using the Prototypical Network training paradigm.
    """
    print("--- Starting Prototypical Network Training ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = SequenceEncoder().to(device)
    optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
    
    dataset = EpisodicManeuverDataset(maneuver_data, N_WAY, K_SHOT, N_QUERY, N_EPISODES)
    # Batch size is 1 because one item = one episode
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    
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
        loss, acc = prototypical_loss(embeddings, N_WAY, K_SHOT, N_QUERY)
        
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

def non_maximum_suppression(candidates: List[Dict[str, Any]], overlap_threshold: float):
    """
    Aggregates consecutive, overlapping segments of the same class ID 
    into a single, longer maneuver segment.

    The input candidates list contains many overlapping windows from the sliding search.
    """
    if not candidates:
        return []

    # 1. Primary Sort: Sort by Scene ID, then by Start Frame. (Crucial for temporal aggregation)
    # The score (distance) is used as a tie-breaker if start frames are identical.
    candidates.sort(key=lambda x: (x['start_frame'], x['score']))

    final_segments = []
    
    # Initialize the first segment as the currently merged result
    current_merged_segment = candidates[0].copy()
    
    # 2. Iterate and Merge
    for next_candidate in candidates[1:]:
        
        # Condition 1: Must be in the same continuous target segment (scene)
        # Condition 2: Must be the same maneuver type
        # Condition 3: Must overlap (or be immediately consecutive, start <= end)
        if (next_candidate['class_id'] == current_merged_segment['class_id'] and
            next_candidate['start_frame'] <= current_merged_segment['end_frame']):
            
            # --- Merge Step ---
            
            # Extend the end frame of the current merged segment
            current_merged_segment['end_frame'] = max(current_merged_segment['end_frame'], next_candidate['end_frame'])
            
            # Since lower score (distance) is better, keep the minimum score found within the merged segment
            current_merged_segment['score'] = min(current_merged_segment['score'], next_candidate['score'])
            
        else:
            # Not overlapping, different ID, or different class. Finalize the current merged segment.
            final_segments.append(current_merged_segment)
            
            # Start a new merged segment
            current_merged_segment = next_candidate.copy()

    # Add the last segment after the loop finishes
    final_segments.append(current_merged_segment)
    
    return final_segments
                


def retrieve_maneuvers(encoder: SequenceEncoder, maneuver_data: Dict[int, List[np.ndarray]], target_segments: List[np.ndarray]):
    """
    Uses the trained encoder to find and label maneuvers in the large target segments.
    """
    print("\n--- Starting Maneuver Retrieval ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.eval()
    
    # 1. Calculate the Class Prototypes
    all_embeddings = []
    class_labels = []
    
    # Flatten the maneuver data for prototype calculation
    for class_id, examples in maneuver_data.items():
        for ex in examples:
            # We must wrap each example in a batch dimension for the encoder
            ex_tensor = torch.from_numpy(ex).unsqueeze(0).to(device) 
            with torch.no_grad():
                embedding = encoder(ex_tensor).cpu().numpy().squeeze()
            all_embeddings.append(embedding)
            class_labels.append(class_id)
            
    all_embeddings = np.stack(all_embeddings)
    
    # Calculate prototypes (mean embedding per class)
    prototypes = {}
    class_names = {0: "Left Turn", 1: "Straight Driving", 2: "Traffic Stop", 3: "Right Turn"}
    
    for class_id in range(N_CLASSES):
        class_embeddings = all_embeddings[np.array(class_labels) == class_id]
        prototypes[class_id] = torch.from_numpy(class_embeddings.mean(axis=0)).to(device)
        print(f"Prototype for {class_names[class_id]} calculated from {len(class_embeddings)} examples.")

    
    # 2. Sliding Window Search on Target Segments
    window_size = WINDOW_SECONDS * REF_FPS
    final_results = {}
    for seg_idx, target_segment in target_segments.items():
        L, D = target_segment.shape
        window_starts = np.arange(0, L - window_size + 1, RETRIEVAL_STRIDE)
        
        print(f"\nProcessing Target Segment {seg_idx} (Length: {L} frames)...")
        candidates = []
        # Process windows in batches for efficiency
        batch_size = 4
        for i in range(0, len(window_starts), batch_size):
            batch_starts = window_starts[i:i + batch_size]
            batch_windows = [target_segment[start:start+window_size, :] for start in batch_starts]
            
            # Convert batch to tensor and encode
            batch_tensor = torch.stack([torch.from_numpy(w) for w in batch_windows]).to(device)
            with torch.no_grad():
                window_embeddings = encoder(batch_tensor) # (Batch_Size, FEATURE_SIZE)
            
            # Compare each window embedding to all prototypes
            for j, embed in enumerate(window_embeddings):
                start_frame = int(batch_starts[j])
                end_frame = int(start_frame + window_size)
                
                min_distance = float('inf')
                best_class = -1
                
                # Calculate distance to all prototypes
                for class_id, prototype in prototypes.items():
                    # Squared Euclidean distance in embedding space
                    dist = torch.sum((embed - prototype) ** 2).item()
                    
                    if dist < min_distance:
                        min_distance = dist
                        best_class = class_id

                # Store candidate if distance is below a threshold (optional, but good practice)
                # Here, we keep all and let NMS filter
                candidates.append({
                    "segment_id": seg_idx,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "class_id": best_class,
                    "maneuver_type": class_names[best_class],
                    "score": min_distance # Lower score is better
                })

        # 3. Apply NMS to the final set of candidates (globally or per segment)
        # We apply NMS globally for all candidates of the same type, but over the whole list.
        results = non_maximum_suppression(candidates, NMS_OVERLAP_THRESHOLD)
        final_results[seg_idx] = results
    with open('prototype_retrieval_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)
    
    return final_results


# --- Main Execution ---

if __name__ == '__main__':
    # A. Simulate Data
    # 50 examples per class for robust prototype calculation
    reference_data = 'dataset/offline_dataset_1.h5'
    offline_data = 'dataset/offline_dataset_2.h5'
    reference_maneuver_data = simulate_maneuvers(reference_data)

    # Simulate 5 large target segments (18s to 25s)
    print("2. Simulating Large Target Segments...")
    target_segments_list = {}   
    with h5py.File(offline_data, 'r') as f:
        scene_list = list(f.keys())
        for scene in scene_list:
            target_data = f[scene]['dynamics'][:]
            target_segments_list[scene] = target_data.astype(np.float32)
    
    # B. Train Model
    start_time = time.time()
    trained_encoder = train_prototypical_network(reference_maneuver_data)
    training_time = time.time() - start_time
    print(f"Total training time: {training_time:.2f} seconds.")

    # C. Retrieval
    retrieval_results = retrieve_maneuvers(trained_encoder, reference_maneuver_data, target_segments_list)

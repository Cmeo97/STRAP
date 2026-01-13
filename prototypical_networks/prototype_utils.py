import os
import h5py
import torch
import numpy as np
import numba as nb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

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

def resize_scipy(data, new_size):
    # Create an index for current data (0 to 399)
    x_old = np.linspace(0, 1, data.shape[0])
    # Create an index for the new target size (0 to 299)
    x_new = np.linspace(0, 1, new_size)
    
    # Kind='linear' or 'cubic' for smoother interpolation
    f = interp1d(x_old, data, axis=0, kind='linear')
    return f(x_new)

def retrieve_maneuver_dtw(prototype_seq, scene_embeddings):
    """
    Finds the best matching subsequence in a scene for a given prototype.
    prototype_seq: (Len_P, Dim) Tensor
    scene_embeddings: (Len_S, Dim) Tensor
    """
    # 1. Calculate Pairwise Distances (PyTorch is faster for this matrix math)
    # We use Squared Euclidean Distance
    distance_matrix = get_distance_matrix(prototype_seq, scene_embeddings)
    accumulated_cost_matrix = compute_accumulated_cost_matrix_subsequence_dtw_standard(
        distance_matrix
    )
    path = compute_optimal_warping_path_subsequence_dtw_standard(accumulated_cost_matrix)
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
    
    return {
        "cost": float(normalized_score),
        "start_idx": int(start),
        "end_idx": int(end)
    }

def calibrate_dtw_thresholds(encoder, prototypes, reference_maneuvers, K_SHOT, std_mult=1.0):
    """
    Calculates the average DTW distance between the Prototype and its own members.
    This gives us a class-specific threshold for 'what a good match looks like'.
    """
    encoder.eval()
    device = next(encoder.parameters()).device
    class_thresholds = {}

    print("--- Calibrating Class-Specific DTW Thresholds ---")
    
    for class_id, sequences in reference_maneuvers.items():
        proto_seq = prototypes[class_id] # (L, Dim)
        scores = []
        
        # We test the prototype against its own support samples
        for seq_np in sequences[:K_SHOT]:
            with torch.no_grad():
                seq_tensor = torch.from_numpy(seq_np).float().unsqueeze(0).to(device)
                # Get (L, Feature_Dim)
                sample_features = encoder(seq_tensor, return_temporal=True).squeeze(0)
                
                # Perform DTW
                match = retrieve_maneuver_dtw(proto_seq.cpu().numpy(), sample_features.cpu().numpy())
                scores.append(match['cost'])
        
        # Threshold = Mean + 2 * Std Dev (Common heuristic for anomaly detection)
        # This allows for some variance while filtering out bad matches.
        mean_dist = np.mean(scores)
        std_dist = np.std(scores)
        
        # We set the threshold slightly above the average to allow for test-time variance
        class_thresholds[class_id] = mean_dist + (std_mult * std_dist)
        
        print(f"Class {class_id} | Mean Dist: {mean_dist:.4f} | Calibrated Threshold: {class_thresholds[class_id]:.4f}")

    return class_thresholds

@nb.jit(nopython=True)
def get_distance_matrix(sub_trajectory, dataset_trajectory):
    """
    Optimized Squared Euclidean Distance Matrix.
    Compatible with Prototypical Training and Standard DTW.
    """
    # 1. Compute squared norms
    # (L_sub, 1)
    sub_squared = np.sum(sub_trajectory**2, axis=1).reshape(-1, 1)
    # (1, L_data)
    dataset_squared = np.sum(dataset_trajectory**2, axis=1).reshape(1, -1)

    # 2. Compute cross term
    cross_term = np.dot(sub_trajectory, dataset_trajectory.T)
    
    # 3. Compute Squared Euclidean Distance
    dist_matrix = sub_squared - 2 * cross_term + dataset_squared
    
    # 4. Clean up numerical noise
    # Ensures no negative values before DTW starts
    dist_matrix = np.maximum(dist_matrix, 0.0)

    # Note: No np.sqrt() here to match the Squared Euclidean used in training
    return dist_matrix

@nb.jit(nopython=True)
def compute_accumulated_cost_matrix_subsequence_dtw_standard(C):
    N, M = C.shape
    D = np.zeros((N + 1, M + 1))
    D[1:, 0] = np.inf
    D[0, :] = 0  # Subsequence DTW: can start anywhere in M
    
    for n in range(N):
        for m in range(M):
            # Standard DTW step: min(Diagonal, Up, Left)
            D[n+1, m+1] = C[n, m] + min(D[n, m], D[n, m+1], D[n+1, m])
    return D[1:, 1:]

@nb.jit(nopython=True)
def compute_optimal_warping_path_subsequence_dtw_standard(D):
    N, M = D.shape
    n = N - 1
    m = np.argmin(D[n, :]) # Best end point
    path = [(n, m)]

    while n > 0:
        if m == 0:
            n = n - 1
        else:
            # Check Diagonal, Up, Left
            v_diag = D[n-1, m-1]
            v_up   = D[n-1, m]
            v_left = D[n, m-1]
            
            if v_diag <= v_up and v_diag <= v_left:
                n, m = n - 1, m - 1
            elif v_up <= v_left:
                n = n - 1
            else:
                m = m - 1
        path.append((n, m))
    
    path.reverse()
    return np.array(path)
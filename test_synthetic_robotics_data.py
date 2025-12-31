# robotics_retrieval_pipeline_v2.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
from unsupervised_retrieval_metrics import UnsupervisedRetrievalEvaluator

# -----------------------------
# 1. Mock proprioceptive data generation
# -----------------------------
import numpy as np

def generate_mock_proprioceptive_data(num_sequences=100, seq_len=100, tasks=10):
    """
    Generates task-diverse proprioceptive sequences with moderate stochasticity.

    Shape: (num_sequences, seq_len, num_features, num_tasks)

    Feature layout:
    0-2   : position (x, y, z)
    3-5   : velocity (vx, vy, vz)
    6-9   : joint angles (4-DOF)
    10-13 : joint torques (4-DOF)
    """
    rng = np.random.default_rng(42)
    n_features = 14
    n_landmarks = 2

    data = np.zeros((num_sequences, seq_len, n_features, tasks), dtype=np.float32)
    samples_per_task = num_sequences // tasks
    total_assigned = 0

    for t in range(tasks):
        if t == tasks - 1:
            num_samples = num_sequences - total_assigned
        else:
            num_samples = samples_per_task
        start_idx = total_assigned
        end_idx = total_assigned + num_samples
        total_assigned += num_samples

        # Uniformly spaced landmarks along the sequence
        landmark_margin = int(0.1 * seq_len)
        landmarks = np.linspace(landmark_margin, seq_len - landmark_margin, n_landmarks, dtype=int)
        lm_strengths = rng.uniform(0.5, 1.0, size=n_landmarks)
        sigmas = np.full(n_landmarks, 5.0)
        xs = np.arange(seq_len)
        lm_masks = np.exp(-0.5 * ((xs[:, None] - landmarks[None, :]) / sigmas[None, :]) ** 2)

        for s in range(num_samples):
            # Initialize sequences
            pos = rng.normal(0, 0.05, size=(seq_len, 3))
            vel = np.zeros((seq_len, 3))
            joints = rng.normal(0, 0.03, size=(seq_len, 4))
            torques = np.zeros((seq_len, 4))

            # Base stochastic trends
            pos_trend = np.cumsum(rng.normal(0, 0.005, size=(seq_len, 3)), axis=0)
            vel_trend = np.gradient(pos_trend, axis=0)
            joint_trend = np.cumsum(rng.normal(0, 0.002, size=(seq_len, 4)), axis=0)

            pos += pos_trend
            vel += vel_trend
            joints += joint_trend
            torques += 0.4 * np.gradient(joints, axis=0)

            # Apply landmark effects
            for li in range(n_landmarks):
                mask = lm_masks[:, li][:, None]  # (seq_len, 1)
                pos += mask * rng.normal(0, 0.01, size=(seq_len, 3))
                joints += mask * lm_strengths[li] * rng.normal(0, 0.01, size=(seq_len, 4))
                torques += mask * lm_strengths[li] * 0.2 * rng.normal(0, 0.01, size=(seq_len, 4))

            signal = np.concatenate([pos, vel, joints, torques], axis=1)  # (seq_len, 14)
            data[start_idx + s, :, :, t] = signal

    return data

# -----------------------------
# 3. STRAP-like retrieval (on raw sequences)
# -----------------------------
def strap_retrieve(query_data, dataset_data, top_k=None):
    """
    Retrieve for each query time series (raw, shape [seq_len]) the most similar dataset time series (raw, shape [seq_len]).
    If top_k is None, retrieve a single index per query (top-1 retrieval).
    If top_k > 1, retrieves top_k indices per query.
    Similarity is computed as negative Euclidean distance.
    """
    # query_data: [num_queries, seq_len]
    # dataset_data: [num_db, seq_len]
    pairwise_dists = cdist(query_data, dataset_data, metric="euclidean")  # [num_queries, num_db]
    if top_k is None or top_k == 1:
        top_indices = np.argmin(pairwise_dists, axis=1)[:, None]
    else:
        top_indices = np.argsort(pairwise_dists, axis=1)[:, :top_k]
    return top_indices

# -----------------------------
# 4. Generate reference dataset (task 6)
# -----------------------------
def get_reference_dataset(data, task_index=5):
    """
    Returns all samples for a specific task (task_index)
    """
    ref_data = data[:, :, :, task_index]
    return ref_data

# -----------------------------
# 5. Pipeline execution
# -----------------------------
if __name__ == "__main__":
    num_samples, seq_len, num_features, num_tasks = 100, 50, 14, 6
    data = generate_mock_proprioceptive_data(num_samples, seq_len, num_tasks)
    print(data.shape)

    # Split general dataset (tasks 1-5)
    general_data = data[:, :, :, :(num_tasks-1)].reshape(num_samples * (num_tasks-1), seq_len, num_features)
    np.random.shuffle(general_data)

    # Reference dataset (task 6)
    reference_data = get_reference_dataset(data, task_index=num_tasks-1)  # shape: [num_samples, seq_len, num_features]
    print(reference_data.shape)
    print(general_data.shape)

    # Now, for each feature, perform retrieval and evaluate on raw time series (shape: [num_samples, seq_len])
    feature_results = {}

    for feat_idx in range(num_features):
        print(f"\n--- Evaluating feature {feat_idx} ---")

        # Get per-feature [num_samples, seq_len] for general and reference data
        general_feat_data = general_data[:, :, feat_idx]
        reference_feat_data = reference_data[:, :, feat_idx]

        # Use STRAP to retrieve from raw time series (no embeddings)
        # Each reference time series retrieves its nearest neighbour from the general set by Euclidean distance
        top_indices = strap_retrieve(reference_feat_data, general_feat_data, top_k=1)
        top_indices_flat = top_indices.flatten()
        retrieved_feat_data = general_feat_data[top_indices_flat]

        # Data-adaptive eps (percentile of pairwise distance between reference and retrieved)
        pairwise_dists = np.linalg.norm(reference_feat_data - retrieved_feat_data, axis=1, keepdims=True)
        eps = np.percentile(pairwise_dists, 10)

        print("Mean of retrieved_feat_data:", np.mean(retrieved_feat_data))
        print("Mean of reference_feat_data:", np.mean(reference_feat_data))

        # Evaluate raw data using UnsupervisedRetrievalEvaluator, shape: [num_samples, seq_len]
        evaluator = UnsupervisedRetrievalEvaluator(
            retrieved_embeddings=retrieved_feat_data,
            reference_embeddings=reference_feat_data
        )
        results = evaluator.full_analysis(
            eps=eps
        )
        feature_results[f"feature_{feat_idx}"] = results

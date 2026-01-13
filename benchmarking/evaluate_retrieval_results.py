import os
import json
import argparse
from typing import Dict, List

import h5py
import numpy as np

from benchmarking.benchmark_eval_metrics import UnsupervisedRetrievalEvaluator

# The 'ee_pos' key in the reference set corresponds to 'ee_pose'
FEATURE_KEYS = ["ee_pose", "gripper_states", "joint_states"]
REFERENCE_FEATURE_KEYS = ["ee_pos", "gripper_states", "joint_states"]

def concat_obs_group(obs_group: h5py.Group, feature_keys=None) -> np.ndarray:
    """
    Concatenate obs features along feature dimension.

    Returns:
        (T, F_total)
    """
    if feature_keys is None:
        feature_keys = FEATURE_KEYS
    features = [obs_group[k][()] for k in feature_keys]
    return np.concatenate(features, axis=-1)


def load_reference_hdf5(path: str) -> Dict[str, np.ndarray]:
    """
    Load reference data.

    Returns:
        episode_name -> (N, T, F)
    """
    out = {}
    with h5py.File(path, "r") as f:
        print(f"Reference HDF5 file keys: {list(f.keys())}")
        for episode in f.keys():
            print(f"Loading episode: {episode}")
            ee_pos = f[episode]["ee_pos"][()]            # (N, T, D1)
            gripper_states = f[episode]["gripper_states"][()]  # (N, T, D2)
            joint_states = f[episode]["joint_states"][()]      # (N, T, D3)
            # Each of these is (N, T, D)
            # Concatenate along the feature dimension (last axis)
            features = [ee_pos, gripper_states, joint_states]
            data = np.concatenate(features, axis=-1)   # (N, T, F)
            print(f"  Concatenated feature shape (N, T, F): {data.shape}")
            out[episode] = data
            print(f"  Final data shape: {out[episode].shape}")
    return out


def load_retrieved_hdf5(path: str) -> Dict[str, np.ndarray]:
    """
    Load retrieved data.

    Returns:
        episode_name -> (N, T, F)
    """
    out = {}
    with h5py.File(path, "r") as f:
        results_group = f["results"]
        for episode in results_group.keys():
            matches: List[np.ndarray] = []
            for k in sorted(results_group[episode].keys()):
                if not k.startswith("match_"):
                    continue
                obs = results_group[episode][k]["obs"]
                matches.append(concat_obs_group(obs, feature_keys=FEATURE_KEYS))
            if not matches:
                raise ValueError(f"No matches found for episode {episode}")
            out[episode] = np.stack(matches, axis=0)
    return out


def run_evaluation(
    retrieved_hdf5: str,
    reference_hdf5: str,
) -> Dict[str, Dict[str, float]]:
    retrieved = load_retrieved_hdf5(retrieved_hdf5)
    reference = load_reference_hdf5(reference_hdf5)
    print(retrieved['bottom_drawer_close'].shape)
    print(reference['bottom_drawer_close'].shape)

    results: Dict[str, Dict[str, float]] = {}

    for episode, retrieval_data in retrieved.items():
        if episode not in reference:
            raise KeyError(f"Episode {episode} missing from reference data")

        # Update: pass arguments by position, not keyword, to match class definition
        evaluator = UnsupervisedRetrievalEvaluator(
            retrieval_data,
            reference[episode],
            viz_dir="visualizations",
        )
        os.makedirs("visualizations", exist_ok=True)
        results[episode] = evaluator.evaluate()
        evaluator.distributional_checker(episode=episode)

    return results


# Example usage:
# python -m benchmarking.evaluate_retrieval_results \
#   --retrieved_path data/retrieval_results/retrieval_results_stumpy.hdf5 \
#   --reference_path data/target_data/target_dataset.hdf5 \
#   --output_json stumpy_results.json
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieved_path", type=str, required=True)
    parser.add_argument("--reference_path", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    results = run_evaluation(
        retrieved_hdf5=args.retrieved_path,
        reference_hdf5=args.reference_path,
    )

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()


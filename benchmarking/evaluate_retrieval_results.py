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
        for episode in f.keys():
            ee_pos = f[episode]["ee_pos"][()]            # (N, T, D1)
            gripper_states = f[episode]["gripper_states"][()]  # (N, T, D2)
            joint_states = f[episode]["joint_states"][()]      # (N, T, D3)
            # Each of these is (N, T, D)
            # Concatenate along the feature dimension (last axis)
            features = [ee_pos, gripper_states, joint_states]
            data = np.concatenate(features, axis=-1)   # (N, T, F)
            out[episode] = data
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
    print(f"[Evaluation] Retrieved data shape: {retrieved['bottom_drawer_close'].shape}")
    print(f"[Evaluation] Reference data shape: {reference['bottom_drawer_close'].shape}")

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

        # CPU parallelism for parallelized metric run. Not necessary cuz this is very fast.
        # results[episode] = evaluator.evaluate(parallel=True) 
        results[episode] = evaluator.evaluate()

        evaluator.distributional_checker(episode=episode)

    return results


# Example usage:
# python benchmarking/evaluate_retrieval_results.py \
#   --retrieved_path data/retrieval_results/retrieval_results_stumpy.hdf5 \
#   --reference_path data/target_data/target_dataset.hdf5 \
#   --output_file stumpy_results.json
def main():
    DEFAULT_OUTPUT_DIR = "retrieval_eval_outputs"
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieved_path", default='data/retrieval_results/libero_retrieval_results_stumpy.hdf5')
    parser.add_argument("--reference_path", default='data/target_data/libero_target_dataset.hdf5')
    parser.add_argument("--output_json", default='data/retrieval_results/libero_stumpy_result.json')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    results = run_evaluation(
        retrieved_hdf5=args.retrieved_path,
        reference_hdf5=args.reference_path,
    )

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results JSON to {output_json}")


if __name__ == "__main__":
    main()


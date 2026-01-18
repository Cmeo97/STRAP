import os
import json
import argparse
from typing import Dict, List

import h5py
import numpy as np

from benchmarking.benchmark_eval_metrics import UnsupervisedRetrievalEvaluator

# The 'ee_pos' key in the reference set corresponds to 'ee_pose'
FEATURE_KEYS = ["ee_pos", "gripper_states", "joint_states"]

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
            episode_data = f[episode]
            # libero dataset structure
            if "obs/ee_pos" in episode_data:
                feature_keys = ["ee_pos", "gripper_states", "joint_states"]
            # nuscene dataset structure
            elif "obs/velocity" in episode_data:
                feature_keys = ["velocity", "acceleration", "yaw_rate"]
            data = concat_obs_group(episode_data["obs"], feature_keys=feature_keys)
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
                if 'ee_pos' in obs:
                    feature_keys = ["ee_pos", "gripper_states", "joint_states"]
                elif 'velocity' in obs:
                    feature_keys = ["velocity", "acceleration", "yaw_rate"]
                matches.append(concat_obs_group(obs, feature_keys=feature_keys))
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
    print(f"[Evaluation] Retrieved data length: {len(retrieved)}")
    print(f"[Evaluation] Reference data length: {len(reference)}")

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='libero')
    parser.add_argument("--metric", default='stumpy')
    parser.add_argument("--retrieved_path", default='data/retrieval_results/')
    parser.add_argument("--reference_path", default='data/target_data/')
    parser.add_argument("--output_path", default='data/evaluation_results/')
    
    args = parser.parse_args()

    target_path = os.path.join(args.reference_path, f"{args.dataset}_target_dataset.hdf5")
    retrieved_path = os.path.join(args.retrieved_path, f"{args.dataset}_retrieval_results_{args.metric}.hdf5")
    output_json = os.path.join(args.output_path, f"{args.dataset}_retrieval_evaluation_{args.metric}.json")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    results = run_evaluation(
        retrieved_hdf5=retrieved_path,
        reference_hdf5=target_path,
    )

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results JSON to {output_json}")


if __name__ == "__main__":
    main()


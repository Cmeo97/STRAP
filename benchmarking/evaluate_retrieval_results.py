import os
import json
import argparse
from typing import Dict, List
from tqdm.auto import tqdm
import h5py
import numpy as np

from benchmarking.benchmark_eval_metrics import UnsupervisedRetrievalEvaluator
from benchmarking.benchmark_utils import concat_obs_group

def load_reference_hdf5(path: str) -> Dict[str, np.ndarray]:
    """
    Load reference data. Each episode is mapped to its observed feature concatenation.
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
    Load retrieved data (from hdf5 with 'results' group). Handles variable-length matches.
    Returns:
        episode_name -> (N, T, F)
    """
    out = {}
    with h5py.File(path, "r") as f:
        results_group = f["results"]
        for episode in results_group:
            matches = []
            for match_key in sorted(results_group[episode].keys()):
                if not match_key.startswith("match_"):
                    continue
                obs = results_group[episode][match_key]["obs"]
                match_arr = concat_obs_group(obs)
                if match_arr.size == 0:
                    continue
                matches.append(match_arr)
            if not matches:
                raise ValueError(f"No matches found for episode {episode}")

            # Pad variable-length episode matches to max length for stacking
            max_len = max(match.shape[0] for match in matches)
            padded_matches = [
                np.pad(match, ((0, max_len - match.shape[0]), (0,0)), mode="constant") if match.shape[0] < max_len else match
                for match in matches
            ]
            out[episode] = np.stack(padded_matches, axis=0)
    return out


def generate_sliding_windows(
    data: np.ndarray, 
    window_size: int, 
    step_size: int = 1
) -> List[np.ndarray]:
    """
    Yield sliding windows (N, window_size, F) from a (N, T, F) array.
    """
    n, T, f = data.shape
    if T < window_size:
        return []
    return [data[:, start:start + window_size, :] for start in range(0, T - window_size + 1, step_size)]


def evaluate_with_sliding_windows(
    retrieved: np.ndarray,
    reference: np.ndarray,
    viz_dir: str = None,
    episode: str = None,
) -> Dict[str, float]:
    """
    Evaluate metrics using sliding windows for sequence pairs of different lengths.
    Returns:
        dict of averaged metrics
    """
    print(f"Shape of retrieved: {retrieved.shape}")
    print(f"Shape of reference: {reference.shape}")
    N_r, T_r, F_r = retrieved.shape
    N_ref, T_ref, F_ref = reference.shape
    if F_r != F_ref:
        raise ValueError(f"Feature dimension mismatch: retrieved has {F_r}, reference has {F_ref}")

    # Standard evaluation if lengths match
    if T_r == T_ref:
        evaluator = UnsupervisedRetrievalEvaluator(retrieved, reference, viz_dir=viz_dir)
        results = evaluator.evaluate()
        if episode is not None:
            evaluator.distributional_checker(episode=episode)
        print(f"[Evaluation] Final metrics (no sliding): {results}")
        return results

    # Use sliding windows: windows from longer sequence, fixed on the shorter one
    window_size = min(T_r, T_ref)
    if T_r > T_ref:
        windows = generate_sliding_windows(retrieved, window_size)
        fixed = reference
        sliding_side = "retrieved"
    else:
        windows = generate_sliding_windows(reference, window_size)
        fixed = retrieved
        sliding_side = "reference"

    print(
        f"[Evaluation] Sliding over {sliding_side} sequence "
        f"with {len(windows)} windows of size {window_size}"
    )

    all_metrics = []

    for i, window in enumerate(
        tqdm(windows, desc=f"Evaluating sliding windows ({episode})", leave=False)
    ):
        if T_r > T_ref:
            window_retrieved, window_reference = window, fixed
        else:
            window_retrieved, window_reference = fixed, window

        run_dist_check = (i == 0) and (episode is not None)

        evaluator = UnsupervisedRetrievalEvaluator(
            window_retrieved, window_reference, viz_dir=viz_dir
        )
        metrics = evaluator.evaluate()
        all_metrics.append(metrics)

        if run_dist_check:
            evaluator.distributional_checker(episode=episode)

    # Average all metrics
    averaged_metrics: Dict[str, float] = {}
    metric_keys = all_metrics[0].keys()

    for key in metric_keys:
        values = [m[key] for m in all_metrics if m.get(key) is not None]
        averaged_metrics[key] = float(np.mean(values)) if values else None

    print(f"[Evaluation] Averaged metrics across {len(windows)} windows:")
    for k, v in averaged_metrics.items():
        print(f"  {k}: {v}")

    return averaged_metrics


def run_evaluation(
    retrieved_hdf5: str,
    reference_hdf5: str,
) -> Dict[str, Dict[str, float]]:
    retrieved = load_retrieved_hdf5(retrieved_hdf5)
    reference = load_reference_hdf5(reference_hdf5)
    print(f"[Evaluation] Retrieved data length: {len(retrieved)}")
    print(f"[Evaluation] Reference data length: {len(reference)}")

    os.makedirs("visualizations", exist_ok=True)
    results: Dict[str, Dict[str, float]] = {}

    for episode, retrieval_data in retrieved.items():
        if episode not in reference:
            raise KeyError(f"Episode {episode} missing from reference data")

        reference_data = reference[episode]
        T_retrieved, T_reference = retrieval_data.shape[1], reference_data.shape[1]
        print(f"[Evaluation] Episode {episode}: retrieved T={T_retrieved}, reference T={T_reference}")

        # Run appropriate evaluation
        if T_retrieved != T_reference:
            metrics = evaluate_with_sliding_windows(
                retrieved=retrieval_data,
                reference=reference_data,
                viz_dir="visualizations",
                episode=episode,
            )
        else:
            evaluator = UnsupervisedRetrievalEvaluator(
                retrieval_data, reference_data, viz_dir="visualizations"
            )
            metrics = evaluator.evaluate()
            evaluator.distributional_checker(episode=episode)
        results[episode] = metrics

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='libero')
    parser.add_argument("--metric", default="stumpy")
    parser.add_argument("--retrieved_path", default="data/retrieval_results/")
    parser.add_argument("--reference_path", default="data/target_data/")
    parser.add_argument("--output_path", default="data/evaluation_results/")
    args = parser.parse_args()

    target_path = os.path.join(
        args.reference_path, f"{args.dataset}_target_dataset.hdf5"
    )
    retrieved_path = os.path.join(
        args.retrieved_path, f"{args.dataset}_retrieval_results_{args.metric}.hdf5"
    )
    output_json = os.path.join(
        args.output_path, f"{args.dataset}_retrieval_evaluation_{args.metric}.json"
    )
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    results = run_evaluation(retrieved_hdf5=retrieved_path, reference_hdf5=target_path)

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results JSON to {output_json}")


if __name__ == "__main__":
    main()


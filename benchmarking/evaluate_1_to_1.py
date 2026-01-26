import os
import numpy as np
from typing import Dict, Any
import h5py
import json
from benchmarking.evaluate_retrieval_results import UnsupervisedRetrievalEvaluator, evaluate_with_sliding_windows

def evaluate_1_to_1_sample_with_window(
    retrieved_sample: np.ndarray,
    reference_sample: np.ndarray,
    task: str = None,
) -> Dict[str, float]:
    """
    Computes metrics for one retrieved and one reference sample, allowing for different time lengths
    by using the sliding window averaging technique from the main evaluator.
    Args:
        retrieved_sample: (T1, F) array
        reference_sample: (T2, F) array
        task: Optional, for verbose output in tqdm
    Returns:
        Dict of computed (averaged) metrics.
    """
    if retrieved_sample.ndim != 2 or reference_sample.ndim != 2:
        raise ValueError("Samples must be 2D arrays of shape (T, F)")
    # Shape to (1, T, F) for UnsupervisedRetrievalEvaluator batch interface
    retrieved = retrieved_sample[np.newaxis, ...]
    reference = reference_sample[np.newaxis, ...]
    T_r = retrieved.shape[1]
    T_ref = reference.shape[1]
    if T_r == T_ref:
        evaluator = UnsupervisedRetrievalEvaluator(retrieved, reference)
        # Just run standard evaluation (no sliding)
        results = evaluator.evaluate()
        # Filter down to common single-match metrics
        filtered = {
            "wasserstein": results.get("wasserstein"),
            "dtw_distance": results.get("dtw_nn"),
            "spectral_wasserstein": results.get("spectral_wasserstein"),
            "temporal_correlation": results.get("temporal_correlation"),
        }
        return filtered
    else:
        # Use exactly the same sliding window averaging logic as main evaluation
        metrics = evaluate_with_sliding_windows(retrieved, reference, task=task)
        filtered = {
            "wasserstein": metrics.get("wasserstein"),
            "dtw_distance": metrics.get("dtw_nn"),
            "spectral_wasserstein": metrics.get("spectral_wasserstein"),
            "temporal_correlation": metrics.get("temporal_correlation"),
        }
        return filtered

def evaluate_all_tasks_and_save_json(
    target_file: str,
    retrieval_files: Dict[str, str],
    match_key: Dict[str, str],
    task_keys: list,
    output_json_path: str,
):
    # Define function for extracting features from an hdf5 file
    def get_feats_from_file(h5f, task, match, is_target=False):
        if is_target:
            data = h5f[task]
            gripper_states = data['obs/gripper_states'][()]
            joint_states = data['obs/joint_states'][()]
            ee_pos = data['obs/ee_pos'][()]
            all_data = np.concatenate([gripper_states, joint_states, ee_pos], axis=-1)
            return all_data[0]
        else:
            task_data = h5f['results'][task][match]
            gripper_states = task_data['obs/gripper_states'][()]
            joint_states = task_data['obs/joint_states'][()]
            ee_pos = task_data['obs/ee_pos'][()]
            all_data = np.concatenate([gripper_states, joint_states, ee_pos], axis=-1)
            return all_data

    # Open target only once for efficiency
    with h5py.File(target_file, 'r') as f_target:
        target_features = {}
        for task in task_keys:
            target_features[task] = get_feats_from_file(f_target, task, match_key[task], is_target=True)

    # For each method (retrieval file), accumulate per-task results
    output = {}
    for method, retrieval_file in retrieval_files.items():
        if retrieval_file is None:
            continue  # skip if missing
        try:
            with h5py.File(retrieval_file, 'r') as f_ret:
                per_task_metrics = {}
                for task in task_keys:
                    try:
                        retrieved_feats = get_feats_from_file(f_ret, task, match_key[task], is_target=False)
                        target_feats = target_features[task]
                        metrics = evaluate_1_to_1_sample_with_window(retrieved_feats, target_feats, task=task)
                        per_task_metrics[task] = metrics
                    except Exception as e:
                        per_task_metrics[task] = {"error": str(e)}
            output[method] = per_task_metrics
        except Exception as e:
            output[method] = {"error": str(e)}

    # Aggregate task-wise metrics to compute average across tasks for each method
    for method, per_task_results in output.items():
        if "error" in per_task_results:
            continue
        valid_metrics = [
            v for v in per_task_results.values()
            if isinstance(v, dict) and "error" not in v and all(x is not None for x in v.values())
        ]
        if valid_metrics:
            avg_metrics = {
                metric: float(np.mean([v[metric] for v in valid_metrics])) for metric in valid_metrics[0].keys()
            }
        else:
            avg_metrics = {}
        output[method]["averaged_metrics"] = avg_metrics

    # Ensure output directory exists in 'data/evaluation_results'
    output_json_path = os.path.join("data", "evaluation_results", os.path.basename(output_json_path))
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(output_json_path, "w") as fout:
        json.dump(output, fout, indent=2)
    print(f"✓ Saved results JSON to {output_json_path}")

# --- Example Usage ---
if __name__ == "__main__":
    # Edit these for your use-case
    target_file = 'data/target_data/libero_target_dataset.hdf5'
    retrieval_files = {
        'roser':   'data/retrieval_results/libero/libero_retrieval_results_prototype.hdf5',
        'stumpy':  'data/retrieval_results/libero/libero_retrieval_results_stumpy.hdf5',
        'dtaidistance': 'data/retrieval_results/libero/libero_retrieval_results_dtaidistance.hdf5'
    }
    match_key = {
        'pnp': 'match_22',
        'microwave_open': 'match_35',
        'bottom_drawer_open': 'match_35',
        'stove_on': 'match_10',
        'stove_off': 'match_37',
        'top_drawer_close': 'match_25',
        'top_drawer_open': 'match_25',
        'bottom_drawer_close': 'match_37',
        'microwave_close': 'match_2'
    }
    task_keys = [
        'pnp', 'microwave_open', 'bottom_drawer_open', 'stove_on',
        'stove_off', 'top_drawer_close', 'top_drawer_open',
        'bottom_drawer_close', 'microwave_close'
    ]
    output_json_path = "results_1to1_eval.json"

    evaluate_all_tasks_and_save_json(
        target_file=target_file,
        retrieval_files=retrieval_files,
        match_key=match_key,
        task_keys=task_keys,
        output_json_path=output_json_path,
    )
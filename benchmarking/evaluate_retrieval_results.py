import os
import json
import traceback
import argparse
from typing import Dict, Any, Optional
from tqdm.auto import tqdm
import numpy as np
import yaml
from benchmarking.benchmark_eval_metrics import UnsupervisedRetrievalEvaluator
from benchmarking.benchmark_utils import (
    generate_sliding_windows,
    load_reference_hdf5,
    load_retrieved_hdf5,
)
from benchmarking.distribution_plots import create_combined_distribution_plots


def evaluate_with_sliding_windows(
    retrieved: np.ndarray,
    reference: np.ndarray,
    task: str = None,
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
        evaluator = UnsupervisedRetrievalEvaluator(retrieved, reference)
        results = evaluator.evaluate()
        print(f"[Evaluation] Final metrics (no sliding): {results}")
        return results

    # Use sliding windows: windows from longer sequence, fixed on the shorter one
    window_size = min(T_r, T_ref)
    MAX_WINDOWS = 50
    
    if T_r > T_ref:
        longer_seq_length = T_r
        sliding_side = "retrieved"
    else:
        longer_seq_length = T_ref
        sliding_side = "reference"
    
    # Calculate number of possible windows with stride=1
    num_possible_windows = longer_seq_length - window_size + 1
    
    # Dynamically adjust stride to cap at MAX_WINDOWS
    if num_possible_windows > MAX_WINDOWS:
        step_size = int(np.ceil(num_possible_windows / MAX_WINDOWS))
        print(
            f"[Evaluation] Capping windows to {MAX_WINDOWS}: "
            f"using stride={step_size} (would generate {num_possible_windows} with stride=1)"
        )
    else:
        step_size = 1
    
    if T_r > T_ref:
        windows = generate_sliding_windows(retrieved, window_size, step_size=step_size)
        fixed = reference
    else:
        windows = generate_sliding_windows(reference, window_size, step_size=step_size)
        fixed = retrieved

    print(
        f"[Evaluation] Sliding over {sliding_side} sequence "
        f"with {len(windows)} windows of size {window_size} (stride={step_size})"
    )

    all_metrics = []

    for window in tqdm(windows, desc=f"Evaluating sliding windows ({task})", leave=False):
        if T_r > T_ref:
            window_retrieved, window_reference = window, fixed
        else:
            window_retrieved, window_reference = fixed, window

        evaluator = UnsupervisedRetrievalEvaluator(window_retrieved, window_reference)
        metrics = evaluator.evaluate()
        all_metrics.append(metrics)

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
    top_k: int = None,
) -> Dict[str, Any]:
    """
    Run evaluation for a single method.
    Handles cases where episodes have no matches by creating JSON entries with null metrics.
    
    Args:
        retrieved_hdf5: Path to retrieved HDF5 file
        reference_hdf5: Path to reference HDF5 file
        top_k: Optional number of top matches to use
        
    Returns:
        Dictionary with episode-level metrics and averaged_metrics
    """
    print(f"[Evaluation] Loading data...")
    print(f"  Retrieved: {retrieved_hdf5}")
    print(f"  Reference: {reference_hdf5}")
    
    try:
        retrieved = load_retrieved_hdf5(retrieved_hdf5, top_k=top_k)
        reference = load_reference_hdf5(reference_hdf5)
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {str(e)}")
    
    print(f"[Evaluation] Retrieved data: {len(retrieved)} episodes (some may have no matches)")
    print(f"[Evaluation] Reference data: {len(reference)} episodes")
    if top_k is not None:
        print(f"[Evaluation] Using top-{top_k} matches")

    results: Dict[str, Dict[str, Any]] = {}
    all_episode_metrics = []

    # Define all possible metric keys for consistent structure
    metric_keys = [
        "wasserstein", "dtw_nn", "spectral_wasserstein", "temporal_correlation",
        "distributional_coverage.density", "distributional_coverage.coverage", "diversity_icd"
    ]

    # Process all reference episodes to ensure we have entries for all tasks
    for episode in reference.keys():
        reference_data = reference[episode]
        
        # Check if this episode has retrieved data
        if episode not in retrieved or retrieved[episode] is None:
            # No matches for this episode - create entry with null metrics
            print(f"[Evaluation] Episode {episode}: No matches found. Creating entry with null metrics.")
            null_metrics = {key: None for key in metric_keys}
            null_metrics["num_matches"] = 0
            null_metrics["has_matches"] = False
            results[episode] = null_metrics
            continue

        retrieval_data = retrieved[episode]

        try:
            T_retrieved, T_reference = retrieval_data.shape[1], reference_data.shape[1]
            num_matches = retrieval_data.shape[0]
            print(f"[Evaluation] Episode {episode}: retrieved shape={retrieval_data.shape}, reference shape={reference_data.shape}")

            # Run appropriate evaluation
            if T_retrieved != T_reference:
                metrics = evaluate_with_sliding_windows(
                    retrieved=retrieval_data,
                    reference=reference_data,
                    task=episode,
                )
            else:
                evaluator = UnsupervisedRetrievalEvaluator(retrieval_data, reference_data)
                metrics = evaluator.evaluate()
            
            # Add metadata about matches
            metrics["num_matches"] = num_matches
            metrics["has_matches"] = True
            
            results[episode] = metrics
            all_episode_metrics.append(metrics)
            print(f"[Evaluation] Episode {episode} completed successfully")
        except Exception as e:
            print(f"Error evaluating episode {episode}: {str(e)}")
            print(traceback.format_exc())
            # Still create an entry with error indicator
            error_metrics = {key: None for key in metric_keys}
            error_metrics["num_matches"] = retrieval_data.shape[0] if retrieval_data is not None else 0
            error_metrics["has_matches"] = retrieval_data is not None
            error_metrics["error"] = str(e)
            results[episode] = error_metrics
            continue

    if not results:
        raise ValueError("No episodes were processed (no reference data found)")

    # Compute averaged metrics across all episodes with valid matches
    averaged_metrics: Dict[str, Any] = {}
    episodes_with_matches = [m for m in all_episode_metrics if m.get("has_matches", False)]
    
    if episodes_with_matches:
        metric_keys = episodes_with_matches[0].keys()
        for key in metric_keys:
            if key in ["num_matches", "has_matches", "error"]:
                continue
            values = [m[key] for m in episodes_with_matches if m.get(key) is not None]
            averaged_metrics[key] = float(np.mean(values)) if values else None

    # Add summary statistics
    total_episodes = len(results)
    episodes_with_matches_count = sum(1 for r in results.values() if r.get("has_matches", False))
    episodes_without_matches_count = total_episodes - episodes_with_matches_count
    
    averaged_metrics["summary"] = {
        "total_episodes": total_episodes,
        "episodes_with_matches": episodes_with_matches_count,
        "episodes_without_matches": episodes_without_matches_count
    }

    print(f"[Evaluation] Computed averaged metrics across {len(episodes_with_matches)} episodes with matches")
    print(f"[Evaluation] Total episodes: {total_episodes}, With matches: {episodes_with_matches_count}, Without matches: {episodes_without_matches_count}")

    # Return results with averaged metrics
    output = {
        **results,
        "averaged_metrics": averaged_metrics
    }
    return output


def get_reference_path(
    benchmark: str,
    method: str,
    config: Dict[str, Any]
) -> str:
    """
    Get the reference data path for a given benchmark and method.
    
    IMPORTANT: modified_strap METHOD (on libero benchmark) ALWAYS uses modified_strap_target_data.hdf5,
    NEVER libero_target_dataset.hdf5, even though its retrieval data is in libero folder.
    
    Args:
        benchmark: Name of the benchmark
        method: Name of the method
        config: Configuration dictionary
        
    Returns:
        Path to reference data file
    """
    # HARDCODED: modified_strap method always uses its own target, never libero's
    if method == 'modified_strap':
        target_path = config['dataset_paths']['modified_strap_target']
        if not os.path.exists(target_path):
            raise FileNotFoundError(
                f"modified_strap target file not found: {target_path}\n"
                f"This file is REQUIRED and must exist. modified_strap method NEVER uses libero_target."
            )
        return target_path
    
    # For all other methods, use the benchmark's target
    target_key = f'{benchmark}_target'
    if target_key not in config['dataset_paths']:
        raise KeyError(
            f"Reference path not found for benchmark '{benchmark}'. "
            f"Expected key '{target_key}' in config['dataset_paths']. "
            f"Available keys: {list(config['dataset_paths'].keys())}"
        )
    return config['dataset_paths'][target_key]


def run_calculations(config: Dict[str, Any]) -> Dict[str, Dict[int, Dict[str, Dict[str, np.ndarray]]]]:
    """
    Run evaluation calculations for all benchmark/method/top_k combinations.
    
    Returns:
        distribution_data: Nested dict structure: benchmark -> top_k -> task -> {method: data}
    """
    benchmarks = config.get('benchmarks', [])
    methods = config.get('methods', [])

    # Define top-K ranges for each dataset
    TOP_K_RANGES = {
        'libero': list(range(50, 151, 20)),  # [50, 70, 90, 110, 130, 150]
        'nuscene': list(range(20, 51, 10)),  # [20, 30, 40, 50]
        'droid': list(range(40, 81, 10)),    # [40, 50, 60, 70, 80]
    }

    # Generate all benchmark-method combinations
    total_combinations = 0
    for benchmark in benchmarks:
        total_combinations += len(methods) * len(TOP_K_RANGES.get(benchmark, []))
    
    print(f"\n{'='*80}")
    print(f"Starting evaluation calculations for {total_combinations} total combinations")
    print(f"Benchmarks: {benchmarks}")
    print(f"Methods: {methods}")
    print(f"{'='*80}\n")

    combination_count = 0
    distribution_data = {}

    for benchmark in benchmarks:
        top_k_values = TOP_K_RANGES.get(benchmark, [])
        if not top_k_values:
            print(f"Warning: No top-K range defined for benchmark '{benchmark}'. Skipping.")
            continue

        base_retrieval_path = config['retrieval_paths'][benchmark]

        # Initialize distribution data structure
        if benchmark not in distribution_data:
            distribution_data[benchmark] = {}
        for top_k in top_k_values:
            if top_k not in distribution_data[benchmark]:
                distribution_data[benchmark][top_k] = {}

        for method in methods:
            # HARDCODED: modified_strap method retrieval files are in libero folder but named libero_retrieval_results_modified_strap.hdf5
            # This is because the retrieval was done using libero data, but evaluation uses modified_strap_target
            if method == 'modified_strap':
                retrieved_path = os.path.join(
                    config['retrieval_paths']['libero'], f'libero_retrieval_results_{method}.hdf5'
                )
                print(f"[modified_strap] Using retrieval file from libero folder: {retrieved_path}")
                print(f"[modified_strap] Will evaluate against modified_strap_target_data.hdf5 (NOT libero_target)")
            else:
                retrieved_path = os.path.join(
                    base_retrieval_path, f'{benchmark}_retrieval_results_{method}.hdf5'
                )
            
            if not os.path.exists(retrieved_path):
                print(f"Warning: Retrieval file not found: {retrieved_path}. Skipping.")
                continue

            for top_k in top_k_values:
                combination_count += 1
                print(f"\n{'='*80}")
                print(f"Combination {combination_count}/{total_combinations}")
                print(f"Benchmark: {benchmark} | Method: {method} | Top-K: {top_k}")
                print(f"{'='*80}\n")
                
                # HARDCODED: modified_strap method outputs to separate folder structure
                if method == 'modified_strap':
                    base_eval_path = config['evaluation_paths']['modified_strap']
                    output_dir = os.path.join(base_eval_path, f"top_{top_k}")
                    os.makedirs(output_dir, exist_ok=True)
                    # Output filename uses modified_strap as benchmark name for separate folder structure
                    output_json = os.path.join(
                        output_dir, f"modified_strap_retrieval_evaluation_{method}.json"
                    )
                else:
                    base_eval_path = config['evaluation_paths'][benchmark]
                    output_dir = os.path.join(base_eval_path, f"top_{top_k}")
                    os.makedirs(output_dir, exist_ok=True)
                    output_json = os.path.join(
                        output_dir, f"{benchmark}_retrieval_evaluation_{method}.json"
                    )

                try:
                    # Load retrieved data for this method/top_k
                    retrieved_data = load_retrieved_hdf5(retrieved_path, top_k=top_k)
                    
                    # Store data for combined distribution plots (skip None values - episodes with no matches)
                    # HARDCODED: modified_strap method stores data under 'modified_strap' key for separate folder structure
                    plot_benchmark = 'modified_strap' if method == 'modified_strap' else benchmark
                    if plot_benchmark not in distribution_data:
                        distribution_data[plot_benchmark] = {}
                    if top_k not in distribution_data[plot_benchmark]:
                        distribution_data[plot_benchmark][top_k] = {}
                    
                    for task, task_retrieved in retrieved_data.items():
                        if task_retrieved is None:
                            continue  # Skip episodes with no matches for plotting
                        if task not in distribution_data[plot_benchmark][top_k]:
                            distribution_data[plot_benchmark][top_k][task] = {}
                        distribution_data[plot_benchmark][top_k][task][method] = task_retrieved
                    
                    # Determine reference path based on benchmark and method
                    # HARDCODED: modified_strap METHOD ALWAYS uses modified_strap_target, NEVER libero_target
                    reference_path = get_reference_path(benchmark, method, config)
                    print(f"[Evaluation] Benchmark: {benchmark}, Method: {method}")
                    print(f"[Evaluation] Reference path: {reference_path}")
                    if method == 'modified_strap':
                        print(f"[Evaluation] IMPORTANT: modified_strap method uses modified_strap_target_data.hdf5, NOT libero_target_dataset.hdf5")
                    
                    # Verify reference file exists (get_reference_path already checks this for modified_strap)
                    if not os.path.exists(reference_path):
                        print(f"ERROR: Reference file not found: {reference_path}")
                        if method == 'modified_strap':
                            print(f"  modified_strap method MUST use: {config['dataset_paths'].get('modified_strap_target', 'NOT FOUND')}")
                            print(f"  modified_strap method MUST NOT use: {config['dataset_paths'].get('libero_target', 'NOT FOUND')}")
                        else:
                            print(f"  Expected path for {benchmark}: {config['dataset_paths'].get(f'{benchmark}_target', 'NOT FOUND')}")
                        raise FileNotFoundError(f"Reference file not found: {reference_path}")
                    
                    # Run evaluation
                    results = run_evaluation(
                        retrieved_hdf5=retrieved_path,
                        reference_hdf5=reference_path,
                        top_k=top_k
                    )

                    with open(output_json, "w") as f:
                        json.dump(results, f, indent=2)
                    print(f"✓ Saved results JSON to {output_json}")
                except Exception as e:
                    print(f"✗ Error evaluating {benchmark}/{method}/top_{top_k}: {e}")
                    print(traceback.format_exc())
                    continue

    print(f"\n{'='*80}")
    print(f"Calculations complete! Processed {combination_count} combinations.")
    print(f"{'='*80}\n")
    
    return distribution_data


def run_plots(config: Dict[str, Any], distribution_data: Dict[str, Dict[int, Dict[str, Dict[str, np.ndarray]]]]) -> None:
    """
    Generate distribution plots from pre-computed distribution data.
    
    Args:
        config: Configuration dictionary
        distribution_data: Nested dict structure: benchmark -> top_k -> task -> {method: data}
    """
    benchmarks = config.get('benchmarks', [])
    methods = config.get('methods', [])
    
    print(f"\n{'='*80}")
    print("Generating distribution plots...")
    print(f"{'='*80}\n")
    
    reference_data_store = {}

    for benchmark in benchmarks:
        target_path = config['dataset_paths'][f'{benchmark}_target']
        
        try:
            reference_data = load_reference_hdf5(target_path)
            reference_data_store[benchmark] = reference_data
            print(f"[Loading] Loaded reference data for {benchmark}: {len(reference_data)} episodes from {target_path}")
        except Exception as e:
            print(f"Error loading reference data for {benchmark}: {e}")
            continue
    
    # HARDCODED: Also load modified_strap_target for modified_strap method (which runs on libero benchmark)
    if 'modified_strap' in methods:
        modified_strap_target_path = config['dataset_paths']['modified_strap_target']
        try:
            modified_strap_reference_data = load_reference_hdf5(modified_strap_target_path)
            # Store it under 'modified_strap' key for distribution plots
            reference_data_store['modified_strap'] = modified_strap_reference_data
            print(f"[Loading] Loaded reference data for modified_strap method: {len(modified_strap_reference_data)} episodes from {modified_strap_target_path}")
        except Exception as e:
            print(f"Error loading modified_strap target data: {e}")
            print(f"  modified_strap method MUST use: {config['dataset_paths'].get('modified_strap_target', 'NOT FOUND')}")
            print(f"  modified_strap method MUST NOT use: {config['dataset_paths'].get('libero_target', 'NOT FOUND')}")

    create_combined_distribution_plots(
        distribution_data,
        reference_data_store,
        config
    )


def load_distribution_data_from_evaluations(config: Dict[str, Any]) -> Dict[str, Dict[int, Dict[str, Dict[str, np.ndarray]]]]:
    """
    Load distribution data by re-reading evaluation results.
    This is used for plots-only mode.
    
    Returns:
        distribution_data: Nested dict structure: benchmark -> top_k -> task -> {method: data}
    """
    benchmarks = config.get('benchmarks', [])
    methods = config.get('methods', [])
    
    TOP_K_RANGES = {
        'libero': list(range(50, 151, 20)),
        'nuscene': list(range(20, 51, 10)),
        'droid': list(range(40, 81, 10)),
    }
    
    distribution_data = {}
    
    for benchmark in benchmarks:
        top_k_values = TOP_K_RANGES.get(benchmark, [])
        if benchmark not in distribution_data:
            distribution_data[benchmark] = {}
        
        base_retrieval_path = config['retrieval_paths'][benchmark]
        
        for method in methods:
            # HARDCODED: modified_strap method retrieval files are in libero folder but named libero_retrieval_results_modified_strap.hdf5
            # This is because the retrieval was done using libero data, but evaluation uses modified_strap_target
            if method == 'modified_strap':
                retrieved_path = os.path.join(
                    config['retrieval_paths']['libero'], f'libero_retrieval_results_{method}.hdf5'
                )
                print(f"[modified_strap method] Using retrieval file from libero folder: {retrieved_path}")
                print(f"[modified_strap method] Will evaluate against modified_strap_target_data.hdf5 (NOT libero_target)")
            else:
                retrieved_path = os.path.join(
                    base_retrieval_path, f'{benchmark}_retrieval_results_{method}.hdf5'
                )
            
            if not os.path.exists(retrieved_path):
                continue
            
            for top_k in top_k_values:
                # HARDCODED: modified_strap method stores data under 'modified_strap' key for separate folder structure
                plot_benchmark = 'modified_strap' if method == 'modified_strap' else benchmark
                if plot_benchmark not in distribution_data:
                    distribution_data[plot_benchmark] = {}
                if top_k not in distribution_data[plot_benchmark]:
                    distribution_data[plot_benchmark][top_k] = {}
                
                try:
                    retrieved_data = load_retrieved_hdf5(retrieved_path, top_k=top_k)
                    for task, task_retrieved in retrieved_data.items():
                        if task_retrieved is None:
                            continue  # Skip episodes with no matches for plotting
                        if task not in distribution_data[plot_benchmark][top_k]:
                            distribution_data[plot_benchmark][top_k][task] = {}
                        distribution_data[plot_benchmark][top_k][task][method] = task_retrieved
                except Exception as e:
                    print(f"Warning: Could not load data for {benchmark}/{method}/top_{top_k}: {e}")
                    continue
    
    return distribution_data


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval results and generate distribution plots"
    )
    parser.add_argument(
        '--mode',
        type=str,
        nargs='+',
        choices=['calculations', 'plots', 'both'],
        default=['both'],
        help='Execution mode(s): calculations only, plots only, both, or metric_plots (default: both). You can specify multiple modes, e.g. --mode plots metric_plots'
    )
    
    args = parser.parse_args()
    config = yaml.safe_load(open('config/config.yaml', 'r'))

    # Convenience: flatten list to set for quick querying
    modes = set(args.mode)

    distribution_data = None

    # "both" acts as shorthand for ['calculations', 'plots']
    if 'both' in modes:
        modes.add('calculations')
        modes.add('plots')
        modes.discard('both')

    # Do calculations (writing output files)
    if 'calculations' in modes:
        distribution_data = run_calculations(config)

    # Do plots (loading from calculation output if not already available)
    if 'plots' in modes:
        if distribution_data is None:
            print("Loading distribution data from evaluation results...")
            distribution_data = load_distribution_data_from_evaluations(config)
        if distribution_data is None:
            raise ValueError("No distribution data available for plotting")
        run_plots(config, distribution_data)

if __name__ == "__main__":
    main()


import json
import os
import glob
import re
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd


# ============================================================
# I/O Utilities
# ============================================================

def load_metrics(path: str) -> Dict[str, Any]:
    """Load metrics JSON file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Metrics file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be episode-keyed dict")

    return data


def parse_filename(filename: str) -> Tuple[str, str, Optional[int]]:
    """
    Parse filename and path to extract dataset, method, and top_k.
    Format: .../[dataset]/top_[k]/[dataset]_retrieval_evaluation_[method].json
    Returns: (dataset, method, top_k)
    """
    # Extract top_k from directory path
    top_k = None
    path_parts = filename.split(os.sep)
    for part in path_parts:
        if part.startswith("top_"):
            try:
                top_k = int(part.split("_")[1])
            except (ValueError, IndexError):
                pass
    
    # Extract dataset and method from filename
    match = re.match(r"(.+)_retrieval_evaluation_(.+)\.json", os.path.basename(filename))
    if not match:
        raise ValueError(f"Filename format not recognized: {filename}")
    return match.group(1), match.group(2), top_k


def find_all_evaluation_files(results_dir: str) -> List[str]:
    """
    Find all evaluation JSON files in the results directory.
    Searches recursively in top_K subdirectories.
    Handles both:
    - Root directory: data/evaluation_results (searches in all benchmarks)
    - Benchmark directory: data/evaluation_results/libero (searches in top_K subdirs)
    """
    files = []
    
    # Pattern 1: Search in benchmark/top_K structure: results_dir/*/top_*/...json
    # This handles root directory like data/evaluation_results
    pattern1 = os.path.join(results_dir, "*", "top_*", "*_retrieval_evaluation_*.json")
    files.extend(glob.glob(pattern1))
    
    # Pattern 2: Search directly in top_K subdirs: results_dir/top_*/...json
    # This handles benchmark directory like data/evaluation_results/libero
    pattern2 = os.path.join(results_dir, "top_*", "*_retrieval_evaluation_*.json")
    files.extend(glob.glob(pattern2))
    
    # Pattern 3: Backward compatibility - search directly in results_dir
    # This handles old structure without top_K subdirectories
    pattern3 = os.path.join(results_dir, "*_retrieval_evaluation_*.json")
    files.extend(glob.glob(pattern3))
    
    # Remove duplicates and sort
    files = sorted(list(set(files)))
    
    return files


# ============================================================
# Data Processing Utilities
# ============================================================

def flatten_metrics(raw: Dict[str, Any], exclude_averaged: bool = True) -> pd.DataFrame:
    """
    Convert metrics dict to DataFrame.
    Handles flattened metric names like "distributional_coverage.density".
    """
    rows = []
    
    for episode, m in raw.items():
        # Skip averaged_metrics if requested
        if exclude_averaged and episode == "averaged_metrics":
            continue
            
        if not isinstance(m, dict):
            continue
            
        try:
            row = {
                "episode": episode,
                "wasserstein": m.get("wasserstein"),
                "dtw_nn": m.get("dtw_nn"),
                "spectral_wasserstein": m.get("spectral_wasserstein"),
                "temporal_correlation": m.get("temporal_correlation"),
                "density": m.get("distributional_coverage.density"),
                "coverage": m.get("distributional_coverage.coverage"),
                "diversity_icd": m.get("diversity_icd"),
            }
            # Only add if all required metrics are present
            if all(v is not None for v in row.values() if isinstance(v, (int, float))):
                rows.append(row)
        except (KeyError, AttributeError) as e:
            # Skip episodes with missing metrics
            continue

    if not rows:
        raise ValueError("No valid episodes found in metrics file")
        
    df = pd.DataFrame(rows).set_index("episode")
    return df


def load_averaged_metrics(raw: Dict[str, Any]) -> Dict[str, float]:
    """Extract averaged metrics if present."""
    if "averaged_metrics" in raw:
        return raw["averaged_metrics"]
    return {}


# ============================================================
# Path Resolution Utilities
# ============================================================

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


# ============================================================
# Data Loading Utilities
# ============================================================

def load_distribution_data_from_evaluations(config: Dict[str, Any]) -> Dict[str, Dict[int, Dict[str, Dict[str, np.ndarray]]]]:
    """
    Load distribution data by re-reading evaluation results.
    This is used for plots-only mode.
    
    Returns:
        distribution_data: Nested dict structure: benchmark -> top_k -> task -> {method: data}
    """
    from benchmarking.benchmark_utils import load_retrieved_hdf5
    
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


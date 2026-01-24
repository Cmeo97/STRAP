"""
Distribution plotting utilities for evaluation results.
Separated from evaluation metrics for better modularity.
"""
import os
import json
import traceback
from typing import Dict, Optional, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from benchmarking.benchmark_utils import get_feature_names


def create_combined_distribution_plots(
    distribution_data: Dict[str, Dict[int, Dict[str, Dict[str, np.ndarray]]]],
    reference_data_store: Dict[str, Dict[str, np.ndarray]],
    config: Dict
) -> None:
    """
    Create combined distribution plots for all benchmark/top_k/task combinations using KDE.
    
    Args:
        distribution_data: Nested dict structure: benchmark -> top_k -> task -> {method: data}
        reference_data_store: Reference data per benchmark
        config: Configuration dictionary
    """
    print(f"\n{'='*80}")
    print("Creating combined (smoothed) distribution plots...")
    print(f"{'='*80}\n")
    
    for benchmark in distribution_data:
        if benchmark not in reference_data_store:
            print(f"Warning: No reference data stored for {benchmark}. Skipping combined plots.")
            continue

        reference_data = reference_data_store[benchmark]

        for top_k in distribution_data[benchmark]:
            for task in distribution_data[benchmark][top_k]:
                method_data_dict = distribution_data[benchmark][top_k][task]

                try:
                    # modified_strap is now a separate benchmark with its own reference data
                    if task not in reference_data:
                        print(f"Warning: Task {task} not in reference data. Skipping combined plot.")
                        continue
                    reference_data_task = reference_data[task]

                    # Create KDE plots (dimension-level and task-level)
                    _plot_kde_distributions(
                        method_data_dict=method_data_dict,
                        reference_data=reference_data_task,
                        benchmark=benchmark,
                        task=task,
                        top_k=top_k
                    )
                except Exception as e:
                    print(f"Error creating combined (smoothed) plot for {benchmark}/{task}/top_{top_k}: {e}")
                    print(traceback.format_exc())
                    continue
        
        # Create benchmark-level aggregation plots
        create_benchmark_level_plots(
            {benchmark: distribution_data[benchmark]},
            {benchmark: reference_data_store[benchmark]},
            config
        )


def _plot_kde_distributions(
    method_data_dict: Dict[str, np.ndarray],
    reference_data: np.ndarray,
    benchmark: str,
    task: str,
    top_k: int
) -> None:
    """
    Plot KDE distributions with:
    - Very small dimension-level subplots
    - Task-level aggregation (all dimensions combined)
    - Benchmark-level aggregation (all tasks combined) - handled separately
    
    Args:
        method_data_dict: Dictionary mapping method names to their retrieved data arrays (N, T, F)
        reference_data: Reference data array (N_ref, T, F)
        benchmark: Benchmark name
        task: Task/episode name
        top_k: Top-K value
    """
    all_methods = list(method_data_dict.keys())
    
    # Flatten data: (N, T, F) -> (N*T, F)
    ref_flat = reference_data.reshape(-1, reference_data.shape[-1])
    method_flat = {
        m: method_data_dict[m].reshape(-1, method_data_dict[m].shape[-1]) 
        for m in all_methods
    }
    
    F = ref_flat.shape[1]
    colors = sns.color_palette("tab10", len(all_methods) + 1)
    
    # Get feature names for this benchmark
    try:
        feature_names = get_feature_names(benchmark)
        # Format feature names: parse underscores and convert to title case
        formatted_feature_names = [
            name.replace("_", " ").title() for name in feature_names
        ]
        # Ensure we have enough feature names
        if len(formatted_feature_names) < F:
            # Pad with generic names if needed
            formatted_feature_names.extend([f"Feature {i}" for i in range(len(formatted_feature_names), F)])
    except (ValueError, KeyError):
        # Fallback to generic names if dataset not recognized
        formatted_feature_names = [f"Feature {f}" for f in range(F)]
    
    # Create figure with two main sections: dimension subplots (small) and task aggregation
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.3)
    
    # Top section: Very small dimension-level subplots
    n_cols_dim = min(12, F)  # Show up to 12 dimensions per row
    n_rows_dim = (F + n_cols_dim - 1) // n_cols_dim
    axs_dim = fig.add_subplot(gs[0])
    axs_dim.axis('off')
    
    # Create a grid for dimension subplots within the top section
    inner_gs = gs[0].subgridspec(n_rows_dim, n_cols_dim, wspace=0.1, hspace=0.3)
    dim_axes = []
    for i in range(n_rows_dim):
        for j in range(n_cols_dim):
            idx = i * n_cols_dim + j
            if idx < F:
                ax = fig.add_subplot(inner_gs[i, j])
                dim_axes.append(ax)
            else:
                break
    
    # Plot dimension-level distributions (very small)
    for f in range(F):
        ax = dim_axes[f]
        
        # Trim to top 99% to remove outliers
        ref_data = ref_flat[:, f]
        ref_99th = np.percentile(ref_data, 99)
        ref_trimmed = ref_data[ref_data <= ref_99th]
        
        sns.kdeplot(
            ref_trimmed, 
            ax=ax, 
            label="Reference", 
            fill=True, 
            linewidth=1, 
            color=colors[0], 
            alpha=0.7
        )
        for m_idx, method in enumerate(all_methods):
            try:
                method_data = method_flat[method][:, f]
                method_99th = np.percentile(method_data, 99)
                method_trimmed = method_data[method_data <= method_99th]
                
                sns.kdeplot(
                    method_trimmed, 
                    ax=ax, 
                    label=method, 
                    fill=True, 
                    linewidth=1, 
                    color=colors[m_idx+1], 
                    alpha=0.45
                )
            except Exception as kde_e:
                print(f"Warning: KDE failed for '{method}' feature {f}: {kde_e}")
        # Use formatted feature name instead of "F{f}"
        ax.set_title(formatted_feature_names[f], fontsize=8)
        ax.tick_params(labelsize=5)
        if f == 0:
            ax.set_ylabel("Density", fontsize=5)
            ax.legend(fontsize=4, loc='upper right')
        else:
            ax.set_ylabel("")  # Remove density label after first plot
    
    # Bottom section: Task-level aggregation (all dimensions combined)
    ax_task = fig.add_subplot(gs[1])
    
    # Flatten all dimensions for task-level view and trim to top 99%
    ref_all_dims = ref_flat.flatten()
    ref_99th = np.percentile(ref_all_dims, 99)
    ref_all_dims_trimmed = ref_all_dims[ref_all_dims <= ref_99th]
    
    for m_idx, method in enumerate(all_methods):
        method_all_dims = method_flat[method].flatten()
        method_99th = np.percentile(method_all_dims, 99)
        method_all_dims_trimmed = method_all_dims[method_all_dims <= method_99th]
        try:
            sns.kdeplot(
                method_all_dims_trimmed,
                ax=ax_task,
                label=method,
                fill=True,
                linewidth=2,
                color=colors[m_idx+1],
                alpha=0.45
            )
        except Exception as kde_e:
            print(f"Warning: KDE failed for '{method}' task aggregation: {kde_e}")
    
    sns.kdeplot(
        ref_all_dims_trimmed,
        ax=ax_task,
        label="Reference",
        fill=True,
        linewidth=2,
        color=colors[0],
        alpha=0.7
    )
    ax_task.set_title(f"Task-level aggregation: {task} (all dimensions)", fontsize=14, fontweight='bold')
    ax_task.set_xlabel("Feature value (all dimensions)", fontsize=10)
    ax_task.set_ylabel("Density", fontsize=10)
    ax_task.legend(fontsize=9)
    ax_task.grid(True, alpha=0.3)
    
    plt.suptitle(f"Distribution Comparison: {task} ({benchmark}, top-{top_k})", fontsize=16, fontweight='bold')
    
    out_dir = os.path.join("data/visualization", benchmark, "all_methods", f"top_{top_k}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{task}_top_{top_k}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved distribution plot to {out_path}")


def create_benchmark_level_plots(
    distribution_data: Dict[str, Dict[int, Dict[str, Dict[str, np.ndarray]]]],
    reference_data_store: Dict[str, Dict[str, np.ndarray]],
    config: Dict
) -> None:
    """
    Create benchmark-level aggregation plots (all tasks combined).
    
    Args:
        distribution_data: Nested dict structure: benchmark -> top_k -> task -> {method: data}
        reference_data_store: Reference data per benchmark
        config: Configuration dictionary
    """
    print(f"\n{'='*80}")
    print("Creating benchmark-level aggregation plots...")
    print(f"{'='*80}\n")
    
    for benchmark in distribution_data:
        if benchmark not in reference_data_store:
            print(f"Warning: No reference data stored for {benchmark}. Skipping benchmark plots.")
            continue

        reference_data = reference_data_store[benchmark]
        
        for top_k in distribution_data[benchmark]:
            # Aggregate all tasks for this benchmark/top_k
            all_ref_data = []
            all_method_data = {}
            
            for task in distribution_data[benchmark][top_k]:
                method_data_dict = distribution_data[benchmark][top_k][task]
                
                # modified_strap is now a separate benchmark with its own reference data
                if task not in reference_data:
                    continue
                ref_task = reference_data[task]
                
                # Flatten and collect reference data
                all_ref_data.append(ref_task.reshape(-1, ref_task.shape[-1]))
                
                # Flatten and collect method data
                for method, method_arr in method_data_dict.items():
                    if method not in all_method_data:
                        all_method_data[method] = []
                    all_method_data[method].append(method_arr.reshape(-1, method_arr.shape[-1]))
            
            if not all_ref_data:
                continue
            
            # Concatenate all tasks
            ref_benchmark = np.concatenate(all_ref_data, axis=0)
            method_benchmark = {
                method: np.concatenate(method_data, axis=0)
                for method, method_data in all_method_data.items()
            }
            
            # Create benchmark-level plot
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = sns.color_palette("tab10", len(method_benchmark) + 1)
            
            # Flatten all dimensions and trim to top 99%
            ref_all = ref_benchmark.flatten()
            ref_99th = np.percentile(ref_all, 99)
            ref_all_trimmed = ref_all[ref_all <= ref_99th]
            
            sns.kdeplot(
                ref_all_trimmed,
                ax=ax,
                label="Reference",
                fill=True,
                linewidth=2,
                color=colors[0],
                alpha=0.7
            )
            
            for m_idx, (method, method_arr) in enumerate(method_benchmark.items()):
                method_all = method_arr.flatten()
                method_99th = np.percentile(method_all, 99)
                method_all_trimmed = method_all[method_all <= method_99th]
                try:
                    sns.kdeplot(
                        method_all_trimmed,
                        ax=ax,
                        label=method,
                        fill=True,
                        linewidth=2,
                        color=colors[m_idx+1],
                        alpha=0.45
                    )
                except Exception as kde_e:
                    print(f"Warning: KDE failed for '{method}' benchmark aggregation: {kde_e}")
            
            ax.set_title(f"Benchmark-level aggregation: {benchmark} (all tasks, top-{top_k})", fontsize=16, fontweight='bold')
            ax.set_xlabel("Feature value (all dimensions, all tasks)", fontsize=12)
            ax.set_ylabel("Density", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            out_dir = os.path.join("visualization", benchmark, "all_methods", f"top_{top_k}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{benchmark}_benchmark_level_top_{top_k}.png")
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved benchmark-level plot to {out_path}")
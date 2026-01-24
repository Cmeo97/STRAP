import json
import os
import glob
import re
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_bar, geom_histogram, geom_point, geom_tile,
    geom_text, geom_vline, geom_label, labs, theme, theme_minimal,
    scale_fill_gradient2, scale_fill_gradientn, scale_fill_discrete, scale_fill_manual,
    scale_color_discrete, facet_wrap, facet_grid, ggsave,
    element_text, element_blank, element_rect, element_line,
    guides, guide_colorbar, guide_legend, coord_flip, geom_line,
    scale_x_discrete, scale_y_discrete, scale_y_continuous, position_dodge
)

# ============================================================
# I/O
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
# Flattening
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
# Plot helpers
# ============================================================

def _save(p, path: str, width: float = 10, height: float = 6, dpi: int = 150):
    """Save plotnine figure."""
    # Ensure directory exists
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    p.save(path, width=width, height=height, dpi=dpi, verbose=False)


# ============================================================
# Single dataset/method plots
# ============================================================

def plot_metric_means(df: pd.DataFrame, out_dir: str):
    """Mean metrics across episodes."""
    mean_df = df.mean().reset_index()
    mean_df.columns = ['metric', 'value']
    
    p = (ggplot(mean_df, aes(x='metric', y='value'))
         + geom_bar(stat='identity', fill='steelblue')
         + labs(x='Metric', y='Mean value', title='Mean retrieval metrics across episodes')
         + theme_minimal()
         + theme(
             axis_text_x=element_text(rotation=45, hjust=1),
             plot_title=element_text(size=14, weight='bold'),
             axis_title=element_text(size=12)
         )
    )
    _save(p, os.path.join(out_dir, "metric_means.png"), width=10, height=5)


def plot_dtw_distribution(df: pd.DataFrame, out_dir: str):
    """DTW-NN distance distribution."""
    dtw_df = pd.DataFrame({'dtw_nn': df['dtw_nn']})
    mean_val = df['dtw_nn'].mean()
    bins = min(30, len(df))
    
    # Create label for mean
    label_df = pd.DataFrame({
        'x': [mean_val], 
        'y': [0], 
        'label': [f'Mean: {mean_val:.2f}']
    })
    
    # Calculate histogram max for label positioning
    hist_max = dtw_df['dtw_nn'].value_counts(bins=bins).max()
    label_df['y'] = hist_max * 0.05  # Position label at 5% of max height
    
    p = (ggplot(dtw_df, aes(x='dtw_nn'))
         + geom_histogram(bins=bins, fill='steelblue', alpha=0.7, color='black')
         + geom_vline(xintercept=mean_val, color='red', linetype='dashed', size=1)
         + geom_text(aes(x='x', y='y', label='label'), data=label_df, 
                     inherit_aes=False, color='red', size=9)
         + labs(x='DTW-NN distance', y='Episode count', 
                title='DTW-NN distance distribution')
         + theme_minimal()
         + theme(
             plot_title=element_text(size=14, weight='bold'),
             axis_title=element_text(size=12)
         )
    )
    _save(p, os.path.join(out_dir, "dtw_nn_distribution.png"), width=8, height=5)


def plot_density_vs_coverage(df: pd.DataFrame, out_dir: str):
    """Density vs Coverage scatter plot."""
    plot_df = df.reset_index()
    plot_df = plot_df[['episode', 'coverage', 'density']]
    
    p = (ggplot(plot_df, aes(x='coverage', y='density'))
         + geom_point(size=3, alpha=0.6, color='steelblue', fill='steelblue', stroke=1)
         + geom_text(aes(label='episode'), size=8, nudge_y=0.01, alpha=0.7)
         + labs(x='Coverage', y='Density', title='Density vs Coverage')
         + theme_minimal()
         + theme(
             plot_title=element_text(size=14, weight='bold'),
             axis_title=element_text(size=12),
             panel_grid=element_line(color='gray', alpha=0.3)
         )
    )
    _save(p, os.path.join(out_dir, "density_vs_coverage.png"), width=7, height=6)


def plot_episode_metric_heatmap(df: pd.DataFrame, out_dir: str):
    """
    Create one heatmap per episode showing metric rankings.
    Rankings: 1 = best, higher = worse (for distance metrics, lower is better; for others, higher is better).
    Averages metrics across all episodes first, then ranks to ensure integer ranks.
    """
    metric_cols = [c for c in df.columns if c not in ['episode']]
    
    # Define which metrics are "lower is better"
    lower_better = ['wasserstein', 'dtw_nn', 'spectral_wasserstein']
    
    # Average metrics across all episodes first
    avg_metrics = df[metric_cols].mean()
    
    # Calculate rankings based on averaged values (one ranking for all episodes)
    rankings = {}
    for metric in metric_cols:
        avg_value = avg_metrics[metric]
        if pd.isna(avg_value):
            rankings[metric] = None
            continue
        
        is_lower_better = any(m in metric.lower() for m in lower_better)
        
        # Get all averaged metric values for ranking context
        all_avg_values = avg_metrics[metric_cols].dropna()
        
        if metric not in all_avg_values.index:
            rankings[metric] = None
            continue
        
        # Rank the averaged values
        if is_lower_better:
            # Lower is better: rank 1 = smallest value (ascending=True)
            ranks = all_avg_values.rank(method='min', ascending=True)
        else:
            # Higher is better: rank 1 = largest value (ascending=False)
            ranks = all_avg_values.rank(method='min', ascending=False)
        
        # Get rank for this metric's averaged value
        rank = int(ranks.loc[metric])
        rankings[metric] = rank
    
    # Create one heatmap per episode using the same rankings (based on averages)
    for episode in df.index:
        
        # Convert to DataFrame for plotting (use same rankings for all episodes)
        rank_df = pd.DataFrame([rankings])
        rank_df['episode'] = episode
        
        # Convert to long format
        rank_long = rank_df.melt(id_vars='episode', var_name='metric', value_name='rank')
        rank_long = rank_long[rank_long['rank'].notna()].copy()
        
        # Format metric names: replace "_" with space and title case
        rank_long['metric_formatted'] = rank_long['metric'].str.replace('_', ' ').str.title()
        
        # Create labels (just the rank number, no decimals - already integers)
        rank_long['label'] = rank_long['rank'].astype(str)
        
        # Determine color scale range
        max_rank = rank_long['rank'].max()
        min_rank = rank_long['rank'].min()
        
        p = (ggplot(rank_long, aes(x='metric_formatted', y='episode', fill='rank'))
             + geom_tile(color='white', size=0.5)
             + geom_text(aes(label='label'), size=10, color='black', fontweight='bold')
             + scale_fill_gradient2(low='#22c55e', mid='#eab308', high='#ef4444', 
                                   midpoint=(max_rank + min_rank) / 2,
                                   name='Rank\n(1=best)')
             + labs(x='Metric', y='Episode', title=f'Metric Rankings: {episode}')
             + theme_minimal()
             + theme(
                 axis_text_x=element_text(rotation=45, hjust=1, size=11),
                 axis_text_y=element_text(size=11),
                 plot_title=element_text(size=14, weight='bold'),
                 axis_title=element_text(size=12),
                 aspect_ratio=0.2
             )
        )
        
        # Save with episode name in filename
        safe_episode_name = str(episode).replace(' ', '_').replace('/', '_')
        heatmap_dir = os.path.join(out_dir, "heatmaps")
        os.makedirs(heatmap_dir, exist_ok=True)
        filename = f"episode_metric_heatmap_{safe_episode_name}.png"
        _save(p, os.path.join(heatmap_dir, filename), width=10, height=3)


def plot_metric_correlation(df: pd.DataFrame, out_dir: str):
    """Metric correlation matrix."""
    corr = df.corr()
    
    # Convert to long format for plotnine
    corr_df = corr.reset_index().melt(id_vars='index', var_name='metric2', value_name='correlation')
    corr_df.columns = ['metric1', 'metric2', 'correlation']
    
    # Create text labels for correlation values
    corr_df['label'] = corr_df['correlation'].apply(lambda x: f'{x:.2f}')
    
    p = (ggplot(corr_df, aes(x='metric1', y='metric2', fill='correlation'))
         + geom_tile(color='white', size=0.5)
         + geom_text(aes(label='label'), size=9, color='black')
         + scale_fill_gradientn(colors=['#3b82f6', '#ffffff', '#ef4444'], 
                               limits=(-1, 1), name='Pearson r')
         + labs(x='Metric', y='Metric', title='Metric correlation matrix')
         + theme_minimal()
         + theme(
             axis_text_x=element_text(rotation=45, hjust=1),
             plot_title=element_text(size=14, weight='bold'),
             axis_title=element_text(size=12),
             aspect_ratio=1
         )
    )
    _save(p, os.path.join(out_dir, "metric_correlation.png"), width=8, height=7)


def plot_diversity_vs_fidelity(df: pd.DataFrame, out_dir: str):
    """Diversity vs Fidelity tradeoff."""
    plot_df = df.reset_index()
    plot_df = plot_df[['episode', 'diversity_icd', 'dtw_nn']]
    
    p = (ggplot(plot_df, aes(x='diversity_icd', y='dtw_nn'))
         + geom_point(size=3, alpha=0.6, color='steelblue', fill='steelblue', stroke=1)
         + geom_text(aes(label='episode'), size=8, nudge_y=0.01, alpha=0.7)
         + labs(x='Diversity (ICD)', y='DTW-NN (↓ better)', 
                title='Diversity vs Fidelity Tradeoff')
         + theme_minimal()
         + theme(
             plot_title=element_text(size=14, weight='bold'),
             axis_title=element_text(size=12),
             panel_grid=element_line(color='gray', alpha=0.3)
         )
    )
    _save(p, os.path.join(out_dir, "diversity_vs_fidelity.png"), width=7, height=6)


# ============================================================
# Comparison plots (multiple datasets/methods)
# ============================================================

def load_all_metrics(results_dir: str) -> pd.DataFrame:
    """
    Load all evaluation files and create a combined DataFrame.
    Returns DataFrame with columns: dataset, method, top_k, episode, and all metrics.
    """
    files = find_all_evaluation_files(results_dir)
    if not files:
        raise ValueError(f"No evaluation files found in {results_dir}")
    
    all_data = []
    
    for filepath in files:
        dataset, method, top_k = parse_filename(filepath)
        raw = load_metrics(filepath)
        df = flatten_metrics(raw, exclude_averaged=True)
        df = df.reset_index()
        df['dataset'] = dataset
        df['method'] = method
        df['top_k'] = top_k if top_k is not None else -1  # Use -1 for files without top_k
        all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    return combined


def load_all_averaged_metrics(results_dir: str) -> pd.DataFrame:
    """
    Load averaged metrics from all evaluation files and create a combined DataFrame.
    Returns DataFrame with columns: dataset, method, top_k, and all metrics.
    """
    files = find_all_evaluation_files(results_dir)
    if not files:
        raise ValueError(f"No evaluation files found in {results_dir}")
    
    all_data = []
    
    for filepath in files:
        dataset, method, top_k = parse_filename(filepath)
        raw = load_metrics(filepath)
        averaged = load_averaged_metrics(raw)
        
        if not averaged:
            continue
        
        # Convert averaged metrics to DataFrame row
        row = {
            'dataset': dataset,
            'method': method,
            'top_k': top_k if top_k is not None else -1,
            'wasserstein': averaged.get('wasserstein'),
            'dtw_nn': averaged.get('dtw_nn'),
            'spectral_wasserstein': averaged.get('spectral_wasserstein'),
            'temporal_correlation': averaged.get('temporal_correlation'),
            'density': averaged.get('distributional_coverage.density'),
            'coverage': averaged.get('distributional_coverage.coverage'),
            'diversity_icd': averaged.get('diversity_icd'),
        }
        all_data.append(row)
    
    if not all_data:
        raise ValueError("No averaged metrics found in any evaluation files")
    
    combined = pd.DataFrame(all_data)
    return combined


def plot_ranking_comparison(combined_df: pd.DataFrame, out_dir: str, top_k_filter: Optional[int] = None):
    """
    Rank methods by each metric per dataset (lower is better for distance metrics, higher for others).
    
    Args:
        combined_df: DataFrame with metrics
        out_dir: Output directory
        top_k_filter: If specified, only use data from this top_k value. If None, aggregate across all top_k.
    """
    # Filter by top_k if specified
    df = combined_df.copy()
    if top_k_filter is not None:
        df = df[df['top_k'] == top_k_filter]
        if len(df) == 0:
            print(f"Warning: No data found for top_k={top_k_filter}. Skipping ranking comparison.")
            return
    
    metric_cols = [c for c in df.columns if c not in ['episode', 'dataset', 'method', 'top_k']]
    
    # Metrics where lower is better
    lower_better = ['wasserstein', 'dtw_nn', 'spectral_wasserstein']
    
    # Compute mean per dataset-method (aggregating across top_k if not filtered)
    means = df.groupby(['dataset', 'method'])[metric_cols].mean().reset_index()
    
    # Prepare data for ranking per dataset
    all_rankings = []
    
    for metric in metric_cols:
        is_lower_better = any(m in metric.lower() for m in lower_better)
        
        # Rank within each dataset using 'average' method to handle ties better
        means_temp = means.copy()
        means_temp['rank'] = means_temp.groupby('dataset')[metric].rank(
            ascending=is_lower_better, method='average'
        )
        
        # Break any ties within each dataset using actual metric values
        for dataset in means_temp['dataset'].unique():
            dataset_data = means_temp[means_temp['dataset'] == dataset]
            metric_values = dataset_data[metric]
            ranks = dataset_data['rank']
            
            # Check for ties
            if ranks.duplicated().any():
                metric_range = metric_values.max() - metric_values.min()
                if metric_range > 0:
                    metric_normalized = (metric_values - metric_values.min()) / metric_range * 1e-8
                    if is_lower_better:
                        epsilon = -metric_normalized
                    else:
                        epsilon = -(1e-8 - metric_normalized)
                    means_temp.loc[means_temp['dataset'] == dataset, 'rank'] = ranks + epsilon
                else:
                    # All methods have same value, use alphabetical order
                    tie_breaker = pd.Series(range(len(dataset_data)), dtype=float) * 1e-10
                    means_temp.loc[means_temp['dataset'] == dataset, 'rank'] = ranks + tie_breaker
        
        # Prepare data for this metric
        for dataset in means_temp['dataset'].unique():
            dataset_data = means_temp[means_temp['dataset'] == dataset][['method', 'rank']].copy()
            dataset_data['metric'] = metric
            dataset_data['dataset'] = dataset
            all_rankings.append(dataset_data)
    
    # Combine all rankings
    rank_long = pd.concat(all_rankings, ignore_index=True)
    rank_long['label'] = rank_long['rank'].apply(lambda x: f'{x:.1f}')
    
    # Get unique methods for consistent ordering
    methods = sorted(rank_long['method'].unique())
    
    # Create separate plot for each dataset
    datasets = sorted(rank_long['dataset'].unique())
    
    for dataset in datasets:
        dataset_data = rank_long[rank_long['dataset'] == dataset].copy()
        dataset_name = dataset
        
        p = (ggplot(dataset_data, aes(x='metric', y='method', fill='rank'))
             + geom_tile(color='white', size=0)
             + geom_text(aes(label='label'), size=9, color='black')
             + scale_fill_gradientn(colors=['#22c55e', '#eab308', '#ef4444'], 
                                   name='Rank\n(lower is better)')
             + scale_y_discrete(limits=methods)
             + labs(x='', y='', 
                    title=f'Method Rankings: {dataset_name.upper()}')
             + theme_minimal()
             + theme(
                 axis_text_x=element_text(rotation=45, hjust=1, size=11),
                 axis_text_y=element_text(size=11),
                 plot_title=element_text(size=14, weight='bold'),
                 strip_text=element_text(size=12, weight='bold'),
                 strip_background=element_rect(fill='lightgray', color='black', size=1)
             )
        )
        
        # Calculate dimensions based on number of methods
        n_methods = len(methods)
        n_metrics = len(metric_cols)
        
        height = max(6, n_methods * 0.5 + 2)
        width = max(10, n_metrics * 1.5)
        
        # Save with dataset name in filename
        safe_dataset_name = dataset_name.replace(' ', '_').lower()
        heatmap_dir = os.path.join(out_dir, "heatmaps")
        os.makedirs(heatmap_dir, exist_ok=True)
        filename = f"method_rankings_{safe_dataset_name}.png"
        _save(p, os.path.join(heatmap_dir, filename), width=width, height=height)


def plot_metric_multibar_per_dataset(combined_df: pd.DataFrame, out_dir: str, top_k_filter: Optional[int] = None):
    """
    For each metric, create ONE multi-bar graph containing all datasets side-by-side.
    Each bar represents a method, with touching bars and a shared legend.
    No x-axis labels - only rely on legend.
    Includes standard deviation error bars.
    
    Args:
        combined_df: DataFrame with metrics
        out_dir: Output directory
        top_k_filter: If specified, only use data from this top_k value. If None, aggregate across all top_k.
    """
    # Filter by top_k if specified
    df = combined_df.copy()
    if top_k_filter is not None:
        df = df[df['top_k'] == top_k_filter]
        if len(df) == 0:
            print(f"Warning: No data found for top_k={top_k_filter}. Skipping multibar plots.")
            return
    
    metric_cols = [c for c in df.columns if c not in ['episode', 'dataset', 'method', 'top_k']]
    datasets = sorted(df['dataset'].unique())
    methods = sorted(df['method'].unique())
    
    # Compute mean per dataset-method combination (aggregating across top_k if not filtered)
    means = df.groupby(['dataset', 'method'])[metric_cols].mean().reset_index()
    
    # Create one plot per metric with all datasets side-by-side
    for metric in metric_cols:
        # Filter datasets that have data for this metric
        valid_datasets = []
        for dataset in datasets:
            dataset_data = means[means['dataset'] == dataset]
            if len(dataset_data) > 0 and metric in dataset_data.columns:
                valid_datasets.append(dataset)
        
        if len(valid_datasets) == 0:
            continue
        
        # Filter data for this metric
        plot_data = means[means['dataset'].isin(valid_datasets)].copy()
        plot_data = plot_data[['dataset', 'method', metric]].copy()
        plot_data['dataset_upper'] = plot_data['dataset'].str.upper()
        
        # Get unique methods and create a muted color palette
        unique_methods = sorted(plot_data['method'].unique())
        # Use a muted, professional color palette (grays and muted blues/greens)
        # Colors from a muted palette that's easy on the eyes
        muted_colors = [
            '#4A90E2',  # Muted blue
            '#7ED321',  # Muted green
            '#F5A623',  # Muted orange
            '#BD10E0',  # Muted purple
            '#50E3C2',  # Muted teal
            '#B8E986',  # Muted light green
            '#9013FE',  # Muted deep purple
            '#D0021B',  # Muted red
            '#8B572A',  # Muted brown
            '#417505',  # Muted dark green
        ]
        # Extend palette if needed by cycling
        if len(unique_methods) > len(muted_colors):
            import itertools
            muted_colors = list(itertools.islice(itertools.cycle(muted_colors), len(unique_methods)))
        else:
            muted_colors = muted_colors[:len(unique_methods)]
        
        # Create color mapping
        color_map = dict(zip(unique_methods, muted_colors))
        plot_data['color'] = plot_data['method'].map(color_map)
        
        # Calculate max value and set y-axis limits (0 to max + small nudge relative to data range)
        max_value = plot_data[metric].max()
        min_value = plot_data[metric].min()
        data_range = max_value - min_value
        y_max = max_value + data_range * 0.05  # Add 5% of data range as nudge on top
        
        p = (ggplot(plot_data, aes(x='method', y=metric, fill='method'))
             + geom_bar(stat='identity', width=1.0)
             + facet_wrap('~dataset_upper', nrow=1, scales='free_x')
             + scale_fill_manual(values=color_map, name='Method')
             + scale_x_discrete(expand=(0, 0))
             + scale_y_continuous(limits=(0, y_max), expand=(0, 0))
             + labs(x='', y=metric.replace('_', ' ').title())
             + theme_minimal()
             + theme(
                 axis_text_x=element_blank(),
                 axis_ticks_x=element_blank(),
                 axis_line_x=element_blank(),
                 strip_text=element_text(size=16, weight='bold'),
                 strip_background=element_rect(fill='lightgray', color='black', size=1),
                 axis_title_y=element_text(size=16, weight='bold'),
                 panel_grid_minor=element_blank(),
                 panel_grid_major_y=element_line(color='gray', alpha=0.3),
                 panel_grid_major_x=element_blank(),
                 legend_position='right',
                 legend_text=element_text(size=14),
                 legend_title=element_text(size=18, weight='bold'),  # Make "Method" larger
                 panel_spacing_x=0.03
             )
        )
        
        # Save the figure
        safe_metric_name = metric.replace('.', '_')
        multibar_dir = os.path.join(out_dir, "multi_bar_graphs")
        os.makedirs(multibar_dir, exist_ok=True)
        filename = f"multibar_{safe_metric_name}.png"
        _save(p, os.path.join(multibar_dir, filename), 
              width=4 * len(valid_datasets), height=6)


# ============================================================
# Top-K analysis plots
# ============================================================

def plot_metrics_vs_topk_faceted(combined_df: pd.DataFrame, out_dir: str):
    """
    Plot metrics vs top-K faceted by dataset.
    Only plots "prototype" method.
    Uses free scales for each benchmark (scales='free_y').
    """
    # Filter out rows without top_k information and filter to prototype only
    df_with_topk = combined_df[(combined_df['top_k'] > 0) & (combined_df['method'] == 'prototype')].copy()
    
    if len(df_with_topk) == 0:
        print("Warning: No data with top_k information for prototype method found. Skipping faceted top-K plots.")
        return
    
    metric_cols = [c for c in df_with_topk.columns if c not in ['episode', 'dataset', 'method', 'top_k']]
    
    # Compute mean per dataset-top_k combination (only prototype method)
    means = df_with_topk.groupby(['dataset', 'top_k'])[metric_cols].mean().reset_index()
    means['dataset_upper'] = means['dataset'].str.upper()
    
    # Create one plot per metric
    for metric in metric_cols:
        # Filter out None values
        plot_data = means[means[metric].notna()].copy()
        
        if len(plot_data) == 0:
            continue
        
        p = (ggplot(plot_data, aes(x='top_k', y=metric))
             + geom_line(size=1.2, alpha=0.8, color='steelblue')
             + geom_point(size=2.5, alpha=0.9, color='steelblue')
             + facet_wrap('~dataset_upper', scales='free', nrow=1)
             + labs(
                 x='Top-K',
                 y=metric.replace('_', ' ').title(),
                 title=f'{metric.replace("_", " ").title()} vs Top-K Across Datasets (Prototype)'
             )
             + theme_minimal()
             + theme(
                 plot_title=element_text(size=14, weight='bold'),
                 axis_title=element_text(size=12),
                 panel_grid=element_line(color='gray', alpha=0.3),
                 strip_text=element_text(size=12, weight='bold'),
                 strip_background=element_rect(fill='lightgray', color='black', size=1)
             )
        )
        
        safe_metric_name = metric.replace('.', '_')
        topk_dir = os.path.join(out_dir, "top_k_prototype")
        os.makedirs(topk_dir, exist_ok=True)
        filename = f"topk_{safe_metric_name}_all_datasets.png"
        _save(p, os.path.join(topk_dir, filename), width=6 * len(plot_data['dataset'].unique()), height=6)


# ============================================================
# Entry point
# ============================================================

def visualize(metrics_path: str, output_dir: Optional[str] = None, 
              comparison_dir: Optional[str] = None):
    """
    Visualize metrics from a single file or all files in a directory.
    
    Args:
        metrics_path: Path to single metrics JSON file or directory containing multiple files
        output_dir: Output directory (default: metric_graphs in same dir as metrics_path)
        comparison_dir: Directory to search for comparison files (default: same as metrics_path parent)
    """
    # Determine output directory
    if output_dir is None:
        if os.path.isfile(metrics_path):
            base_dir = os.path.dirname(metrics_path)
        else:
            base_dir = metrics_path
        output_dir = os.path.join(base_dir, "metric_graphs")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Single file visualization (only if a specific file is provided)
    if os.path.isfile(metrics_path):
        raw = load_metrics(metrics_path)
        df = flatten_metrics(raw, exclude_averaged=True)
        averaged = load_averaged_metrics(raw)
        
        # Save averaged metrics if available
        if averaged:
            avg_df = pd.DataFrame([averaged])
            avg_df.to_csv(os.path.join(output_dir, "averaged_metrics.csv"), index=False)
        
        # Save table
        df.to_csv(os.path.join(output_dir, "metrics_table.csv"))
        
        # Single dataset/method plots
        plot_metric_means(df, output_dir)
        plot_dtw_distribution(df, output_dir)
        plot_density_vs_coverage(df, output_dir)
        plot_episode_metric_heatmap(df, output_dir)
        plot_metric_correlation(df, output_dir)
        plot_diversity_vs_fidelity(df, output_dir)
        
        print(f"Saved single-file visualizations to {output_dir}")
    
    # Comparison visualizations (always run if directory is provided or for comparisons)
    if comparison_dir is None:
        if os.path.isfile(metrics_path):
            comparison_dir = os.path.dirname(metrics_path)
        else:
            comparison_dir = metrics_path
    
    try:
        combined_df = load_all_metrics(comparison_dir)
        
        # Method rankings (aggregate across all top_k values)
        plot_ranking_comparison(combined_df, output_dir)
        
        # Multi-bar graphs for each metric per dataset (aggregate across all top_k values)
        plot_metric_multibar_per_dataset(combined_df, output_dir)
        
        # Top-K analysis plots (only if top_k data exists)
        # Only create "all_datasets" plots (per-dataset plots removed)
        if 'top_k' in combined_df.columns and (combined_df['top_k'] > 0).any():
            plot_metrics_vs_topk_faceted(combined_df, output_dir)
            # Heatmaps removed per user request
        
        print(f"Saved comparison visualizations to {output_dir}")
        
    except (ValueError, FileNotFoundError) as e:
        print(f"Comparison visualizations skipped: {e}")
    
    # List all generated files
    print(f"\nGenerated files in {output_dir}:")
    for f in sorted(os.listdir(output_dir)):
        print(f"  {f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Visualize retrieval evaluation metrics")
    parser.add_argument("--metrics_path", type=str, default=None,
                       help="Path to metrics JSON file or directory containing evaluation files (default: data/evaluation_results)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: metric_graphs in same dir as metrics_path)")
    parser.add_argument("--comparison_dir", type=str, default=None,
                       help="Directory to search for comparison files (default: same as metrics_path parent)")

    args = parser.parse_args()
    
    # Default to data/evaluation_results if no path provided
    if args.metrics_path is None:
        # Get the project root (assuming script is in benchmarking/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        args.metrics_path = os.path.join(project_root, "data")
    
    # If metrics_path is a file, use its directory for comparison
    if os.path.isfile(args.metrics_path):
        comparison_dir = os.path.dirname(args.metrics_path)
    else:
        comparison_dir = args.metrics_path
    
    if args.comparison_dir is None:
        args.comparison_dir = comparison_dir
    
    print(f"Processing evaluation files from: {args.comparison_dir}")
    visualize(args.metrics_path, args.output_dir, args.comparison_dir)

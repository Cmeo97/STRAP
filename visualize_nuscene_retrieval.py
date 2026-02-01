from time import time
import momentfm
import numpy as np
import yaml
import argparse
import os
import h5py
import glob
import pandas as pd
from plotnine import *

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def resize_to_mean_length(task_data_list, length=None):
    """
    Resizes a list of 2D numpy arrays to their mean length via interpolation.
    
    Args:
        task_data_list (list of np.ndarray): List of 2D arrays (time_steps, features).
    
    Returns:
        np.ndarray: 2D array of shape (num_arrays, length).
    """
    if not task_data_list:
        return np.array([]).reshape(0, 0)
    
    resized = []
    for arr in task_data_list:
        old_len = arr.shape[0]
        new_arr = np.zeros((length))
        new_arr = np.interp(np.linspace(0, old_len - 1, length),
                            np.arange(old_len),
                            arr)  # Assuming single feature for yaw_rate
        resized.append(new_arr)
    return np.array(resized)

def plot_retrieval_seaborn(target_yaw, retrieved_data_dict, time_axis=None, fontsize=14):
    """
    Plots retrieved trajectories against the target to show alignment quality.
    
    Args:
        target_yaw (np.array): Shape (T,)
        retrieved_data_dict (dict): { 'MethodName': np.array(30, T) }
        time_axis (np.array): Optional time steps.
    """
    methods = list(retrieved_data_dict.keys())
    n_methods = len(methods)
    
    if time_axis is None:
        time_axis = np.arange(len(target_yaw))

    # Setup the figure
    fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4), sharey=True)
    if n_methods == 1: axes = [axes] # Handle single method case

    # Define colors
    target_color = 'black'
    sample_color_base = {
        'ROSER': 'green',
        'Stumpy': 'blue',
        'Dtaidistance': 'red', 
    }

    for ax, method in zip(axes, methods):
        # 1. Get data for this method
        samples = retrieved_data_dict[method] # Shape (30, T)
        
        # 2. Plot individual retrieved samples (Thin, transparent)
        color = sample_color_base.get(method, 'gray')
        for i in range(len(samples)):
            ax.plot(time_axis, samples[i], color=color, alpha=0.15, linewidth=1)
            
        # 3. Plot the Mean of retrieved samples (Bold, same color)
        mean_yaw = np.mean(samples, axis=0)
        ax.plot(time_axis, mean_yaw, color=color, linestyle='--', linewidth=2, label=f'{method} Mean')

        # 4. Plot the Target (Bold, Black)
        ax.plot(time_axis, target_yaw, color=target_color, linewidth=2.5, linestyle='-', label='Reference')

        # Styling
        ax.set_title(f"{method}", fontsize=fontsize, fontweight='bold')
        ax.set_xlabel("Time Step", fontsize=fontsize)
        #if ax == axes[0]:
        ax.set_ylabel("Yaw Rate (rad/s)", fontsize=fontsize)
        
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', fontsize='medium')

    plt.tight_layout()
    plt.savefig("retrieval_quality_horizontal.pdf", dpi=300)
    print("Plot saved to retrieval_quality_horizontal.pdf")

def plot_retrieval_comparison(target_yaw, retrieved_data_dict, time_axis=None):
    if time_axis is None:
        time_axis = np.arange(len(target_yaw))

    all_data = []
    methods = list(retrieved_data_dict.keys())
    
    for method in methods:
        samples = retrieved_data_dict[method] 
        n_samples, n_timesteps = samples.shape
        
        # Add individual sample lines
        for i in range(n_samples):
            all_data.append(pd.DataFrame({
                'Time': time_axis,
                'YawRate': samples[i],
                'Method': method,
                'Type': 'Sample',
                'Group': f"{method}_sample_{i}"
            }))
            
        # Add Mean line
        all_data.append(pd.DataFrame({
            'Time': time_axis,
            'YawRate': np.mean(samples, axis=0),
            'Method': method,
            'Type': 'Mean',
            'Group': f"{method}_mean"
        }))

        # Add Target line
        all_data.append(pd.DataFrame({
            'Time': time_axis,
            'YawRate': target_yaw,
            'Method': method,
            'Type': 'Target',
            'Group': f"{method}_target"
        }))

    df = pd.concat(all_data, ignore_index=True)
    df['Method'] = pd.Categorical(df['Method'], categories=methods)

    # Calculate y-axis limits with some padding
    y_min = -0.6
    y_max = 0.6
    y_range = y_max - y_min
    y_padding = y_range * 0.1  # 10% padding
    
    # Create color mapping for each method
    method_colors = {
        'ROSER': '#2ecc71',
        'Stumpy': '#3498db',
        'Dtaidistance': '#e74c3c'
    }
    
    # Create a legend label column with unique keys per method
    df['LegendLabel'] = df.apply(
        lambda row: 'Original Data' if row['Type'] == 'Sample' 
                   else 'Mean' if row['Type'] == 'Mean'
                   else 'Reference',
        axis=1
    )
    
    # Create combined color key: Method_Type for samples/mean, just type for target
    df['ColorKey'] = df.apply(
        lambda row: f"{row['Method']}_Sample" if row['Type'] == 'Sample'
                   else f"{row['Method']}_Mean" if row['Type'] == 'Mean'
                   else 'Reference',
        axis=1
    )
    
    # Build color mapping
    color_map = {}
    for method in methods:
        color_map[f"{method}_Sample"] = method_colors[method]
        color_map[f"{method}_Mean"] = method_colors[method]
    color_map['Reference'] = 'black'
    
    p = (
        ggplot(df, aes(x='Time', y='YawRate'))
        + geom_line(
            data=df[df['Type'] == 'Sample'],
            mapping=aes(group='Group', color='ColorKey', linetype='LegendLabel'),
            alpha=0.20, size=0.4
        )
        + geom_line(
            data=df[df['Type'] == 'Mean'],
            mapping=aes(group='Group', color='ColorKey', linetype='LegendLabel'),
            size=1.0
        )
        + geom_line(
            data=df[df['Type'] == 'Target'],
            mapping=aes(group='Group', color='ColorKey', linetype='LegendLabel'),
            size=1.0
        )
        + facet_wrap('~Method', nrow=1)
        + theme_minimal()
        + scale_color_manual(values=color_map)
        + scale_linetype_manual(
            values={
                'Original Data': 'solid',
                'Mean': 'dashed',
                'Reference': 'solid'
            },
            breaks=['Original Data', 'Mean', 'Reference'],
            name=''
        )
        + scale_y_continuous(limits=(y_min - y_padding, y_max + y_padding))
        + labs(
            x="Time Step",
            y="Yaw Rate (rad/s)"
        )
        + theme(
            figure_size=(10, 3.5), 
            legend_position='inside',
            legend_position_inside=(0.15, 0.85),
            legend_background=element_rect(fill='white', alpha=0.8),
            legend_box_spacing=0.1,
            strip_background=element_rect(fill='#f0f0f0'),
            strip_text=element_text(size=10, weight='bold'),
            panel_spacing=0.01,
            plot_title=element_text(hjust=0.5),
            plot_subtitle=element_text(hjust=0.5)
        )
    )
    
    p.save("retrieval_quality_horizontal_fixed.pdf", dpi=600)
    print("Plot saved to retrieval_quality_horizontal_fixed.pdf")

# --- Mock Data Generation for Demo ---
# (Replace this with your actual data loading logic)
if __name__ == "__main__":
    target_file = 'data/target_data/nuscene_target_dataset.hdf5'
    ours_file = 'data/retrieval_results/nuscene/nuscene_retrieval_results_prototype.hdf5'
    stumpy_file = 'data/retrieval_results/nuscene/nuscene_retrieval_results_stumpy.hdf5'
    dtaidistance_file = 'data/retrieval_results/nuscene/nuscene_retrieval_results_dtaidistance.hdf5'

    with h5py.File(target_file, 'r') as f:
        target_yaw = f['right turn']
        target_yaw = target_yaw['obs/yaw_rate'][:][3][:,0]

    with h5py.File(ours_file, 'r') as f:
        data = f['results/right turn']
        ours = []
        for i in range(30):
            match_id = f'match_{i}'
            ours.append(data[match_id]['obs/yaw_rate'][:][:,0])
        ours = resize_to_mean_length(ours, length=len(target_yaw))  # Shape (30, T)
    
    with h5py.File(stumpy_file, 'r') as f:
        data = f['results/right turn']
        stumpy = []
        for i in range(30):
            match_id = f'match_{i}'
            stumpy.append(data[match_id]['obs/yaw_rate'][:][:,0])
        stumpy = resize_to_mean_length(stumpy, length=len(target_yaw))  # Shape (30, T)
    
    with h5py.File(dtaidistance_file, 'r') as f:
        data = f['results/right turn']
        dtaidistance = []
        for i in range(30):
            match_id = f'match_{i}'
            dtaidistance.append(data[match_id]['obs/yaw_rate'][:][:,0])
        dtaidistance = resize_to_mean_length(dtaidistance, length=len(target_yaw))  # Shape (30, T)
    
    data = {
        'ROSER': ours,
        'Stumpy': stumpy,
        'Dtaidistance': dtaidistance
    }
    
    plot_retrieval_seaborn(target_yaw, data)
    #plot_retrieval_comparison(target_yaw, data)
import os
import h5py
import numpy as np
from string import ascii_uppercase
from scipy.interpolate import interp1d
from strap.configs.libero_file_functions import get_libero_lang_instruction

def resize_scipy(data, new_size):
    # Create an index for current data (0 to 399)
    x_old = np.linspace(0, 1, data.shape[0])
    # Create an index for the new target size (0 to 299)
    x_new = np.linspace(0, 1, new_size)
    
    # Kind='linear' or 'cubic' for smoother interpolation
    f = interp1d(x_old, data, axis=0, kind='linear')
    return f(x_new)

def get_demo_data(hdf5_dataset, demo_key):
    demo_data = hdf5_dataset[demo_key]
    # Libero dataset structure
    if 'obs/ee_pos' in demo_data and 'obs/gripper_states' in demo_data and 'obs/joint_states' in demo_data:
        ee_pose = demo_data['obs/ee_pos'][:]
        gripper_states = demo_data['obs/gripper_states'][:]
        joint_states = demo_data['obs/joint_states'][:]
        series = np.concatenate((ee_pose, gripper_states, joint_states), axis=-1)
        return series
    # Nuscene dataset structure
    if 'obs/velocity' in demo_data and 'obs/acceleration' in demo_data and 'obs/yaw_rate' in demo_data:
        velocity = demo_data['obs/velocity'][:]
        acceleration = demo_data['obs/acceleration'][:]
        yaw_rate = demo_data['obs/yaw_rate'][:]
        series =  np.concatenate((velocity, acceleration, yaw_rate), axis=-1)
        return series

def process_retrieval_results(episode_results, length = None, top_k=None, max_distance=None):
    episode_results.sort(key=lambda x: x['cost'])
    num_retrieved = len(episode_results)
    if not top_k:
        top_k = num_retrieved//2
    episode_results = episode_results[:top_k]

    if max_distance is not None:
        episode_results = [res for res in episode_results if res['cost'] <= max_distance]

    output = {}
    for i, result in enumerate(episode_results):
        output[f"match_{i}"] = {}
        with h5py.File(result['offline_file'], 'r') as f:
            demo_data = f['data'][result['demo_key']]

            # Libero dataset structure
            if 'obs/ee_pos' in demo_data:
                actions = demo_data['actions'][:]
                ee_pose = demo_data['obs/ee_pos'][:]
                gripper_states = demo_data['obs/gripper_states'][:]
                joint_states = demo_data['obs/joint_states'][:]
                robot_states = demo_data['robot_states'][:]
                output[f"match_{i}"]['actions'] = actions[result['start_idx']:result['end_idx']]
                output[f"match_{i}"]['robot_states'] = robot_states[result['start_idx']:result['end_idx']]
                output[f"match_{i}"]['obs/ee_pos'] = ee_pose[result['start_idx']:result['end_idx']]
                output[f"match_{i}"]['obs/gripper_states'] = gripper_states[result['start_idx']:result['end_idx']]
                output[f"match_{i}"]['obs/joint_states'] = joint_states[result['start_idx']:result['end_idx']]
                output[f"match_{i}"]['file_path'] = result['offline_file']
                output[f"match_{i}"]['demo_key'] = result['demo_key']
                output[f"match_{i}"]['lang_instruction'] = get_libero_lang_instruction(f, result['demo_key'])
            # Nuscene dataset structure
            elif 'obs/velocity' in demo_data:
                velocity = demo_data['obs/velocity'][:]
                acceleration = demo_data['obs/acceleration'][:]
                yaw_rate = demo_data['obs/yaw_rate'][:]
                output[f"match_{i}"]['obs/velocity'] = velocity[result['start_idx']:result['end_idx']]
                output[f"match_{i}"]['obs/acceleration'] = acceleration[result['start_idx']:result['end_idx']]
                output[f"match_{i}"]['obs/yaw_rate'] = yaw_rate[result['start_idx']:result['end_idx']]
                output[f"match_{i}"]['file_path'] = result['offline_file']
                output[f"match_{i}"]['demo_key'] = result['demo_key']
                output[f"match_{i}"]['lang_instruction'] = f.attrs.get('lang_instruction', 'No instruction available')
    return output


def transform_series_to_text(series, use_sax=False, sax_num_bins=26):
    """
    Transpose a (seq_length, features) time series to (features, seq_length),
    and return a string representation where each feature is a row and timesteps are columns.
    If use_sax is True, represent each feature's time series using SAX (Symbolic Aggregate approXimation).
    """
    feature_names = [
        "ee_pose_x", "ee_pose_y", "ee_pose_z",
        "gripper_state_0", "gripper_state_1",
        "joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"
    ]
    series_t = series.T  # shape: (features, seq_len)
    seq_len = series_t.shape[1]

    header = "step\t" + "\t".join(str(i) for i in range(seq_len))
    lines = [header]

    if use_sax:
        sax_symbols = ascii_uppercase[:sax_num_bins]

        for i in range(series_t.shape[0]):
            x = series_t[i]
            x_mean = np.mean(x)
            x_std = np.std(x)
            if x_std == 0:
                norm_x = np.zeros_like(x)
            else:
                norm_x = (x - x_mean) / x_std
            # Define bin edges at quantiles, so bins have equal support in normal distribution
            bin_edges = np.quantile(norm_x, np.linspace(0, 1, sax_num_bins + 1))
            # To avoid duplicate bin edges from constant value series
            bin_edges = np.unique(bin_edges)
            # Assign each value to a bin based on bin_edges
            digitized = np.digitize(norm_x, bin_edges[1:-1], right=False)  # Returns ints in [0, bins-1]
            # Map bins to symbols
            symbols = [sax_symbols[idx] if idx < len(sax_symbols) else sax_symbols[-1] for idx in digitized]
            values = "\t".join(symbols)
            # Add "[SAX]" prefix to indicate to the LLM this is a SAX representation
            lines.append(f"{feature_names[i]}\t[SAX Representation]\t{values}")
    else:
        for i in range(series_t.shape[0]):
            values = "\t".join(f"{series_t[i, t]:.4f}" for t in range(seq_len))
            lines.append(f"{feature_names[i]}\t{values}")

    return "\n".join(lines)
import os
import glob
import traceback
import h5py
import numpy as np
from typing import Dict, List, Optional
from string import ascii_uppercase
from scipy.interpolate import interp1d
from strap.configs.libero_file_functions import get_libero_lang_instruction


def get_feature_keys_from_group(group: h5py.Group) -> Optional[List[str]]:
    """
    Determine feature keys based on the observation group structure.
    
    Args:
        group: HDF5 group containing observation data
        
    Returns:
        List of feature key names, or None if structure is unknown
    """
    if "ee_pos" in group:
        return ["ee_pos", "gripper_states", "joint_states"]
    elif "velocity" in group:
        return ["velocity", "acceleration", "yaw_rate"]
    elif "cartesian_positions" in group:
        return ["cartesian_positions", "gripper_states", "joint_states"]
    else:
        return None


def concat_obs_group(obs_group: h5py.Group, feature_keys=None) -> np.ndarray:
    """
    Concatenate obs features along feature dimension.
    
    Returns:
        (T, F_total)
    """
    features = [obs_group[k][()] for k in feature_keys]
    return np.concatenate(features, axis=-1)

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
        series = concat_obs_group(demo_data['obs'], feature_keys=['ee_pos', 'gripper_states', 'joint_states'])
        return series
    # Nuscene dataset structure
    if 'obs/velocity' in demo_data and 'obs/acceleration' in demo_data and 'obs/yaw_rate' in demo_data:
        series = concat_obs_group(demo_data['obs'], feature_keys=['acceleration', 'velocity', 'yaw_rate'])
        return series
    # Droid dataset structure
    if 'obs/cartesian_positions' in demo_data and 'obs/gripper_states' in demo_data and 'obs/joint_states' in demo_data:
        series = concat_obs_group(demo_data['obs'], feature_keys=['cartesian_positions', 'gripper_states', 'joint_states'])
        return series
    raise ValueError("Unknown dataset structure or missing expected keys.")

def process_retrieval_results(episode_results, top_k=None, max_distance=None):
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
                output[f"match_{i}"]['lang_instruction'] = f.attrs.get('language_instruction', 'No instruction available')
            # Droid dataset structure
            elif 'obs/cartesian_positions' in demo_data:
                cartesian_positions = demo_data['obs/cartesian_positions'][:]
                gripper_states = demo_data['obs/gripper_states'][:]
                joint_states = demo_data['obs/joint_states'][:]
                output[f"match_{i}"]['obs/cartesian_positions'] = cartesian_positions[result['start_idx']:result['end_idx']]
                output[f"match_{i}"]['obs/gripper_states'] = gripper_states[result['start_idx']:result['end_idx']]
                output[f"match_{i}"]['obs/joint_states'] = joint_states[result['start_idx']:result['end_idx']]
                output[f"match_{i}"]['file_path'] = result['offline_file']
                output[f"match_{i}"]['demo_key'] = result['demo_key']
                output[f"match_{i}"]['lang_instruction'] = f.attrs.get('language_instruction', 'No instruction available')
            else:
                raise ValueError("Unknown dataset structure or missing expected keys.")
    return output

# Feature names for each supported dataset
FEATURE_SETS = {
    "libero": [
        "ee_pose_x", "ee_pose_y", "ee_pose_z",
        "gripper_state_0", "gripper_state_1",
        "joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"
    ],
    "nuscene": [
        "acceleration", "velocity", "yaw_rate"
    ],
    "droid": [
        "cartesian_position_0", "cartesian_position_1", "cartesian_position_2",
        "cartesian_position_3", "cartesian_position_4", "cartesian_position_5",
        "gripper_state",
        "joint_state_0", "joint_state_1", "joint_state_2", "joint_state_3",
        "joint_state_4", "joint_state_5", "joint_state_6"
    ]
}


def get_feature_names(dataset: str) -> List[str]:
    """
    Get feature names for a given dataset.
    
    Args:
        dataset: Dataset name ("libero", "nuscene", or "droid")
        
    Returns:
        List of feature names
    """
    if dataset not in FEATURE_SETS:
        raise ValueError(f"Dataset {dataset} not recognized. Choose from: {list(FEATURE_SETS.keys())}")
    return FEATURE_SETS[dataset]


def transform_series_to_text(series, sax_num_bins=26, dataset="libero"):
    """
    Transform a (seq_length, features) time series to a compact string representation.

    Args:
        series: np.ndarray of shape (seq_len, features)
        sax_num_bins: number of bins to quantize into SAX symbols (default=26)
        dataset: str, one of ["libero", "nuscene", "droid"]

    Returns:
        str: Compact text representation of the time-series, one line per feature.
    """
    feature_sets = FEATURE_SETS

    if len(feature_sets[dataset]) != series.shape[1]:
        raise ValueError(f"Number of features in dataset {dataset} does not match the number of features in the series")

    feature_names = feature_sets[dataset]

    series_t = series.T  # shape: (features, seq_len)
    lines = []
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
        bin_edges = np.unique(bin_edges)
        digitized = np.digitize(norm_x, bin_edges[1:-1], right=False)
        symbols = [sax_symbols[idx] if idx < len(sax_symbols) else sax_symbols[-1] for idx in digitized]
        values = "".join(symbols)
        fname = feature_names[i]
        lines.append(f"{fname} [SAX Representation]: {values}")
    return "\n".join(lines)


def generate_sliding_windows(
    data: np.ndarray, 
    window_size: int, 
    step_size: int = 1
) -> List[np.ndarray]:
    """
    Generate sliding windows from a multi-dimensional array.
    
    Args:
        data: Array of shape (N, T, F) where N is batch size, T is time steps, F is features
        window_size: Size of each sliding window
        step_size: Step size between windows (stride)
        
    Returns:
        List of arrays, each of shape (N, window_size, F)
    """
    n, T, f = data.shape
    if T < window_size:
        return []
    return [data[:, start:start + window_size, :] for start in range(0, T - window_size + 1, step_size)]


def load_reference_hdf5(path: str) -> Dict[str, np.ndarray]:
    """
    Load reference data from HDF5 file(s). Each episode is mapped to its observed feature concatenation.
    Supports both single file paths and glob patterns (for loading multiple files).
    
    Args:
        path: Path to HDF5 file or glob pattern matching multiple files
        
    Returns:
        Dictionary mapping episode_name -> (T, F) array of concatenated features
    """
    out = {}
    
    # Check if path contains glob pattern
    if '*' in path or '?' in path:
        file_paths = glob.glob(path)
        if not file_paths:
            raise FileNotFoundError(f"No files found matching pattern: {path}")
        print(f"[Loading] Found {len(file_paths)} files matching pattern: {path}")
    else:
        file_paths = [path]
    
    # Load data from all matching files
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}. Skipping.")
            continue
        
        with h5py.File(file_path, "r") as f:
            # Handle both direct episode groups and data/episode structure
            if "data" in f:
                demo_group = f["data"]
            else:
                demo_group = f
            
            for episode, episode_data in demo_group.items():
                # Skip if episode already exists (from a previous file)
                if episode in out:
                    print(f"Warning: Episode '{episode}' already exists. Skipping duplicate from {file_path}.")
                    continue
                
                # Handle both obs group and direct episode data
                obs_group = episode_data.get("obs", episode_data)
                feature_keys = get_feature_keys_from_group(obs_group)
                if feature_keys is None:
                    print(f"Warning: Episode '{episode}' in {file_path} does not contain known observation keys. Skipping.")
                    continue
                out[episode] = concat_obs_group(obs_group, feature_keys=feature_keys)
    
    if not out:
        raise ValueError(f"No valid episodes found in files matching: {path}")
    
    return out


def load_retrieved_hdf5(path: str, top_k: Optional[int] = None) -> Dict[str, Optional[np.ndarray]]:
    """
    Load retrieved data from HDF5 file with 'results' group. Handles variable-length matches.
    Only loads top-K matches (match_0 to match_{top_k-1}) if top_k is specified.
    
    Args:
        path: Path to HDF5 file containing retrieval results
        top_k: Optional number of top matches to load (None loads all)
        
    Returns:
        Dictionary mapping episode_name -> (N, T, F) array where N is number of matches,
        or None if no matches found for that episode
    """
    out = {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Retrieval file not found: {path}")
    
    try:
        with h5py.File(path, "r") as f:
            if "results" not in f:
                raise KeyError(f"File {path} does not contain 'results' group. Available groups: {list(f.keys())}")
            
            results_group = f["results"]
            if len(results_group) == 0:
                # Return empty dict - will be handled by caller
                print(f"Warning: No episodes found in results group of {path}")
                return out
            
            for episode, episode_group in results_group.items():
                matches = []
                # Determine feature_keys for this episode using the first available match
                feature_keys = None
                for match_key, match_group in episode_group.items():
                    if match_key.startswith("match_"):
                        if "obs" not in match_group:
                            continue
                        obs_group = match_group["obs"]
                        feature_keys = get_feature_keys_from_group(obs_group)
                        if feature_keys is not None:
                            break
                
                if feature_keys is None:
                    # No valid match structure found - mark as None
                    print(f"Warning: Episode '{episode}' does not contain known observation keys. No matches available.")
                    out[episode] = None
                    continue
                
                # Collect all match keys and sort them by index
                match_keys = []
                for match_key in episode_group.keys():
                    if match_key.startswith("match_"):
                        try:
                            match_idx = int(match_key.split("_")[1])
                            match_keys.append((match_idx, match_key))
                        except (ValueError, IndexError):
                            continue
                
                if not match_keys:
                    print(f"Warning: No match_* keys found for episode {episode}. Marking as no matches.")
                    out[episode] = None
                    continue
                
                # Sort by index and take top-K if specified
                match_keys.sort(key=lambda x: x[0])
                if top_k is not None:
                    match_keys = match_keys[:top_k]
                    if len(match_keys) < top_k:
                        print(f"Info: Episode {episode} has {len(match_keys)} matches, requested top-{top_k}")
                
                for _, match_key in match_keys:
                    if match_key not in episode_group:
                        continue
                    match_group = episode_group[match_key]
                    if "obs" not in match_group:
                        continue
                    obs = match_group["obs"]
                    match_arr = concat_obs_group(obs, feature_keys=feature_keys)
                    if match_arr.size == 0:
                        continue
                    matches.append(match_arr)
                
                if not matches:
                    print(f"Warning: No valid matches found for episode {episode}. Marking as no matches.")
                    out[episode] = None
                    continue
                
                # Pad variable-length episode matches to max length for stacking
                max_len = max(match.shape[0] for match in matches)
                padded_matches = [
                    np.pad(match, ((0, max_len - match.shape[0]), (0, 0)), mode="constant")
                    if match.shape[0] < max_len else match
                    for match in matches
                ]
                out[episode] = np.stack(padded_matches, axis=0)
    except Exception as e:
        raise RuntimeError(f"Error loading retrieved HDF5 from {path}: {str(e)}\n{traceback.format_exc()}")
    
    return out

def pad_zeros_to_length(series, MOMENT_LENGTH=512):
    """
    Pad a time series with zeros to reach the target length.

    Args:
        series: np.ndarray of shape (seq_length, features)
        MOMENT_LENGTH: int, desired sequence length after padding

    Returns:
        np.ndarray of shape (MOMENT_LENGTH, features)
    """
    current_length = series.shape[0]
    if current_length >= MOMENT_LENGTH:
        return resize_scipy(series, MOMENT_LENGTH)
    padding_length = MOMENT_LENGTH - current_length
    padding = np.zeros((padding_length, series.shape[1]))
    padded_series = np.vstack((series, padding))
    padded_mask = np.hstack((np.ones(current_length), np.zeros(padding_length)))
    return padded_series, padded_mask
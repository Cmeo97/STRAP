import os
import h5py
import numpy as np

from strap.configs.libero_file_functions import get_libero_lang_instruction


def get_demo_data(hdf5_dataset, demo_key):
    demo_data = hdf5_dataset[demo_key]
    ee_pose = demo_data['obs/ee_pos'][:]
    gripper_states = demo_data['obs/gripper_states'][:]
    joint_states = demo_data['obs/joint_states'][:]
    series = np.concatenate((ee_pose, gripper_states, joint_states), axis=1)
    return series

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
            actions = demo_data['actions'][:]
            agentview_rgb = demo_data['obs/agentview_rgb'][:]
            ee_pose = demo_data['obs/ee_pos'][:]
            gripper_states = demo_data['obs/gripper_states'][:]
            joint_states = demo_data['obs/joint_states'][:]
            robot_states = demo_data['robot_states'][:]
            output[f"match_{i}"]['actions'] = actions[result['start_idx']:result['end_idx']]
            output[f"match_{i}"]['robot_states'] = robot_states[result['start_idx']:result['end_idx']]
            #output[f"match_{i}"]['obs/agentview_rgb'] = agentview_rgb[result['start_idx']:result['end_idx']]
            output[f"match_{i}"]['obs/ee_pose'] = ee_pose[result['start_idx']:result['end_idx']]
            output[f"match_{i}"]['obs/gripper_states'] = gripper_states[result['start_idx']:result['end_idx']]
            output[f"match_{i}"]['obs/joint_states'] = joint_states[result['start_idx']:result['end_idx']]
            output[f"match_{i}"]['file_path'] = result['offline_file']
            output[f"match_{i}"]['demo_key'] = result['demo_key']
            output[f"match_{i}"]['lang_instruction'] = get_libero_lang_instruction(f, result['demo_key'])
    return output
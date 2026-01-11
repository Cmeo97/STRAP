import os
import glob
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from strap.configs.libero_file_functions import get_libero_lang_instruction
from benchmarking.benchmark_models import stumpy_matching, dtaidistance_matching

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

def stumpy_dtaidistance_retrieval(output_path, stumpy=True, dtaidistance=False):
    with h5py.File(output_path, 'w') as outfile:
        results = outfile.create_group('results')
        with h5py.File(target_data_path, 'r') as f:
            for episode_key in f.keys():
                episode = results.create_group(episode_key)
                episode_data = f[episode_key]
                ee_pose = episode_data['ee_pos'][:][0]
                gripper_states = episode_data['gripper_states'][:][0]
                joint_states = episode_data['joint_states'][:][0]
                target_series = np.concatenate((ee_pose, gripper_states, joint_states), axis=1)
                
                episode_results = []
                for offline_file in tqdm(offline_file_list, desc=f"Retrieving data for {episode_key}"):
                    with h5py.File(offline_file, 'r') as offline_f:
                        offline_data = offline_f['data']
                        for demo_key in list(offline_data.keys()):
                            demo_data = offline_data[demo_key]
                            ee_pose = demo_data['obs/ee_pos'][:]
                            gripper_states = demo_data['obs/gripper_states'][:]
                            joint_states = demo_data['obs/joint_states'][:]
                            offline_series = np.concatenate((ee_pose, gripper_states, joint_states), axis=1)
                            
                            if stumpy:
                                 result = stumpy_matching(target_series, offline_series, top_k=1)
                            if dtaidistance:
                                 result = dtaidistance_matching(target_series, offline_series, top_k=1)
                            for match in result:
                                if match:
                                    episode_results.append({
                                        'cost': match[0],
                                        'start_idx': match[1],
                                        'end_idx': match[2],
                                        'demo_key': demo_key,
                                        'offline_file': offline_file
                                    })
                processed_results = process_retrieval_results(episode_results)
                for match_key, data in processed_results.items():
                    match_group = episode.create_group(match_key)
                    match_group.attrs['ep_meta'] = json.dumps({
                            "lang": data['lang_instruction']
                        })
                    match_group.attrs['file_path'] = data['file_path']
                    match_group.attrs['demo_key'] = data['demo_key']
                    for data_key, value in data.items():
                        if data_key not in ['file_path', 'demo_key', 'lang_instruction']:
                            match_group.create_dataset(data_key, data=value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Retrieval Benchmarking')
    parser.add_argument('--target_data_path', default="data/target_data/target_dataset.hdf5",
                        help='Path of the target dataset hdf5 file')
    parser.add_argument('--offline_data_dir', default="data/LIBERO/libero_90/",
                        help='Directory of libero_90 datasets')
    
    args = parser.parse_args()
    target_data_path = args.target_data_path
    offline_data_dir = args.offline_data_dir
    offline_file_list = glob.glob(os.path.join(offline_data_dir, "*.hdf5"))
    
    stumpy_dtaidistance_retrieval("data/retrieval_results/retrieval_results_stumpy.hdf5", stumpy=True, dtaidistance=False)
    stumpy_dtaidistance_retrieval("data/retrieval_results/retrieval_results_dtaidistance.hdf5", stumpy=False, dtaidistance=True)

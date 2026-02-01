import numpy as np
import h5py
import os
import re

close_drawer_ids = os.listdir('data/droid/videos/close_drawer')
close_cabinet_ids = os.listdir('data/droid/videos/close_cabinet')
open_cabinet_ids = os.listdir('data/droid/videos/open_cabinet')
pnp_ids = os.listdir('data/droid/videos/pnp')
turn_ids = os.listdir('data/droid/videos/turn')
all_ids = {'close_drawer': close_drawer_ids,
           'close_cabinet': close_cabinet_ids,
           'open_cabinet': open_cabinet_ids,
           'pnp': pnp_ids,
           'turn': turn_ids
          }

pattern = r"(.*?\.hdf5).*episode_(\d+)"
target_ids = set()

def resize_to_mean_length(task_data_list):
    """
    Resizes a list of 2D numpy arrays to their mean length via interpolation.
    
    Args:
        task_data_list (list of np.ndarray): List of 2D arrays (time_steps, features).
    
    Returns:
        np.ndarray: 3D array of shape (num_arrays, mean_length, features).
    """
    if not task_data_list:
        return np.array([]).reshape(0, 0, 0)
    
    lengths = [arr.shape[0] for arr in task_data_list]
    mean_len = int(np.round(np.mean(lengths)))
    
    resized = []
    for arr in task_data_list:
        if arr.shape[0] == mean_len:
            resized.append(arr)
        else:
            old_len = arr.shape[0]
            new_arr = np.zeros((mean_len, arr.shape[1]))
            for feat in range(arr.shape[1]):
                new_arr[:, feat] = np.interp(
                    np.linspace(0, old_len - 1, mean_len),
                    np.arange(old_len),
                    arr[:, feat]
                )
            resized.append(new_arr)
    
    return np.stack(resized, axis=0)


def create_droid_target_dataset():
    """
    Creates a target dataset for Droid maneuvers by collecting specific episodes
    and resizing them to the mean length.
    """
    with h5py.File('data/target_data/droid_target_dataset.hdf5', 'w') as target_hf:
        for task_id in all_ids.keys():
            task_data = {
                'gripper_states': [],
                'joint_states': [],
                'cartesian_positions': []
            }
            for file_name in all_ids[task_id]:
                match = re.search(pattern, file_name)
                if match:
                    hdf5_name = match.group(1)
                    episode_id = match.group(2)
                    print(f"File: {hdf5_name} | Episode ID: {episode_id}")
                    target_ids.add(f"{hdf5_name}+episode_{episode_id}")
                with h5py.File(os.path.join('data/droid/droid_dataset', hdf5_name), 'r') as hf:
                    episode_data = hf[f'episode_{episode_id}']
                    cartesian_positions = episode_data['cartesian_positions'][:]
                    gripper_states = episode_data['gripper_states'][:]
                    joint_states = episode_data['joint_positions'][:]
                    task_data['cartesian_positions'].append(cartesian_positions)
                    task_data['gripper_states'].append(gripper_states)
                    task_data['joint_states'].append(joint_states)
            
            print(f"Mean sequence length: {np.mean([len(a) for a in task_data['cartesian_positions']])} for task {task_id}")
            target_hf.create_group(task_id)
            target_hf[task_id].create_dataset(
                'obs/cartesian_positions', 
                data=resize_to_mean_length(task_data['cartesian_positions'])
            )
            target_hf[task_id].create_dataset(
                'obs/gripper_states', 
                data=resize_to_mean_length(task_data['gripper_states'])
            )
            target_hf[task_id].create_dataset(
                'obs/joint_states', 
                data=resize_to_mean_length(task_data['joint_states'])
            )
        
    print(f"Total target IDs collected: {len(target_ids)}")

def create_droid_offline_dataset():
    """
    Creates an offline dataset for Droid maneuvers by collecting episodes
    not included in the target dataset.
    """
    chunk_size = 100
    chunk_count = 1
    output_dir = f'data/droid/offline_dataset{chunk_count}.hdf5'
    with h5py.File(output_dir, 'w') as offline_hf:
        offline_hf.create_group('data')
        scene_number = 0
        for file_name in os.listdir('data/droid/droid_dataset'):
            with h5py.File(os.path.join('data/droid/droid_dataset', file_name), 'r') as hf:
                for episode_key in hf.keys():
                    full_id = f"{file_name}+{episode_key}"
                    if full_id not in target_ids:
                        print(f"Adding episode {full_id} to offline dataset.")
                        episode_data = hf[episode_key]
                        grp = offline_hf['data'].create_group(f'scene_{scene_number}')
                        lang = episode_data.attrs['instructions'][0]
                        uuid = episode_data.attrs['uuid']
                        grp.create_dataset('obs/cartesian_positions', data=episode_data['cartesian_positions'][:])
                        grp.create_dataset('obs/gripper_states', data=episode_data['gripper_states'][:])
                        grp.create_dataset('obs/joint_states', data=episode_data['joint_positions'][:])
                        grp.attrs['language_instruction'] = lang
                        grp.attrs['uuid'] = uuid
                        scene_number += 1
                        if scene_number >= chunk_size:
                            chunk_count += 1
                            output_dir = f'data/droid/offline_dataset{chunk_count}.hdf5'
                            offline_hf.close()
                            offline_hf = h5py.File(output_dir, 'w')
                            offline_hf.create_group('data')
                            scene_number = 0

if __name__ == '__main__':
    create_droid_target_dataset()

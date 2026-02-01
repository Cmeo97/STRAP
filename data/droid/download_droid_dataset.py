import pickle
import tensorflow_datasets as tfds
import os
import re
import h5py
import numpy as np
from typing import List, Dict
import json
from datetime import datetime

#ds = tfds.load("droid", data_dir="gs://gresearch/robotics", split="train")
chunk_size = 100

with open('data/droid/unique_task_ids.json', 'r') as f:
    unique_task_ids = json.load(f)

all_ids = set()
for task, ids in unique_task_ids.items():
    for id in ids:
        all_ids.add(id)

with open('data/droid/unique_org_ids.json', 'r') as f:
    org_dict = json.load(f)

def extract_trajectory_id(path):
    parts = path.split('/')
    project_name = parts[5]
    if project_name not in org_dict.keys():
        return ""
    unique_id = org_dict[project_name]
    # 1. Extract the date/time parts using regex
    # We look for the YYYY-MM-DD pattern and the HH:MM:SS pattern
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', path)
    time_match = re.search(r'(\d{2}:\d{2}:\d{2})', path)
    
    if not date_match or not time_match:
        return ""
    
    date_str = date_match.group(1)
    time_str = time_match.group(1)
    
    # 2. Convert to a datetime object to ensure formatting is correct
    # The path uses ':' but the target ID uses 'h', 'm', 's'
    dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    
    # 3. Format into the desired string: Prefix + UniqueID + FormattedTime
    formatted_time = dt.strftime("%Y-%m-%d-%Hh-%Mm-%Ss")
    
    return f"{project_name}+{unique_id}+{formatted_time}"

def download_droid_dataset(output_dir: str, max_demos: int = 100) -> None:
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    chunk_count = 0
    episode_in_file = 0
    
    output_file_name = os.path.join(output_dir, f'droid_dataset_{chunk_count}.hdf5')
    hdf5_file = h5py.File(output_file_name, 'w')
    
    for idx, episode in enumerate(ds.shuffle(1, seed=999)):
        file_path_bytes = episode['episode_metadata']['file_path'].numpy()
        file_path = file_path_bytes.decode('utf-8') # Decode file_path from bytes to string
        id = extract_trajectory_id(file_path)
        if len(id) == 0 or id not in all_ids:
            continue
        
        print(f"Processing episode {idx+1}: {file_path} (File: {chunk_count}, Count: {episode_in_file})")
        grp = hdf5_file.create_group(f'episode_{episode_in_file}')
        grp.attrs['file_path'] = file_path
        all_ids.remove(id)
        
        cartesian_positions = []
        gripper_states = []
        joint_positions = []
        images = []
        instructions = []
        for i, step in enumerate(episode['steps']):
            if i == 0:
                instruction_bytes = step['language_instruction'].numpy()
                instruction = instruction_bytes.decode('utf-8') if instruction_bytes else ""
                instructions.append(instruction)
            cartesian_positions.append(step['observation']['cartesian_position'].numpy())
            gripper_states.append(step['observation']['gripper_position'].numpy())
            joint_positions.append(step['observation']['joint_position'].numpy())
            images.append(step['observation']['exterior_image_1_left'].numpy())
        grp.create_dataset('images', data=np.array(images))
        grp.create_dataset('cartesian_positions', data=np.array(cartesian_positions))
        grp.create_dataset('gripper_states', data=np.array(gripper_states))
        grp.create_dataset('joint_positions', data=np.array(joint_positions))
        instr_dtype = h5py.string_dtype(encoding='utf-8')
        grp.attrs.create('instructions', data=np.array(instructions, dtype=instr_dtype))
        
        episode_in_file += 1
        count += 1
        
        # When we reach max_demos, close current file and start a new one
        if episode_in_file >= max_demos:
            hdf5_file.close()
            print(f"Saved {output_file_name} with {episode_in_file} episodes")
            chunk_count += 1
            episode_in_file = 0
            output_file_name = os.path.join(output_dir, f'droid_dataset_{chunk_count}.hdf5')
            hdf5_file = h5py.File(output_file_name, 'w')
    
    # Close the last file if it's still open
    if episode_in_file > 0:
        hdf5_file.close()
        print(f"Saved {output_file_name} with {episode_in_file} episodes")
    with open('remaining_ids.json', 'w') as f:
        json.dump(list(all_ids), f)


if __name__ == "__main__":
    download_droid_dataset('data')
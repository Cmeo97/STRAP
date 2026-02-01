import json
import os
import h5py
import numpy as np
from datetime import datetime
import cv2

unique_tasks = set(['Turn twistable object (ex: faucets, lamps, stove knobs)', 
                 'open or close cabinet door and pick up/ place object from cabinet', 
                 'Open a drawer and take some items  out', 'open/close cabinets and move objects', 
                 'put brick in drawer shelf and close drawer', 'put utensil in drawer', 
                 'put in or take out crockery from a microwave', 
                 'load the microwave', 
                 'Open or close hinged object (ex: hinged door, microwave, oven, book, dryer, toilet, box)', 
                 'open/close cabineet and place/pick objects from cabinet', 'Place objects in drawer and close it.', 
                 'Place objectas in cabinet and close it', 'put small stuff into an open drawer and close it', 
                 'Open cabinet and place objects inside',
                 'Open or close slidable objects (ex: toaster, drawers, sliding doors, dressers)', 
                 'put in or take out crockery from microwave', 
                 'Move object into or out of container (ex: drawer, clothes hamper, plate, trashcan, washer)', 
                 'Put brick in drawer shelf and close drawer'
                 ])

def sort_json_by_timestamp(input_file, output_file):
    # 1. Load the data
    with open(input_file, 'r') as f:
        data = json.load(f)

    # 2. Define a helper function to parse your specific date format
    # Format: 2023-12-09-15h-38m-03s
    def parse_timestamp(ts_str):
        return datetime.strptime(ts_str, "%Y-%m-%d-%Hh-%Mm-%Ss")

    # 3. Sort the dictionary items
    # We sort by the nested "timestamp" value inside each object
    sorted_items = sorted(
        data.items(), 
        key=lambda item: parse_timestamp(item[1]['timestamp'])
    )

    # 4. Convert back to a dictionary (Python 3.7+ maintains insertion order)
    sorted_dict = dict(sorted_items)

    # 5. Save the sorted result
    with open(output_file, 'w') as f:
        json.dump(sorted_dict, f, indent=4)
    
    print(f"Successfully sorted {len(sorted_dict)} entries and saved to {output_file}")
    return sorted_dict

def get_unique_tasks():
    unique_tasks = set()
    for key, val in sorted_dict.items():
        if val['current_task'] not in unique_tasks:
            if 'Do' in val['current_task']:
                continue
            unique_tasks.add(val['current_task'])
    print(f"Total unique tasks in sorted metadata: {len(unique_tasks)}")

def get_ids_for_unique_tasks():
    unique_tasks_ids = {}
    for key, val in sorted_dict.items():
        task = val['current_task']
        if task not in unique_tasks:
            continue
        if task not in unique_tasks_ids:
            unique_tasks_ids[task] = []
        if len(unique_tasks_ids[task]) < 100:
            unique_tasks_ids[task].append(key)
    return unique_tasks_ids

def save_videos():
    files = os.listdir('data/droid/droid_dataset1')
    for file in files:
        path = os.path.join('data/droid/droid_dataset1', file)
        with h5py.File(path, 'r') as hf:
            for key in hf.keys():
                images = hf[key]['images'][:]
                uuid = hf[key].attrs['uuid']
                instructions = hf[key].attrs['instructions']
                lang = '_'.join([ch for ch in instructions[0].split(' ')])
                height, width = images.shape[1], images.shape[2]
                video_path = os.path.join('data/droid/videos1', f'{file}_{lang}_{key}.mp4')
                os.makedirs('data/droid/videos1', exist_ok=True)
                out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
                for i in range(images.shape[0]):
                    frame = images[i]
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                out.release()
                print(f"Saved video: {video_path}")

if __name__ == "__main__":
    save_videos()


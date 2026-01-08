import numpy as np
import h5py
import os

target_data_dict = {
    'pnp': {
        'sources': {
            'data/LIBERO/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5': [
                ('demo_0', 120, 280), ('demo_1', 120, 280), ('demo_11', 120, 280), ('demo_12', 130, 270)
            ],
            'data/LIBERO/libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo.hdf5': [
                ('demo_0', 0, 135), ('demo_0', 135, 280), ('demo_1', 0, 135), ('demo_1', 135, 280)
            ],
            'data/LIBERO/libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo.hdf5': [
                ('demo_0', 0, 150), ('demo_0', 140, 280), ('demo_1', 0, 150), ('demo_1', 140, 280)
            ]
        }
    },
    'stove_on': {
        'sources': {
            'data/LIBERO/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5': [
                ('demo_0', 10, 120), ('demo_1', 10, 120), ('demo_2', 10, 120), ('demo_10', 10, 110),
                ('demo_11', 10, 120), ('demo_12', 10, 110), ('demo_13', 10, 120), ('demo_14', 10, 110),
                ('demo_15', 10, 100), ('demo_16', 10, 130)
            ]
        }
    },
    'stove_off': {
        'sources': {
            'data/LIBERO/libero_90/KITCHEN_SCENE8_turn_off_the_stove_demo.hdf5': [
                ('demo_0', 0, 280), ('demo_1', 0, 280), ('demo_2', 0, 280), ('demo_10', 0, 280),
                ('demo_11', 0, 280), ('demo_12', 0, 280), ('demo_13', 0, 280), ('demo_14', 0, 280),
                ('demo_15', 0, 280), ('demo_16', 0, 280)
            ]
        }
    },
    'top_drawer_open': {
        'sources': {
            'data/LIBERO/libero_90/KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_demo.hdf5': [
                ('demo_1', 0, 280), ('demo_2', 0, 280), ('demo_10', 0, 280), ('demo_11', 0, 280),
                ('demo_12', 0, 280), ('demo_13', 0, 280), ('demo_14', 0, 280), ('demo_15', 0, 280),
                ('demo_17', 0, 280), ('demo_18', 0, 280)
            ]
        }
    },
    'top_drawer_close': {
        'sources': {
            'data/LIBERO/libero_90/KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo.hdf5': [
                ('demo_0', 0, 280), ('demo_1', 0, 280), ('demo_10', 0, 280), ('demo_11', 0, 280),
                ('demo_12', 0, 280)
            ],
            'data/LIBERO/libero_90/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_demo.hdf5': [
                ('demo_0', 0, 280), ('demo_1', 0, 280), ('demo_10', 0, 280), ('demo_11', 0, 280),
                ('demo_12', 0, 280)
            ]
        }
    },
    'bottom_drawer_open': {
        'sources': {
            'data/LIBERO/libero_90/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_demo.hdf5': [
                ('demo_0', 0, 280), ('demo_1', 0, 280), ('demo_10', 0, 280), ('demo_11', 0, 280),
                ('demo_12', 0, 280), ('demo_13', 0, 280), ('demo_14', 0, 280), ('demo_17', 0, 280),
                ('demo_18', 0, 280), ('demo_19', 0, 280)
            ]
        }
    },
    'bottom_drawer_close': {
        'sources': {
            'data/LIBERO/libero_90/KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_demo.hdf5': [
                ('demo_0', 0, 280), ('demo_1', 0, 280), ('demo_10', 0, 280), ('demo_11', 0, 280),
                ('demo_12', 0, 280)
            ],
            'data/LIBERO/libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5': [
                ('demo_0', 180, 280), ('demo_1', 170, 280), ('demo_2', 150, 280), ('demo_10', 170, 280),
                ('demo_11', 160, 280)
            ]
        }
    },
    'microwave_open': {
        'sources': {
            'data/LIBERO/libero_90/KITCHEN_SCENE7_open_the_microwave_demo.hdf5': [
                ('demo_0', 0, 280), ('demo_1', 0, 280), ('demo_10', 0, 280), ('demo_11', 0, 280),
                ('demo_12', 0, 280), ('demo_13', 0, 280), ('demo_14', 0, 280), ('demo_15', 0, 280),
                ('demo_16', 0, 280), ('demo_17', 0, 280)
            ]
        }
    },
    'microwave_close': {
        'sources': {
            'data/LIBERO/libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo.hdf5': [
                ('demo_0', 260, 330), ('demo_1', 200, 330), ('demo_2', 200, 330), ('demo_10', 200, 330),
                ('demo_11', 200, 330), ('demo_12', 260, 330), ('demo_13', 220, 330), ('demo_14', 220, 330),
                ('demo_15', 200, 330), ('demo_19', 200, 330)
            ]
        }
    }
}

def extract_data_segment(data, start_idx, end_idx):
    actions = data['actions'][:][start_idx:end_idx]
    gripper_states = data['obs/gripper_states'][:][start_idx:end_idx]
    joint_states = data['obs/joint_states'][:][start_idx:end_idx]
    ee_pos = data['obs/ee_pos'][:][start_idx:end_idx]
    robot_states = data['robot_states'][:][start_idx:end_idx]

    return actions, gripper_states, joint_states, ee_pos, robot_states

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

def create_target_data():
    with h5py.File('data/retrieved_results/target_dataset.hdf5', 'w') as f:
        for task in target_data_dict.keys():
            task_data = {
                'actions': [],
                'gripper_states': [],
                'joint_states': [],
                'ee_pos': [],
                'robot_states': []
            }
            f.create_group(task)
            for source_file in target_data_dict[task]['sources'].keys():
                with h5py.File(source_file, 'r') as source:
                    for demos in target_data_dict[task]['sources'][source_file]:
                        demo_name, start, end = demos
                        data = source['data'][demo_name]
                        actions, gripper_states, joint_states, ee_pos, robot_states = extract_data_segment(data, start, end)
                        
                        task_data['actions'].append(actions)
                        task_data['gripper_states'].append(gripper_states)
                        task_data['joint_states'].append(joint_states)
                        task_data['ee_pos'].append(ee_pos)
                        task_data['robot_states'].append(robot_states)
            print(f"Task: {task}, Segments Collected: {len(task_data['actions'])}")
            print(f"Mean sequence length: {np.mean([len(a) for a in task_data['actions']])}")
            print("---------------------------------------------------")
            # Resize all data to mean length
            f[task].create_dataset('actions', data=resize_to_mean_length(task_data['actions']))
            f[task].create_dataset('gripper_states', data=resize_to_mean_length(task_data['gripper_states']))
            f[task].create_dataset('joint_states', data=resize_to_mean_length(task_data['joint_states']))
            f[task].create_dataset('ee_pos', data=resize_to_mean_length(task_data['ee_pos']))
            f[task].create_dataset('robot_states', data=resize_to_mean_length(task_data['robot_states']))

if __name__ == "__main__":
    create_target_data()

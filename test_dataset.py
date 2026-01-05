import h5py
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

data_file = "data/LIBERO/libero_90/KITCHEN_SCENE7_open_the_microwave_demo.hdf5"

def create_videos(dataset):
    input_dir = 'data/LIBERO'
    files = os.listdir(os.path.join(input_dir, dataset))
    for file in files:
        if not file.endswith('.hdf5'):
            continue
        file_path = os.path.join(input_dir, dataset, file)
        output_dir = os.path.join(f"data/videos/{dataset}", os.path.splitext(file)[0])
        with h5py.File(file_path, 'r') as f:
            data = f['data']

            for demo_name in list(data.keys())[:15]:
                demo_data = data[demo_name]
                images = demo_data['obs/agentview_rgb'][:]
                video_path = os.path.join(output_dir, f"{demo_name}.mp4")

                # Create a video writer object
                os.makedirs(output_dir, exist_ok=True)
                height, width = 480, 640
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

                for img in images:
                    img = np.flip(img, axis=0)  # Flip the image vertically
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
                    img = img.astype(np.uint8)  # Ensure uint8 type
                    img = cv2.resize(img, (width, height))  # Resize to desired dimensions
                    video_writer.write(img)

                video_writer.release()
                print(f"Video saved to {video_path}")

def plot_data(data):
    fig, axs = plt.subplots(12, 1, figsize=(15, 12))

    for i, demo in enumerate(data):
        if i >= 4:  # Only process first 4 demos
            break
        actions = data[demo]['actions'][:]
        gripper_states = data[demo]['obs/gripper_states'][: ]
        joint_states = data[demo]['obs/joint_states'][: ]
        robot_states = data[demo]['robot_states'][: ]
        ee_pos = data[demo]['obs/ee_pos'][: ]

        base_idx = i * 3
        axs[base_idx].plot(np.diff(ee_pos, axis=0))
        axs[base_idx].set_title(f'Demo {i+1}: End Effector Velocities over Time')

        axs[base_idx + 1].plot(np.diff(gripper_states, axis=0))
        axs[base_idx + 1].set_title(f'Demo {i+1}: Gripper State Changes over Time')

        axs[base_idx + 2].plot(np.diff(joint_states, axis=0))
        axs[base_idx + 2].set_title(f'Demo {i+1}: Joint Velocities over Time')

    plt.tight_layout()
    plt.show()

with h5py.File(data_file, "r") as f:
    data = f['data']
    #create_videos("libero_90")
    plot_data(data)



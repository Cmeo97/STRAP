import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

data_file = "data/LIBERO/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5"

with h5py.File(data_file, "r") as f:
    data = f['data']
    for demo in data:
        actions = data[demo]['actions'][:]
        gripper_states = data[demo]['obs/gripper_states'][: ]
        joint_states = data[demo]['obs/joint_states'][: ]
        robot_states = data[demo]['robot_states'][: ]

        fig, axs = plt.subplots(4, 1, figsize=(10, 8))

        axs[0].plot(actions)
        axs[0].set_title('Actions over Time')

        axs[1].plot(gripper_states)
        axs[1].set_title('Gripper States over Time')

        axs[2].plot(joint_states)
        axs[2].set_title('Joint Positions over Time')

        axs[3].plot(robot_states)
        axs[3].set_title('Robot States over Time')

        plt.tight_layout()
        plt.show()
        break  # Visualize only the first demo for brevity


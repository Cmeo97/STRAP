import os
import h5py
import numpy as np
import pandas as pd
from plotnine import *
import yaml
import yaml
import argparse
import cv2
import matplotlib.pyplot as plt

def get_images(img_array, max_images=6):
    """ Utility function to extract a fixed number of images from an array for visualization. """
    total_images = img_array.shape[0]
    if total_images <= max_images:
        return img_array
    else:
        indices = np.linspace(0, total_images - 1, max_images).astype(int)
        return img_array[indices]

def array_from_videos(video_dir):
    cap = cv2.VideoCapture(video_dir)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))  # Resize for consistency
        frames.append(frame)
    cap.release()
    return np.array(frames)

def plot_image_grid_plotnine(image_data_list, method_names, save_path):
    """
    Uses plotnine to create a highly structured image grid with vertical 
    labels using faceting.
    """
    rows = []
    for m_idx, (method_imgs, method_name) in enumerate(zip(image_data_list, method_names)):
        for f_idx, img in enumerate(method_imgs):
            # To use geom_raster, we ideally want coordinates, but for speed 
            # and complexity in plotnine, we can also use a "hack" or 
            # simply display the image data. 
            # However, plotnine is best at handling the layout logic.
            # We'll create a dataframe of images.
            rows.append({
                'Method': method_name,
                'Frame': f"Frame {f_idx+1}",
                'Image': img,
                'RowIdx': m_idx
            })
    
    df_images = pd.DataFrame(rows)
    # Ensure Method order is preserved
    df_images['Method'] = pd.Categorical(df_images['Method'], categories=method_names)

    # Note: plotnine isn't natively built to "imshow" arrays inside cells efficiently.
    # The standard way is to use subplots, but since you want the plotnine "look":
    # We will use a custom approach where we draw the facets and then 
    # place the images using an inset or a specialized mapping.
    #
    # Because geom_raster is extremely slow for high-res images in plotnine,
    # the most "plotnine" way to get beautiful vertical labels and 
    # consistent margins is using facet_grid.
    
    # Let's use a specialized function to render the final visual using the 
    # layout engine of plotnine but the image rendering of matplotlib for performance.
    
    import matplotlib.pyplot as plt

    # We use plotnine's theme and facet structure logic
    p = (
        ggplot(df_images)
        + facet_grid('Method ~ Frame')
        + theme_minimal()
        + theme(
            axis_text=element_blank(),
            axis_ticks=element_blank(),
            axis_title=element_blank(),
            panel_grid=element_blank(),
            strip_text_x=element_blank(), # Hide top frame labels
            strip_text_y=element_text(rotation=0, weight='bold', size=10), # Vertical labels
            panel_spacing=0.02
        )
    )

    # Render to matplotlib to inject the actual images into the facet areas
    fig = p.draw()
    axes = fig.get_axes()
    
    # The axes are ordered by facet (Row 1 Col 1, Row 1 Col 2...)
    for i, ax in enumerate(axes):
        # Only inject images into the main panels, not the strip labels
        if i < len(df_images):
            img = df_images.iloc[i]['Image']
            ax.imshow(img, aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])

    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Plotnine-styled grid saved to {save_path}")

def plot_image_grid(image_data_list, method_names, save_path):
    """
    Creates a compact grid with vertical labels on the far left.
    The labels no longer occupy an image slot.
    """
    n_methods = len(image_data_list)
    n_frames = 6
    
    # Tightened height further to collapse vertical gap
    fig, axes = plt.subplots(n_methods, n_frames, figsize=(12, 2 * n_methods))
    
    # hspace=0.02 for near-zero vertical gap
    # wspace=0.01 for near-zero horizontal gap
    plt.subplots_adjust(wspace=0.01, hspace=0.02)

    for i, (method_imgs, method_name) in enumerate(zip(image_data_list, method_names)):
        for j in range(n_frames):
            ax = axes[i, j]
            if j < len(method_imgs):
                ax.imshow(method_imgs[j])
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Clean frame edges
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Vertical labels on the last column (right side)
            if j == n_frames - 1:
                # rotation=270 for vertical text on the right
                # va='center', ha='left' to align it closely to the image edge
                ax.set_ylabel(method_name, 
                              fontsize=12, 
                              #fontweight='bold', 
                              rotation=270, 
                              labelpad=15, 
                              va='center', 
                              ha='center')
                ax.yaxis.set_label_position("right")


    plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.05)
    plt.close()
    print(f"Compact grid with vertical labels saved to {save_path}")

def plot_figures(df, graph_path):   
    plot = (
        ggplot(df, aes(x='Time', y='Value', color='Model', linetype='Model'))
        + geom_line(size=1.0)
        + facet_wrap('~Variable', scales='free_y', ncol=1)
        + scale_color_manual(values={
            'Target': '#000000',
            'Dtaidistance': '#E69F00',
            'Stumpy': '#56B4E9',
            'ROSER': '#009E73'
        })
        + scale_linetype_manual(values={
            'Target': 'solid',
            'Dtaidistance': 'dashed',
            'Stumpy': 'dotted',
            'ROSER': 'dashdot'
        })
        + theme_bw()
        + labs(x='Time Steps', y='Normalized Values')
        + theme(
            figure_size=(12, 10),
            # Legend inside: positioned relative to the whole plot grid
            legend_position=(1.0, 0.95), 
            legend_background=element_rect(fill='white', alpha=0.8, color='gray'),
            legend_box_margin=5,
            panel_grid_minor=element_blank(),
            strip_background=element_rect(fill='#f0f0f0'),
            subplots_adjust={'hspace': 0.3}
        )
    )

    plot.save(graph_path, width=12, height=10, units='in', dpi=300)

def plot_nuscenes_results(match_key, task_key='left turn', save_plot=True, plot_no=1):
    """
    Docstring for plot_nuscenes_results
    
    :param match_key: Description
    :param task_key: Description
    :param save_plot: Description
    :param plot_no: Description

    right turn: match_15
    """
    target_scenes = {
        'left turn': ('scene-0068', (500, 870)),
        'right turn': ('scene-0072', (600, 1200)),
        'straight driving': ('scene-0065', (600, 1000)),
        'regular stop': ('scene-0073', (300, 700))
    }
    
    all_data = []
    image_data_list = []
    method_names = ['Reference']

    # 1. Extract Target Data
    with h5py.File(target_file, 'r') as f:
        target_data = f[task_key]
        yaw_rate = target_data['obs/yaw_rate'][:][0]
        velocity = target_data['obs/velocity'][:][0]
        acceleration = target_data['obs/acceleration'][:][0]
        
        def get_df(velocity, acceleration, yaw_rate, model_name):
            n = len(yaw_rate)
            return pd.DataFrame({
                'Time': np.tile(np.arange(n), 3),
                'Value': np.concatenate([velocity[:, 0], acceleration[:, 0], yaw_rate[:, 0]]), # Plotting first dimension/index
                'Variable': np.repeat(['Velocity', 'Acceleration', 'Yaw Rate'], n),
                'Model': model_name
            })
        all_data.append(get_df(velocity, acceleration, yaw_rate, 'Reference'))

        # Images
        scene = target_scenes[task_key][0]
        start_idx, end_idx = target_scenes[task_key][1]
        video_path = os.path.join('/home/zillur/programs/nuscene/output_videos2', f"{scene}.avi")
        if os.path.exists(video_path):
            video_frames = array_from_videos(video_path)
            selected_frames = video_frames[start_idx//5:end_idx//5]
            image_data_list.append(get_images(selected_frames))
        else:
            image_data_list.append(np.zeros((5, 224, 224, 3), dtype=np.uint8))

    # 2. Extract Retrieval Results
    for file_path in baseline_list:
        method_name = os.path.basename(file_path).split('.')[0].split('_')[3].capitalize()
        if method_name == 'Prototype':
            method_name = 'ROSER'
        method_names.append(method_name)

        
        with h5py.File(file_path, 'r') as f:
            task = f['results'][task_key][match_key]
            yaw_rate = task['obs/yaw_rate'][:]
            velocity = task['obs/velocity'][:]
            acceleration = task['obs/acceleration'][:]
            scene_key = task.attrs['demo_key']
            start_idx = int(task.attrs['start_idx'])
            end_idx = int(task.attrs['end_idx'])

            all_data.append(get_df(velocity, acceleration, yaw_rate, method_name))

            # Images
            video_path = os.path.join('/home/zillur/programs/nuscene/output_videos2', f"{scene_key}.avi")
            if os.path.exists(video_path):
                video_frames = array_from_videos(video_path)
                selected_frames = video_frames[start_idx//5:end_idx//5]
                image_data_list.append(get_images(selected_frames))
            else:
                image_data_list.append(np.zeros((6, 224, 224, 3), dtype=np.uint8))

    # 3. Create Graph (Plotnine)
    df = pd.concat(all_data, ignore_index=True)
    ds_type = args.dataset_type
    graph_path = f"data/qualitative_results/{ds_type}_graph_{plot_no}.pdf"
    grid_path = f"data/qualitative_results/{ds_type}_images_{plot_no}.pdf"
    os.makedirs(os.path.dirname(grid_path), exist_ok=True)
    #plot_figures(df, graph_path)

    plot_image_grid(image_data_list, method_names, grid_path)

def plot_libero_results(match_key, task_key='pnp', save_plot=True, plot_no=1):
    target_scenes = {
        'pnp': ('data/LIBERO/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5','demo_0', 120, 280),
        'stove_on': ('data/LIBERO/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5','demo_0', 10, 120),
        'stove_off': ('data/LIBERO/libero_90/KITCHEN_SCENE8_turn_off_the_stove_demo.hdf5','demo_0', 0, 280),
        'top_drawer_open': ('data/LIBERO/libero_90/KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_demo.hdf5','demo_1', 0, 280),
        'top_drawer_close': ('data/LIBERO/libero_90/KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo.hdf5','demo_0', 0, 280),
        'microwave_close': ('data/LIBERO/libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo.hdf5','demo_0', 260, 330),
        'microwave_open': ('data/LIBERO/libero_90/KITCHEN_SCENE7_open_the_microwave_demo.hdf5','demo_0', 0, 280),
        'bottom_drawer_close': ('data/LIBERO/libero_90/KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_demo.hdf5','demo_0', 0, 280),
        'bottom_drawer_open': ('data/LIBERO/libero_90/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_demo.hdf5','demo_0', 0, 280)
    }

    all_data = []
    image_data_list = []
    method_names = ['Reference']

    # 1. Extract Target Data
    with h5py.File(target_scenes[task_key][0], 'r') as f:
        demo_key = target_scenes[task_key][1]
        start_idx = target_scenes[task_key][2]
        end_idx = target_scenes[task_key][3]
        target_data = f['data'][demo_key]
        ee_pos = target_data['obs/ee_pos'][:][start_idx:end_idx]
        gripper_states = target_data['obs/gripper_states'][:][start_idx:end_idx]
        joint_states = target_data['obs/joint_states'][:][start_idx:end_idx]

        # Helper to melt data for facets
        def get_df(ee, gr, jt, model_name):
            n = len(ee)
            return pd.DataFrame({
                'Time': np.tile(np.arange(n), 3),
                'Value': np.concatenate([ee[:, 0], gr[:, 0], jt[:, 0]]), # Plotting first dimension/index
                'Variable': np.repeat(['EE Position (X)', 'Gripper State', 'Joint Position (0)'], n),
                'Model': model_name
            })

        all_data.append(get_df(ee_pos, gripper_states, joint_states, 'Reference'))
        images = target_data['obs/agentview_rgb'][:][start_idx:end_idx]
        selected_images = get_images(images)
        selected_images = np.flip(selected_images, axis=1) 
        image_data_list.append(selected_images)

    # 2. Extract Retrieval Results
    for file_path in baseline_list:
        method_name = os.path.basename(file_path).split('.')[0].split('_')[3].capitalize()
        if method_name == 'Prototype':
            method_name = 'ROSER'
        method_names.append(method_name)
        
        with h5py.File(file_path, 'r') as f:
            task = f['results'][task_key][match_key]
            ee_pos = task['obs/ee_pos'][:]
            gripper_states = task['obs/gripper_states'][:]
            joint_states = task['obs/joint_states'][:]
            start_idx = int(task.attrs['start_idx']) 
            end_idx = int(task.attrs['end_idx']) + 20  # Slightly extend for visualization

            all_data.append(get_df(ee_pos, 
                                   gripper_states, 
                                   joint_states, 
                                   method_name))
            
            
            image_file = task.attrs['file_path']
            demo_key = task.attrs['demo_key']
            with h5py.File(image_file, 'r') as img_f:
                task_img = img_f['data'][demo_key]
                images = task_img['obs/agentview_rgb'][:][start_idx:end_idx]
            selected_images = get_images(images)
            selected_images = np.flip(selected_images, axis=1)
            image_data_list.append(selected_images)
    
    # 3. Create Graph (Plotnine)
    df = pd.concat(all_data, ignore_index=True)
    ds_type = args.dataset_type
    graph_path = f"data/qualitative_results/{ds_type}_graph_{plot_no}.pdf"
    grid_path = f"data/qualitative_results/{ds_type}_images_{plot_no}.pdf"
    os.makedirs(os.path.dirname(grid_path), exist_ok=True)
    #plot_figures(df, graph_path)

    plot_image_grid(image_data_list, method_names, grid_path)

def plot_droid_results(match_key, task_key='turn', save_plot=True, plot_no=1):
    close_drawer_ids = os.listdir('data/droid/videos/close_drawer')
    close_cabinet_ids = os.listdir('data/droid/videos/close_cabinet')
    open_cabinet_ids = os.listdir('data/droid/videos/open_cabinet')
    pnp_ids = os.listdir('data/droid/videos/pnp')
    turn_ids = os.listdir('data/droid/videos/turn')
    all_ids = {'close_drawer': close_drawer_ids[0],
            'close_cabinet': close_cabinet_ids[5],
            'open_cabinet': open_cabinet_ids[5],
            'pnp': pnp_ids[8],
            'turn': turn_ids[0]
            }
    
    image_data_list = []
    method_names = ['Reference']
    # 1. Extract Target Data
    video_path = os.path.join('data/droid/videos', task_key, all_ids[task_key])
    images = array_from_videos(video_path)
    selected_images = get_images(images)
    image_data_list.append(selected_images)

    
    raw_droid_data_list = os.listdir('data/droid/droid_dataset')
    # 2. Extract Retrieval Results
    for file_path in baseline_list:
        method_name = os.path.basename(file_path).split('.')[0].split('_')[3].capitalize()
        if method_name == 'Prototype':
            method_name = 'ROSER'
        method_names.append(method_name)
        
        with h5py.File(file_path, 'r') as f:
            task = f['results'][task_key][match_key]
            file_path = task.attrs['file_path']
            demo_key = task.attrs['demo_key']
            start_idx = int(task.attrs['start_idx'])
            end_idx = int(task.attrs['end_idx']) + 20  # Slightly extend for visualization
            with h5py.File(file_path, 'r') as img_f:
                task_img = img_f['data'][demo_key]
                uuid = task_img.attrs['uuid']
                for raw_file in raw_droid_data_list:
                    with h5py.File(os.path.join('data/droid/droid_dataset', raw_file), 'r') as raw_f:
                        for episode_keys in raw_f.keys():
                            if raw_f[episode_keys].attrs['uuid'] == uuid:
                                images = raw_f[episode_keys]['images'][:][start_idx:end_idx]
                                selected_images = get_images(images)
                                selected_images = [cv2.resize(img, (224, 224)) for img in selected_images]
                                image_data_list.append(selected_images)
                                break
    ds_type = args.dataset_type
    grid_path = f"data/qualitative_results/{ds_type}_images_{plot_no}.pdf"
    os.makedirs(os.path.dirname(grid_path), exist_ok=True)
    plot_image_grid(image_data_list, method_names, grid_path)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Euclidean Distance-based Maneuver Retrieval')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file.')
    parser.add_argument('--dataset_type', default='nuscene', choices=['libero', 'nuscene', 'droid'], 
                       help='Type of dataset to use.')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))

    if args.dataset_type == 'libero':
        target_file = config['dataset_paths']['libero_target']
        offline_data_dir = config['dataset_paths']['libero_offline']
        prototype_output = os.path.join(config['retrieval_paths']['libero'], 'libero_retrieval_results_prototype.hdf5')
        stumpy_output = os.path.join(config['retrieval_paths']['libero'], 'libero_retrieval_results_stumpy.hdf5')
        dtaidistance_output = os.path.join(config['retrieval_paths']['libero'], 'libero_retrieval_results_dtaidistance.hdf5')
    elif args.dataset_type == 'nuscene':
        target_file = config['dataset_paths']['nuscene_target']
        offline_data_dir = config['dataset_paths']['nuscene_offline']
        prototype_output = os.path.join(config['retrieval_paths']['nuscene'], 'nuscene_retrieval_results_prototype.hdf5')
        stumpy_output = os.path.join(config['retrieval_paths']['nuscene'], 'nuscene_retrieval_results_stumpy.hdf5')
        dtaidistance_output = os.path.join(config['retrieval_paths']['nuscene'], 'nuscene_retrieval_results_dtaidistance.hdf5')
    elif args.dataset_type == 'droid':
        target_file = config['dataset_paths']['droid_target']
        offline_data_dir = config['dataset_paths']['droid_offline']
        prototype_output = os.path.join(config['retrieval_paths']['droid'], 'droid_retrieval_results_prototype.hdf5')
        stumpy_output = os.path.join(config['retrieval_paths']['droid'], 'droid_retrieval_results_stumpy.hdf5')
        dtaidistance_output = os.path.join(config['retrieval_paths']['droid'], 'droid_retrieval_results_dtaidistance.hdf5')
    else:
        raise ValueError("Unsupported dataset type!")
    
    baseline_list = [dtaidistance_output, stumpy_output, prototype_output]
    if args.dataset_type == 'libero':
        # for i in range(0, 30, 2):
        #     plot_libero_results(match_key=f'match_{i}', task_key='microwave_close', save_plot=True, plot_no=i)
        plot_libero_results(match_key=f'match_22', task_key='pnp', save_plot=True, plot_no='pnp')
        plot_libero_results(match_key=f'match_35', task_key='microwave_open', save_plot=True, plot_no='microwave_open')
        plot_libero_results(match_key=f'match_35', task_key='bottom_drawer_open', save_plot=True, plot_no='bottom_drawer_open')
        plot_libero_results(match_key=f'match_10', task_key='stove_on', save_plot=True, plot_no='stove_on')
        plot_libero_results(match_key=f'match_37', task_key='stove_off', save_plot=True, plot_no='stove_off')
        plot_libero_results(match_key=f'match_25', task_key='top_drawer_close', save_plot=True, plot_no='top_drawer_close')
        plot_libero_results(match_key=f'match_25', task_key='top_drawer_open', save_plot=True, plot_no='top_drawer_open')
        plot_libero_results(match_key=f'match_37', task_key='bottom_drawer_close', save_plot=True, plot_no='bottom_drawer_close')
        plot_libero_results(match_key=f'match_2', task_key='microwave_close', save_plot=True, plot_no='microwave_close')
    elif args.dataset_type == 'nuscene':
        # for i in range(5, 40, 3):
        #     plot_nuscenes_results(match_key=f'match_{i}', task_key='regular stop', save_plot=True, plot_no=i)
        plot_nuscenes_results(match_key='match_15', task_key='right turn', save_plot=True, plot_no='right_turn')
        plot_nuscenes_results(match_key='match_26', task_key='left turn', save_plot=True, plot_no='left_turn')
        plot_nuscenes_results(match_key='match_10', task_key='straight driving', save_plot=True, plot_no='straight_driving')
        plot_nuscenes_results(match_key='match_17', task_key='regular stop', save_plot=True, plot_no='regular_stop')
    elif args.dataset_type == 'droid':
        # for i in range(0, 30, 2):
        #     plot_droid_results(match_key=f'match_{i}', task_key='turn', save_plot=True, plot_no=i)
        plot_droid_results(match_key=f'match_7', task_key='pnp', save_plot=True, plot_no='pnp')
        plot_droid_results(match_key=f'match_13', task_key='close_cabinet', save_plot=True, plot_no='close_cabinet')
        plot_droid_results(match_key=f'match_10', task_key='open_cabinet', save_plot=True, plot_no='open_cabinet')
        plot_droid_results(match_key=f'match_28', task_key='close_drawer', save_plot=True, plot_no='close_drawer')
        plot_droid_results(match_key=f'match_0', task_key='turn', save_plot=True, plot_no='turn')
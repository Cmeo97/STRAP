import numpy as np
import h5py
import json
import os
import argparse
import yaml
from typing import List, Tuple, Dict, Any
import random

from prototypical_networks.prototype_datasets import process_target_data

# Should be set before importing keras
os.environ["KERAS_BACKEND"] = "torch"

from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.shapelets import LearningShapelets, \
    grabocka_params_to_shapelet_size_dict
from tslearn.utils import ts_size, to_time_series_dataset

def model(X_train, y_train, epochs=10, model_path='shapelet_model.json'):
    # Normalize each of the timeseries in the Trace dataset
    X_train = TimeSeriesScalerMinMax().fit_transform(X_train)
    
    # 4 shapelets, each of size 300 timsteps
    #shapelet_sizes = {300:1}

    # Get statistics of the dataset
    n_ts, ts_sz = X_train.shape[:2]
    n_tasks = len(set(y_train))

    # Set the number of shapelets per size as done in the original paper
    shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=n_ts,
                                                        ts_sz=ts_sz,
                                                        n_classes=n_tasks,
                                                        l=0.1,
                                                        r=1)

    # Initialize the LearningShapelets model
    shp_clf = LearningShapelets(n_shapelets_per_size=shapelet_sizes,
                                optimizer='adam',
                                weight_regularizer=0.001,
                                max_iter=epochs,
                                verbose=1,
                                batch_size=4,
                                random_state=999)

    # Fit the model
    shp_clf.fit(X_train, y_train)
    shp_clf.to_json(model_path)

    return shp_clf


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Shapelet-based Maneuver Retrieval')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file.')
    parser.add_argument('--dataset_type', default='droid', choices=['libero', 'nuscene', 'droid'], 
                       help='Type of dataset to use.')
    parser.add_argument('--epochs', type=int, default=250, help='Number of training epochs for shapelet model.')
    parser.add_argument('--pretrained', default=False, action='store_true', help='Use pretrained model for retrieval.')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))

    if args.dataset_type == 'libero':
        target_data = config['dataset_paths']['libero_target']
        offline_data_dir = config['dataset_paths']['libero_offline']
        retrieved_output = os.path.join(config['retrieval_paths'], 'libero_retrieval_results_shapelet.hdf5')
        checkpoint_name = config['shapelet_ckpt_paths']['libero']
    elif args.dataset_type == 'nuscene':
        target_data = config['dataset_paths']['nuscene_target']
        offline_data_dir = config['dataset_paths']['nuscene_offline']
        retrieved_output = os.path.join(config['retrieval_paths'], 'nuscene_retrieval_results_shapelet.hdf5')
        checkpoint_name = config['shapelet_ckpt_paths']['nuscene']
    elif args.dataset_type == 'droid':
        target_data = config['dataset_paths']['droid_target']
        offline_data_dir = config['dataset_paths']['droid_offline']
        retrieved_output = os.path.join(config['retrieval_paths'], 'droid_retrieval_results_shapelet.hdf5')
        checkpoint_name = config['shapelet_ckpt_paths']['droid']
    else:
        raise ValueError("Unsupported dataset type!")
    
    reference_maneuver_data, episode_names = process_target_data(target_data)
    N_DIM =reference_maneuver_data[0][0].shape[-1]
    N_CLASSES = len(reference_maneuver_data)
    Y_train = [x for x in range(N_CLASSES) for _ in range(len(reference_maneuver_data[0]))]

    if not args.pretrained:
        X_train = []
        for i in range(len(reference_maneuver_data)):
            for maneuver in reference_maneuver_data[i]:
                X_train.append(maneuver)
        X_train = to_time_series_dataset(X_train)
        shp_clf = model(X_train, Y_train, epochs=args.epochs, model_path=checkpoint_name)
    else:
        shp_clf = LearningShapelets.from_json(checkpoint_name)


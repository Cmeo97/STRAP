import numpy as np
import h5py
import json
import os
import argparse
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
    parser = argparse.ArgumentParser(description='Prototype Libero Retrieval')
    parser.add_argument('--dataset', default='libero', choices=['libero', 'nuscene', 'droid'],
                        help='Dataset to use for shapelet learning')
    
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Use pretrained model for retrieval without training')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train shapelet model if not using pretrained')
    
    args = parser.parse_args()

    if args.dataset == 'libero':
        target_data = 'data/target_data/libero_target_dataset.hdf5'
        offline_data_dir = 'data/LIBERO/libero_90/*demo.hdf5'
        output_path = 'data/retrieval_results/libero_retrieval_results_shapelet.hdf5'
        model_path = 'shapelet_libero_model.json'
    elif args.dataset == 'nuscene':
        target_data = 'data/target_data/nuscene_target_dataset.hdf5'
        offline_data_dir = 'data/nuscene/*.hdf5'
        output_path = 'data/retrieval_results/nuscene_retrieval_results_shapelet.hdf5'
        model_path = 'shapelet_nuscene_model.json'
    elif args.dataset == 'droid':
        target_data = 'data/target_data/droid_target_dataset.hdf5'
        offline_data_dir = 'data/droid/droid_dataset/*.hdf5'
        output_path = 'data/retrieval_results/droid_retrieval_results_shapelet.hdf5'
        model_path = 'shapelet_droid_model.json'
    
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
        shp_clf = model(X_train, Y_train, epochs=args.epochs, model_path=model_path)
    else:
        shp_clf = LearningShapelets.from_json(model_path)


import os
import h5py
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d


def reduce_features_pca(time_series_2d, n_components=10):
    """
    Reduces the feature dimension of a single 2D time series sample 
    while keeping the number of timestamps exactly the same.
    
    Parameters:
    time_series_2d: np.array of shape (Time_steps, Features)
    n_components: Target number of features (default 10)
    
    Returns:
    reduced_series: np.array of shape (Time_steps, n_components)
    """
    # 1. Check if the requested components are valid
    n_features = time_series_2d.shape[1]
    if n_components > n_features:
        print(f"Warning: n_components ({n_components}) is greater than "
              f"original features ({n_features}). Returning original data.")
        return time_series_2d

    # 2. Standardize features (Scale along the Time axis)
    # This ensures each feature has mean=0 and variance=1 across time
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(time_series_2d)
    
    # 3. Initialize and fit PCA on the feature dimension
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(scaled_data)

    return reduced_data

def resize_scipy(data, new_size):
    # Create an index for current data (0 to 399)
    x_old = np.linspace(0, 1, data.shape[0])
    # Create an index for the new target size (0 to 299)
    x_new = np.linspace(0, 1, new_size)
    
    # Kind='linear' or 'cubic' for smoother interpolation
    f = interp1d(x_old, data, axis=0, kind='linear')
    return f(x_new)
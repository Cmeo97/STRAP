import json
import os
import subprocess
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader
from transformers import AutoVideoProcessor, AutoModel

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.attentive_pooler import AttentiveClassifier
from src.models.vision_transformer import vit_giant_xformers_rope

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def video_to_numpy_array(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return None
    
    frames_list = []
    
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        
        # 'ret' is True if the frame was read correctly, False otherwise (e.g., end of video)
        if not ret:
            break
        
        frame = cv2.resize(frame, (256,256))
        # Append the frame (which is already a NumPy array) to the list
        frames_list.append(frame)
        
    # Release the video capture object
    cap.release()
    
    # Convert the list of frames into a single NumPy array
    # The resulting array will have the shape (num_frames, height, width, channels)
    video_array = np.array(frames_list)
    
    return video_array[:20]


def forward_vjepa_video(model_hf, hf_transform):
    # Run a sample inference with VJEPA
    with torch.inference_mode():
        # Read and pre-process the image
        video = video_to_numpy_array('demo_0.mp4')  # T x H x W x C
        video = torch.from_numpy(video).permute(0, 3, 1, 2)  # T x C x H x W
        print(f"Shape of the video: {video.shape}")
        x_hf = hf_transform(video, return_tensors="pt")["pixel_values_videos"].to("cuda")
        # Extract the patch-wise features from the last layer
        out_patch_features_hf = model_hf.get_vision_features(x_hf)

    return out_patch_features_hf

if __name__=='__main__':
    # HuggingFace model repo name
    hf_model_name = (
        "facebook/vjepa2-vitl-fpc64-256"  # Replace with your favored model, e.g. facebook/vjepa2-vitg-fpc64-384
    )
    
    # Initialize the HuggingFace model, load pretrained weights
    model_hf = AutoModel.from_pretrained(hf_model_name)
    model_hf.cuda().eval()
    
    # Build HuggingFace preprocessing transform
    hf_transform = AutoVideoProcessor.from_pretrained(hf_model_name)

    video_embeddings = forward_vjepa_video(model_hf, hf_transform)

    if video_embeddings.dim() == 3 and video_embeddings.shape[0] == 1:
        feature_vector = video_embeddings.squeeze(0).mean(dim=0)
    elif video_embeddings.dim() == 2 and video_embeddings.shape[0] == 1:
        feature_vector = video_embeddings.squeeze(0)

    print(f"Embedding Shape: {video_embeddings.shape}, Feature vector Shape: {feature_vector.shape}")
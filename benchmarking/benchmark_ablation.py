import os
import glob
import h5py
import json
import time
import random
import yaml
import numpy as np
import matplotlib.pyplot as plt
import argparse
from itertools import accumulate
from tqdm import tqdm
from strap.configs.libero_file_functions import get_libero_lang_instruction
from benchmarking.benchmark_models import (
    stumpy_single_matching,
    dtaidistance_single_matching,
    modified_strap_single_matching,
    shaplet_matching,
    llm_matching
)
from sentence_transformers import SentenceTransformer
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.shapelets import LearningShapelets
from benchmarking.benchmark_utils import get_demo_data, process_retrieval_results, transform_series_to_text
from strap.utils.retrieval_utils import segment_trajectory_by_derivative, merge_short_segments




def stumpy_dtaidistance_retrieval(
    target_path, 
    offline_list, 
    output_path, 
    stumpy=False, 
    dtaidistance=False, 
    qwen_embedder=False, 
    llama_embedder=False, 
    gemma_embedder=False,
    TOP_K=150,  # Changed to use max value 
    dataset="libero",
    top_k_range=None  # New parameter: list of TOP_K values to save
):
    """
    Performs retrieval and saves results for multiple TOP_K values in separate files.
    
    Args:
        top_k_range: List of TOP_K values to save (e.g., [50, 70, 90, 110, 130, 150])
                     If None, defaults to [50, 70, 90, 110, 130, 150]
    """
    # Set default TOP_K range if not provided
    if top_k_range is None:
        top_k_range = list(range(50, 151, 20))  # [50, 70, 90, 110, 130, 150]
    
    # Validate that TOP_K is at least the maximum value in the range
    max_top_k = max(top_k_range)
    if TOP_K < max_top_k:
        TOP_K = max_top_k
        print(f"TOP_K adjusted to {max_top_k} to match the maximum in top_k_range")
    
    # Sort top_k_range in descending order for efficient processing
    top_k_range_sorted = sorted(top_k_range, reverse=True)
    
    # Prepare output paths for each TOP_K value
    base_output_path = output_path.replace('.hdf5', '')
    output_paths = {k: f"{base_output_path}_top{k}.hdf5" for k in top_k_range}
    
    # Create directories for all output files
    for path in output_paths.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Initialize embedder model
    embedder_model = None
    if qwen_embedder:
        embedder_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    if llama_embedder:
        embedder_model = SentenceTransformer("nvidia/llama-nemotron-embed-1b-v2", trust_remote_code=True)
    if gemma_embedder:
        embedder_model = SentenceTransformer("google/embeddinggemma-300m")
    
    # Open all output files
    outfiles = {k: h5py.File(output_paths[k], 'w') for k in top_k_range}
    results_groups = {k: outfiles[k].create_group('results') for k in top_k_range}
    
    try:
        with h5py.File(target_path, 'r') as f:
            for episode_key in list(f.keys()):
                # Create episode groups in all output files
                episodes = {k: results_groups[k].create_group(episode_key) for k in top_k_range}
                
                target_series = get_demo_data(f, episode_key)

                # -----For LLM matching-----
                if embedder_model:
                    target_text = transform_series_to_text(target_series[0], dataset=dataset)
                    query_prompt = (
                        "Which multivariate SAX sequences are similar to this?\n"
                        f"{target_text}"
                    )
                    if qwen_embedder:
                        target_embeddings = embedder_model.encode(query_prompt, prompt_name="query")
                    if llama_embedder:
                        target_embeddings = embedder_model.encode_query(query_prompt, convert_to_tensor=True)
                    if gemma_embedder:
                        target_embeddings = embedder_model.encode(query_prompt)
                # ----------------------------
                
                episode_results = []
                for offline_file in tqdm(offline_list, desc=f"Stumpy/Dtaidistance -- Retrieving data for {episode_key}"):
                    with h5py.File(offline_file, 'r') as offline_f:
                        offline_data = offline_f['data']
                        for demo_key in list(offline_data.keys()):
                            offline_series = get_demo_data(offline_data, demo_key)
                            if stumpy:
                                 result = stumpy_single_matching(target_series[0], offline_series, top_k=1)
                            if dtaidistance:
                                 result = dtaidistance_single_matching(target_series[0], offline_series, top_k=1)
                            if qwen_embedder or llama_embedder or gemma_embedder:
                                 result = llm_matching(
                                    target_embeddings, 
                                    target_series[0], 
                                    offline_series, 
                                    top_k=1, 
                                    embedder_model=embedder_model,
                                    model_type="qwen" if qwen_embedder else "llama" if llama_embedder else "gemma",
                                    max_windows=10,
                                    dataset=dataset
                                )
                            for match in result:
                                if match:
                                    episode_results.append({
                                        'cost': match[0],
                                        'start_idx': match[1],
                                        'end_idx': match[2],
                                        'demo_key': demo_key,
                                        'offline_file': offline_file
                                    })
                
                # Process and save results for each TOP_K value
                for k in top_k_range_sorted:
                    processed_results = process_retrieval_results(episode_results, top_k=k)
                    
                    for match_key, data in processed_results.items():
                        match_group = episodes[k].create_group(match_key)
                        match_group.attrs['ep_meta'] = json.dumps({
                                "lang": data['lang_instruction']
                            })
                        match_group.attrs['file_path'] = data['file_path']
                        match_group.attrs['demo_key'] = data['demo_key']
                        for data_key, value in data.items():
                            if data_key not in ['file_path', 'demo_key', 'lang_instruction']:
                                match_group.create_dataset(data_key, data=value)
    
    finally:
        # Close all output files
        for outfile in outfiles.values():
            outfile.close()
    
    print(f"\nRetrieval results saved for TOP_K values: {top_k_range}")
    for k, path in output_paths.items():
        print(f"  TOP_K={k}: {path}")
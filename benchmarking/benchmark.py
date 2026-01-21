import os
import glob
import h5py
import json
import time
import random
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
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
def stumpy_dtaidistance_retrieval(
    target_path, 
    offline_list, 
    output_path, 
    stumpy=False, 
    dtaidistance=False, 
    qwen_embedder=False, 
    llama_embedder=False, 
    gemma_embedder=False,
    TOP_K=100, 
    dataset="libero"
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    embedder_model = None
    if qwen_embedder:
        embedder_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    if llama_embedder:
        embedder_model = SentenceTransformer("nvidia/llama-nemotron-embed-1b-v2", trust_remote_code=True)
    if gemma_embedder:
        embedder_model = SentenceTransformer("google/embeddinggemma-300m")
    with h5py.File(output_path, 'w') as outfile:
        results = outfile.create_group('results')
        with h5py.File(target_path, 'r') as f:
            episodes = next(iter(f.keys()))
            for episode_key in [episodes]:
                episode = results.create_group(episode_key)
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
                processed_results = process_retrieval_results(episode_results, top_k=TOP_K)
                for match_key, data in processed_results.items():
                    match_group = episode.create_group(match_key)
                    match_group.attrs['ep_meta'] = json.dumps({
                            "lang": data['lang_instruction']
                        })
                    match_group.attrs['file_path'] = data['file_path']
                    match_group.attrs['demo_key'] = data['demo_key']
                    for data_key, value in data.items():
                        if data_key not in ['file_path', 'demo_key', 'lang_instruction']:
                            match_group.create_dataset(data_key, data=value)

def modified_strap_retrieval(libero_10_list, libero_90_list, output_file, demo_per_task=1, min_length=60):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with h5py.File(output_file, 'w') as outfile:
        results = outfile.create_group('results')
        episode_id = 0

        # Read all hdf5 file in libero_10 dataset
        for target_file in libero_10_list:
            with h5py.File(target_file, 'r') as target_f:
                target_data = target_f['data']
                demo_list = list(target_data.keys())
                select_demo_keys = random.sample(demo_list, demo_per_task)
                for d_key in select_demo_keys:
                    demo_data = target_data[d_key]
                    ee_pose = demo_data['obs/ee_pos'][:]
                    gripper_states = demo_data['obs/gripper_states'][:]
                    joint_states = demo_data['obs/joint_states'][:]
                    target_series = np.concatenate((ee_pose, gripper_states, joint_states), axis=1)
                    segments = segment_trajectory_by_derivative(ee_pose)
                    merged_segments = merge_short_segments(segments, min_length=min_length)
                    
                    # extract slice indexes
                    seg_idcs = [0] + list(accumulate(len(seg) for seg in merged_segments))
                    for i in range(len(seg_idcs) - 1):
                        episode_results = []
                        episode = results.create_group(f"episode_{episode_id}")
                        start = seg_idcs[i]
                        end = seg_idcs[i+1]

                        # save target data used for retrieval
                        td_grp = episode.create_group("target_data")
                        td_grp.attrs["file_path"] = target_file
                        td_grp.attrs["demo_key"] = d_key
                        td_grp.create_dataset('actions', data=demo_data['obs/ee_pos'][:][start:end])
                        td_grp.create_dataset('obs/ee_pos', data=demo_data['obs/ee_pos'][:][start:end])
                        td_grp.create_dataset('obs/gripper_states', data=demo_data['obs/gripper_states'][:][start:end])
                        td_grp.create_dataset('obs/joint_states', data=demo_data['obs/joint_states'][:][start:end])
                        td_grp.create_dataset('robot_states', data=demo_data['robot_states'][:][start:end])
                        language_instruction = get_libero_lang_instruction(target_f, d_key)
                        td_grp.attrs['ep_meta'] = json.dumps({
                            "lang": language_instruction
                        })

                        # Query used for retrieval
                        query = target_series[start: end]
                        for offline_file in tqdm(libero_90_list, desc=f'Modified STRAP -- Retrieving Data for {episode_id}'):
                            with h5py.File(offline_file, 'r') as offline_f:
                                offline_data = offline_f['data']
                                for demo_key in list(offline_data.keys()):
                                    offline_series = get_demo_data(offline_data, demo_key)
                                    match = modified_strap_single_matching(query, offline_series)
                                    episode_results.append({
                                        'cost': match[0],
                                        'start_idx': match[1],
                                        'end_idx': match[2],
                                        'demo_key': demo_key,
                                        'offline_file': offline_file
                                    })
                        processed_results = process_retrieval_results(episode_results, top_k=100)
                        for match_key, data in processed_results.items():
                            match_group = episode.create_group(match_key)
                            match_group.attrs['ep_meta'] = json.dumps({
                                    "lang": data['lang_instruction']
                                })
                            match_group.attrs['file_path'] = data['file_path']
                            match_group.attrs['demo_key'] = data['demo_key']
                            for data_key, value in data.items():
                                if data_key not in ['file_path', 'demo_key', 'lang_instruction']:
                                    match_group.create_dataset(data_key, data=value)
                        episode_id += 1

def shaplet_retrieval(target_file, offline_list, output_path, 
                      shapelet_model: LearningShapelets, window_size, 
                      stride, TOP_K=100):
    
    with h5py.File(target_file, 'r') as target_f:
        label_names = list(target_f.keys())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, 'w') as outfile:
        results = outfile.create_group('results')
                
        episode_results = [[] for _ in range(len(label_names))]
        for offline_file in tqdm(offline_list, desc=f"Shapelet -- Retrieving data"):
            with h5py.File(offline_file, 'r') as offline_f:
                offline_data = offline_f['data']
                for demo_key in list(offline_data.keys()):
                    offline_series = get_demo_data(offline_data, demo_key)
                    matches = shaplet_matching(offline_series, shapelet_model, window_size, stride)
                    for match in matches:
                        episode_results[match['type']].append({
                            'cost': match['cost'],
                            'start_idx': match['start_idx'],
                            'end_idx': match['end_idx'],
                            'demo_key': demo_key,
                            'offline_file': offline_file
                        })
        for i in range(len(episode_results)):
            episode = results.create_group(label_names[i])
            processed_results = process_retrieval_results(episode_results[i], top_k=TOP_K)
            for match_key, data in processed_results.items():
                match_group = episode.create_group(match_key)
                match_group.attrs['ep_meta'] = json.dumps({
                        "lang": data['lang_instruction']
                    })
                match_group.attrs['file_path'] = data['file_path']
                match_group.attrs['demo_key'] = data['demo_key']
                for data_key, value in data.items():
                    if data_key not in ['file_path', 'demo_key', 'lang_instruction']:
                        match_group.create_dataset(data_key, data=value)

def benchmark_libero(args):
    target_data_path = args.libero_target
    libero_90_dir = args.libero_90_dir
    libero_10_dir = args.libero_10_dir
    libero_10_list = glob.glob(os.path.join(libero_10_dir, "*demo.hdf5"))
    libero_90_list = glob.glob(os.path.join(libero_90_dir, "*demo.hdf5"))
    
    start_time = time.time()
    # modified_strap_retrieval(libero_10_list, libero_90_list, f'{args.output_dir}/libero_retrieval_results_modified_strap.hdf5', 
    #                          demo_per_task=1, min_length=60)
    # stumpy_dtaidistance_retrieval(target_data_path, libero_90_list, f'{args.output_dir}/libero_retrieval_results_stumpy.hdf5', 
    #                               stumpy=True, dtaidistance=False)
    # stumpy_dtaidistance_retrieval(target_data_path, libero_90_list, f'{args.output_dir}/libero_retrieval_results_dtaidistance.hdf5', 
    #                               stumpy=False, dtaidistance=True)
    # stumpy_dtaidistance_retrieval(target_data_path, libero_90_list, f'{args.output_dir}/libero_retrieval_results_qwen.hdf5', 
    #                               stumpy=False, dtaidistance=False, qwen_embedder=True, dataset="libero")
    # stumpy_dtaidistance_retrieval(target_data_path, libero_90_list, f'{args.output_dir}/libero_retrieval_results_llama.hdf5',
    #                               stumpy=False, dtaidistance=False, llama_embedder=True, dataset="libero")
    stumpy_dtaidistance_retrieval(target_data_path, libero_90_list, f'{args.output_dir}/libero_retrieval_results_gemma.hdf5',
                                    stumpy=False, dtaidistance=False, gemma_embedder=True, dataset="libero")
    # load the pre-trained shapelet model
    # shapelet_model = LearningShapelets.from_json('shapelet_libero_model.json')
    # shaplet_retrieval(target_data_path, libero_90_list, f'{args.output_dir}/libero_retrieval_results_shapelet.hdf5',
    #                   shapelet_model, window_size=100, stride=30, TOP_K=100)
    
    end_time = time.time()
    print(f"Total Benchmarking Time for Libero Dataset: {(end_time - start_time)/60:.2f} minutes.")

def benchmark_nuscene(args):
    target_data_path = args.nuscene_target
    offline_data_dir = args.nuscene_offline
    offline_list = glob.glob(os.path.join(offline_data_dir, "*.hdf5"))
    
    start_time = time.time()
    # stumpy_dtaidistance_retrieval(target_data_path, offline_list, f'{args.output_dir}/nuscene_retrieval_results_stumpy.hdf5', 
    #                               stumpy=True, dtaidistance=False, TOP_K=40)
    # stumpy_dtaidistance_retrieval(target_data_path, offline_list, f'{args.output_dir}/nuscene_retrieval_results_dtaidistance.hdf5',
    #                               stumpy=False, dtaidistance=True, TOP_K=40)
    # stumpy_dtaidistance_retrieval(target_data_path, offline_list, f'{args.output_dir}/nuscene_retrieval_results_llm.hdf5',
    #                               stumpy=False, dtaidistance=False, qwen_embedder=True, TOP_K=40, dataset="nuscene")
    # stumpy_dtaidistance_retrieval(target_data_path, offline_list, f'{args.output_dir}/nuscene_retrieval_results_llama.hdf5',
    #                               stumpy=False, dtaidistance=False, llama_embedder=True, TOP_K=40, dataset="nuscene")
    stumpy_dtaidistance_retrieval(target_data_path, offline_list, f'{args.output_dir}/nuscene_retrieval_results_gemma.hdf5',
                                    stumpy=False, dtaidistance=False, gemma_embedder=True, TOP_K=40, dataset="nuscene")
    # load the pre-trained shapelet model
    # shapelet_model = LearningShapelets.from_json('shapelet_nuscene_model.json')
    # shaplet_retrieval(target_data_path, offline_list, f'{args.output_dir}/nuscene_retrieval_results_shapelet.hdf5',
    #                   shapelet_model, window_size=800, stride=200, TOP_K=20)
    
    end_time = time.time()
    print(f"Total Benchmarking Time for Nuscene Dataset: {(end_time - start_time)/60:.2f} minutes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Retrieval Benchmarking')
    parser.add_argument('--libero_target', default="data/target_data/libero_target_dataset.hdf5",
                        help='Path of the target dataset hdf5 file')
    parser.add_argument('--libero_90_dir', default="data/LIBERO/libero_90/",
                        help='Directory of libero_90 datasets')
    parser.add_argument('--libero_10_dir', default="data/LIBERO/libero_10/",
                        help='Directory of libero_10 datasets')
    parser.add_argument('--nuscene_target', default="data/target_data/nuscene_target_dataset.hdf5",
                        help='Path of the target dataset hdf5 file')
    parser.add_argument('--nuscene_offline', default="data/nuscene/",
                        help='Path of the offline dataset hdf5 file')
    parser.add_argument('--output_dir', default="data/retrieval_results/",
                        help='Directory to save retrieval results')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    benchmark_libero(args)
    benchmark_nuscene(args)
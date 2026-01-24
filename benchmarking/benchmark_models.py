import os
import numpy as np
import stumpy
import torch
import torch.nn.functional as F
from dtaidistance.subsequence.dtw import subsequence_search
from strap.utils.retrieval_utils import (
   compute_accumulated_cost_matrix_subsequence_dtw_21,
   compute_optimal_warping_path_subsequence_dtw_21,
   get_distance_matrix
)
from benchmarking.benchmark_utils import transform_series_to_text, pad_zeros_to_length
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.shapelets import LearningShapelets
stumpy.config.STUMPY_EXCL_ZONE_DENOM = np.inf
from momentfm import MOMENTPipeline

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def stumpy_single_matching(query, series, top_k=None, dist_thres=None):
    if query.shape[0] > query.shape[1]:
        query = np.swapaxes(query, 0, 1)
        series = np.swapaxes(series, 0, 1)

    if query.shape[1] > series.shape[1]:
        return []

    if not dist_thres:
        dist_thres = lambda D: max(np.mean(D) - 2 * np.std(D), np.min(D))

    matches = stumpy.match(query, series, max_matches=top_k, max_distance=dist_thres)
    result = [[] for _ in range(len(matches))]
    for i, match in enumerate(matches):
        result[i].append(match[0])
        result[i].append(int(match[1]))
        result[i].append(int(match[1] + query.shape[1]))

    return result

def dtaidistance_single_matching(query, series, top_k=None, dist_thres=None):
    if query.ndim == 3:
        query = query[0]
    if series.ndim == 3:
        series = series[0]
    w = query.shape[0]
    stride = int(np.floor(w / 2))
    if stride < 1:
        stride = 1
    wn = int(np.floor((len(series) - (w - stride)) / stride))
    if not top_k:
        top_k = wn

    s = []
    start_indexes = []
    si, ei = 0, w
    for i in range(wn):
        s.append(series[si:ei])
        start_indexes.append(si)
        si += stride
        ei += stride

    sa = subsequence_search(query, s, max_dist=dist_thres, use_lb=False)
    best = sa.kbest_matches(k=top_k)

    result = [[] for _ in range(len(best))]
    for i, match in enumerate(best):
        result[i].append(match.distance)
        result[i].append(start_indexes[match.idx])
        result[i].append(start_indexes[match.idx] + w)

    return result

def modified_strap_single_matching(query, series):
    distance_matrix = get_distance_matrix(query, series)
    accumulated_cost_matrix = compute_accumulated_cost_matrix_subsequence_dtw_21(
        distance_matrix
    )
    path = compute_optimal_warping_path_subsequence_dtw_21(accumulated_cost_matrix)
    start = path[0, 1]
    if start < 0:
        assert start == -1
        start = 0
    end = path[-1, 1]
    cost = accumulated_cost_matrix[-1, end]
    # Note that the actual end index is inclusive in this case so +1 to use python : based indexing
    end = end + 1

    return [cost, start, end]

def shaplet_matching(series, shapelet_model: LearningShapelets, window_size, stride):
    L, D = series.shape
    window_starts = np.arange(0, L - window_size + 1, stride)
    
    candidates = []
    for i in range(0, len(window_starts)):
        start_frame = int(window_starts[i])
        end_frame = int(start_frame + window_size)
        window = series[start_frame:end_frame, :]
        window = TimeSeriesScalerMinMax().fit_transform(np.expand_dims(window, axis=0))
        
        probability = shapelet_model.predict_proba(window)
        match = np.argmax(probability)

        # Store candidate if distance is below a threshold (optional, but good practice)
        # Here, we keep all and let NMS filter
        candidates.append({
            "cost": -probability[0][match],
            "start_idx": start_frame,
            "end_idx": end_frame,
            "demo_key": None,
            "type": match,
        })
    return candidates

def llm_matching(query_embedding, query, series, top_k=None, embedder_model=None, model_type="qwen", max_windows=10, dataset="libero"):
    """
    Sliding-window matching with dynamically adapted stride so that
    at most `max_windows` windows are evaluated.
    """
    query_length = query.shape[0]
    series_length = series.shape[0]

    # Correct window count formula
    max_possible_windows = series_length - query_length + 1
    if query_length > series_length or max_possible_windows <= 0:
        return []

    stride = int(np.ceil(max(1, max_possible_windows / max_windows)))

    start_indices = list(range(0, max_possible_windows, stride))
    start_indices = start_indices[:max_windows]

    all_series = [transform_series_to_text(series[i:i + query_length], dataset=dataset) for i in start_indices]
    if model_type == "qwen" or model_type == "gemma":
        all_series_embeddings = embedder_model.encode(all_series)  # (W, D)
    elif model_type == "llama":
        all_series_embeddings = embedder_model.encode_document(all_series, convert_to_tensor=True)  # (W, D)

    cosine_sim_matrix = embedder_model.similarity(query_embedding, all_series_embeddings)  # (1, W)
    similarities = cosine_sim_matrix[0]
    costs = 1.0 - similarities

    window_indices = [(i, i + query_length) for i in start_indices]

    if top_k is None or top_k > len(costs):
        top_k = len(costs)

    top_k_indices = np.argsort(costs)[:top_k]

    results = []
    for idx in top_k_indices:
        start_idx, end_idx = window_indices[idx]
        results.append([costs[idx], start_idx, end_idx])
    print("Finished results...")
    return results

def momentfm_matching(query_embedding, series, momentfm_model: MOMENTPipeline, query_length):
    L, D = series.shape
    window_size = query_length
    stride = int(np.floor(window_size / 4))
    window_starts = np.arange(0, L - window_size + 1, stride)
    
    candidates = []
    for i in range(0, len(window_starts)):
        start_frame = int(window_starts[i])
        end_frame = int(start_frame + window_size)
        window = series[start_frame:end_frame, :]
        window, pad_mask = pad_zeros_to_length(window, 512)
        window = window.to(DEVICE)
        pad_mask = pad_mask.to(DEVICE)

        series_embedding = momentfm_model(x_enc=window, input_mask=pad_mask)
        similarity = F.cosine_similarity(query_embedding.embeddings, series_embedding.embeddings, dim=-1)

        candidates.append([1.0 - similarity.item(), start_frame, end_frame])
    return candidates
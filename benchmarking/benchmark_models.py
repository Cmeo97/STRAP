import os
import numpy as np
import stumpy
from dtaidistance.subsequence.dtw import subsequence_search
from strap.utils.retrieval_utils import (
   compute_accumulated_cost_matrix_subsequence_dtw_21,
   compute_optimal_warping_path_subsequence_dtw_21,
   get_distance_matrix
)
from benchmarking.benchmark_utils import transform_series_to_text
stumpy.config.STUMPY_EXCL_ZONE_DENOM = np.inf


def stumpy_single_matching(query, series, top_k=None, dist_thres = None):
  # stumpy requires query and series is of dimension x length shape
  if query.shape[0] > query.shape[1]:
    query = np.swapaxes(query, 0, 1)
    series = np.swapaxes(series, 0, 1)

  if query.shape[1] > series.shape[1]:
    return []

  if not dist_thres:
    dist_thres = lambda D: max(np.mean(D) - 2 * np.std(D), np.min(D))

  matches = stumpy.match(query, series, max_matches=top_k,
                      max_distance=dist_thres
                      )
  result = [[] for _ in range(len(matches))]
  for i, match in enumerate(matches):
    result[i].append(match[0])
    result[i].append(int(match[1]))
    result[i].append(int(match[1] + query.shape[1]))

  return result

def dtaidistance_single_matching(query, series, top_k=None, dist_thres = None):
  s = []
  w = query.shape[0]
  stride = int(np.floor(w/2))
  wn = int(np.floor((len(series) - (w - stride)) / stride))
  if not top_k:
    top_k = wn

  start_indexes = []
  si, ei = 0, w
  for i in range(wn):
      s.append(series[si:ei])
      si += stride
      ei += stride
      start_indexes.append(si)

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


def llm_matching(query, series, top_k=None, embedder_model=None, use_sax=False):
    """
    Perform sliding window over `series` (the document), transform each window to text for embedding,
    and match with the query's embedding. Returns a list of [cost, start_idx, end_idx] for the top_k best matches.
    The start_idx and end_idx returned are from the series, that is, the indices in the series that are cutoff by the sliding window.
    """
    query_length = query.shape[0]
    total_windows = series.shape[0] - query_length + 1
    if query_length > series.shape[0] or total_windows <= 0:
        return []

    query_text = transform_series_to_text(query, use_sax=use_sax)
    all_series = [transform_series_to_text(series[i:i+query_length], use_sax=use_sax) for i in range(total_windows)]

    query_embedding = embedder_model.encode([query_text])      # shape (1, D)
    all_series_embeddings = embedder_model.encode(all_series)  # shape (N, D)

    cosine_sim_matrix = embedder_model.match(query_embedding, all_series_embeddings)  # shape (1, N)
    similarities = cosine_sim_matrix[0]  # shape (N,)
    costs = 1.0 - similarities  # shape (N,)

    window_indices = [(i, i + query_length) for i in range(total_windows)]

    # Select the top_k lowest costs (= highest cosine similarities)
    if top_k is None or top_k > len(costs):
        top_k = len(costs)
    top_k_indices = np.argsort(costs)[:top_k]

    results = []
    for idx in top_k_indices:
        start_idx, end_idx = window_indices[idx]
        cost = costs[idx]
        results.append([cost, start_idx, end_idx])

    return results
import os
import numpy as np
import stumpy
from dtaidistance.subsequence.dtw import subsequence_search

stumpy.config.STUMPY_EXCL_ZONE_DENOM = np.inf

def stumpy_matching(query, series, top_k=None, dist_thres = None):

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

def dtaidistance_matching(query, series, top_k=None, dist_thres = None):
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

def llm_matching(query, series, top_k=None):
    # Placeholder for LLM-based matching implementation
    # This function should interface with a language model to perform matching
    raise NotImplementedError("LLM-based matching is not implemented yet.")
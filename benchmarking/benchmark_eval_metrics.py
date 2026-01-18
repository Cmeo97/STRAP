import numpy as np
from typing import Dict, Any
from scipy.stats import wasserstein_distance
from scipy.signal import correlate
from scipy.spatial.distance import cdist
from dtaidistance import dtw_ndim
import os
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

class UnsupervisedRetrievalEvaluator:
    """
    Unsupervised evaluation for multivariate time-series retrieval.

    All inputs are raw NumPy arrays.

    retrieved: (N_r, T, F)
    reference: (N_ref, T, F)
    """

    def __init__(self, retrieved: np.ndarray, reference: np.ndarray, viz_dir: str = None):
        """
        Args:
            retrieved (np.ndarray): (N_r, T, F)
            reference (np.ndarray): (N_ref, T, F)
            viz_dir (str, optional): directory for distributional checker visualizations; can be None if not used
        """
        retrieved = np.asarray(retrieved, dtype=np.float64)
        reference = np.asarray(reference, dtype=np.float64)

        if retrieved.ndim != 3 or reference.ndim != 3:
            raise ValueError("Inputs must be rank-3 arrays (N, T, F)")

        if retrieved.shape[1:] != reference.shape[1:]:
            raise ValueError("Retrieved and reference must share (T, F)")

        self.retrieved = retrieved
        self.reference = reference

        self.N_r, self.T, self.F = retrieved.shape
        self.N_ref = reference.shape[0]
        self.viz_dir = viz_dir

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flatten(self, x: np.ndarray) -> np.ndarray:
        """
        (N, T, F) -> (N, T*F)
        """
        return x.reshape(x.shape[0], -1)

    def _flatten_2d(self, x: np.ndarray) -> np.ndarray:
        """
        (N, T, F) -> (N*T, F)
        """
        return x.reshape(-1, x.shape[-1])

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def wasserstein(self) -> float:
        """
        Feature-wise 1-Wasserstein distance over marginal distributions.

        Operates on flattened samples:
        (N*T, F)
        """
        r = self.retrieved.reshape(-1, self.F)
        ref = self.reference.reshape(-1, self.F)

        result = float(np.mean([
            wasserstein_distance(r[:, f], ref[:, f])
            for f in range(self.F)
        ]))
        return result

    def dtw_nn(self) -> float:
        """
        Nearest-neighbor multivariate DTW (C-optimized).

        For each retrieved trajectory, compute minimum DTW distance
        to any reference trajectory.
        """
        dists = []

        # Ensure contiguous float64 for C backend
        reference = [np.ascontiguousarray(ref, dtype=np.float64) for ref in self.reference]

        for r in self.retrieved:
            r = np.ascontiguousarray(r, dtype=np.float64)
            best = np.inf
            for ref in reference:
                d = dtw_ndim.distance_fast(r, ref)  # C optimization
                if d < best:
                    best = d
            dists.append(best)

        result = float(np.mean(dists))
        return result


    def spectral_wasserstein(self, eps: float = 1e-8) -> float:
        """
        Power spectral density comparison via Wasserstein distance.

        PSD computed per feature, averaged over dataset.
        """
        def avg_psd(x: np.ndarray) -> np.ndarray:
            # x: (N, T, F) -> (F, T)
            fft = np.fft.fft(x, axis=1)
            psd = np.abs(fft) ** 2
            return psd.mean(axis=0).T

        p_ret = avg_psd(self.retrieved)
        p_ref = avg_psd(self.reference)

        dists = []
        for f in range(self.F):
            p = p_ret[f]
            q = p_ref[f]

            p = p / (p.sum() + eps)
            q = q / (q.sum() + eps)

            dists.append(wasserstein_distance(p, q))

        result = float(np.mean(dists))
        return result

    def temporal_correlation(self) -> float:
        """
        Max normalized cross-correlation, averaged over
        all (retrieved, reference, feature) tuples.
        """
        vals = []

        for f in range(self.F):
            for r in self.retrieved:
                r_f = r[:, f]
                r_norm = np.linalg.norm(r_f) + 1e-8

                for ref in self.reference:
                    ref_f = ref[:, f]
                    ref_norm = np.linalg.norm(ref_f) + 1e-8

                    c = correlate(r_f, ref_f, mode="valid")
                    vals.append(np.max(c) / (r_norm * ref_norm))

        result = float(np.mean(vals))
        return result

    def distributional_coverage(self, k: int = 5) -> Dict[str, float]:
        """
        Density and coverage metrics from:
        Naeem et al., "Reliable Fidelity and Diversity Metrics for Generative Models" (ICML 2020).

        Inputs:
            retrieved: (M, T, F)
            reference: (N, T, F)

        Operates in flattened feature space.
        """
        r = self._flatten(self.retrieved)   # (M, D)
        ref = self._flatten(self.reference) # (N, D)

        M, N = r.shape[0], ref.shape[0]

        # --- k-NN radii for reference samples ---
        d_ref_ref = cdist(ref, ref)
        np.fill_diagonal(d_ref_ref, np.inf)
        ref_radii = np.partition(d_ref_ref, k, axis=1)[:, k]  # (N,)

        # --- cross distances ---
        d_ref_r = cdist(ref, r)  # (N, M)

        # =====================
        # Density
        # =====================
        # For each retrieved sample, count how many real neighborhoods contain it
        density = (
            (d_ref_r <= ref_radii[:, None]).sum(axis=0).mean() / k
        )

        # =====================
        # Coverage
        # =====================
        # Fraction of real samples that lie in at least one retrieved neighborhood

        coverage = (d_ref_r <= ref_radii[:, None]).any(axis=1).mean()


        return {
            "density": float(density),
            "coverage": float(coverage),
        }

    def diversity_icd(
        self,
        window: int | None = None,
    ) -> float:
        """
        Intra-set DTW diversity over retrieved samples.

        Optimized version:
        - Uses C-optimized distance_matrix_fast
        - Computes all pairwise distances once
        - Aggregates upper triangle only

        Args:
            window: Sakoe-Chiba window (int). If None, full DTW.

        Returns:
            Mean pairwise DTW distance.
        """
        X = np.asarray(self.retrieved, dtype=np.float64)

        # Compute full distance matrix (NxN)
        D = dtw_ndim.distance_matrix_fast(
            X,
            window=window,
        )

        # Extract upper triangle without diagonal
        iu = np.triu_indices(D.shape[0], k=1)
        result = float(D[iu].mean())

        return result

    def distributional_checker(self, episode: str = None):
        """
        Generates a visualization of the flattened feature marginals
        comparing retrieved and reference distributions.
        The visualization is output in self.viz_dir/episode.png (NOT a subdirectory).
        Args:
            episode (str): episode name to use as filename in output path.
        """

        if self.viz_dir is None:
            raise ValueError("viz_dir must be specified for distributional_checker")

        os.makedirs(self.viz_dir, exist_ok=True)

        r = self._flatten_2d(self.retrieved)  # (N_r * T, F)
        ref = self._flatten_2d(self.reference)  # (N_ref * T, F)

        F = r.shape[1]
        selected_feats = list(range(F))

        # Set up a 4 by 3 grid
        n_rows, n_cols = 4, 3
        total_plots = n_rows * n_cols
        # Cap at F features shown, but still set grid up to 3x4 in any case
        feats_to_show = min(F, total_plots)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 12))
        axs = axs.flatten()

        for idx in range(total_plots):
            ax = axs[idx]
            if idx < F:
                f = selected_feats[idx]
                ax.hist(ref[:, f], bins=80, alpha=0.65, color="tab:blue", label="Reference", density=True)
                ax.hist(r[:, f], bins=80, alpha=0.65, color="tab:orange", label="Retrieved", density=True)
                ax.set_title(f"Feature {f} marginal")
                ax.legend()
            else:
                # Hide unused subplot axes
                ax.axis('off')
        plt.tight_layout()

        # Output as viz_dir/episode.png if episode is given, else viz_dir/distributional_checker.png
        if episode is not None:
            out_path = os.path.join(self.viz_dir, f"{episode}.png")
        else:
            out_path = os.path.join(self.viz_dir, "distributional_checker.png")
        plt.savefig(out_path)
        plt.close()

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    # Parallel evaluator using ThreadPoolExecutor

    class _MetricJobRunner:
        def __init__(self, evaluator: "UnsupervisedRetrievalEvaluator"):
            self.evaluator = evaluator
            # (name, fn) pairs
            self.metrics = [
                ("wasserstein", evaluator.wasserstein),
                ("dtw_nn", evaluator.dtw_nn),
                ("spectral_wasserstein", evaluator.spectral_wasserstein),
                ("temporal_correlation", evaluator.temporal_correlation),
                ("distributional_coverage", evaluator.distributional_coverage),
                ("diversity_icd", evaluator.diversity_icd),
            ]

        def run_parallel(self, max_workers: int = None) -> Dict[str, Any]:
            results: Dict[str, Any] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_metric = {
                    executor.submit(fn): name for name, fn in self.metrics
                }
                for future in as_completed(future_to_metric):
                    name = future_to_metric[future]
                    try:
                        res = future.result()
                        # Handle dict-valued metric (distributional_coverage)
                        if isinstance(res, dict):
                            for k, v in res.items():
                                results[f"{name}.{k}"] = v
                        else:
                            results[name] = res
                    except Exception:
                        results[name] = None
            return results

    def evaluate(self, parallel: bool = False, max_workers: int = None) -> Dict[str, float]:
        """
        Run all metrics serially (default) or concurrently (parallel=True).
        """
        if parallel:
            runner = self._MetricJobRunner(self)
            return runner.run_parallel(max_workers=max_workers)
        else:
            res = {
                "wasserstein": self.wasserstein(),
                "dtw_nn": self.dtw_nn(),
                "spectral_wasserstein": self.spectral_wasserstein(),
                "temporal_correlation": self.temporal_correlation(),
                "distributional_coverage": self.distributional_coverage(),
                "diversity_icd": self.diversity_icd(),
            }
            # If any values are dicts (e.g., distributional_coverage), flatten them
            flat_results = {}
            for k, v in res.items():
                if isinstance(v, dict):
                    for subk, subv in v.items():
                        flat_results[f"{k}.{subk}"] = subv
                else:
                    flat_results[k] = v
            return flat_results

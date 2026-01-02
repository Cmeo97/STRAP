import os
import json
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cdist
from scipy.signal import correlate
from dtaidistance import dtw_ndim
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE


class UnsupervisedRetrievalEvaluator:
    def __init__(self, retrieved_data: np.ndarray, reference_data: np.ndarray, results_dir="results"):
        """
        retrieved_data: (N_r, T, F)
        reference_data: (N_ref, T, F)
        """
        self.retrieved = retrieved_data
        self.reference = reference_data

        self.results_dir = results_dir
        self.vis_dir = os.path.join(results_dir, "visualizations")

        os.makedirs(self.vis_dir, exist_ok=True)

        self.N_r, self.T, self.F = self.retrieved.shape
        self.N_ref = self.reference.shape[0]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _flatten_time(self, x):
        return x.reshape(x.shape[0], -1)

    def _summary_features(self, x):
        """
        Flatten input so that the output is (num_samples * length_of_ts, num_features).
        """
        num_samples, length_of_ts, num_features = x.shape
        return x.reshape(num_samples * length_of_ts, num_features)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def wasserstein_distance(self):
        """
        Computes the (average) 1-Wasserstein distance (Earth Mover's Distance) between distributions, feature-wise.
        Uses the scipy.stats.wasserstein_distance for each feature, then averages.
        """
        ret_flat = self.retrieved.reshape(-1, self.F)
        ref_flat = self.reference.reshape(-1, self.F)

        distances = []
        for f in range(self.F):
            d = wasserstein_distance(ret_flat[:, f], ref_flat[:, f])
            distances.append(d)
        return float(np.mean(distances))

    def dynamic_time_warping(self):
        """
        Dataset-level temporal similarity using nearest-neighbor multivariate DTW.

        Each retrieved trajectory is matched to the closest reference trajectory
        in DTW distance. Preserves temporal structure and dynamics.
        """
        distances = []

        for r in self.retrieved:
            r = np.asarray(r, dtype=np.float64)
            min_dist = np.inf
            for ref in self.reference:
                ref = np.asarray(ref, dtype=np.float64)
                dist = dtw_ndim.distance(r, ref, use_c=True, psi=0)
                if dist < min_dist:
                    min_dist = dist
            distances.append(min_dist)
        return float(np.mean(distances))

    def psd_spectral_distance(self, eps=1e-8):
        """
        Computes a temporal dynamics metric via power spectral density (PSD) comparison.
        """

        N_r, T_r, F = self.retrieved.shape
        N_s, T_s, F_ref = self.reference.shape

        def compute_avg_psd(data):
            """
            data: (N, T, F)
            returns: avg PSD per feature: (F, T)
            """
            N, T, F = data.shape
            avg_psd = np.zeros((F, T), dtype=np.float64)
            for f in range(F):
                psd_sum = np.zeros(T, dtype=np.float64)
                for i in range(N):
                    fft_vals = np.fft.fft(data[i, :, f])
                    psd_sum += np.abs(fft_vals) ** 2
                avg_psd[f] = psd_sum / N
            return avg_psd

        avg_psd_ret = compute_avg_psd(self.retrieved)
        avg_psd_ref = compute_avg_psd(self.reference)

        distances = []
        for f in range(F):
            p = avg_psd_ret[f]
            q = avg_psd_ref[f]

            p /= p.sum() + eps
            q /= q.sum() + eps

            distances.append(wasserstein_distance(p, q))
        return float(np.mean(distances))

    def temporal_cross_correlation(self):
        """
        Average max normalized cross-correlation per feature.
        """
        corrs = []
        for f in range(self.F):
            for r in self.retrieved:
                for ref in self.reference:
                    c = correlate(r[:, f], ref[:, f], mode="valid")
                    corrs.append(np.max(c) / (np.linalg.norm(r[:, f]) * np.linalg.norm(ref[:, f]) + 1e-8))
        return float(np.mean(corrs))


    def distributional_coverage(self, k=5):
        ref = self.reference
        ret = self.retrieved
        num_ref = ref.shape[0]
        num_ret = ret.shape[0]

        # --- Precompute k-NN radii ---
        ref_radii = [
            np.max(np.partition(pairwise_distances(seq, seq, n_jobs=-1), k+1, axis=1)[:, :k+1], axis=1)
            for seq in ref
        ]
        ret_radii = [
            np.max(np.partition(pairwise_distances(seq, seq, n_jobs=-1), k+1, axis=1)[:, :k+1], axis=1)
            for seq in ret
        ]

        precision_list, recall_list, density_list, coverage_list = [], [], [], []

        # --- Only compute cross distances per pair ---
        for i in range(num_ref):
            ref_seq = ref[i]
            ref_r = ref_radii[i]

            for j in range(num_ret):
                ret_seq = ret[j]
                ret_r = ret_radii[j]

                dist_ref_ret = pairwise_distances(ref_seq, ret_seq, n_jobs=-1)

                precision = (dist_ref_ret < ref_r[:, None]).any(axis=0).mean()
                recall = (dist_ref_ret < ret_r[None, :]).any(axis=1).mean()
                density = (1.0 / k * (dist_ref_ret < ref_r[:, None]).sum(axis=0)).mean()
                coverage = (dist_ref_ret.min(axis=1) < ref_r).mean()

                precision_list.append(precision)
                recall_list.append(recall)
                density_list.append(density)
                coverage_list.append(coverage)

        metrics = {}
        for name, values in zip(
            ["precision", "recall", "density", "coverage"],
            [precision_list, recall_list, density_list, coverage_list]
        ):
            values = np.array(values)
            metrics[name] = {"mean": values.mean(), "std": values.std()}

        return metrics


    def diversity_icd(self):
        """
        Computes DTW-based Intra-Class Distance (ICD) for multivariate time series
        using dtw_ndim.

        Args:
            retrieved: np.ndarray [num_samples, seq_len, num_features]

        Returns:
            float: ICD (average pairwise DTW distance; higher = more diverse)
        """
        num_samples = self.retrieved.shape[0]

        # Initialize pairwise distance matrix
        dists = np.zeros((num_samples, num_samples), dtype=np.float32)

        # Compute distances for all pairs (i<j) using dtw_ndim
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                d = dtw_ndim.distance(self.retrieved[i], self.retrieved[j])
                dists[i, j] = d
                dists[j, i] = d

        # Average of all distinct pairwise distances (exclude diagonal)
        icd = dists.sum() / (num_samples * (num_samples - 1))
        return float(icd)

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------
    def tsne_visualization(self):
        feats_r = self._summary_features(self.retrieved)
        feats_ref = self._summary_features(self.reference)

        X = np.concatenate([feats_r, feats_ref], axis=0)
        y = np.array([0] * feats_r.shape[0] + [1] * feats_ref.shape[0])

        tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=42)
        X_emb = tsne.fit_transform(X)

        plt.figure(figsize=(6, 6))
        plt.scatter(X_emb[y == 0, 0], X_emb[y == 0, 1], label="Retrieved", alpha=0.6)
        plt.scatter(X_emb[y == 1, 0], X_emb[y == 1, 1], label="Reference", alpha=0.6)
        plt.legend()
        plt.title("t-SNE Summary Feature Space")
        plt.savefig(os.path.join(self.vis_dir, "tsne.png"))
        plt.close()

    def distribution_plot(self):
        for f in range(self.F):
            retrieved_data = self.retrieved[:, :, f].reshape(-1)
            reference_data = self.reference[:, :, f].reshape(-1)

            plt.figure(figsize=(6, 4))

            # Compute histogram bins jointly for fairness
            all_data = np.concatenate([retrieved_data, reference_data])
            bins = np.histogram_bin_edges(all_data, bins=50)

            # Plot normalized (density=True) histograms
            plt.hist(retrieved_data, bins=bins, alpha=0.7, label="Retrieved", density=True)
            plt.hist(reference_data, bins=bins, alpha=0.7, label="Reference", density=True)
            plt.title(f"Feature {f} Distribution (Normalized)")
            plt.legend()
            plt.savefig(os.path.join(self.vis_dir, f"distribution_feature_{f}.png"))
            plt.close()

    # ------------------------------------------------------------------
    # Full Analysis
    # ------------------------------------------------------------------
    def full_analysis(self):
        results = {}

        results["wasserstein_distance"] = self.wasserstein_distance()
        results["dtw"] = self.dynamic_time_warping()
        results["psd_spectral_distance"] = self.psd_spectral_distance()
        results["temporal_cross_correlation"] = self.temporal_cross_correlation()
        results["distributional_coverage"] = self.distributional_coverage()
        results["diversity_icd"] = self.diversity_icd()

        self.tsne_visualization()
        self.distribution_plot()

        with open(os.path.join(self.results_dir, "metrics.json"), "w") as f:
            json.dump(results, f, indent=2)

        return results

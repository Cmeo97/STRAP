# unsupervised_retrieval_metrics.py

"""
Evaluator for unsupervised retrieval of multivariate embedding sequences.
All metrics operate on embedding sequences directly.
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from scipy.stats import skew, kurtosis
from dtaidistance import dtw
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class UnsupervisedRetrievalEvaluator:
    """
    Evaluates retrieval quality for embedding sequences.
    """

    def __init__(self, retrieved_embeddings: np.ndarray, reference_embeddings: np.ndarray):
        """
        Args:
            retrieved_embeddings: (num_retrieved, embedding_dim) or (sequence_length, embedding_dim)
            reference_embeddings: (num_reference, embedding_dim) or (sequence_length, embedding_dim)
        """
        self.retrieved = retrieved_embeddings
        self.reference = reference_embeddings

    # -------------------------
    # Distributional Metrics
    # -------------------------
    def wasserstein_distance(self):
        """1D Wasserstein distance per dimension (mean over all dimensions)."""
        r, ref = self.retrieved, self.reference
        n_r, n_ref = r.shape[0], ref.shape[0]
        dists = []
        for i in range(r.shape[1]):
            s_r, s_ref = np.sort(r[:, i]), np.sort(ref[:, i])
            if n_r != n_ref:
                # Linear interpolation to match sizes
                if n_r > n_ref:
                    x = np.linspace(0, 1, n_ref)
                    xi = np.linspace(0, 1, n_r)
                    s_ref = np.interp(xi, x, s_ref)
                else:
                    x = np.linspace(0, 1, n_r)
                    xi = np.linspace(0, 1, n_ref)
                    s_r = np.interp(xi, x, s_r)
            dists.append(np.mean(np.abs(s_r - s_ref)))
        return np.mean(dists)

    def mahalanobis_distance(self):
        """Mahalanobis distance between mean embeddings."""
        mu_r, mu_ref = self.retrieved.mean(axis=0), self.reference.mean(axis=0)
        cov_ref = np.cov(self.reference, rowvar=False)
        cov_inv = np.linalg.pinv(cov_ref)
        diff = mu_r - mu_ref
        return np.sqrt(diff.T @ cov_inv @ diff)

    # -------------------------
    # Coverage & Redundancy
    # -------------------------
    def coverage_ratio(self, k=1, eps=0.1):
        """Fraction of reference embeddings covered by retrieved embeddings within eps."""
        nbrs = NearestNeighbors(n_neighbors=k).fit(self.retrieved)
        distances, _ = nbrs.kneighbors(self.reference)
        covered = np.sum(np.any(distances < eps, axis=1))
        return covered / len(self.reference)

    def redundancy_ratio(self, threshold=1e-3):
        """Fraction of retrieved embeddings that are duplicates (distance below threshold)."""
        pairwise = cdist(self.retrieved, self.retrieved, metric='euclidean')
        np.fill_diagonal(pairwise, np.inf)
        duplicates = np.sum(pairwise < threshold)
        return duplicates / (len(self.retrieved) ** 2)

    def knn_precision_recall_f1(self, k=1, eps=0.1):
        """Threshold-based k-NN Precision, Recall, and F1 using embeddings."""
        nbrs_ref = NearestNeighbors(n_neighbors=k).fit(self.reference)
        distances, _ = nbrs_ref.kneighbors(self.retrieved)
        precision = np.mean(np.any(distances < eps, axis=1))

        nbrs_ret = NearestNeighbors(n_neighbors=k).fit(self.retrieved)
        distances, _ = nbrs_ret.kneighbors(self.reference)
        recall = np.mean(np.any(distances < eps, axis=1))

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    # -------------------------
    # Temporal/Sequence-like Metrics
    # -------------------------
    def dtw_distance(self):
        """
        DTW distance applied over embedding sequences.
        If embeddings are multivariate, sum DTW over dimensions.
        """
        r, ref = self.retrieved, self.reference
        if r.ndim == 1:
            return dtw.distance(r, ref)
        return sum(dtw.distance(r[:, d], ref[:, d]) for d in range(r.shape[1]))

    def temporal_cross_correlation(self):
        """Average cross-correlation over all dimensions of embeddings."""
        r, ref = self.retrieved, self.reference
        min_len = min(r.shape[0], ref.shape[0])
        r_cut, ref_cut = r[:min_len], ref[:min_len]
        corrs = [np.corrcoef(r_cut[:, d], ref_cut[:, d])[0, 1] for d in range(r_cut.shape[1])]
        return np.mean(corrs)

    # -------------------------
    # Visualization & Stats
    # -------------------------
    def visualize(self, bins=30, save_dir="visualizations"):
        """
        Histogram distributions and embedding statistics, saved to disk.

        Creates:
        - Distribution plot: histograms of the marginal distribution (all values flattened) for retrieved vs reference embeddings
        - t-SNE plot: 2D projection of embeddings (use feature dimension directly if multivariate), then reports mean, std, skewness, kurtosis over the 2D projected samples
        """

        os.makedirs(save_dir, exist_ok=True)

        # --- Marginal Distribution plot: flatten all values ---
        flat_retrieved = self.retrieved.flatten()
        flat_reference = self.reference.flatten()

        dist_path = os.path.join(save_dir, "distribution_plot_stats")
        os.makedirs(dist_path, exist_ok=True)

        hist_retrieved, bin_edges = np.histogram(flat_retrieved, bins=bins, density=True)
        hist_reference, _ = np.histogram(flat_reference, bins=bin_edges, density=True)

        plt.figure(figsize=(8, 4))
        plt.bar(bin_edges[:-1], hist_reference, width=np.diff(bin_edges), alpha=0.5, label="Reference")
        plt.bar(bin_edges[:-1], hist_retrieved, width=np.diff(bin_edges), alpha=0.5, label="Retrieved")
        plt.title("Marginal Value Distribution (all values)")
        plt.xlabel("Embedding Value")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        dist_plot_path = os.path.join(dist_path, "distribution.png")
        plt.savefig(dist_plot_path)
        plt.close()

        # --- t-SNE 2D projection then compute stats on the 2D projections ---
        emb_all = np.vstack([self.reference, self.retrieved])
        labels = np.array([0]*len(self.reference) + [1]*len(self.retrieved))  # 0=ref, 1=retrieved

        reducer = TSNE(n_components=2, random_state=42, perplexity=20)
        embedding_2d = reducer.fit_transform(emb_all)

        plt.figure(figsize=(8, 6))
        plt.scatter(embedding_2d[labels == 0, 0], embedding_2d[labels == 0, 1], alpha=0.7, label="Reference")
        plt.scatter(embedding_2d[labels == 1, 0], embedding_2d[labels == 1, 1], alpha=0.7, label="Retrieved")
        plt.title("Embedding t-SNE Projection")
        plt.xlabel("t-SNE-1")
        plt.ylabel("t-SNE-2")
        plt.legend()
        tsne_path = os.path.join(save_dir, "tsne_projection")
        os.makedirs(tsne_path, exist_ok=True)
        tsne_plot_path = os.path.join(tsne_path, "tsne.png")
        plt.savefig(tsne_plot_path)
        plt.close()

        # Compute statistics (mean, std, skew, kurtosis) over the t-SNE 2d embeddings, separately for ref and retrieved
        from scipy.stats import skew, kurtosis

        emb2d_ref = embedding_2d[labels == 0]
        emb2d_ret = embedding_2d[labels == 1]

        # Compute for axis 1 (features=2) across all samples
        def tsne_stats(arr):
            means = np.mean(arr, axis=0)
            stds = np.std(arr, axis=0)
            skews = skew(arr, axis=0)
            kurts = kurtosis(arr, axis=0)
            return {"mean": means, "std": stds, "skewness": skews, "kurtosis": kurts}

        tsne_stats_retrieved = tsne_stats(emb2d_ret)
        tsne_stats_reference = tsne_stats(emb2d_ref)

        stat_names = ["mean", "std", "skewness", "kurtosis"]

        return {
            "hist_retrieved": hist_retrieved,
            "hist_reference": hist_reference,
            "bin_edges": bin_edges,
            "stat_names": ["Marginal Value Distribution"],
            "distribution_plot_path": dist_plot_path,
            "tsne_plot_path": tsne_plot_path,
            "tsne_stats_retrieved": tsne_stats_retrieved,
            "tsne_stats_reference": tsne_stats_reference,
            "tsne_stat_names": stat_names
        }
    
    def full_analysis(self, eps=0.1, save_visualizations=True):
        """
        Compute all metrics on embedding sequences and optionally save visualization plots.
        """
        results = {
            "wasserstein": self.wasserstein_distance(),
            "mahalanobis": self.mahalanobis_distance(),
            "coverage_ratio": self.coverage_ratio(eps=eps),
            "redundancy_ratio": self.redundancy_ratio(),
            "knn_precision_recall_f1": self.knn_precision_recall_f1(eps=eps),
            "dtw_distance": self.dtw_distance(),
            "temporal_cross_correlation": self.temporal_cross_correlation(),
        }

        # Save plots and include paths
        if save_visualizations:
            viz_results = self.visualize()
            results.update(viz_results)

        return results


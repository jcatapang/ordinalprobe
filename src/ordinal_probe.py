# src/ordinal_probe.py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from typing import List

class DimensionalityReducer:
    """Handles dimensionality reduction using PCA."""
    
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.reducer = PCA(n_components=self.n_components)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Applies PCA-based dimensionality reduction."""
        return self.reducer.fit_transform(embeddings)

class OrdinalCluster:
    """Performs agglomerative hierarchical clustering for ordinal analysis."""

    def __init__(self, linkage: str = 'average', n_clusters: int = None):
        self.linkage = linkage
        self.n_clusters = n_clusters

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Clusters embeddings into groups based on the chosen linkage method."""
        clustering = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage
        )
        return clustering.fit_predict(embeddings)

class OrdinalEvaluator:
    """Evaluates the extent to which embeddings encode ordinal information."""

    @staticmethod
    def spearman_rank(true_order: List[int], predicted_order: List[int]) -> float:
        """Calculates Spearman's rank correlation coefficient."""
        rho, _ = spearmanr(true_order, predicted_order)
        return rho

    @staticmethod
    def mean_abs_error(true_order: List[int], predicted_order: List[int]) -> float:
        """Calculates mean absolute error (MAE)."""
        return mean_absolute_error(true_order, predicted_order)

    @staticmethod
    def cem(true_order: List[int], predicted_order: List[int], num_classes: int) -> float:
        """Calculates the Closeness Evaluation Measure (CEM) for ordinal data."""
        mae = mean_absolute_error(true_order, predicted_order)
        cem_score = 1 - (mae / (num_classes - 1))
        return cem_score

    @staticmethod
    def purity(clusters: List[int], labels: List[int]) -> float:
        """Calculates clustering purity."""
        cluster_labels = np.array(clusters)
        label_set = set(labels)
        purity_sum = 0
        for cluster in set(cluster_labels):
            indices = np.where(cluster_labels == cluster)[0]
            majority_label_count = max(
                np.sum(np.array(labels)[indices] == label)
                for label in label_set
            )
            purity_sum += majority_label_count
        return purity_sum / len(labels)

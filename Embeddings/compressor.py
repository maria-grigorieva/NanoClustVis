from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import hdbscan
from fastcluster import linkage
from sklearn.cluster import Birch, OPTICS, MeanShift, AffinityPropagation, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, pairwise_distances
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from gensim.models import Word2Vec
import torch
# from transformers import BertTokenizer, BertModel
import warnings
from collections import defaultdict
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial import cKDTree
from Embeddings import SequenceEmbedding

warnings.filterwarnings('ignore')


class ClusteringMethod(ABC):
    """Abstract base class for clustering methods"""

    @abstractmethod
    def fit_predict(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform clustering and return cluster assignments"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the clustering method"""
        pass


class KMeansClustering(ClusteringMethod):
    def fit_predict(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        return KMeans(n_clusters=n_clusters, random_state=42).fit_predict(features)

    def get_name(self) -> str:
        return "KMeans"


class HierarchicalClustering(ClusteringMethod):

    def reduce_samples(self, features: np.ndarray, n_representative: int = 5000) -> np.ndarray:
        """Reduce the number of samples by selecting representative points using MiniBatchKMeans."""
        if len(features) > n_representative:
            kmeans = MiniBatchKMeans(n_clusters=n_representative, random_state=42, batch_size=1024)
            kmeans.fit(features)
            return kmeans.cluster_centers_  # Use cluster centers as representative points
        return features

    def fit_predict(self, features: np.ndarray, n_clusters: int) -> np.ndarray:

        linkage_matrix = linkage(features, method='ward')
        return fcluster(linkage_matrix, t=n_clusters, criterion='maxclust') - 1

    def get_name(self) -> str:
        return "Hierarchical"


class DBSCANClustering(ClusteringMethod):
    def fit_predict(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        eps = self._estimate_eps(features)
        return DBSCAN(eps=eps, min_samples=10).fit_predict(features)

    def _estimate_eps(self, features: np.ndarray) -> float:
        """Estimate eps parameter for DBSCAN"""
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=2)
        nn_dist = nn.fit(features).kneighbors(features)[0]
        return np.percentile(nn_dist[:, 1], 90)

    def get_name(self) -> str:
        return "DBSCAN"

class HDBSCANClustering(ClusteringMethod):
    def fit_predict(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        min_samples = self._estimate_min_samples(features)
        return hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=10).fit_predict(features)

    def _estimate_min_samples(self, features: np.ndarray) -> int:
        """Estimate min_samples parameter for HDBSCAN"""
        nn = NearestNeighbors(n_neighbors=2)
        nn_dist = nn.fit(features).kneighbors(features)[0]
        estimated_value = int(np.percentile(nn_dist[:, 1], 90))

        return max(estimated_value, 1)

    def get_name(self) -> str:
        return "HDBSCAN"

class BIRCHClustering(ClusteringMethod):
    def fit_predict(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        threshold = self._estimate_threshold(features)
        return Birch(n_clusters=n_clusters, threshold=threshold).fit_predict(features)

    def _estimate_threshold(self, features: np.ndarray) -> float:
        """Estimate threshold parameter for BIRCH"""
        nn = NearestNeighbors(n_neighbors=2)
        nn_dist = nn.fit(features).kneighbors(features)[0]
        return np.percentile(nn_dist[:, 1], 75)  # 75th percentile of nearest neighbor distances

    def get_name(self) -> str:
        return "BIRCH"

class OPTICSClustering(ClusteringMethod):
    def fit_predict(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        min_samples = self._estimate_min_samples(features)
        model = OPTICS(min_samples=min_samples, n_jobs=-1)  # Parallel execution
        return model.fit_predict(features)

    def _estimate_min_samples(self, features: np.ndarray) -> int:
        """Estimate min_samples parameter for OPTICS"""
        nn = NearestNeighbors(n_neighbors=2, n_jobs=-1)
        nn_dist = nn.fit(features).kneighbors(features)[0]
        return max(2, int(np.percentile(nn_dist[:, 1], 90)))  # At least 2

    def get_name(self) -> str:
        return "OPTICS"

class GMMClustering(ClusteringMethod):
    def fit_predict(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        if n_clusters is None:
            n_clusters = self._estimate_n_components(features)
        return GaussianMixture(n_components=n_clusters, covariance_type='full').fit_predict(features)

    def _estimate_n_components(self, features: np.ndarray) -> int:
        """Estimate number of components using a simple heuristic"""
        n_samples = features.shape[0]
        return min(max(2, int(np.sqrt(n_samples) / 2)), 10)  # Rough heuristic

    def get_name(self) -> str:
        return "GMM"

class MeanShiftClustering(ClusteringMethod):
    def fit_predict(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        bandwidth = self._estimate_bandwidth(features)
        return MeanShift(bandwidth=bandwidth).fit_predict(features)

    def _estimate_bandwidth(self, features: np.ndarray) -> float:
        """Estimate bandwidth parameter for Mean Shift"""
        nn = NearestNeighbors(n_neighbors=5)
        nn_dist = nn.fit(features).kneighbors(features)[0]
        return np.percentile(nn_dist[:, 4], 50)  # Median distance to 5th neighbor

    def get_name(self) -> str:
        return "MeanShift"

class AffinityPropagationClustering(ClusteringMethod):
    def fit_predict(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        damping = 0.5  # Default value, could be tuned
        preference = self._estimate_preference(features)
        return AffinityPropagation(damping=damping, preference=preference).fit_predict(features)

    def _estimate_preference(self, features: np.ndarray) -> float:
        """Estimate preference parameter for Affinity Propagation"""
        nn = NearestNeighbors(n_neighbors=2)
        nn_dist = nn.fit(features).kneighbors(features)[0]
        median_dist = np.median(nn_dist[:, 1])
        return -median_dist * 10  # Negative scaling factor for preference

    def get_name(self) -> str:
        return "AffinityPropagation"

class SpectralClusteringMethod(ClusteringMethod):
    def fit_predict(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        if n_clusters is None:
            n_clusters = self._estimate_n_clusters(features)
        return SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors').fit_predict(features)

    def _estimate_n_clusters(self, features: np.ndarray) -> int:
        """Estimate number of clusters for Spectral Clustering"""
        n_samples = features.shape[0]
        return min(max(2, int(np.sqrt(n_samples) / 2)), 10)  # Rough heuristic

    def get_name(self) -> str:
        return "SpectralClustering"


class IntelligentCompressor:
    def __init__(self,
                 target_clusters: int = 100,
                 clustering_method: Optional[ClusteringMethod] = None):
        self.target_clusters = target_clusters
        self.clustering_method = clustering_method or HierarchicalClustering()
        self.scaler = StandardScaler()


    def calculate_cluster_diversity(self, features: np.ndarray) -> float:
        """
        Calculate the diversity score of a cluster based on feature distances.
        Returns a score between 0 (uniform) and 1 (diverse).
        """
        if len(features) <= 1:
            return 0.0

        # Calculate pairwise distances
        center = np.mean(features, axis=0)
        distances = np.linalg.norm(features - center, axis=1)

        # Normalize distances and calculate diversity score
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0.0

        normalized_distances = distances / max_dist
        diversity_score = np.mean(normalized_distances)

        return diversity_score

    def select_representative_sequences(self, features: np.ndarray,
                                        n_samples: int) -> List[int]:
        """
        Select representative sequences using distance-based clustering.
        Returns indices of representative sequences.
        """
        if len(features) <= n_samples:
            return list(range(len(features)))

        # Calculate center of the cluster
        center = np.mean(features, axis=0)

        # Calculate distances from center
        distances = np.linalg.norm(features - center, axis=1)

        # Initialize with the point closest to center
        selected_indices = [np.argmin(distances)]
        remaining_indices = set(range(len(features))) - set(selected_indices)

        # Select remaining representatives
        while len(selected_indices) < n_samples:
            # Calculate minimum distance to any selected point for each remaining point
            min_distances = []
            for idx in remaining_indices:
                dist_to_selected = [np.linalg.norm(features[idx] - features[sel_idx])
                                    for sel_idx in selected_indices]
                min_distances.append(min(dist_to_selected))

            # Select the point with maximum minimum distance
            remaining_indices_list = list(remaining_indices)
            next_point = remaining_indices_list[np.argmax(min_distances)]
            selected_indices.append(next_point)
            remaining_indices.remove(next_point)

        return sorted(selected_indices)

    def compress_sequences(self,
                           embeddings: List[float],
                           features: np.ndarray,
                           max_width: int,
                           save_results: bool = False,
                           output_file: str = "clustering_results.csv") -> Tuple[List[float], np.ndarray]:
        """Compress sequences using selected embedding and clustering methods"""
        if len(embeddings) <= self.target_clusters:
            return embeddings, np.arange(len(embeddings))


        # Perform clustering
        clusters = self.clustering_method.fit_predict(features, self.target_clusters)

        # Save clustering results to CSV file if save_results is True
        if save_results:
            import pandas as pd
            results = []
            for i, match in enumerate(embeddings):
                results.append({
                    "Sequence": str(match),
                    "Cluster": clusters[i]
                })
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            print(f"Clustering results saved to {output_file}")

        # Select representative sequences with diversity check
        compressed_embeddings = []
        diversity_threshold = 0.3  # Adjust this threshold based on your needs

        for cluster_id in range(max(clusters) + 1):
            cluster_indices = np.where(clusters == cluster_id)[0]

            if len(cluster_indices) > 0:
                cluster_features = features[cluster_indices]
                diversity_score = self.calculate_cluster_diversity(cluster_features)

                if diversity_score > diversity_threshold:
                    # Cluster is diverse - select multiple representatives
                    n_samples = max(1, int(len(cluster_indices) * 0.1))  # 10% of cluster size
                    representative_indices = self.select_representative_sequences(
                        cluster_features,
                        n_samples
                    )

                    # Add selected representatives to compressed matches
                    for idx in representative_indices:
                        compressed_embeddings.append(embeddings[cluster_indices[idx]])

                    print(f"Cluster {cluster_id}: Diversity = {diversity_score:.3f}, "
                          f"Selected {len(representative_indices)} representatives "
                          f"from {len(cluster_indices)} sequences")
                else:
                    # Cluster is uniform - select single representative
                    cluster_center = np.mean(cluster_features, axis=0)
                    distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
                    representative_idx = cluster_indices[np.argmin(distances)]
                    compressed_embeddings.append(embeddings[representative_idx])

                    print(f"Cluster {cluster_id}: Diversity = {diversity_score:.3f}, "
                          f"Selected 1 representative from {len(cluster_indices)} sequences")

        # Calculate clustering quality
        if len(np.unique(clusters)) > 1:
            silhouette_avg = silhouette_score(features, clusters)
            print(f"Clustering quality (silhouette score): {silhouette_avg:.3f}")

        return compressed_embeddings, clusters

    def evaluate_clustering_methods(self,
                                    embeddings: List[float],
                                    max_width: int) -> Dict[str, float]:
        """Evaluate different clustering methods"""
        features = self.create_feature_vectors(embeddings, max_width)
        features_scaled = self.scaler.fit_transform(features)

        clustering_methods = [
            KMeansClustering(),
            HierarchicalClustering(),
            DBSCANClustering(),
            HDBSCANClustering(),
            OPTICSClustering(),
            BIRCHClustering(),
            MeanShiftClustering(),
            GMMClustering(),
            SpectralClusteringMethod(),
            AffinityPropagationClustering()
        ]

        scores = {}
        for method in clustering_methods:
            clusters = method.fit_predict(features_scaled, self.target_clusters)
            if len(np.unique(clusters)) > 1:
                score = silhouette_score(features_scaled, clusters)
                scores[method.get_name()] = score

        return scores

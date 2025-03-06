from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from gensim.models import Word2Vec
import torch
# from transformers import BertTokenizer, BertModel
import warnings
from matcher import SequenceMatch
from collections import defaultdict

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
    def fit_predict(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        linkage_matrix = linkage(features, method='ward')
        return fcluster(linkage_matrix, t=n_clusters, criterion='maxclust') - 1

    def get_name(self) -> str:
        return "Hierarchical"


class DBSCANClustering(ClusteringMethod):
    def fit_predict(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        eps = self._estimate_eps(features)
        return DBSCAN(eps=eps, min_samples=5).fit_predict(features)

    def _estimate_eps(self, features: np.ndarray) -> float:
        """Estimate eps parameter for DBSCAN"""
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=2)
        nn_dist = nn.fit(features).kneighbors(features)[0]
        return np.percentile(nn_dist[:, 1], 90)

    def get_name(self) -> str:
        return "DBSCAN"


class SequenceEmbedding(ABC):
    """Abstract base class for sequence embedding methods"""

    @abstractmethod
    def embed_sequences(self, sequences: List[List[SequenceMatch]]) -> np.ndarray:
        """Convert sequences to embeddings"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the embedding method"""
        pass


class Word2VecEmbedding(SequenceEmbedding):
    def __init__(self, vector_size: int = 5):
        self.vector_size = vector_size

    def embed_sequences(self, sequences: List[List[SequenceMatch]]) -> np.ndarray:
        # Convert sequences to "words" (query names in order of appearance)
        sequence_words = [
            [match.query_name for match in seq_matches]
            for seq_matches in sequences
        ]

        # Train Word2Vec model
        model = Word2Vec(sentences=sequence_words,
                         vector_size=self.vector_size,
                         window=5,
                         min_count=1,
                         workers=4)

        # Create sequence embeddings by averaging word vectors
        embeddings = np.zeros((len(sequences), self.vector_size))
        for i, seq_words in enumerate(sequence_words):
            if seq_words:
                embeddings[i] = np.mean([model.wv[word] for word in seq_words], axis=0)

        return embeddings

    def get_name(self) -> str:
        return "Word2Vec"

class Bio2VecEmbedding(SequenceEmbedding):
    """Embedding using Bio2Vec approach"""
    def __init__(self,
                 vector_size: int = 100,
                 window: int = 5,
                 k_mer_size: int = 3):
        self.vector_size = vector_size
        self.window = window
        self.k_mer_size = k_mer_size
        self.model = None

    def _sequence_to_kmers(self, sequence: str) -> List[str]:
        """Convert sequence to k-mers"""
        return [sequence[i:i+self.k_mer_size]
                for i in range(len(sequence)-self.k_mer_size+1)]

    def _extract_sequence(self, match: SequenceMatch) -> str:
        """Extract sequence from match"""
        # This is a placeholder - you'll need to implement actual sequence extraction
        # based on your data structure
        return match.sequence if hasattr(match, 'sequence') else ''

    def _train_bio2vec(self, sequences: List[List[str]]) -> Word2Vec:
        """Train Bio2Vec model on sequences"""
        return Word2Vec(sequences,
                       vector_size=self.vector_size,
                       window=self.window,
                       min_count=1,
                       workers=4)

    def embed_sequences(self, sequences: List[List[SequenceMatch]]) -> np.ndarray:
        """Convert sequences to Bio2Vec embeddings"""
        # Convert sequences to k-mers
        kmer_sequences = []
        for seq_matches in sequences:
            seq_kmers = []
            for match in seq_matches:
                sequence = self._extract_sequence(match)
                if sequence:
                    seq_kmers.extend(self._sequence_to_kmers(sequence))
            kmer_sequences.append(seq_kmers)

        # Train Bio2Vec model if not already trained
        if self.model is None:
            self.model = self._train_bio2vec(kmer_sequences)

        # Create embeddings
        embeddings = np.zeros((len(sequences), self.vector_size))
        for i, kmers in enumerate(kmer_sequences):
            if kmers:
                kmer_vectors = [self.model.wv[kmer] for kmer in kmers if kmer in self.model.wv]
                if kmer_vectors:
                    embeddings[i] = np.mean(kmer_vectors, axis=0)

        return embeddings

    def get_name(self) -> str:
        return f"Bio2Vec_k{self.k_mer_size}"

# class BERTEmbedding(SequenceEmbedding):
#     def __init__(self):
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         self.model = BertModel.from_pretrained('bert-base-uncased')
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model.to(self.device)
#
#     def embed_sequences(self, sequences: List[List[SequenceMatch]]) -> np.ndarray:
#         # Convert sequences to text representation
#         sequence_texts = [
#             ' '.join([f"{match.query_name}_{match.position}" for match in seq_matches])
#             for seq_matches in sequences
#         ]
#
#         embeddings = []
#         with torch.no_grad():
#             for text in sequence_texts:
#                 inputs = self.tokenizer(text,
#                                         return_tensors="pt",
#                                         padding=True,
#                                         truncation=True,
#                                         max_length=512).to(self.device)
#                 outputs = self.model(**inputs)
#                 # Use [CLS] token embedding as sequence representation
#                 embeddings.append(outputs.last_hidden_state[0, 0].cpu().numpy())
#
#         return np.array(embeddings)
#
#     def get_name(self) -> str:
#         return "BERT"


class IntelligentCompressor:
    def __init__(self,
                 target_clusters: int = 100,
                 clustering_method: Optional[ClusteringMethod] = None,
                 embedding_method: Optional[SequenceEmbedding] = None):
        self.target_clusters = target_clusters
        self.clustering_method = clustering_method or HierarchicalClustering()
        self.embedding_method = embedding_method or Word2VecEmbedding()
        self.scaler = StandardScaler()

    def create_feature_vectors(self,
                               sequence_matches: List[List[SequenceMatch]],
                               max_width: int) -> np.ndarray:
        """Create traditional feature vectors based on match statistics"""
        n_sequences = len(sequence_matches)
        n_queries = len(set(match.query_name for matches in sequence_matches
                            for match in matches))

        features = np.zeros((n_sequences, n_queries * 4))  # 4 features per query type

        for seq_idx, matches in enumerate(sequence_matches):
            query_stats = defaultdict(lambda: {'positions': [], 'scores': [], 'lengths': []})

            for match in matches:
                query_stats[match.query_name]['positions'].append(match.position / max_width)
                query_stats[match.query_name]['scores'].append(match.score)
                query_stats[match.query_name]['lengths'].append(match.length)

            for query_idx, query_name in enumerate(sorted(set(match.query_name
                                                              for matches in sequence_matches
                                                              for match in matches))):
                if query_name in query_stats:
                    stats = query_stats[query_name]
                    base_idx = query_idx * 4
                    features[seq_idx, base_idx] = np.mean(stats['positions']) if stats['positions'] else 0
                    features[seq_idx, base_idx + 1] = np.std(stats['positions']) if len(stats['positions']) > 1 else 0
                    features[seq_idx, base_idx + 2] = np.mean(stats['scores']) if stats['scores'] else 0
                    features[seq_idx, base_idx + 3] = np.mean(stats['lengths']) if stats['lengths'] else 0

        return features

    def compress_sequences(self,
                           sequence_matches: List[List[SequenceMatch]],
                           max_width: int) -> Tuple[List[List[SequenceMatch]], np.ndarray]:
        """Compress sequences using selected embedding and clustering methods"""
        if len(sequence_matches) <= self.target_clusters:
            return sequence_matches, np.arange(len(sequence_matches))

        # Get both traditional features and embeddings
        traditional_features = self.create_feature_vectors(sequence_matches, max_width)
        embedded_features = self.embedding_method.embed_sequences(sequence_matches)

        # Combine features
        combined_features = np.hstack([
            self.scaler.fit_transform(traditional_features),
            self.scaler.fit_transform(embedded_features)
        ])

        # Perform clustering
        clusters = self.clustering_method.fit_predict(combined_features, self.target_clusters)

        # Select representative sequences
        compressed_matches = []
        for cluster_id in range(max(clusters) + 1):
            cluster_indices = np.where(clusters == cluster_id)[0]
            if len(cluster_indices) > 0:
                cluster_features = combined_features[cluster_indices]
                cluster_center = np.mean(cluster_features, axis=0)
                distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
                representative_idx = cluster_indices[np.argmin(distances)]
                compressed_matches.append(sequence_matches[representative_idx])

        # Calculate clustering quality
        if len(np.unique(clusters)) > 1:
            silhouette_avg = silhouette_score(combined_features, clusters)
            print(f"Clustering quality (silhouette score): {silhouette_avg:.3f}")

        return compressed_matches, clusters

    def evaluate_clustering_methods(self,
                                    sequence_matches: List[List[SequenceMatch]],
                                    max_width: int) -> Dict[str, float]:
        """Evaluate different clustering methods"""
        features = self.create_feature_vectors(sequence_matches, max_width)
        features_scaled = self.scaler.fit_transform(features)

        clustering_methods = [
            KMeansClustering(),
            HierarchicalClustering(),
            DBSCANClustering()
        ]

        scores = {}
        for method in clustering_methods:
            clusters = method.fit_predict(features_scaled, self.target_clusters)
            if len(np.unique(clusters)) > 1:
                score = silhouette_score(features_scaled, clusters)
                scores[method.get_name()] = score

        return scores

from gensim.models import Word2Vec
from typing import List, Tuple, Dict, Optional
from abc import ABC, abstractmethod
from sklearn.cluster import MiniBatchKMeans
from matcher import SequenceMatch
import numpy as np
from tqdm import tqdm

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
        sequence_words = []
        for seq_matches in tqdm(sequences, desc="Processing sequences"):
            sequence_words.append([match.query_name for match in seq_matches])

        # Train Word2Vec model
        model = Word2Vec(sentences=sequence_words,
                         vector_size=self.vector_size,
                         window=5,
                         min_count=1,
                         workers=4)

        # Create sequence embeddings by averaging word vectors
        embeddings = np.zeros((len(sequences), self.vector_size))
        for i, seq_words in enumerate(tqdm(sequence_words, desc="Creating embeddings")):
            if seq_words:
                embeddings[i] = np.mean([model.wv[word] for word in seq_words], axis=0)

        return embeddings

    def reduce_samples(self, features: np.ndarray, n_representative: int = 3000) -> Tuple[np.ndarray, List[int]]:
        """
        Reduce the number of samples by selecting representative points using MiniBatchKMeans.

        Returns:
        - reduced_features: np.ndarray -> The reduced feature set (cluster centers).
        - selected_indices: List[int] -> Indices of selected representative samples.
        """
        num_samples = len(features)

        if num_samples <= n_representative:
            return features, list(range(num_samples))  # If samples are already small, return all

        # Apply clustering to select representative samples
        kmeans = MiniBatchKMeans(n_clusters=n_representative, random_state=42, batch_size=1024)
        kmeans.fit(features)

        # Find closest points to each cluster center
        closest_indices = []
        for center in kmeans.cluster_centers_:
            index = np.argmin(np.linalg.norm(features - center, axis=1))
            closest_indices.append(index)

        return kmeans.cluster_centers_, closest_indices

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
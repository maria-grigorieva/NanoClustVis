from gensim.models import Word2Vec
from typing import List, Tuple, Dict, Optional
from abc import ABC, abstractmethod
from sklearn.cluster import MiniBatchKMeans
from matcher import SequenceMatch
import numpy as np
from tqdm import tqdm
import json
import os
from Bio import SeqIO
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

class QueryAutoencoder(nn.Module):
    def __init__(self, vocab_size, emb_size=8, hidden_size=16, latent_size=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.encoder = nn.LSTM(emb_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, latent_size)
        self.decoder = nn.LSTM(emb_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def encode(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.encoder(x)
        return self.fc(h_n.squeeze(0))

    def forward(self, x):
        z = self.encode(x)
        # Decoder expects embedded input; during training you can also use teacher forcing
        x_emb = self.embedding(x)
        output, _ = self.decoder(x_emb)
        return self.output(output)

class SequenceEmbedding(ABC):
    """Abstract base class for sequence embedding methods"""

    @abstractmethod
    def embed_sequences(self,
                        sequences: List[List[SequenceMatch]]) -> np.ndarray:
        """Convert sequences to embeddings"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the embedding method"""
        pass

    def reduce_samples(self, features: np.ndarray, n_representative: int = 5000) -> Tuple[np.ndarray, List[int]]:
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

class TransformerSequenceEmbedding(SequenceEmbedding):
    def __init__(self, embed_dim=8, vector_size=30, transformer_dim=32, max_seq_len=20, nhead=4, num_layers=2):
        self.embed_dim = embed_dim
        self.output_dim = vector_size
        self.max_seq_len = max_seq_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Vocab will be built later
        self.query2id = {}

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim * 2,
            nhead=nhead,
            dim_feedforward=transformer_dim,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(self.device)

        # Embeddings
        self.query_embedding = None  # created after vocab is built
        self.position_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        ).to(self.device)

        # Output projection to desired embedding size
        self.output_projector = nn.Linear(embed_dim * 2, vector_size).to(self.device)

    def _build_vocab(self, sequences: List[List[SequenceMatch]]):
        vocab = {match.query_name for seq in sequences for match in seq}
        self.query2id = {name: idx for idx, name in enumerate(sorted(vocab))}
        self.query_embedding = nn.Embedding(len(self.query2id), self.embed_dim).to(self.device)

    def embed_sequences(self, sequences: List[List[SequenceMatch]]) -> np.ndarray:
        self._build_vocab(sequences)
        embedded_all = []

        for seq in sequences:
            if len(seq) < self.max_seq_len:
                seq = seq + [SequenceMatch("<PAD>", 0, 0, 0)] * (self.max_seq_len - len(seq))
            else:
                seq = seq[:self.max_seq_len]

            query_ids = []
            positions = []

            for match in seq:
                query_id = self.query2id.get(match.query_name, 0)
                query_ids.append(query_id)
                positions.append([match.position])

            # Normalize positions
            scaler = MinMaxScaler()
            positions = scaler.fit_transform(positions)

            query_ids = torch.tensor(query_ids, dtype=torch.long, device=self.device)
            positions = torch.tensor(positions, dtype=torch.float32, device=self.device)

            query_embeds = self.query_embedding(query_ids)                   # (seq_len, embed_dim)
            position_embeds = self.position_encoder(positions)              # (seq_len, embed_dim)

            fused = torch.cat([query_embeds, position_embeds], dim=1).unsqueeze(0)  # (1, seq_len, embed_dim*2)

            with torch.no_grad():
                encoded = self.encoder(fused)                               # (1, seq_len, embed_dim*2)
                pooled = encoded.mean(dim=1)                                # (1, embed_dim*2)
                output = self.output_projector(pooled)                      # (1, output_dim)

            embedded_all.append(output.cpu().numpy())

        return np.vstack(embedded_all)

    def get_name(self) -> str:
        return "transformer_query_pos_fusion"

# LSTM-based embedding implementation
class LSTMEmbedding(SequenceEmbedding):
    def __init__(self, embed_dim = 5, vector_size=30, max_seq_len=20, batch_size=32):
        self.embed_dim = embed_dim
        self.hidden_dim = vector_size
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.query2id: Dict[str, int] = {}
        self.model = None
        self._is_trained = False

    def _build_vocab(self, sequences: List[List[SequenceMatch]]):
        vocab = {match.query_name for seq in sequences for match in seq}
        self.query2id = {name: idx for idx, name in enumerate(sorted(vocab))}

    def _preprocess_sequence(self, sequence: List[SequenceMatch]) -> torch.Tensor:
        tokens = []
        max_position = max((m.position for m in sequence), default=1)
        for match in sequence[:self.max_seq_len]:
            query_id = self.query2id.get(match.query_name, 0)
            norm_pos = match.position / max_position if max_position > 0 else 0.0
            tokens.append((query_id, norm_pos))
        # Pad if necessary
        while len(tokens) < self.max_seq_len:
            tokens.append((0, 0.0))
        return torch.tensor(tokens)

    def _build_model(self):
        self.model = nn.Sequential(
            nn.Embedding(len(self.query2id), self.embed_dim),
        )
        self.lstm = nn.LSTM(input_size=self.embed_dim + 1, hidden_size=self.hidden_dim, batch_first=True)
        self.model.to(self.device)
        self.lstm.to(self.device)
        self._is_trained = True

    def embed_sequences(self, sequences: List[List[SequenceMatch]]) -> np.ndarray:
        if not self._is_trained:
            self._build_vocab(sequences)
            self._build_model()

        embeddings = []
        for i in range(0, len(sequences), self.batch_size):
            batch = sequences[i:i + self.batch_size]
            batch_tensors = torch.stack([
                self._preprocess_sequence(seq) for seq in batch
            ]).to(self.device)  # Shape: [B, L, 2]

            token_ids = batch_tensors[:, :, 0].long()
            positions = batch_tensors[:, :, 1].float().unsqueeze(-1)

            token_embeds = self.model[0](token_ids)
            lstm_input = torch.cat([token_embeds, positions], dim=-1)
            _, (hidden, _) = self.lstm(lstm_input)
            embedding = hidden.squeeze(0)  # shape: [B, hidden_dim]

            embeddings.append(embedding.cpu().detach().numpy())

        return np.vstack(embeddings)

    def get_name(self) -> str:
        return "LSTMEmbedding"

class PositionalEncodingWord2VecEmbedding(SequenceEmbedding):
    def __init__(self, vector_size: int = 30, position_encoding_dim: int = 10, max_position: int = 500):
        self.vector_size = vector_size
        self.position_encoding_dim = position_encoding_dim
        self.max_position = max_position  # Maximum expected position value

    def _positional_encoding(self, position: int) -> np.ndarray:
        """Generates a positional encoding vector."""
        encoding = np.zeros(self.position_encoding_dim)
        for i in range(0, self.position_encoding_dim, 2):
            encoding[i] = np.sin(position / (10000 ** (i / self.position_encoding_dim)))
            if i + 1 < self.position_encoding_dim:
                encoding[i + 1] = np.cos(position / (10000 ** (i / self.position_encoding_dim)))
        return encoding

    def embed_sequences(self, sequences: List[List[SequenceMatch]]) -> np.ndarray:
        # Convert sequences to "words" (query names in order of appearance)
        sequence_words = []
        for seq_matches in tqdm(sequences, desc="Processing sequences"):
            sequence_words.append([(match.query_name, match.position) for match in seq_matches])

        # Train Word2Vec model
        model = Word2Vec(sentences=[[word[0] for word in seq] for seq in sequence_words],  # Train on query names only
                         vector_size=self.vector_size,
                         window=5,
                         min_count=1,
                         workers=4)

        # Create sequence embeddings by concatenating word vectors with positional encodings
        embedding_dim = self.vector_size + self.position_encoding_dim
        embeddings = np.zeros((len(sequences), embedding_dim))

        for i, seq_words in enumerate(tqdm(sequence_words, desc="Creating embeddings")):
            if seq_words:
                combined_vectors = []
                for word, position in seq_words:
                    word_vector = model.wv[word]
                    positional_encoding = self._positional_encoding(position)
                    combined_vector = np.concatenate((word_vector, positional_encoding))
                    combined_vectors.append(combined_vector)

                embeddings[i] = np.mean(combined_vectors, axis=0)

        return embeddings

    def get_name(self) -> str:
        return "PositionalEncodingWord2Vec"

class WeightedWord2VecEmbedding(SequenceEmbedding):
    def __init__(self, vector_size: int = 30, position_weight: float = 0.1):
        self.vector_size = vector_size
        self.position_weight = position_weight  # Adjust this value

    def embed_sequences(self, sequences: List[List[SequenceMatch]]) -> np.ndarray:
        # Convert sequences to "words" (query names in order of appearance)
        sequence_words = []
        for seq_matches in tqdm(sequences, desc="Processing sequences"):
            sequence_words.append([(match.query_name, match.position) for match in seq_matches])

        # Train Word2Vec model
        model = Word2Vec(sentences=[[word[0] for word in seq] for seq in sequence_words],  # Train on query names only
                         vector_size=self.vector_size,
                         window=5,
                         min_count=1,
                         workers=4)

        # Create sequence embeddings by weighted averaging of word vectors
        embeddings = np.zeros((len(sequences), self.vector_size))
        for i, seq_words in enumerate(tqdm(sequence_words, desc="Creating embeddings")):
            if seq_words:
                weighted_vectors = []
                for word, position in seq_words:
                    # # Weight based on position (linear scaling - adjust as needed)
                    # weight = 1 - (position / max(1, max(pos for _, pos in seq_words))) * self.position_weight # Avoid division by zero and normalize
                    # weighted_vectors.append(model.wv[word] * weight)
                    positions = [pos for _, pos in seq_words]
                    min_pos = min(positions)
                    max_pos = max(positions)

                    # To avoid division by zero
                    range_pos = max(1e-6, max_pos - min_pos)

                    min_weight = 0.1  # lowest allowed weight (for highest position)
                    for word, position in seq_words:
                        weight = 1.0 - (position - min_pos) / range_pos * (1.0 - min_weight)
                        weighted_vectors.append(model.wv[word] * weight)
                embeddings[i] = np.mean(weighted_vectors, axis=0)

        return embeddings

    def get_name(self) -> str:
        return "WeightedWord2Vec"

# class LSTMAutoencoderEmbedding(SequenceEmbedding):
#     def __init__(self, vector_size: int = 32, hidden_dim: int = 64, num_layers: int = 1, learning_rate: float = 0.001,
#                  epochs: int = 10, device: str = "cpu"):
#         self.vector_size = vector_size  # Renamed to vector_size
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.learning_rate = learning_rate
#         self.epochs = epochs
#         self.device = device
#         self.word_to_index = None  # Initialize vocabulary mappings
#         self.index_to_word = None
#         self.vocab_size = None  # Will be determined in _prepare_data
#         self.model = None  # Initialize model to None
#         self.optimizer = None
#         self.loss_fn = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for sequence prediction
#
#     def _prepare_data(self, sequences: List[List[SequenceMatch]]) -> Tuple[List[torch.Tensor], List[int]]:
#         """Converts sequences of SequenceMatch objects to sequences of integer indices."""
#
#         # Create a vocabulary mapping query_name to integer index
#         # Use a tuple of attributes to make SequenceMatch hashable
#         unique_matches = set(
#             (match.seq_id, match.query_name, match.position, match.length, match.score) for seq in sequences for match
#             in seq)
#
#         self.word_to_index = {
#             "<PAD>": 0,  # Padding token
#             **{str(match_tuple): i + 1 for i, match_tuple in enumerate(unique_matches)}
#             # Use string representation of tuple
#         }
#         self.index_to_word = {i: word for word, i in self.word_to_index.items()}
#         self.vocab_size = len(self.word_to_index)  # Determine vocab_size
#
#         indexed_sequences = []
#         sequence_lengths = []
#         for seq in sequences:
#             indexed_seq = []
#             for match in seq:
#                 match_tuple = (match.seq_id, match.query_name, match.position, match.length, match.score)
#                 indexed_seq.append(self.word_to_index[str(match_tuple)])  # Use string representation of tuple
#             indexed_sequences.append(torch.tensor(indexed_seq, dtype=torch.long))  # Ensure dtype is torch.long
#             sequence_lengths.append(len(seq))
#
#         return indexed_sequences, sequence_lengths
#
#     def embed_sequences(self, sequences: List[List[SequenceMatch]]) -> np.ndarray:
#         """Embeds sequences using the trained LSTM autoencoder."""
#         indexed_sequences, sequence_lengths = self._prepare_data(sequences)
#
#         # Pad sequences to the maximum length in the batch
#         padded_sequences = pad_sequence(indexed_sequences, batch_first=True, padding_value=0).to(
#             self.device).long()  # 0 is the padding index and cast to long
#
#         # Initialize the model here, after vocab_size is known
#         self.model = LSTMAutoencoder(self.vocab_size, self.vector_size, self.hidden_dim, self.num_layers).to(
#             self.device)
#         self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
#
#         # Train the autoencoder
#         self._train(padded_sequences, sequence_lengths)
#
#         # Get embeddings (encoder output)
#         with torch.no_grad():
#             self.model.eval()
#             _, embeddings = self.model.encode(padded_sequences)  # Get the embeddings from the encoder
#             embeddings = embeddings.cpu().numpy()
#
#         return embeddings
#
#     def _train(self, padded_sequences: torch.Tensor, sequence_lengths: List[int]):
#         """Trains the LSTM autoencoder."""
#         self.model.train()
#         for epoch in tqdm(range(self.epochs), desc="Training Autoencoder"):
#             self.optimizer.zero_grad()
#             padded_sequences = padded_sequences.long()  # Ensure dtype is torch.long BEFORE passing to the model
#             reconstructed_sequences, _ = self.model(padded_sequences)  # Pass padded sequences through the autoencoder
#
#             # Flatten the output and target for CrossEntropyLoss
#             reconstructed_sequences = reconstructed_sequences.view(-1, self.vocab_size)
#             padded_sequences = padded_sequences.view(-1)  # No need to cast to long here, as it's already done above
#
#             loss = self.loss_fn(reconstructed_sequences, padded_sequences)
#             loss.backward()
#             self.optimizer.step()
#             print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")
#
#     def get_name(self) -> str:
#         return "LSTMAutoencoder"


class LSTMAutoencoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int):
        super(LSTMAutoencoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)  # Map back to vocabulary size

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes the input sequence."""
        output, (hidden, cell) = self.encoder(x)  # x уже embedded
        return output, hidden[-1]  # Return the last hidden state as the embedding

    def decode(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Decodes the hidden state to reconstruct the sequence."""
        output, _ = self.decoder(x, (hidden, hidden))  # Use the same hidden state for both hidden and cell states
        return output

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the autoencoder."""
        embedded = self.embedding(x)
        encoded_output, (hidden, cell) = self.encoder(embedded)
        decoded_output, _ = self.decoder(encoded_output, (hidden, cell))
        output = self.linear(decoded_output)
        return output, encoded_output

class Word2VecEmbedding(SequenceEmbedding):
    def __init__(self, vector_size: int = 30):
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

    def get_name(self) -> str:
        return "Word2Vec"

class EnhancedAutoencoder(SequenceEmbedding):
    def __init__(self, vector_size: int = 30):
        self.vector_size = vector_size

    def extract_sequences(self, data: List[List[SequenceMatch]]):
        queries_seq = []
        distances_seq = []
        for group in data:
            sorted_group = sorted(group, key=lambda m: m.position)
            queries = [m.query_name for m in sorted_group]
            positions = [m.position for m in sorted_group]
            distances = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
            queries_seq.append(queries)
            distances_seq.append(distances)
        return queries_seq, distances_seq

class Bio2VecEmbedding(SequenceEmbedding):
    """Embedding using Bio2Vec approach"""
    def __init__(self,
                 vector_size: int = 30,
                 window: int = 5,
                 k_mer_size: int = 5):
        self.vector_size = vector_size
        self.window = window
        self.k_mer_size = k_mer_size
        # self.model = None

    def _sequence_to_kmers(self, sequence: str) -> List[str]:
        """Convert sequence to k-mers"""
        return [sequence[i:i+self.k_mer_size]
                for i in range(len(sequence)-self.k_mer_size+1)]

    def _train_bio2vec(self, sequences: List[List[str]]) -> Word2Vec:
        """Train Bio2Vec model on sequences"""
        return Word2Vec(sentences=sequences,
                       vector_size=self.vector_size,
                       window=self.window,
                       min_count=1,
                       workers=4)

    def embed_sequences(self, sequences: List[List[str]]) -> np.ndarray:
        """Convert sequences to Bio2Vec embeddings"""
        kmer_sequences = [self._sequence_to_kmers(s) for s in sequences]

        model = self._train_bio2vec(kmer_sequences)

        # Create embeddings
        embeddings = np.zeros((len(sequences), self.vector_size))
        for i, kmers in enumerate(kmer_sequences):
            if kmers:
                kmer_vectors = [model.wv[kmer] for kmer in kmers if kmer in model.wv]
                if kmer_vectors:
                    embeddings[i] = np.mean(kmer_vectors, axis=0)

        return embeddings

    def get_name(self) -> str:
        return f"Bio2Vec_k{self.k_mer_size}"

from typing import List, Tuple, Dict, Optional
from abc import ABC, abstractmethod
from sklearn.cluster import MiniBatchKMeans
from matcher import SequenceMatch
import numpy as np
from tqdm import tqdm
import torch
import json
import os
from Bio import SeqIO
from BioSeqDataset import BioSeqDataset
from torch.utils.data import DataLoader
from SequenceAutoencoder import Autoencoder
import torch.nn as nn
import torch.optim as optim

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

    def reduce_samples(self, features: np.ndarray, n_representative: int = 10000) -> Tuple[np.ndarray, List[int]]:
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
class BioSeqEmbedding(SequenceEmbedding):

    def __init__(self, vector_size: int = 32):
        self.vector_size = vector_size

    def embed_sequences(self, sequences: List[List[SequenceMatch]]) -> np.ndarray:

        dataset = BioSeqDataset(sequences)
        vocab = dataset.get_vocab()
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        # Initialize the autoencoder model, loss function, and optimizer
        input_dim = len(vocab) + 1  # +1 for padding token
        embedding_dim = 32
        model = Autoencoder(input_dim, embedding_dim, dataset.max_length)
        print('AE model initialized')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print('Start training...')
        # Train the autoencoder model
        for epoch in range(10):
            print('Epoch {}'.format(epoch))
            model.train()
            total_loss = 0
            for batch in train_loader:
                input_data = batch['input']
                target_data = batch['target']
                optimizer.zero_grad()
                output = model(input_data)
                loss = criterion(output.view(-1, input_dim), target_data.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')

        embeddings = []
        with torch.no_grad():
            for ds in dataset.padded_data:
                input_data = torch.tensor(ds, dtype=torch.long).unsqueeze(0)  # Add a batch dimension
                encoded = model.encoder(input_data)
                embeddings.append(encoded.numpy().squeeze(0))

        return embeddings
    def get_name(self) -> str:
        return "BioSeq2Vec"
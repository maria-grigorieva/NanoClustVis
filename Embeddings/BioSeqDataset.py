import re
import pandas as pd
import torch
import json
import os
from torch.utils.data import Dataset
from Bio import SeqIO
import numpy as np

class BioSeqDataset(Dataset):

    def __init__(self, fastq_file, max_length=None, vocab=None, z_score_threshold=2):
        """
        Args:
            fastq_file (str): Path to the fastq file
            max_length (int, optional): Fixed sequence length for padding
            vocab (dict, optional): Pre-existing vocabulary dictionary
            z_score_threshold (int, optional): Threshold for Z-score to remove outliers
        """
        self.raw_datasets = []
        self.sequence_lengths = []
        for record in SeqIO.parse(fastq_file, "fastq"):
            self.raw_datasets.append(str(record.seq))
            self.sequence_lengths.append(len(record.seq))

        # Calculate statistics and remove outliers
        self.mean_length = np.mean(self.sequence_lengths)
        self.std_length = np.std(self.sequence_lengths)
        self.z_scores = [(x - self.mean_length) / self.std_length for x in self.sequence_lengths]
        self.filtered_datasets = [x for x, z in zip(self.raw_datasets, self.z_scores) if abs(z) < z_score_threshold]

        self.tokenized = [self._tokenize(ds) for ds in self.filtered_datasets]

        # Initialize special tokens
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"

        # Handle vocabulary
        if vocab is not None:
            self.vocab = vocab
            self._validate_vocab()
        else:
            self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)

        self.index_to_token = {idx: token for token, idx in self.vocab.items()}  # Reverse lookup

        # Sequence processing
        self.max_length = max_length if max_length else self._get_max_length()
        self.padded_data = self._numericalize_and_pad()

    def _tokenize(self, dataset_name):
        """Tokenize a single dataset name"""
        return list(dataset_name)

    def _build_vocab(self):
        """Build vocabulary from all tokens"""
        vocabulary = set()
        for tokens in self.tokenized:
            vocabulary.update(tokens)
        vocab = {token: i + 2 for i, token in enumerate(vocabulary)}
        vocab[self.pad_token] = 0
        vocab[self.unk_token] = 1
        return vocab

    def _validate_vocab(self):
        """Ensure special tokens exist in vocabulary"""
        if self.pad_token not in self.vocab:
            self.vocab[self.pad_token] = 0
        if self.unk_token not in self.vocab:
            self.vocab[self.unk_token] = 1

    def save(self, save_dir):
        """Save the dataset processor to disk"""
        os.makedirs(save_dir, exist_ok=True)

        # Save vocabulary
        with open(os.path.join(save_dir, 'vocab.json'), 'w') as f:
            json.dump(self.vocab, f)

        # Save configuration
        config = {
            'max_length': self.max_length,
            'pad_token': self.pad_token,
            'unk_token': self.unk_token
        }
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f)

        print(f"Saved dataset processor to {save_dir}")

    @classmethod
    def load(cls, fastq_file, save_dir):
        """Load a saved dataset processor"""
        # Load vocabulary
        with open(os.path.join(save_dir, 'vocab.json'), 'r') as f:
            vocab = json.load(f)

        # Load configuration
        with open(os.path.join(save_dir, 'config.json'), 'r') as f:
            config = json.load(f)

        return cls(
            fastq_file=fastq_file,
            max_length=config['max_length'],
            vocab=vocab
        )

    def save_torch(self, save_path):
        """Save the entire processor as a torch object"""
        torch.save({
            'vocab': self.vocab,
            'max_length': self.max_length,
            'pad_token': self.pad_token,
            'unk_token': self.unk_token
        }, save_path)
        print(f"Saved torch processor to {save_path}")

    @classmethod
    def load_torch(cls, fastq_file, save_path):
        """Load a torch-saved processor"""
        data = torch.load(save_path)
        return cls(
            fastq_file=fastq_file,
            max_length=data['max_length'],
            vocab=data['vocab']
        )

    # [Rest of your existing methods...]
    def _get_max_length(self):
        return max(len(tokens) for tokens in self.tokenized)

    def _numericalize_and_pad(self):
        numerical_data = []
        for tokens in self.tokenized:
            indices = [self.vocab.get(token, 1) for token in tokens]
            if self.max_length:
                if len(indices) > self.max_length:
                    indices = indices[:self.max_length]
                else:
                    indices += [0] * (self.max_length - len(indices))
            numerical_data.append(indices)
        return numerical_data

    def get_vocab(self):
        return self.vocab

    def decode(self, indices):
        idx_to_token = {v: k for k, v in self.vocab.items()}
        return [idx_to_token.get(idx, self.unk_token) for idx in indices]

    def __len__(self):
        return len(self.padded_data)

    def __getitem__(self, idx):
        tensor = torch.tensor(self.padded_data[idx], dtype=torch.long)
        return {'input': tensor, 'target': tensor.clone()}

    def print_sequence_length_stats(self):
        print(f"Mean sequence length: {self.mean_length}")
        print(f"Standard deviation of sequence lengths: {self.std_length}")
        print(f"Number of sequences: {len(self.raw_datasets)}")
        print(f"Number of filtered sequences: {len(self.filtered_datasets)}")


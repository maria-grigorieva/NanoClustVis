from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.signal import find_peaks
from rapidfuzz import fuzz
import itertools
import numpy as np
import random

@dataclass
class SequenceMatch:
    query_name: str
    position: int
    score: float
    length: int


class SequenceAnalyzer:
    def __init__(self, similarity_threshold: float = 0.9, window_size: int = 50):
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size

    def compare_with_wildcards(self, seq1: str, seq2: str) -> float:
        """
        Compare two sequences with wildcards (*).
        Returns similarity score between 0 and 1.
        """
        if len(seq1) != len(seq2):
            return 0.0

        matches = 0
        total = 0

        for c1, c2 in zip(seq1, seq2):
            if c1 == '*' or c2 == '*':
                continue
            total += 1
            if c1 == c2:
                matches += 1

        return matches / total if total > 0 else 0.0

    def find_matches_in_sequence(self, sequence: str, query_dict: Dict[str, str]) -> List[SequenceMatch]:
        """
        Find all matches for queries in a sequence.

        Args:
            sequence: Target sequence to search in
            query_dict: Dictionary of query names and sequences
            with_wildcards: If True, enable wildcard (*) matching
        """
        matches = []
        for query_name, query_seq in query_dict.items():
            positions = []
            scores = []

            # Calculate similarity scores along the sequence
            for pos in range(len(sequence) - len(query_seq) + 1):
                window = sequence[pos:pos + len(query_seq)]
                score = fuzz.ratio(window, query_seq) / 100.0
                if score >= self.similarity_threshold:
                    positions.append(pos)
                    scores.append(score)

            if len(positions) > 0:
                # Find peaks in similarity scores
                peaks, properties = find_peaks(scores, width=len(query_seq))

                if len(peaks) == 0:
                    fragments = [[0]]
                    for i in range(1,len(positions)):
                        if (positions[i] - positions[fragments[-1][-1]] <= len(query_seq)):
                            fragments[-1].append(i)
                        else:
                            fragments.append([i])
                    peaks = [
                        [i for i in fragment if scores[i] == max(scores[j] for j in fragment)][0]
                        for fragment in fragments
                    ]

                for peak in peaks:
                    matches.append(SequenceMatch(
                        query_name=query_name,
                        position=positions[peak],
                        length=len(query_seq),
                        score=scores[peak]
                    ))
        # ✅ Sort matches by position (ascending)
        matches.sort(key=lambda m: m.position)

        return matches
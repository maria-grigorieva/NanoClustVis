from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.signal import find_peaks
from rapidfuzz import fuzz

@dataclass
class SequenceMatch:
    query_name: str
    position: int
    score: float
    length: int


class SequenceAnalyzer:
    def __init__(self, similarity_threshold: float = 0.95, window_size: int = 50):
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size

    def find_matches_in_sequence(self, sequence: str, query_dict: Dict[str, str]) -> List[SequenceMatch]:
        """Find all matches for queries in a sequence using RapidFuzz."""
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
                peaks, properties = find_peaks(scores)

                # Manually check if the first value is a peak
                if len(scores) > 1:
                    if scores[0] > scores[1]:
                        peaks = [0]
                    if scores[len(scores)-1] > scores[len(scores)-2]:
                        peaks = [len(scores)-1]

                if len(peaks) == 0:
                    peaks = [idx for idx,pos in enumerate(positions) if scores[idx] > self.similarity_threshold]

                for peak in peaks:
                    matches.append(SequenceMatch(
                        query_name=query_name,
                        position=positions[peak],
                        length=len(query_seq),
                        score=scores[peak]
                    ))

        return matches
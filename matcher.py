from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.signal import find_peaks
from rapidfuzz import fuzz
import itertools
import numpy as np


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

    def find_query_segments(self, query_seq: str) -> List[tuple]:
        """
        Split query sequence into segments based on wildcards.
        Returns list of (segment, start_pos, is_wildcard).
        """
        segments = []
        current_segment = ""
        current_start = 0

        for i, char in enumerate(query_seq):
            if char == '*':
                if current_segment:
                    segments.append((current_segment, current_start, False))
                    current_segment = ""
                current_segment = '*'
                current_start = i
            else:
                if current_segment == '*':
                    segments.append(('*', current_start, True))
                    current_segment = ""
                    current_start = i
                current_segment += char

        if current_segment:
            segments.append((current_segment, current_start,
                             current_segment == '*'))

        return segments

    def find_matches_in_sequence(self, sequence: str, query_dict: Dict[str, str],
                                 with_wildcards: bool = True) -> List[SequenceMatch]:
        """
        Find all matches for queries in a sequence.

        Args:
            sequence: Target sequence to search in
            query_dict: Dictionary of query names and sequences
            with_wildcards: If True, enable wildcard (*) matching
        """
        if not with_wildcards:
            # Original implementation for exact matching
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
                            peaks = np.append(peaks, 0)
                        if scores[-1] > scores[-2]:
                            peaks = np.append(peaks, len(scores) - 1)

                    if len(peaks) == 0:
                        peaks = [idx for idx, pos in enumerate(positions)
                                 if scores[idx] > self.similarity_threshold]

                    for peak in peaks:
                        matches.append(SequenceMatch(
                            query_name=query_name,
                            position=positions[peak],
                            length=len(query_seq),
                            score=scores[peak]
                        ))

            return matches

        else:
            # Implementation for wildcard matching
            matches = []
            for query_name, query_seq in query_dict.items():
                # Skip if query doesn't contain wildcards
                if '*' not in query_seq:
                    # Use original matching for queries without wildcards
                    positions = []
                    scores = []
                    for pos in range(len(sequence) - len(query_seq) + 1):
                        window = sequence[pos:pos + len(query_seq)]
                        score = fuzz.ratio(window, query_seq) / 100.0
                        if score >= self.similarity_threshold:
                            positions.append(pos)
                            scores.append(score)

                    if positions:
                        peaks, _ = find_peaks(scores)
                        if len(scores) > 1:
                            if scores[0] > scores[1]:
                                peaks = np.append(peaks, 0)
                            if scores[-1] > scores[-2]:
                                peaks = np.append(peaks, len(scores) - 1)

                        if len(peaks) == 0:
                            peaks = [idx for idx, pos in enumerate(positions)
                                     if scores[idx] > self.similarity_threshold]

                        for peak in peaks:
                            matches.append(SequenceMatch(
                                query_name=query_name,
                                position=positions[peak],
                                length=len(query_seq),
                                score=scores[peak]
                            ))
                    continue

                # Split query into segments
                segments = self.find_query_segments(query_seq)

                # Find potential matches for each non-wildcard segment
                potential_matches = []
                for segment, start_pos, is_wildcard in segments:
                    if not is_wildcard and len(segment) >= 3:
                        segment_matches = []
                        for pos in range(len(sequence) - len(segment) + 1):
                            window = sequence[pos:pos + len(segment)]
                            score = fuzz.ratio(window, segment) / 100.0
                            if score >= self.similarity_threshold:
                                segment_matches.append({
                                    'position': pos,
                                    'score': score,
                                    'segment_start': start_pos,
                                    'segment_length': len(segment)
                                })
                        if segment_matches:
                            potential_matches.append(segment_matches)

                # Process potential matches
                if potential_matches:
                    for match_combination in itertools.product(*potential_matches):
                        valid_combination = True
                        for i in range(len(match_combination) - 1):
                            curr_match = match_combination[i]
                            next_match = match_combination[i + 1]

                            expected_distance = (next_match['segment_start'] -
                                                 (curr_match['segment_start'] +
                                                  curr_match['segment_length']))
                            actual_distance = (next_match['position'] -
                                               (curr_match['position'] +
                                                curr_match['segment_length']))

                            if (actual_distance < 0 or
                                    abs(actual_distance - expected_distance) > 3):
                                valid_combination = False
                                break

                        if valid_combination:
                            start_pos = match_combination[0]['position']
                            end_pos = (match_combination[-1]['position'] +
                                       match_combination[-1]['segment_length'])

                            if end_pos - start_pos == len(query_seq):
                                complete_window = sequence[start_pos:end_pos]
                                final_score = self.compare_with_wildcards(
                                    complete_window, query_seq)

                                if final_score >= self.similarity_threshold:
                                    matches.append(SequenceMatch(
                                        query_name=query_name,
                                        position=start_pos,
                                        length=len(query_seq),
                                        score=final_score
                                    ))

            return matches
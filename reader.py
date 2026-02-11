from matcher import SequenceAnalyzer, SequenceMatch
from fuzzy_searcher import FuzzySearcher
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from Bio import SeqIO
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

class FastqProcessor:
    def __init__(self, fastq_path: Path, query_dict: Dict[str, str], similarity_threshold=0.9):
        self.fastq_path = fastq_path
        self.query_dict = query_dict
        #self.sequence_analyzer = SequenceAnalyzer()
        self.sequence_analyzer = FuzzySearcher(similarity_threshold)
        self.max_sequence_length = 0
        self.sequences = None

    def process_file(self) -> List[List[SequenceMatch]]:
        """Process FASTQ file and return matches for each sequence."""
        all_sequence_matches = []
        sequence_lengths = []

        # Count total records for tqdm progress bar
        total_records = sum(1 for _ in SeqIO.parse(self.fastq_path, "fastq"))
        sequences = []

        for record in tqdm(SeqIO.parse(self.fastq_path, "fastq"), total=total_records, desc="Processing FASTQ"):
        # for record in SeqIO.parse(self.fastq_path, "fastq"):
            sequence = str(record.seq)
            sequences.append(sequence)
            sequence_lengths.append(len(sequence))
            self.max_sequence_length = max(self.max_sequence_length, len(sequence))
            #
            # matches = self.sequence_analyzer.find_matches_in_sequence(sequence, self.query_dict)
            #
            # if matches:
            #     all_sequence_matches.append(matches)
            # else:
            #     all_sequence_matches.append([])
        all_sequence_matches = self.sequence_analyzer.find_matches_in_sequences(sequences, query_dict=self.query_dict)
        sequence_lengths_df = pd.DataFrame(sequence_lengths)
        self.lengths_stats = sequence_lengths_df.describe()
        self.sequence_lengths = sequence_lengths
        self.total_records = total_records
        self.sequences = sequences
        return all_sequence_matches

    def remove_empty_records(self, data: List[List[SequenceMatch]]) -> List[List[SequenceMatch]]:
        to_remove = []
        for idx,match in enumerate(data):
            if len(match) <= 1:
                to_remove.append(idx)
        return [item for i,item in enumerate(data) if i not in to_remove]

    def filter_by_distance(self, data: List[List[SequenceMatch]], threshold: int = 40) -> List[List[SequenceMatch]]:
        filtered_data = []

        for sequence_list in data:
            new_sequence = []
            for i in range(len(sequence_list) - 1):
                current = sequence_list[i]
                next_ = sequence_list[i + 1]
                gap = next_.position - (current.position + current.length)

                if gap >= threshold:
                    new_sequence.append(current)
                    new_sequence.append(next_)

            # # Optionally: check if last element should be added
            # # E.g., if last gap was > threshold, keep last
            # if len(sequence_list) >= 2:
            #     last = sequence_list[-1]
            #     second_last = sequence_list[-2]
            #     if last.position - (second_last.position + second_last.length) > threshold:
            #         new_sequence.append(last)

            filtered_data.append(new_sequence)

        return filtered_data

    def matches_stats(self, matches):
        # Dictionary to store statistics
        stats = defaultdict(lambda: {'count': 0, 'total_score': 0, 'total_length': 0})

        total_elements = 0

        # Process the data
        for match_list in matches:
            for match in match_list:
                query = match.query_name
                stats[query]['count'] += 1
                stats[query]['total_score'] += match.score
                stats[query]['total_length'] += match.length
                total_elements += 1

        # Calculate averages
        for query in stats:
            stats[query]['average_score'] = stats[query]['total_score'] / stats[query]['count']
            stats[query]['average_length'] = stats[query]['total_length'] / stats[query]['count']
            # Remove intermediate sums
            del stats[query]['total_score']
            del stats[query]['total_length']

        # Add total elements count
        # stats['total_elements'] = total_elements
        stats['number_of_reads'] = self.total_records

        return dict(stats)

    def compute_pairwise_distances(self, records: List[List[SequenceMatch]]) -> List[List[Tuple[str, str, int]]]:
        """
        For each record (list of SequenceMatch), sort by position and compute distances between neighbors.
        Returns list of lists of tuples (query1, query2, distance).
        """
        all_distances = []

        for record in records:

            filtered_record = [m for m in record if "barcode" not in m.query_name]

            # Skip empty records or records with only one element
            if len(filtered_record) < 2:
                continue

            # Sort by position
            sorted_record = sorted(filtered_record, key=lambda match: match.position)

            # Compute distances between each neighbor
            distances = []
            for i in range(len(sorted_record) - 1):
                m1 = sorted_record[i]
                m2 = sorted_record[i + 1]
                dist = m2.position - m1.position
                distances.append((m1.query_name, m2.query_name, dist))

            all_distances.append(distances)

        return [item for sublist in all_distances for item in sublist]
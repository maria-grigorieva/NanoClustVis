from matcher import SequenceAnalyzer, SequenceMatch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from Bio import SeqIO
from collections import defaultdict
from tqdm import tqdm

class FastqProcessor:
    def __init__(self, fastq_path: Path, query_dict: Dict[str, str]):
        self.fastq_path = fastq_path
        self.query_dict = query_dict
        self.sequence_analyzer = SequenceAnalyzer()
        self.max_sequence_length = 0

    def process_file(self) -> List[List[SequenceMatch]]:
        """Process FASTQ file and return matches for each sequence."""
        all_sequence_matches = []

        # Count total records for tqdm progress bar
        total_records = sum(1 for _ in SeqIO.parse(self.fastq_path, "fastq"))

        for record in tqdm(SeqIO.parse(self.fastq_path, "fastq"), total=total_records, desc="Processing FASTQ"):
        # for record in SeqIO.parse(self.fastq_path, "fastq"):
            sequence = str(record.seq)
            self.max_sequence_length = max(self.max_sequence_length, len(sequence))

            matches = self.sequence_analyzer.find_matches_in_sequence(sequence, self.query_dict)

            if matches:  # Only append if matches were found
                all_sequence_matches.append(matches)
            else:
                all_sequence_matches.append([])

        return all_sequence_matches

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
        stats['total_elements'] = total_elements

        return stats
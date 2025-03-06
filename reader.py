from matcher import SequenceAnalyzer, SequenceMatch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from Bio import SeqIO

class FastqProcessor:
    def __init__(self, fastq_path: Path, query_dict: Dict[str, str]):
        self.fastq_path = fastq_path
        self.query_dict = query_dict
        self.sequence_analyzer = SequenceAnalyzer()
        self.max_sequence_length = 0

    def process_file(self) -> List[List[SequenceMatch]]:
        """Process FASTQ file and return matches for each sequence."""
        all_sequence_matches = []

        for record in SeqIO.parse(self.fastq_path, "fastq"):
            sequence = str(record.seq)
            self.max_sequence_length = max(self.max_sequence_length, len(sequence))

            matches = self.sequence_analyzer.find_matches_in_sequence(sequence, self.query_dict)
            if matches:  # Only append if matches were found
                all_sequence_matches.append(matches)

        return all_sequence_matches
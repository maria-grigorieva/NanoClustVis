import tempfile
import subprocess
from dataclasses import dataclass
from Bio.Align import PairwiseAligner
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class SequenceMatch:
    seq_id: str
    query_name: str
    position: int
    length: int
    score: float


class FuzzySearcher:
    def __init__(self,
                 similarity_threshold: float = 0.9):
        """
        :param similarity_threshold: минимальный относительный score (0–1), чтобы считать совпадение валидным
        """
        self.similarity_threshold = similarity_threshold
        self.aligner = PairwiseAligner()
        self.aligner.mode = "local"
        self.aligner.match_score = 1
        self.aligner.mismatch_score = -1
        self.aligner.open_gap_score = -2
        self.aligner.extend_gap_score = -0.5

    def find_matches_in_sequences(
        self,
        sequences: List[str],
        query_dict: Dict[str, str],
    ) -> List[SequenceMatch]:

        all_matches = []

        for seq_idx, target_seq in enumerate(sequences, start=0):
            matches = []

            for query_name, query_seq in query_dict.items():
                alignments = self.aligner.align(target_seq, query_seq)

                for aln in alignments:
                    if len(aln.aligned) < 2 or len(aln.aligned[0]) == 0 or len(aln.aligned[1]) == 0:
                        continue

                    target_start, target_end = aln.aligned[0][0]  # на целевой последовательности
                    query_start, query_end = aln.aligned[1][0]  # на запросе

                    aln_len = target_end - target_start
                    score = aln.score / len(query_seq)  # нормируем по длине запроса

                    if score >= self.similarity_threshold:
                        matches.append(SequenceMatch(
                            seq_id=seq_idx,
                            query_name=query_name,
                            position=target_start,
                            length=aln_len,
                            score=round(score, 3) if self.similarity_threshold < 1 else 1.0
                        ))
            matches.sort(key=lambda m: (m.seq_id, m.position))
            all_matches.append(matches)

        # Удалим дубли по позиции, если нужно — можно оставить только лучшие
        # matches = self._filter_overlaps(matches)
        return all_matches

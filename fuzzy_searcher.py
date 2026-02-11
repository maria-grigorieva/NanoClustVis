import tempfile
import subprocess
from dataclasses import dataclass, asdict
from Bio.Align import PairwiseAligner
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
import edlib
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from Bio import SeqIO

@dataclass
class SequenceMatch:
    seq_id: str
    query_name: str
    position: int
    length: int
    score: float


class FuzzySearcher:
    """
    High-performance fuzzy searcher for HUGE FASTQ files.

    Features:
    - streaming FASTQ
    - constant RAM usage
    - batched parquet writing
    - edlib C++ backend
    """

    def __init__(
        self,
        fastq_path: Path,
        output_parquet: Path,
        query_dict: Dict[str, str],
        similarity_threshold: float = 0.9,
        batch_size: int = 50_000,
    ):
        self.fastq_path = Path(fastq_path)
        self.output_parquet = Path(output_parquet)
        self.query_dict = query_dict
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size

        self.schema = pa.schema([
            ("seq_id", pa.string()),
            ("query_name", pa.string()),
            ("position", pa.int32()),
            ("length", pa.int16()),
            ("score", pa.float32()),
        ])

    # ✅ converts similarity → max edit distance
    def _max_edits(self, query_len: int):
        return int(query_len * (1 - self.similarity_threshold))

    def search(self):

        writer = pq.ParquetWriter(self.output_parquet, self.schema)

        batch = []

        try:
            for record in tqdm(
                SeqIO.parse(self.fastq_path, "fastq"),
                desc="Scanning FASTQ",
            ):
                seq = str(record.seq)

                for query_name, query_seq in self.query_dict.items():

                    max_edits = self._max_edits(len(query_seq))

                    result = edlib.align(
                        query_seq,
                        seq,
                        mode="HW",          # substring search
                        task="locations",
                        k=max_edits
                    )

                    if result["editDistance"] == -1:
                        continue

                    score = 1 - (result["editDistance"] / len(query_seq))

                    for start, end in result["locations"]:

                        batch.append({
                            "seq_id": record.id,
                            "query_name": query_name,
                            "position": start,
                            "length": end - start,
                            "score": score,
                        })

                # 🔥 flush batch
                if len(batch) >= self.batch_size:
                    table = pa.Table.from_pylist(batch, schema=self.schema)
                    writer.write_table(table)
                    batch.clear()

            # final flush
            if batch:
                table = pa.Table.from_pylist(batch, schema=self.schema)
                writer.write_table(table)

        finally:
            writer.close()

        print("✅ Finished. Parquet saved to:", self.output_parquet)

    # def find_matches_in_sequences(
    #     self,
    #     sequences: List[str],
    #     query_dict: Dict[str, str],
    # ) -> List[SequenceMatch]:
    #
    #     all_matches = []
    #
    #     for seq_idx, target_seq in enumerate(sequences, start=0):
    #         matches = []
    #
    #         for query_name, query_seq in query_dict.items():
    #             alignments = self.aligner.align(target_seq, query_seq)
    #
    #             for aln in alignments:
    #                 if len(aln.aligned) < 2 or len(aln.aligned[0]) == 0 or len(aln.aligned[1]) == 0:
    #                     continue
    #
    #                 target_start, target_end = aln.aligned[0][0]  # на целевой последовательности
    #                 query_start, query_end = aln.aligned[1][0]  # на запросе
    #
    #                 aln_len = target_end - target_start
    #                 score = aln.score / len(query_seq)  # нормируем по длине запроса
    #
    #                 if score >= self.similarity_threshold:
    #                     matches.append(SequenceMatch(
    #                         seq_id=seq_idx,
    #                         query_name=query_name,
    #                         position=target_start,
    #                         length=aln_len,
    #                         score=round(score, 3) if self.similarity_threshold < 1 else 1.0
    #                     ))
    #         matches.sort(key=lambda m: (m.seq_id, m.position))
    #         all_matches.append(matches)
    #
    #     # Удалим дубли по позиции, если нужно — можно оставить только лучшие
    #     # matches = self._filter_overlaps(matches)
    #     return all_matches

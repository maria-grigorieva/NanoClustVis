from Bio import SeqIO
from matcher import SequenceAnalyzer
from reader import FastqProcessor
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from matcher import SequenceMatch
from Bio.SeqRecord import SeqRecord

# Define primers
PL = "CTTCATGGATCCTGCTCTCG"

def reverse_complement(seq: str) -> str:
    """Generate reverse complement of a DNA sequence."""
    complement = str.maketrans("ATGC", "TACG")
    return seq.translate(complement)[::-1]

RC_PR = reverse_complement("GGCCCTAAAGCTTAGCACGA")

# Compute minimum required distance
min_distance = min(len(PL), len(RC_PR))

# def split_strings(sequence_matches: List[List[SequenceMatch]],
#                   input_fastq):
#
#     split_by = []
#
#     for idx,matches in enumerate(sequence_matches):
#         if len(matches) > 0:
#             # Step 1: Remove 'Barcode' matches
#             matches = [match for match in matches if "Barcode" not in match.query_name]
#
#             # Step 2: Sort matches by position
#             matches.sort(key=lambda match: match.position)
#
#             # Step 3: Filter sequences
#             for i in range(len(matches)-1):
#                 if matches[i].position == matches[i+1].position + 1:
#                     split_by.append(matches[i].position)
#
#     for index, record in enumerate(SeqIO.parse(input_fastq, "fastq")):
#
#         split_seq = record.seq[rec['start_pos']:rec['end_pos']]
#         trimmed_qual = record.letter_annotations["phred_quality"][rec['start_pos']:rec['end_pos']]
#
#         # Create a new FASTQ record
#         new_record = SeqRecord(
#             trimmed_seq,
#             id=record.id,
#             description=record.description,
#             letter_annotations={"phred_quality": trimmed_qual}  # Preserve quality scores
#         )
#
#         selected_records.append(new_record)
#
#     return split_by

def filter_sequences(sequence_matches: List[List[SequenceMatch]], min_distance_threshold: int) -> List[
    List[SequenceMatch]]:
    """
    Filters sequences based on the distance between matches.

    1) Removes matches where 'query_name' contains 'Barcode'.
    2) Sorts matches by position in ascending order.
    3) Calculates distances between all pairs of matches.
    4) Filters out sequences where any distance is below a threshold.

    :param sequence_matches: List of lists of SequenceMatch objects
    :param min_distance_threshold: Minimum allowed distance between any two matches
    :return: Filtered list of sequences
    """
    filtered_sequences = []

    for idx,matches in enumerate(sequence_matches):
        if len(matches) > 0:
            # Step 1: Remove 'Barcode' matches
            matches = [match for match in matches if "Barcode" not in match.query_name]

            # Step 2: Sort matches by position
            matches.sort(key=lambda match: match.position)

            # Step 3: Filter sequences
            for i in range(len(matches)-1):
                # left - right
                if (matches[i].query_name == 'Left' or
                        matches[i].query_name == 'RCRight') and (matches[i+1].query_name == 'Right' or
                        matches[i+1].query_name == 'RCLeft'):
                    distance = abs(matches[i].position + matches[i].length - matches[i+1].position)
                    if distance >= min_distance_threshold:
                        filtered_sequences.append({'id': idx,
                                                   'start_pos': matches[i].position,
                                                   'end_pos': matches[i+1].position + matches[i+1].length})
            #
            # # Step 3: Compute distances between all pairs
            # distances = [
            #     abs(matches[i].position + matches[i].length - matches[j].position)
            #     for i in range(len(matches))
            #     for j in range(i + 1, len(matches))
            # ]
            #
            # # Step 4: Filter out sequences where any distance is too small
            # if any(dist >= min_distance_threshold for dist in distances):
            #     filtered_sequences.append(idx)

    return filtered_sequences


fastq_file = "FBA01901_barcode04_filtered_indices.fastq"
query_dict = {
    "Left": "CTTCATGGATCCTGCTCTCG",
    "RCLeft": reverse_complement("CTTCATGGATCCTGCTCTCG"),
    "Right": "GGCCCTAAAGCTTAGCACGA",
    "RCRight": reverse_complement("GGCCCTAAAGCTTAGCACGA"),
}

# Process FASTQ file
processor = FastqProcessor(fastq_file, query_dict)
sequence_matches = processor.process_file()

# Example usage
filtered = filter_sequences(sequence_matches, min_distance_threshold=30)
print(filtered)

from pathlib import Path

def filter_out_sequences(input_fastq: str, output_fastq: str, filtered: list):
    """
    Filters sequences in a FASTQ file based on provided indices and position ranges.

    :param input_fastq: Path to the input FASTQ file
    :param output_fastq: Path to the output FASTQ file
    :param filtered: List of dictionaries containing 'id', 'start_pos', 'end_pos'
    """
    selected_records = []
    n_records = 0

    for index, record in enumerate(SeqIO.parse(input_fastq, "fastq")):
        n_records += 1
        for rec in filtered:
            if rec['id'] == index:  # Match sequence index
                trimmed_seq = record.seq[rec['start_pos']:rec['end_pos']]
                trimmed_qual = record.letter_annotations["phred_quality"][rec['start_pos']:rec['end_pos']]

                # Create a new FASTQ record
                new_record = SeqRecord(
                    trimmed_seq,
                    id=record.id,
                    description=record.description,
                    letter_annotations={"phred_quality": trimmed_qual}  # Preserve quality scores
                )

                selected_records.append(new_record)

    print(f'Selected records: {len(selected_records)}')
    print(f'Number of records: {n_records}')

    # Save filtered sequences in FASTQ format
    with open(output_fastq, 'w') as outfile:
        SeqIO.write(selected_records, outfile, "fastq")

def filter_fastq_by_indices(input_fastq: str, output_fastq: str, indices: list):
    """
    Filters sequences by their indices from a FASTQ file and writes them to a new file.

    :param input_fastq: Path to the input FASTQ file
    :param output_fastq: Path to the output FASTQ file with filtered sequences
    :param indices: List of sequence indices to retain
    """
    indices_set = set(indices)  # Convert to set for faster lookup
    selected_records = []

    with open(input_fastq, 'r') as infile:
        for index, record in enumerate(zip(*[infile] * 1000)):  # Read FASTQ in chunks of 1000 lines
            if index in indices_set:
                selected_records.extend(record)  # Append matching records

    # Save filtered sequences to a new FASTQ file
    with open(output_fastq, 'w') as outfile:
        outfile.writelines(selected_records)

output_fastq_path = "FBA01901_barcode04_trimmed_filtered.fastq"
filter_out_sequences(fastq_file, output_fastq_path, filtered)

def fastq_to_fasta(input_fastq: str, output_fasta: str):
    """
    Convert a FASTQ file to a FASTA file.
    """
    with open(output_fasta, "w") as fasta_handle:
        SeqIO.write(SeqIO.parse(input_fastq, "fastq"), fasta_handle, "fasta")

fastq_to_fasta(output_fastq_path, 'FBA01901_barcodes_5_6_7_8_trimmed_filtered.fasta')

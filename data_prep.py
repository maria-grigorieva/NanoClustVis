import pandas as pd
from pathlib import Path
# from dataclasses import dataclass
from reader import FastqProcessor, SequenceMatch
from HeatMapVis import SequenceVisualizer
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict
from Bio import SeqIO
# from sklearn.metrics import silhouette_score
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

def reverse_complement(seq: str) -> str:
    """Generate reverse complement of a DNA sequence."""
    complement = str.maketrans("ATGC", "TACG")
    return seq.translate(complement)[::-1]

def record_reverse_complement(record: SeqRecord) -> SeqRecord:
    """
    Return a new SeqRecord with the reverse complement of the given sequence.
    """
    return SeqRecord(
        seq=record.seq.reverse_complement(),
        id=record.id + "_RC",
        description=record.description + " reverse complement",
        letter_annotations=record.letter_annotations
    )

def save_sequence_matches_to_csv(sequence_matches: List[List[SequenceMatch]], output_csv: str):
    """Save sequence matches to a CSV file in an efficient format."""
    records = []

    for idx, matches in enumerate(sequence_matches):
        for match in matches:
            records.append({
                "index": idx,
                "query_name": match.query_name,
                "position": match.position,
                "score": match.score,
                "length": match.length
            })

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

def select_valid_indices(sequence_matches: List[List[SequenceMatch]],
                         auto_rc = False) -> List[int]:
    """Return indices where both Aptamer/1 and Aptamer/2 or their RC versions exist."""
    selected_indices_direct = []
    selected_indices_rc = []

    for idx, matches in enumerate(sequence_matches):
        query_names = {match.query_name for match in matches}

        if not auto_rc:
            if ({"Aptamer/1", "Aptamer/2"} <= query_names) or \
                    ({"Aptamer/1/RC", "Aptamer/2/RC"} <= query_names):
                selected_indices_direct.append(idx)
        else:
            if ({"Aptamer/1", "Aptamer/2"} <= query_names):
                selected_indices_direct.append(idx)
            if ({"Aptamer/1/RC", "Aptamer/2/RC"} <= query_names):
                selected_indices_rc.append(idx)

    return selected_indices_direct, selected_indices_rc

def filter_fastq_by_indices(input_fastq: str, output_fastq: str,
                            selected_valid_indices_direct: List[int],
                            selected_valid_indices_rc: List[int]) -> List[int]:
    """
    Read a FASTQ file, select only records with indices in select_valid_indices,
    and write them to a new FASTQ file.
    """
    selected_records = []

    for index, record in enumerate(SeqIO.parse(input_fastq, "fastq")):
        if index in selected_valid_indices_direct:
            selected_records.append(record)
        if len(selected_valid_indices_rc) > 0:
            if index in selected_valid_indices_rc:
                selected_records.append(record_reverse_complement(record))

    # Write the selected records to a new FASTQ file
    with open(output_fastq, "w") as out_handle:
        SeqIO.write(selected_records, out_handle, "fastq")

def filter_sequences(sequence_matches: List[List[SequenceMatch]],
                     min_distance_threshold: int) -> List[
    List[SequenceMatch]]:

    filtered_sequences = []

    for idx, matches in enumerate(sequence_matches):
        if len(matches) > 0:
            # Step 1: Remove 'Barcode' matches
            matches = [match for match in matches if "Barcode" not in match.query_name]

            # Step 2: Sort matches by position
            matches.sort(key=lambda match: match.position)

            # Step 3: Filter sequences
            for i in range(len(matches) - 1):
                # left - right
                if ((matches[i].query_name == 'RCRight' and matches[i + 1].query_name == 'RCLeft')
                       or
                        (matches[i].query_name == 'Left' and matches[i + 1].query_name == 'Right')):
                    #     or
                    # (matches[i].query_name == 'Right' and matches[i + 1].query_name == 'RCLeft')):
                # if (matches[i].query_name == 'Left' or
                #     matches[i].query_name == 'RCRight') and (matches[i + 1].query_name == 'Right' or
                #                                              matches[i + 1].query_name == 'RCLeft'):
                    distance = abs(matches[i].position + matches[i].length - matches[i + 1].position)
                    if distance >= min_distance_threshold:
                        filtered_sequences.append({'id': idx,
                                                   'start_pos': matches[i].position + matches[i].length,
                                                   'end_pos': matches[i + 1].position})
                        # filtered_sequences.append({'id': idx,
                        #                            'start_pos': matches[i].position,
                        #                            'end_pos': matches[i + 1].position + matches[i + 1].length})

    return filtered_sequences

def remove_duplicates(dict_list):
    seen = set()
    unique_dicts = []
    for d in dict_list:
        # Convert dict to a hashable form (tuple of sorted key-value pairs)
        dict_tuple = tuple(sorted(d.items()))
        if dict_tuple not in seen:
            seen.add(dict_tuple)
            unique_dicts.append(d)
    return unique_dicts

def filter_by_neighbour_distances(sequence_matches: List[List[SequenceMatch]],
                     rules: List[Dict],
                     min_distance_threshold: int,
                     max_distance_threshold: int,
                     include=False) -> List[
    List[SequenceMatch]]:

    filtered_sequences = []

    for idx, matches in enumerate(sequence_matches):
        if len(matches) > 2:
            matches.sort(key=lambda match: match.position)
            # print(len(matches))
            # print(matches)
            for rule in rules:
                start = rule['start']
                end = rule['end']
                orientation = rule['orientation']
                for i in range(len(matches) - 1):
                    if (matches[i].query_name == start and matches[i+1].query_name == end):
                        distance = matches[i+1].position - (matches[i].position + matches[i].length)
                        if (distance >= min_distance_threshold and distance <= max_distance_threshold):
                            filtered_sequences.append({'id': matches[i].seq_id,
                                                       'start_pos': matches[i].position + matches[i].length if not include else matches[i].position,
                                                       'end_pos': matches[i + 1].position if not include else matches[i + 1].position + matches[i+1].length,
                                                       'orientation': orientation})

    return remove_duplicates(filtered_sequences)

def filter_by_distance_from_primer(sequence_matches: List[List[SequenceMatch]],
                     rules: List[Dict]) -> List[List[SequenceMatch]]:

    filtered_sequences = []

    for idx, matches in enumerate(sequence_matches):
        if len(matches) > 2:
            matches.sort(key=lambda match: match.position)
            for rule in rules:
                primer = rule['primer']
                dist_min = rule['distance_min']
                dist_max = rule['distance_max']
                orientation = rule['orientation']
                for match in matches:
                    if (match.query_name == primer):
                        filtered_sequences.append({'id': match.seq_id,
                                                   'start_pos': match.position + match.length,
                                                   'end_pos': match.position + match.length + dist_max,
                                                   'orientation': orientation})
    return remove_duplicates(filtered_sequences)

def trim_sequences(input_fastq: str, output_fastq: str, filtered: list):

    selected_records = []
    n_records = 0
    unique_id_counter = 1  # Counter for generating unique numeric IDs


    for index, record in enumerate(SeqIO.parse(input_fastq, "fastq")):
        n_records += 1
        for rec in filtered:
            if rec['id'] == index:  # Match sequence index
                trimmed_seq = record.seq[rec['start_pos']:rec['end_pos']]
                if len(str(trimmed_seq)) >= rec['end_pos'] - rec['start_pos']:
                    trimmed_qual = record.letter_annotations["phred_quality"][rec['start_pos']:rec['end_pos']]

                    # Ensure it's a Seq object
                    seq_obj = Seq(trimmed_seq)

                    # Then reverse complement if needed
                    seq_final = seq_obj if rec['orientation'] == 'direct' else seq_obj.reverse_complement()

                    # Create a new FASTQ record
                    new_record = SeqRecord(
                        seq_final,
                        id=str(unique_id_counter),
                        description="",
                        letter_annotations={"phred_quality": trimmed_qual}  # Preserve quality scores
                    )

                    selected_records.append(new_record)
                    unique_id_counter += 1

    print(f'Selected records: {len(selected_records)}')
    print(f'Number of records: {n_records}')

    # Save filtered sequences in FASTQ format
    with open(output_fastq, 'w') as outfile:
        SeqIO.write(selected_records, outfile, "fastq")

# def filter_fastq_by_indices(input_fastq: str, output_fastq: str, indices: list):
#     """
#     Filters sequences by their indices from a FASTQ file and writes them to a new file.
#
#     :param input_fastq: Path to the input FASTQ file
#     :param output_fastq: Path to the output FASTQ file with filtered sequences
#     :param indices: List of sequence indices to retain
#     """
#     indices_set = set(indices)  # Convert to set for faster lookup
#     selected_records = []
#
#     with open(input_fastq, 'r') as infile:
#         for index, record in enumerate(zip(*[infile] * 1000)):  # Read FASTQ in chunks of 1000 lines
#             if index in indices_set:
#                 selected_records.extend(record)  # Append matching records
#
#     # Save filtered sequences to a new FASTQ file
#     with open(output_fastq, 'w') as outfile:
#         outfile.writelines(selected_records)

def fastq_to_fasta(input_fastq: str, output_fasta: str):
    """
    Convert a FASTQ file to a FASTA file.
    """
    with open(output_fasta, "w") as fasta_handle:
        SeqIO.write(SeqIO.parse(input_fastq, "fastq"), fasta_handle, "fasta")
def main():
    fastq_file = Path("data_samples/new_sequencing_may2025/reads/barcode14.fastq")
    query_dict = {
        "RCRight|R": "CTTTAGGGCC",
        "RCLeft|L": "CGAGAGCAG",
        "Left|R": "CCTGCTCTCG",
        "Right|L": "GGCCCTAAAG",
        "RCLeft|R": "ATCCATGAAG",
        "Left|L": "CTTCATGGAT",
        "Right|R": "CTTAGCACGA",
        "RCRight|L": "TCGTGCTAAG",
        # "Right/c": "TAAAGCTTAG",
        # "Left/c": "TGGATCCTGC",
        # "RCRight/c": "CTAAGCTTTA",
        # "RCLeft/c": "GCAGGATCCA"
    }

    # query_dict = {
    #     "Left": "CTTCATGGATCCTGCTCTCG",
    #     "RCLeft": "CGAGAGCAGGATCCATGAAG",
    #     "Right": "GGCCCTAAAGCTTAGCACGA",
    #     "RCRight": "TCGTGCTAAGCTTTAGGGCC",
    #     # "RCLeft-Left": "CGAGAGCNGGATCCNGCTCTCG",
    #     # "Right-RCRight": "GGCCCNAAAGCTTTAGGGCC"
    # }
    # query_dict = {
    #     # "Aptamer/1": "TGC**CGAGAGC",
    #     # "Aptamer/2": "GGGCC*GCA",
    #     # "Aptamer/1/RC": reverse_complement("TGC**CGAGAGC"),
    #     # "Aptamer/2/RC": reverse_complement("GGGCC*GCA"),
    #     "Left": "CTTCATGGATCCTGCTCTCG",
    #     "RCLeft": reverse_complement("CTTCATGGATCCTGCTCTCG"),
    #     "Right": "GGCCCTAAAGCTTAGCACGA",
    #     "RCRight": reverse_complement("GGCCCTAAAGCTTAGCACGA"),
    #     "Barcode": "CGTCAACTGACAGTGGTTCGTACT",
    #     "Barcode RC": "AGTACGAACCACTGTCAGTTGACG"
    # }

    # Process FASTQ file
    processor = FastqProcessor(fastq_file, query_dict,1.0)
    sequence_matches = processor.process_file()
    df = pd.DataFrame(sequence_matches)
    # df.to_csv('sequence_matches.csv')
    #print(sequence_matches)
    # rules = [
    #     {"primer": "Left/r", "distance_min": 50, "distance_max": 100, "orientation": "direct"},]
    #          # {"primer": "Right/r", "distance_min": 50, "distance_max": 100, "orientation": "rc"}]
    # filtered = filter_by_distance_from_primer(sequence_matches, rules)
    rules = [{"start": "Left|R", "end": "Right|L", "orientation": "direct"}]
            # {"start": "RCRight/r", "end": "RCLeft/l", "orientation": "rc"}]
    filtered = filter_by_neighbour_distances(sequence_matches, rules, 0, 0, True)
    # filtered = filter_sequences(sequence_matches, 30)
    print(filtered)
    output_fastq_path = "junctions/barcode14_trimmed_left_right_0.fastq"
    trim_sequences(fastq_file, output_fastq_path, filtered)
    fastq_to_fasta(output_fastq_path, 'junctions/barcode14_trimmed_left_right_0.fasta')


    # direct_indices, rc_indices = select_valid_indices(sequence_matches)
    # filter_fastq_by_indices(fastq_file, 'filtered.fastq',
    #                         direct_indices, rc_indices)
    #
    # filtered = filter_sequences(sequence_matches, min_distance_threshold=30)
    # print(filtered)
    #
    # output_fastq_path = "trimmed.fastq"
    # trim_sequences(fastq_file, output_fastq_path, filtered)
    # fastq_to_fasta(output_fastq_path, 'trimmed.fasta')

if __name__ == "__main__":
    main()
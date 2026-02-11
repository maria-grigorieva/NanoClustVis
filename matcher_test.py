from reader import FastqProcessor
import glob
import os
from pathlib import Path

fastq_file = Path("data_samples/new_sequencing_may2025/reads/barcode06.fastq")

artificial_sequences = {
    #"Left": "CTTCATGGATCCNGCTCTCG",
    #"RCLeft": "CGAGAGCAGGATCCATGAAG",
    "Right": "GGCCCNAAAGCTTAGCACGA",
    #"RCRight": "TCGTGCTAAGCTTTAGGGCC",
    #"RCLeft-Left": "CGAGAGCNGGATCCNGCTCTCG",
    #"Right-RCRight": "GGCCCNAAAGCTTTAGGGCC",
    #"Barcode06": "AAGGTTAAGACTACTTTCTGCCTTTGCGAGAACAGCACCT"
}

processor = FastqProcessor(fastq_file, artificial_sequences)

sequence_matches = processor.process_file()

sequence_matches
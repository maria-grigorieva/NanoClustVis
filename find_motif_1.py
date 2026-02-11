from gimmemotifs.motif import Motif
from gimmemotifs.motif import read_motifs
from gimmemotifs.tools import Meme
from Bio import SeqIO

# Читаем последовательности
fastq_file = "barcode06_trimmed_direct.fastq"
sequences = [str(record.seq) for record in SeqIO.parse(fastq_file, "fastq")]

# Найти минимальную длину и обрезать последовательности
min_length = min(len(seq) for seq in sequences)
trimmed_sequences = [seq[:min_length] for seq in sequences]

# Записываем последовательности во временный файл для MEME
with open("sequences.fasta", "w") as f:
    for i, seq in enumerate(trimmed_sequences):
        f.write(f">seq{i+1}\n{seq}\n")

# # Запускаем MEME через Python (может потребоваться установить `meme-suite`)
# meme = Meme()
# motifs = meme.run("sequences.fasta")
#
# # Вывод найденных мотивов
# for motif in motifs:
#     print("Motif:", motif.consensus)

import subprocess

# Run MEME from the command line
subprocess.run(["meme", "sequences.fasta", "-o", "output", "-mod", "zoops", "-minlength", "6", "-maxlength", "50"])
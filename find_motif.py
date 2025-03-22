from Bio import SeqIO
from Bio import motifs

# Читаем последовательности из FASTQ-файла
fastq_file = "cd133_trimmed.fastq"
sequences = [str(record.seq) for record in SeqIO.parse(fastq_file, "fastq")]

# Найти минимальную длину последовательности
min_length = min(len(seq) for seq in sequences)
trimmed_sequences = [seq[:min_length] for seq in sequences]

# Создаем мотив
m = motifs.create(trimmed_sequences)

# Создаем позиционно-весовую матрицу (PWM)
pwm = m.counts.normalize()

# Поиск консенсусного мотива
print("Consensus Motif:", m.consensus)

# Вывод PWM для анализа
print("Position Weight Matrix:")
print(pwm)
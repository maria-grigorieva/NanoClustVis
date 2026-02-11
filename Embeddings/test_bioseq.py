from BioSeqDataset import BioSeqDataset
from SequenceAutoencoder import Autoencoder
fastq_file = 'data_samples/merged.fastq'
dataset = BioSeqDataset(fastq_file)
print(dataset.get_vocab())
print(dataset.decode([1, 2, 3]))

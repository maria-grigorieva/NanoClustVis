import pandas as pd
from compressor import (IntelligentCompressor, KMeansClustering, HierarchicalClustering, DBSCANClustering, \
    # Word2VecEmbedding,
                        HDBSCANClustering, BIRCHClustering, OPTICSClustering, GMMClustering, MeanShiftClustering, \
    AffinityPropagationClustering, SpectralClusteringMethod)
from pathlib import Path
from reader import FastqProcessor, SequenceMatch
from HeatMapVis import SequenceVisualizer
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Embeddings import Word2VecEmbedding


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


def main():
    fastq_file = Path("data_samples/bigfile.fastq")
    # query_dict = {
    #     "Left": "ACCTAGAGGTAAGGCAGGGG",
    #     "LeftRC": reverse_complement("ACCTAGAGGTAAGGCAGGGG"),
    #     "Right": "TGACAGGAAAAGGTACGGGC",
    #     "RightRC": reverse_complement("TGACAGGAAAAGGTACGGGC"),
    #     "Barcode": "AAGGTTACACAAACCCTGGACAAG",
    #     "BarcodeRC": "CTTGTCCAGGGTTTGTGTAACCTT"
    # }
    query_dict = {
        "Left": "CTCCTCTGACTGTAACCACG",
        "Left/RC": reverse_complement("CTCCTCTGACTGTAACCACG"),
        "Right": "GGCTTCTGGACTACCTATGC",
        "Right/RC": reverse_complement("GGCTTCTGGACTACCTATGC"),
        # "Barcode 01": "AAGGTTAACACAAAGACACCGACAACTTTCTTCAGCACCT",
        # "Barcode 01/RC": reverse_complement("AAGGTTAACACAAAGACACCGACAACTTTCTTCAGCACCT"),
        # "Barcode 03": "AAGGTTAACCTGGTAACTGGGACACAAGACTCAGCACCT",
        # "Barcode 03/RC": reverse_complement("AAGGTTAACCTGGTAACTGGGACACAAGACTCAGCACCT"),
        # "Barcode 05": "AAGGTTAAAAGGTTACACAAACCCTGGACAAGCAGCACCT",
        # "Barcode 05/RC": reverse_complement("AAGGTTAAAAGGTTACACAAACCCTGGACAAGCAGCACCT"),
        # "Barcode 08": "ACGTAACTTGGTTTGTTCCCTGAA",
        # "Barcode 08/RC": reverse_complement("ACGTAACTTGGTTTGTTCCCTGAA"),
        # "Barcode 10": "GAGAGGACAAAGGTTTCAACGCTT",
        # "Barcode 10/RC": reverse_complement("GAGAGGACAAAGGTTTCAACGCTT"),
        # "Barcode 11": "TCCATTCCCTCCGATAGATGAAAC",
        # "Barcode 11/RC": reverse_complement("TCCATTCCCTCCGATAGATGAAAC"),
        # "Barcode 12": "TCCGATTCTGCTTCTTTCTACCTG",
        # "Barcode 12/RC": reverse_complement("TCCGATTCTGCTTCTTTCTACCTG"),
        # "Barcode 13": "AGAACGACTTCCATACTCGTGTGA",
        # "Barcode 13/RC": reverse_complement("AGAACGACTTCCATACTCGTGTGA"),
        # "Barcode 14": "AACGAGTCTCTTGGGACCCATAGA",
        # "Barcode 14/RC": reverse_complement("AACGAGTCTCTTGGGACCCATAGA"),
        # "Barcode 15": "AGGTCTACCTCGCTAACACCACTG",
        # "Barcode 15/RC": reverse_complement("AGGTCTACCTCGCTAACACCACTG"),
        # "Barcode 16": "CGTCAACTGACAGTGGTTCGTACT",
        # "Barcode 16/RC": reverse_complement("CGTCAACTGACAGTGGTTCGTACT")
    }
    # query_dict = {
    #     "Left": "CTCCTCTGACTGTAACCACG",
    #     "RCLeft": reverse_complement("CTCCTCTGACTGTAACCACG"),
    #     "Right": "GGCTTCTGGACTACCTATGC",
    #     "RCRight": reverse_complement("GGCTTCTGGACTACCTATGC"),
    #     "Barcode 01": "AAGGTTAACACAAAGACACCGACAACTTTCTTCAGCACCT",
    #     "Barcode 01/RC": reverse_complement("AAGGTTAACACAAAGACACCGACAACTTTCTTCAGCACCT"),
    #     "Barcode 02": "AAGGTTAAACAGACGACTACAAACGGAATCGACAGCACCT",
    #     "Barcode 02/RC": reverse_complement("AAGGTTAAACAGACGACTACAAACGGAATCGACAGCACCT"),
    #     "Barcode 03": "AAGGTTAACCTGGTAACTGGGACACAAGACTCAGCACCT",
    #     "Barcode 03/RC": reverse_complement("AAGGTTAACCTGGTAACTGGGACACAAGACTCAGCACCT"),
    #     "Barcode 04": "AAGGTTAATAGGGAAACACGATAGAATCCGAACAGCACCT",
    #     "Barcode 04/RC": reverse_complement("AAGGTTAATAGGGAAACACGATAGAATCCGAACAGCACCT"),
    #     "Barcode 05": "AAGGTTAAAAGGTTACACAAACCCTGGACAAGCAGCACCT",
    #     "Barcode 05/RC": reverse_complement("AAGGTTAAAAGGTTACACAAACCCTGGACAAGCAGCACCT"),
    #     "Barcode 11": "AAGGTTAATCCATTCCCTCCGATAGATGAAACCAGCACCT",
    #     "Barcode 11/RC": reverse_complement("AAGGTTAATCCATTCCCTCCGATAGATGAAACCAGCACCT"),
    #     "Barcode 12": "AAGGTTAATCCGATTCTGCTTCTTTCTACCTGCAGCACCT",
    #     "Barcode 12/RC": reverse_complement("AAGGTTAATCCGATTCTGCTTCTTTCTACCTGCAGCACCT"),
    #     "Aptamer": "TAGGGAAACACGATAGAATCCGAACAGCACC",
    #     "Motif": "GTGCGTGTTGGGGTGTGTATGTTTTGCGTG"
    # }
    # query_dict = {
    #     "Aptamer/1": "TGC**CGAGAGC",
    #     "Aptamer/2": "GGGCC*GCA",
    #     "Aptamer/1/RC": reverse_complement("TGC**CGAGAGC"),
    #     "Aptamer/2/RC": reverse_complement("GGGCC*GCA"),
    #     "Left": "CTTCATGGATCCTGCTCTCG",
    #     "RCLeft": reverse_complement("CTTCATGGATCCTGCTCTCG"),
    #     "Right": "GGCCCTAAAGCTTAGCACGA",
    #     "RCRight": reverse_complement("GGCCCTAAAGCTTAGCACGA"),
    #     "Barcode": "CGTCAACTGACAGTGGTTCGTACT",
    #     "Barcode RC": "AGTACGAACCACTGTCAGTTGACG"
    # }

    # Process FASTQ file
    processor = FastqProcessor(fastq_file, query_dict)
    sequence_matches = processor.process_file()

    print(processor.matches_stats(sequence_matches))

    # visualize RAW fastq file
    # visualizer = SequenceVisualizer(query_dict)
    # fig = visualizer.plot_raw_heatmap(sequence_matches)
    # fig.savefig(f'heatmap_raw.png',
    #             bbox_inches='tight',
    #             dpi=300)
    # plt.close()


    # Try different clustering methods
    clustering_methods = {
        'hierarchical': HierarchicalClustering(),
        # 'kmeans': KMeansClustering(),
        # 'dbscan': DBSCANClustering(),
        'hdbscan': HDBSCANClustering(),
        # 'birch': BIRCHClustering(),
        'optics': OPTICSClustering(),
        # 'GMM': GMMClustering(),
        # 'meanshift': MeanShiftClustering(),
        # 'affinity': AffinityPropagationClustering(),
        # 'spectral': SpectralClusteringMethod()
    }

    embedding_methods = {
        'word2vec': Word2VecEmbedding(),
        # 'bert': BERTEmbedding()
    }

    embed = Word2VecEmbedding()
    features = embed.embed_sequences(sequence_matches)
    if len(features) >= 10000:
        features, indices = embed.reduce_samples(features)
        sequence_matches = [sequence_matches[i] for i in indices]

    results = {}

    for clustering_name, clustering_method in clustering_methods.items():
        for embedding_name, embedding_method in embedding_methods.items():
            print(f"\nTesting {clustering_name} clustering with {embedding_name} embeddings:")

            compressor = IntelligentCompressor(
                target_clusters=100,
                clustering_method=clustering_method,
                embedding_method=embedding_method
            )

            try:
                # Perform compression and clustering
                compressed_matches, cluster_assignments = compressor.compress_sequences(
                    sequence_matches,
                    features,
                    max_width=max(match.position + match.length
                                  for matches in sequence_matches
                                  for match in matches),
                    combined=False,
                    save_results=True,
                    output_file=f'{clustering_name}_results.csv'
                )

                # Create visualization
                visualizer = SequenceVisualizer(query_dict)
                vis_matrix = visualizer.create_visualization_matrix(compressed_matches)

                # Save plot with method names
                fig = visualizer.plot_heatmap(
                    vis_matrix,
                    cluster_assignments,
                    title=f'Sequences clustered with {clustering_name} and {embedding_name}'
                )
                fig.savefig(f'heatmap_{clustering_name}_{embedding_name}.png',
                            bbox_inches='tight',
                            dpi=300)
                plt.close()

                # Store results
                results[f"{clustering_name}_{embedding_name}"] = {
                    'n_clusters': len(np.unique(cluster_assignments)),
                    'compressed_size': len(compressed_matches)
                }

                # Print clustering statistics
                print("\nClustering Statistics:")
                print(f"Original sequences: {len(sequence_matches)}")
                print(f"Compressed to: {len(compressed_matches)} clusters")
                # print(f"Silhouette score: {silhouette_score(vis_matrix, cluster_assignments):.3f}")

                # Print cluster size distribution
                unique_clusters = np.unique(cluster_assignments)
                print("\nCluster size distribution:")
                for cluster in unique_clusters:
                    size = np.sum(cluster_assignments == cluster)
                    print(f"Cluster {cluster}: {size} sequences")

            except Exception as e:
                print(f"Error: {str(e)}")
                import traceback
                traceback.print_exc()

    # Print comparison results
    print("\nClustering Results Comparison:")
    for method, stats in results.items():
        print(f"\n{method}:")
        print(f"Number of clusters: {stats['n_clusters']}")
        print(f"Compressed sequences: {stats['compressed_size']}")


if __name__ == "__main__":
    main()
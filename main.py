from compressor import IntelligentCompressor, KMeansClustering, HierarchicalClustering, DBSCANClustering, \
    Word2VecEmbedding, HDBSCANClustering, BIRCHClustering, OPTICSClustering, GMMClustering, MeanShiftClustering, \
    AffinityPropagationClustering, SpectralClusteringMethod
from pathlib import Path
# from dataclasses import dataclass
from reader import FastqProcessor, SequenceMatch
from HeatMapVis import SequenceVisualizer
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.metrics import silhouette_score


def reverse_complement(seq: str) -> str:
    """Generate reverse complement of a DNA sequence."""
    complement = str.maketrans("ATGC", "TACG")
    return seq.translate(complement)[::-1]

def main():
    fastq_file = Path("data_samples/FBA01901_barcode06.fastq")
    query_dict = {
        "Left_Primer": "CTTCATGGATCCTGCTCTCG",
        "Left_Rev_Compl": reverse_complement("CTTCATGGATCCTGCTCTCG"),
        "Right_Primer": "GGCCCTAAAGCTTAGCACGA",
        "Right_Rev_Compl": reverse_complement("GGCCCTAAAGCTTAGCACGA"),
        "Barcode_6": "GACTACTTTCTGCCTTTGCGAGAA",
        "Barcode_6_rev": "TTCTCGCAAAGGCAGAAAGTAGTC"
    }

    # Process FASTQ file
    processor = FastqProcessor(fastq_file, query_dict)
    sequence_matches = processor.process_file()

    visualizer = SequenceVisualizer(query_dict)
    fig = visualizer.plot_raw_heatmap(sequence_matches)
    fig.savefig(f'heatmap_raw.png',
                bbox_inches='tight',
                dpi=300)
    plt.close()


    # Try different clustering methods
    clustering_methods = {
        'hierarchical': HierarchicalClustering(),
        'kmeans': KMeansClustering(),
        'dbscan': DBSCANClustering(),
        'hdbscan': HDBSCANClustering(),
        'birch': BIRCHClustering(),
        'optics': OPTICSClustering(),
        'GMM': GMMClustering(),
        'meanshift': MeanShiftClustering(),
        'affinity': AffinityPropagationClustering(),
        'spectral': SpectralClusteringMethod()
    }

    embedding_methods = {
        'word2vec': Word2VecEmbedding(),
        # 'bert': BERTEmbedding()
    }

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
                    max_width=max(match.position + match.length
                                  for matches in sequence_matches
                                  for match in matches)
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
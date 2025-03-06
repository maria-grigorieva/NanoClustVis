from Bio import SeqIO
from rapidfuzz import fuzz
from scipy.signal import find_peaks
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from collections import defaultdict
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler

@dataclass
class SequenceMatch:
    query_name: str
    position: int
    score: float
    length: int


class IntelligentCompressor:
    def __init__(self, target_clusters: int = 100):
        self.target_clusters = target_clusters

    def create_feature_vectors(self, sequence_matches: List[List[SequenceMatch]],
                               max_width: int) -> np.ndarray:
        """
        Create feature vectors for each sequence based on:
        1. Position of matches
        2. Types of matches
        3. Pattern of matches
        """
        n_sequences = len(sequence_matches)
        n_queries = len(set(match.query_name for matches in sequence_matches
                            for match in matches))

        # Initialize features matrix
        features = np.zeros((n_sequences, n_queries * 3))  # 3 features per query type

        for seq_idx, matches in enumerate(sequence_matches):
            query_stats = defaultdict(lambda: {'positions': [], 'scores': []})

            # Collect statistics for each query type
            for match in matches:
                query_stats[match.query_name]['positions'].append(match.position / max_width)
                query_stats[match.query_name]['scores'].append(match.score)

            # Calculate features for each query type
            for query_idx, query_name in enumerate(sorted(set(match.query_name
                                                              for matches in sequence_matches
                                                              for match in matches))):
                if query_name in query_stats:
                    positions = query_stats[query_name]['positions']
                    scores = query_stats[query_name]['scores']

                    # Features: mean position, std of positions, mean score
                    features[seq_idx, query_idx * 3] = np.mean(positions) if positions else 0
                    features[seq_idx, query_idx * 3 + 1] = np.std(positions) if len(positions) > 1 else 0
                    features[seq_idx, query_idx * 3 + 2] = np.mean(scores) if scores else 0

        return features

    def compress_sequences(self, sequence_matches: List[List[SequenceMatch]],
                           max_width: int) -> Tuple[List[List[SequenceMatch]], np.ndarray]:
        """
        Compress sequences using intelligent clustering.
        Returns compressed matches and cluster assignments.
        """
        if len(sequence_matches) <= self.target_clusters:
            return sequence_matches, np.arange(len(sequence_matches))

        # Create feature vectors
        features = self.create_feature_vectors(sequence_matches, max_width)

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Perform hierarchical clustering
        linkage_matrix = linkage(features_scaled, method='ward')
        clusters = fcluster(linkage_matrix,
                            t=self.target_clusters,
                            criterion='maxclust')

        # Create compressed representation
        compressed_matches = []
        cluster_sizes = []

        for cluster_id in range(1, self.target_clusters + 1):
            cluster_indices = np.where(clusters == cluster_id)[0]
            if len(cluster_indices) > 0:
                # Select representative sequence (closest to cluster center)
                cluster_features = features_scaled[cluster_indices]
                cluster_center = np.mean(cluster_features, axis=0)
                distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
                representative_idx = cluster_indices[np.argmin(distances)]

                compressed_matches.append(sequence_matches[representative_idx])
                cluster_sizes.append(len(cluster_indices))

        return compressed_matches, clusters

class SequenceVisualizer:
    def __init__(self, query_dict: Dict[str, str], target_clusters: int = 100):
        self.query_dict = query_dict
        self.colors = ['#FFFFFF', '#FF4444', '#4444FF', '#44FF44',
                       '#FFFF44', '#FF44FF', '#44FFFF', '#CCCCCC']
        self.compressor = IntelligentCompressor(target_clusters)

    def calculate_visualization_width(self, sequence_matches: List[List[SequenceMatch]]) -> int:
        """Calculate the mean maximum position of matches plus padding."""
        max_positions = []
        for matches in sequence_matches:
            if matches:
                max_pos = max(match.position + match.length for match in matches)
                max_positions.append(max_pos)

        if not max_positions:
            return 0

        # Calculate mean max position and add 10% padding
        mean_max_pos = int(np.mean(max_positions))
        return int(mean_max_pos * 1.1)  # Add 10% padding

    def create_visualization_matrix(self, sequence_matches: List[List[SequenceMatch]]) -> Tuple[np.ndarray, np.ndarray]:
        """Create a visualization matrix using intelligent compression."""
        # Calculate max width based on matches
        max_width = max(match.position + match.length
                        for matches in sequence_matches
                        for match in matches)

        # Compress sequences
        compressed_matches, cluster_assignments = self.compressor.compress_sequences(
            sequence_matches, max_width)

        # Create visualization matrix
        vis_width = self.calculate_visualization_width(compressed_matches)
        vis_matrix = np.zeros((len(compressed_matches), vis_width))

        # Fill visualization matrix
        for row_idx, matches in enumerate(compressed_matches):
            for match in matches:
                if match.position < vis_width:
                    query_idx = list(self.query_dict.keys()).index(match.query_name) + 1
                    pos_start = match.position
                    pos_end = min(pos_start + match.length, vis_width)
                    vis_matrix[row_idx, pos_start:pos_end] = query_idx

        return vis_matrix, cluster_assignments

    def plot_heatmap(self, vis_matrix: np.ndarray,
                     cluster_assignments: np.ndarray,
                     title: str = 'Sequence Match Heatmap'):
        """Plot the heatmap visualization with cluster information."""
        plt.figure(figsize=(20, 12))

        # Create custom colormap
        n_queries = len(self.query_dict)
        colors = self.colors[:n_queries + 1]
        cmap = ListedColormap(colors)

        # Plot main heatmap
        ax_main = plt.subplot2grid((1, 20), (0, 0), colspan=15)
        sns.heatmap(vis_matrix,
                    cmap=cmap,
                    cbar=False,
                    xticklabels=50,
                    yticklabels=False,
                    ax=ax_main)

        # Plot cluster sizes
        ax_clusters = plt.subplot2grid((1, 20), (0, 16), colspan=3)
        unique_clusters = np.unique(cluster_assignments)
        cluster_sizes = [np.sum(cluster_assignments == c) for c in unique_clusters]
        ax_clusters.barh(range(len(cluster_sizes)), cluster_sizes)
        ax_clusters.set_title('Sequences per Cluster')

        # Add legend
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=self.colors[i + 1])
                           for i in range(len(self.query_dict))]
        ax_main.legend(legend_elements,
                       list(self.query_dict.keys()),
                       bbox_to_anchor=(1.05, 1),
                       loc='upper left')

        ax_main.set_title(title)
        ax_main.set_xlabel('Position in Sequence (bp)')
        ax_main.set_ylabel('Sequence Clusters')

        plt.tight_layout()
        return plt.gcf()


class SequenceAnalyzer:
    def __init__(self, similarity_threshold: float = 0.7, window_size: int = 50):
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size

    def find_matches_in_sequence(self, sequence: str, query_dict: Dict[str, str]) -> List[SequenceMatch]:
        """Find all matches for queries in a sequence using RapidFuzz."""
        matches = []

        for query_name, query_seq in query_dict.items():
            positions = []
            scores = []

            # Calculate similarity scores along the sequence
            for pos in range(len(sequence) - len(query_seq) + 1):
                window = sequence[pos:pos + len(query_seq)]
                score = fuzz.ratio(window, query_seq) / 100.0
                if score >= self.similarity_threshold:
                    positions.append(pos)
                    scores.append(score)

            if len(positions) > 0:
                # Find peaks in similarity scores
                peaks, properties = find_peaks(scores, distance=self.window_size)

                for peak in peaks:
                    matches.append(SequenceMatch(
                        query_name=query_name,
                        position=positions[peak],
                        length=len(query_seq),
                        score=scores[peak]
                    ))

        return matches


class FastqProcessor:
    def __init__(self, fastq_path: Path, query_dict: Dict[str, str]):
        self.fastq_path = fastq_path
        self.query_dict = query_dict
        self.sequence_analyzer = SequenceAnalyzer()
        self.max_sequence_length = 0

    def process_file(self) -> List[List[SequenceMatch]]:
        """Process FASTQ file and return matches for each sequence."""
        all_sequence_matches = []

        for record in SeqIO.parse(self.fastq_path, "fastq"):
            sequence = str(record.seq)
            self.max_sequence_length = max(self.max_sequence_length, len(sequence))

            matches = self.sequence_analyzer.find_matches_in_sequence(sequence, self.query_dict)
            if matches:  # Only append if matches were found
                all_sequence_matches.append(matches)

        return all_sequence_matches


def reverse_complement(seq: str) -> str:
    """Generate reverse complement of a DNA sequence."""
    complement = str.maketrans("ATGC", "TACG")
    return seq.translate(complement)[::-1]


def main():
    # Example usage
    fastq_file = "/Users/maria/PycharmProjects/nanopore-analysis/data_samples/FBA01901/barcode01/FBA01901_barcode01.fastq"  # Replace with your file path
    query_dict = {
        "Left Primer": "CTTCATGGATCCTGCTCTCG",
        "Left Rev Compl": str(reverse_complement("CTTCATGGATCCTGCTCTCG")),
        "Right Primer": "GGCCCTAAAGCTTAGCACGA",
        "Right Rev Compl": str(reverse_complement("GGCCCTAAAGCTTAGCACGA")),
        "Barcode 1": "CACAAAGACACCGACAACTTTCTT",
        "Barcode_1_rev": "AAGAAAGTTGTCGGTGTCTTTGTG"
    }  # Replace with your sequences

    # Process FASTQ file
    processor = FastqProcessor(fastq_file, query_dict)
    sequence_matches = processor.process_file()

    # Create visualization
    visualizer = SequenceVisualizer(query_dict)

    try:
        # vis_matrix = visualizer.create_visualization_matrix(sequence_matches)
        vis_matrix, cluster_assignments = visualizer.create_visualization_matrix(sequence_matches)

        # Plot and save the heatmap
        #fig = visualizer.plot_heatmap(vis_matrix)
        fig = visualizer.plot_heatmap(vis_matrix, cluster_assignments)
        fig.savefig('sequence_matches_heatmap.png', bbox_inches='tight', dpi=300)
        plt.close()

        # Print summary statistics
        print(f"Found matches in {len(sequence_matches)} sequences")
        print(f"Compressed to: {len(vis_matrix)} clusters")
        print(f"Visualization width: {vis_matrix.shape[1]} bp")

        # Print cluster statistics
        unique_clusters = np.unique(cluster_assignments)
        print("\nCluster statistics:")
        for cluster in unique_clusters:
            size = np.sum(cluster_assignments == cluster)
            print(f"Cluster {cluster}: {size} sequences")

        # Print first few matches as example
        for i, matches in enumerate(sequence_matches[:5], 1):
            print(f"\nSequence {i} matches:")
            for match in matches:
                print(f"  {match.query_name} at position {match.position} "
                      f"(score: {match.score:.2f})")

    except ValueError as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()
from compressor import IntelligentCompressor
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from matcher import SequenceMatch
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns

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

    def create_visualization_matrix(self,
                                    compressed_matches: List[List[SequenceMatch]]) -> np.ndarray:
        """Create visualization matrix from compressed matches"""
        # Calculate visualization width
        # max_width = max(match.position + match.length
        #                 for matches in compressed_matches
        #                 for match in matches)
        # vis_width = int(max_width * 1.1)  # Add 10% padding

        # Calculate visualization width based on mean max position
        vis_width = self.calculate_visualization_width(compressed_matches)
        if vis_width == 0:
            raise ValueError("No matches found in sequences")

        # Create visualization matrix
        vis_matrix = np.zeros((len(compressed_matches), vis_width))

        # Fill visualization matrix
        for row_idx, matches in enumerate(compressed_matches):
            for match in matches:
                if match.position < vis_width:
                    query_idx = list(self.query_dict.keys()).index(match.query_name) + 1
                    pos_start = match.position
                    pos_end = min(pos_start + match.length, vis_width)
                    vis_matrix[row_idx, pos_start:pos_end] = query_idx

        return vis_matrix

    def plot_heatmap(self,
                     vis_matrix: np.ndarray,
                     cluster_assignments: np.ndarray,
                     title: str = 'Sequence Match Heatmap') -> plt.Figure:
        """Plot heatmap with cluster information"""
        plt.figure(figsize=(22, 12))  # Increased width for better spacing

        # Create subplot layout with more columns
        gs = plt.GridSpec(1, 24)  # More columns to shift the cluster plot further right

        # Main heatmap
        ax_main = plt.subplot(gs[0, :17])  # Extend the main heatmap

        # Create custom colormap
        n_queries = len(self.query_dict)
        colors = self.colors[:n_queries + 1]
        cmap = ListedColormap(colors)

        # Plot heatmap
        sns.heatmap(vis_matrix,
                    cmap=cmap,
                    cbar=False,
                    xticklabels=50,
                    yticklabels=False,
                    ax=ax_main)

        # Cluster information (moved further right)
        ax_clusters = plt.subplot(gs[0, 19:])  # Shifted right by starting at 19

        unique_clusters = np.unique(cluster_assignments)
        cluster_sizes = [np.sum(cluster_assignments == c) for c in unique_clusters]

        # Plot cluster sizes
        cluster_positions = range(len(cluster_sizes))
        ax_clusters.barh(cluster_positions, cluster_sizes)
        ax_clusters.set_title('Sequences per Cluster')
        ax_clusters.set_xlabel('Number of Sequences')
        ax_clusters.set_ylabel('Cluster ID')

        # Add legend for queries
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=self.colors[i + 1])
                           for i in range(len(self.query_dict))]
        ax_main.legend(legend_elements,
                       list(self.query_dict.keys()),
                       bbox_to_anchor=(0.85, 1),
                       loc='upper left')

        # Set titles and labels
        ax_main.set_title(title)
        ax_main.set_xlabel('Position in Sequence (bp)')
        ax_main.set_ylabel('Sequence Clusters')

        # Add statistics text
        stats_text = (
            f'Total Sequences: {sum(cluster_sizes)}\n'
            f'Number of Clusters: {len(unique_clusters)}\n'
            f'Average Cluster Size: {np.mean(cluster_sizes):.1f}\n'
            f'Max Cluster Size: {max(cluster_sizes)}'
        )
        plt.figtext(0.94, 0.15, stats_text,  # Moved text further right
                    bbox=dict(facecolor='white', alpha=0.8),
                    verticalalignment='center')

        plt.tight_layout()
        return plt.gcf()
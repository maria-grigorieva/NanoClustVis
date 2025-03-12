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
        self.colors = ['#FFFFFF',  # Background color (white)
                       '#FF4444',  # Red
                       '#4444FF',  # Blue
                       '#44FF44',  # Green
                       '#FFFF44',  # Yellow
                       '#FF44FF',  # Magenta
                       '#44FFFF',  # Cyan
                       '#FFA500',  # Orange
                       '#8A2BE2',  # Purple
                       '#FF1493',  # Deep Pink
                       '#20B2AA',  # Light Sea Green
                       '#DAA520']  # Goldenrod
        self.compressor = IntelligentCompressor(target_clusters)

    def calculate_visualization_width(self, sequence_matches: List[List[SequenceMatch]],
                                      average=False) -> int:
        """Calculate the mean maximum position of matches plus padding."""
        max_positions = []
        for matches in sequence_matches:
            if len(matches) > 0:
                max_pos = max(match.position + match.length for match in matches)
                max_positions.append(max_pos)
        if not max_positions:
            return 0

        if average:
            mean_max_pos = int(np.mean(max_positions))
            return int(mean_max_pos * 1.1)  # Add 10% padding
        else:
            return int(np.max(max_positions))

    def create_visualization_matrix(self, compressed_matches: List[List[SequenceMatch]]) -> np.ndarray:
        """Create visualization matrix from compressed matches with length information"""
        vis_width = self.calculate_visualization_width(compressed_matches)
        if vis_width == 0:
            raise ValueError("No matches found in sequences")

        vis_matrix = np.zeros((len(compressed_matches), vis_width, 2))

        # Debug print to check query processing
        query_counts = {key: 0 for key in self.query_dict.keys()}

        for row_idx, matches in enumerate(compressed_matches):
            for match in matches:
                if match.position < vis_width:
                    try:
                        query_idx = list(self.query_dict.keys()).index(match.query_name) + 1
                        query_counts[match.query_name] += 1
                        pos_start = match.position
                        pos_end = min(pos_start + match.length, vis_width)
                        vis_matrix[row_idx, pos_start:pos_end, 0] = query_idx
                        vis_matrix[row_idx, pos_start:pos_end, 1] = match.length
                    except ValueError as e:
                        print(f"Warning: Query name {match.query_name} not found in query_dict")

        # Print query statistics
        print("\nQuery match statistics:")
        for query, count in query_counts.items():
            print(f"Query {query}: {count} matches")

        return vis_matrix

    @staticmethod
    def hex_to_rgba(hex_color: str, alpha: float = 0.5) -> tuple:
        """Convert hex color to RGBA tuple with specified alpha"""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))
        return rgb + (alpha,)

    def plot_heatmap(self,
                     vis_matrix: np.ndarray,
                     cluster_assignments: np.ndarray,
                     title: str = 'Sequence Match Heatmap') -> plt.Figure:
        """Plot heatmap with variable-width boxes"""
        plt.figure(figsize=(22, 12))
        gs = plt.GridSpec(1, 24)
        ax_main = plt.subplot(gs[0, :17])
        ax_main.clear()

        ax_main.set_ylim(0, len(vis_matrix))
        max_x = vis_matrix.shape[1]
        ax_main.set_xlim(0, max_x)

        for row_idx in range(len(vis_matrix)):
            col_idx = 0
            while col_idx < max_x:
                query_idx = int(vis_matrix[row_idx, col_idx, 0])
                if query_idx > 0:
                    length = int(vis_matrix[row_idx, col_idx, 1])

                    base_color = self.colors[query_idx]
                    if isinstance(base_color, str):
                        color = (*self.hex_to_rgba(base_color)[:3], 0.8)
                    else:
                        color = (*base_color[:3], 0.8)

                    rect = plt.Rectangle(
                        (col_idx, row_idx),
                        length,
                        1,
                        facecolor=color,
                        edgecolor='gray',
                        linewidth=0.5
                    )
                    ax_main.add_patch(rect)
                    #
                    # if length > 20:
                    #     ax_main.text(
                    #         col_idx + length / 2,
                    #         row_idx + 0.5,
                    #         str(length),
                    #         ha='center',
                    #         va='center',
                    #         fontsize=8,
                    #         color='black',
                    #         weight='bold'
                    #     )

                    col_idx += length
                else:
                    col_idx += 1

        # Add grid
        ax_main.grid(True, which='major', color='gray', linestyle='-', alpha=0.2)
        ax_main.set_xticks(np.arange(0, max_x, 50))

        # Cluster information
        ax_clusters = plt.subplot(gs[0, 19:])
        unique_clusters = np.unique(cluster_assignments)
        cluster_sizes = [np.sum(cluster_assignments == c) for c in unique_clusters]

        # Plot cluster sizes
        cluster_positions = range(len(cluster_sizes))
        ax_clusters.barh(cluster_positions, cluster_sizes)
        ax_clusters.set_title('Sequences per Cluster')
        ax_clusters.set_xlabel('Number of Sequences')
        ax_clusters.set_ylabel('Cluster ID')

        # Add legend
        legend_elements = []
        query_keys = list(self.query_dict.keys())

        # Debug print to check queries
        print(f"Total queries: {len(query_keys)}")

        for i, query_key in enumerate(query_keys):
            color = self.colors[i + 1]  # Skip the first color (background)
            if isinstance(color, str):
                rgb = self.hex_to_rgba(color)[:3]
                legend_elements.append(
                    plt.Rectangle((0, 0), 1, 1, facecolor=(*rgb, 1.0))
                )
            else:
                legend_elements.append(
                    plt.Rectangle((0, 0), 1, 1, facecolor=(*color[:3], 1.0))
                )
            # Debug print for legend creation
            print(f"Creating legend for query {i + 1}: {query_key}")

        # Create legend with all query keys
        ax_main.legend(legend_elements,
                       query_keys,
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
        plt.figtext(0.94, 0.15, stats_text,
                    bbox=dict(facecolor='white', alpha=0.8),
                    verticalalignment='center')

        plt.tight_layout()
        return plt.gcf()


    def plot_raw_heatmap(self, matches: list, title: str = 'Sequence Match Positions') -> plt.Figure:
        """Plot raw heatmap with variable-width boxes"""
        plt.figure(figsize=(15, 10))
        ax_main = plt.gca()
        ax_main.clear()

        # Calculate visualization width
        vis_width = self.calculate_visualization_width(matches)
        if vis_width == 0:
            raise ValueError("No matches found in sequences")

        # Set the axis limits
        ax_main.set_ylim(0, len(matches))
        ax_main.set_xlim(0, vis_width)

        # Create a dictionary to map query names to colors
        color_dict = {query_name: self.colors[idx + 1]
                      for idx, query_name in enumerate(self.query_dict.keys())}

        # Plot rectangles for each match
        for row_idx, sequence in enumerate(matches):
            for match in sequence:
                color = color_dict[match.query_name]

                # Create rectangle
                rect = plt.Rectangle(
                    (match.position, row_idx),  # (x, y)
                    match.length,  # width
                    1,  # height
                    facecolor=color,
                    edgecolor='none'
                )
                ax_main.add_patch(rect)

                # Add length text if box is wide enough
                if match.length > 20:  # Adjust threshold as needed
                    ax_main.text(
                        match.position + match.length / 2,
                        row_idx + 0.5,
                        str(match.length),
                        ha='center',
                        va='center',
                        fontsize=8
                    )

        # Add legend for queries
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color_dict[query])
                           for query in self.query_dict.keys()]
        ax_main.legend(legend_elements,
                       list(self.query_dict.keys()),
                       bbox_to_anchor=(1.02, 1),
                       loc='upper left')

        # Set titles and labels
        ax_main.set_title(title)
        ax_main.set_xlabel('Position in Sequence (bp)')
        ax_main.set_ylabel('Sequence Number')

        # Add grid
        ax_main.grid(True, which='major', color='gray', linestyle='-', alpha=0.2)

        # Add x-axis ticks every 50 positions
        ax_main.set_xticks(np.arange(0, vis_width, 50))

        plt.tight_layout()
        return plt.gcf()
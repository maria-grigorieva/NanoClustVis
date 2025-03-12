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
                       '#FFFF44', '#FF44FF', '#44FFFF', '#CCCCCC',
                       '#FFA500', '#8A2BE2']
        self.compressor = IntelligentCompressor(target_clusters)

    def calculate_visualization_width(self, sequence_matches: List[List[SequenceMatch]],
                                      average = False) -> int:
        """Calculate the mean maximum position of matches plus padding."""
        max_positions = []
        for matches in sequence_matches:
            if len(matches) > 0:
                max_pos = max(match.position + match.length for match in matches)
                max_positions.append(max_pos)
        if not max_positions:
            return 0

        if average:
            # Calculate mean max position and add 10% padding
            mean_max_pos = int(np.mean(max_positions))
            return int(mean_max_pos * 1.1)  # Add 10% padding
        else:
            return int(np.max(max_positions))

    def create_visualization_matrix(self,
                                  compressed_matches: List[List[SequenceMatch]]) -> np.ndarray:
        """Create visualization matrix from compressed matches with length information"""
        vis_width = self.calculate_visualization_width(compressed_matches)
        if vis_width == 0:
            raise ValueError("No matches found in sequences")

        # Create visualization matrix with additional dimension for length information
        vis_matrix = np.zeros((len(compressed_matches), vis_width, 2))

        # Fill visualization matrix
        for row_idx, matches in enumerate(compressed_matches):
            for match in matches:
                if match.position < vis_width:
                    query_idx = list(self.query_dict.keys()).index(match.query_name) + 1
                    pos_start = match.position
                    pos_end = min(pos_start + match.length, vis_width)
                    # Store both query index and length
                    vis_matrix[row_idx, pos_start:pos_end, 0] = query_idx
                    vis_matrix[row_idx, pos_start:pos_end, 1] = match.length

        return vis_matrix

    @staticmethod
    def hex_to_rgba(hex_color: str, alpha: float = 0.5) -> tuple:
        """Convert hex color to RGBA tuple with specified alpha"""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))
        return rgb + (alpha,)

    def find_overlaps(self, matches: List[SequenceMatch]) -> Dict[tuple, int]:
        """
        Find overlapping regions in a sequence and count overlap depth
        Returns a dictionary with position ranges and overlap count
        """
        overlap_map = {}

        # Sort matches by position
        sorted_matches = sorted(matches, key=lambda x: x.position)

        # Check each position for overlaps
        for match in sorted_matches:
            start = match.position
            end = start + match.length

            # Update overlap count for each position in the match range
            for pos in range(start, end):
                overlap_map[pos] = overlap_map.get(pos, 0) + 1

        return overlap_map

    def plot_heatmap(self,
                     vis_matrix: np.ndarray,
                     cluster_assignments: np.ndarray,
                     title: str = 'Sequence Match Heatmap') -> plt.Figure:
        """Plot heatmap with variable-width boxes and overlap transparency"""
        plt.figure(figsize=(22, 12))
        gs = plt.GridSpec(1, 24)
        ax_main = plt.subplot(gs[0, :17])
        ax_main.clear()

        # Set the y-axis limits
        ax_main.set_ylim(0, len(vis_matrix))

        # Calculate x-axis limits
        max_x = vis_matrix.shape[1]  # Use the width of the matrix
        ax_main.set_xlim(0, max_x)

        # Plot rectangles for each row
        for row_idx in range(len(vis_matrix)):
            # Create a map of overlapping regions
            overlap_map = {}

            # First pass: count overlaps
            for col_idx in range(max_x):
                if vis_matrix[row_idx, col_idx, 0] > 0:  # If there's a match
                    overlap_map[col_idx] = overlap_map.get(col_idx, 0) + 1

            # Second pass: plot rectangles
            col_idx = 0
            while col_idx < max_x:
                query_idx = int(vis_matrix[row_idx, col_idx, 0])
                if query_idx > 0:
                    # Get the length of this match
                    length = int(vis_matrix[row_idx, col_idx, 1])

                    # Calculate overlap for this region
                    max_overlap = max(overlap_map.get(pos, 1)
                                      for pos in range(col_idx, min(col_idx + length, max_x)))

                    # Adjust alpha based on overlap count
                    alpha = min(0.8 / max_overlap, 0.8)

                    # Create color with adjusted alpha
                    base_color = self.colors[query_idx]
                    if isinstance(base_color, str):
                        # Convert hex to rgba if needed
                        color = (*self.hex_to_rgba(base_color)[:3], alpha)
                    else:
                        color = (*base_color[:3], alpha)

                    # Create rectangle
                    rect = plt.Rectangle(
                        (col_idx, row_idx),
                        length,
                        1,
                        facecolor=color,
                        edgecolor='gray',
                        linewidth=0.5
                    )
                    ax_main.add_patch(rect)

                    # Add length text if box is wide enough
                    if length > 20:
                        ax_main.text(
                            col_idx + length / 2,
                            row_idx + 0.5,
                            str(length),
                            ha='center',
                            va='center',
                            fontsize=8,
                            color='black',
                            weight='bold'
                        )

                    col_idx += length
                else:
                    col_idx += 1

        # Rest of the method remains the same...
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

        # Add legend for queries with solid colors for better visibility
        legend_elements = []
        for i in range(len(self.query_dict)):
            color = self.colors[i + 1]
            if isinstance(color, str):
                # Convert hex to RGB
                rgb = self.hex_to_rgba(color)[:3]  # Get only RGB values
                legend_elements.append(
                    plt.Rectangle((0, 0), 1, 1, facecolor=(*rgb, 1.0))
                )
            else:
                # If color is already in RGB/RGBA format
                legend_elements.append(
                    plt.Rectangle((0, 0), 1, 1, facecolor=(*color[:3], 1.0))
                )
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
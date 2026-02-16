import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import pyarrow.parquet as pq
import pyarrow as pa
import time
import gc
from tqdm import tqdm


@dataclass
class SequenceMatch:
    """Класс для представления совпадения последовательности (оптимизированная версия)."""
    __slots__ = ['query_name', 'position', 'length', 'seq_id']  # Экономия памяти
    query_name: str
    position: int
    length: int
    seq_id: str


class OptimizedSequenceVisualizer:
    """
    Оптимизированная версия SequenceVisualizer для больших данных.
    Использует векторизованные операции и минимизирует использование памяти.
    """

    def __init__(self, query_dict: Dict[str, str], target_clusters: int = 100):
        self.query_dict = query_dict
        self.query_to_idx = {name: i + 1 for i, name in enumerate(query_dict.keys())}
        self.target_clusters = target_clusters

        # Оптимизированная цветовая палитра (предварительно конвертированная в RGB)
        self.colors_rgb = self._prepare_colors()

        # Кэш для результатов
        self._cache = {}

    def _prepare_colors(self) -> List[Tuple[float, float, float, float]]:
        """Предварительная конвертация цветов в RGB формат."""
        hex_colors = ['#FFFFFF', '#FF4444', '#4444FF', '#44FF44', '#FFFF44',
                      '#FF44FF', '#44FFFF', '#FFA500', '#8A2BE2', '#FF1493',
                      '#20B2AA', '#DAA520', '#000000', '#A52A2A', '#00CED1',
                      '#9400D3', '#008000', '#FF4500', '#1E90FF', '#FFD700',
                      '#C71585', '#4682B4', '#808000', '#DC143C', '#00FF7F',
                      '#B22222', '#5F9EA0', '#9932CC', '#FF6347', '#40E0D0',
                      '#32CD32', '#BDB76B', '#FF69B4', '#191970', '#ADFF2F',
                      '#7B68EE', '#D2691E', '#8B0000']

        colors_rgb = []
        for hex_color in hex_colors:
            hex_color = hex_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))
            colors_rgb.append((*rgb, 0.8))

        return colors_rgb

    def load_data_chunked(self,
                          matches_path: str,
                          clusters_path: str,
                          chunk_size: int = 100000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Загружает данные чанками для экономии памяти.

        Returns:
            Tuple[cluster_ids, positions, lengths, query_indices, seq_ids]
        """
        print(f"Загрузка данных чанками по {chunk_size} записей...")

        # Загружаем кластеры (они обычно небольшие)
        clusters_df = pd.read_parquet(clusters_path)
        seq_to_cluster = clusters_df.set_index('seq_id')['cluster'].to_dict()

        # Освобождаем память
        del clusters_df
        gc.collect()

        # Подготавливаем массивы для результатов
        all_clusters = []
        all_positions = []
        all_lengths = []
        all_query_indices = []
        all_seq_ids = []

        # Читаем matches файл чанками
        parquet_file = pq.ParquetFile(matches_path)

        for batch in tqdm(parquet_file.iter_batches(batch_size=chunk_size),
                          desc="Обработка чанков"):
            df = batch.to_pandas()

            # Фильтруем только те seq_id, которые есть в кластерах
            mask = df['seq_id'].isin(seq_to_cluster)
            df_filtered = df[mask]

            if len(df_filtered) == 0:
                continue

            # Преобразуем query_name в индексы
            query_indices = df_filtered['query_name'].map(self.query_to_idx).values
            valid_mask = ~np.isnan(query_indices)
            df_filtered = df_filtered[valid_mask]
            query_indices = query_indices[valid_mask].astype(np.int32)

            # Получаем кластеры
            clusters = np.array([seq_to_cluster[seq_id] for seq_id in df_filtered['seq_id']])

            # Добавляем в общие массивы
            all_clusters.append(clusters)
            all_positions.append(df_filtered['position'].values.astype(np.int32))
            all_lengths.append(df_filtered['length'].values.astype(np.int32))
            all_query_indices.append(query_indices)
            all_seq_ids.append(df_filtered['seq_id'].values)

        # Объединяем все чанки
        if all_clusters:
            return (np.concatenate(all_clusters),
                    np.concatenate(all_positions),
                    np.concatenate(all_lengths),
                    np.concatenate(all_query_indices),
                    np.concatenate(all_seq_ids))
        else:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    def create_cluster_groups(self,
                              clusters: np.ndarray,
                              positions: np.ndarray,
                              lengths: np.ndarray,
                              query_indices: np.ndarray,
                              seq_ids: np.ndarray) -> Dict:
        """
        Создает группы по кластерам используя numpy для скорости.
        """
        print("Группировка по кластерам...")
        start_time = time.time()

        # Получаем уникальные кластеры
        unique_clusters = np.unique(clusters)

        # Создаем словарь для хранения индексов
        cluster_groups = {}

        for cluster in tqdm(unique_clusters, desc="Группировка кластеров"):
            mask = clusters == cluster
            cluster_groups[int(cluster)] = {
                'positions': positions[mask],
                'lengths': lengths[mask],
                'query_indices': query_indices[mask],
                'seq_ids': seq_ids[mask]
            }

        elapsed = time.time() - start_time
        print(f"Группировка завершена за {elapsed:.2f} сек")

        return cluster_groups

    def compress_cluster_fast(self,
                              cluster_data: Dict,
                              method: str = 'merge',
                              max_seq_per_cluster: int = 50) -> List[Tuple]:
        """
        Быстрое сжатие кластера с ограничением на количество последовательностей.
        """
        if method == 'merge':
            # Объединяем все совпадения
            # Создаем уникальные комбинации
            unique_mask = np.ones(len(cluster_data['positions']), dtype=bool)

            # Оптимизированное удаление дубликатов
            combined = np.column_stack((
                cluster_data['query_indices'],
                cluster_data['positions'],
                cluster_data['lengths']
            ))

            # Получаем уникальные строки
            _, unique_indices = np.unique(combined, axis=0, return_index=True)

            # Сортируем по позиции
            sorted_indices = unique_indices[np.argsort(cluster_data['positions'][unique_indices])]

            return [(cluster_data['query_indices'][i],
                     cluster_data['positions'][i],
                     cluster_data['lengths'][i])
                    for i in sorted_indices]

        elif method == 'representative':
            # Берем репрезентативную последовательность (с максимальным количеством совпадений)
            seq_ids_unique, counts = np.unique(cluster_data['seq_ids'], return_counts=True)
            rep_seq_id = seq_ids_unique[np.argmax(counts)]

            mask = cluster_data['seq_ids'] == rep_seq_id
            positions = cluster_data['positions'][mask]
            lengths = cluster_data['lengths'][mask]
            query_indices = cluster_data['query_indices'][mask]

            # Сортируем по позиции
            sorted_idx = np.argsort(positions)

            return [(query_indices[i], positions[i], lengths[i])
                    for i in sorted_idx]

        else:  # 'sample'
            # Сэмплируем случайные последовательности из кластера
            seq_ids_unique = np.unique(cluster_data['seq_ids'])

            if len(seq_ids_unique) > max_seq_per_cluster:
                # Выбираем случайные seq_id
                selected = np.random.choice(seq_ids_unique,
                                            size=max_seq_per_cluster,
                                            replace=False)
            else:
                selected = seq_ids_unique

            result = []
            for seq_id in selected:
                mask = cluster_data['seq_ids'] == seq_id
                positions = cluster_data['positions'][mask]
                lengths = cluster_data['lengths'][mask]
                query_indices = cluster_data['query_indices'][mask]

                sorted_idx = np.argsort(positions)
                for i in sorted_idx:
                    result.append((query_indices[i], positions[i], lengths[i]))

            return result

    def prepare_visualization_data(self,
                                   matches_path: str,
                                   clusters_path: str,
                                   compression_method: str = 'merge',
                                   max_seq_per_cluster: int = 50,
                                   use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Оптимизированная подготовка данных для визуализации.
        """
        cache_key = f"{matches_path}_{clusters_path}_{compression_method}"

        if use_cache and cache_key in self._cache:
            print("Используем кэшированные данные...")
            return self._cache[cache_key]

        # Загружаем данные чанками
        clusters, positions, lengths, query_indices, seq_ids = self.load_data_chunked(
            matches_path, clusters_path
        )

        if len(clusters) == 0:
            raise ValueError("Нет данных для визуализации")

        print(f"Загружено {len(clusters)} совпадений")

        # Группируем по кластерам
        cluster_groups = self.create_cluster_groups(
            clusters, positions, lengths, query_indices, seq_ids
        )

        # Освобождаем память
        del clusters, positions, lengths, query_indices, seq_ids
        gc.collect()

        # Сжимаем каждый кластер
        print(f"Сжатие данных методом '{compression_method}'...")
        compressed_data = []
        cluster_ids = []

        for cluster_id in tqdm(sorted(cluster_groups.keys()), desc="Сжатие кластеров"):
            compressed = self.compress_cluster_fast(
                cluster_groups[cluster_id],
                method=compression_method,
                max_seq_per_cluster=max_seq_per_cluster
            )

            if compression_method == 'sample':
                # Для сэмплирования добавляем несколько строк на кластер
                seq_count = len(np.unique(cluster_groups[cluster_id]['seq_ids']))
                actual_count = min(seq_count, max_seq_per_cluster)
                for _ in range(actual_count):
                    compressed_data.append(compressed)
                    cluster_ids.append(cluster_id)
            else:
                compressed_data.append(compressed)
                cluster_ids.append(cluster_id)

        result = (compressed_data, np.array(cluster_ids))

        if use_cache:
            self._cache[cache_key] = result

        return result

    def calculate_visualization_width(self,
                                      compressed_data: List[List[Tuple]],
                                      average: bool = False) -> int:
        """Векторизованное вычисление ширины визуализации."""
        if not compressed_data:
            return 0

        max_positions = []
        for matches in compressed_data:
            if matches:
                # Векторизованная операция
                matches_array = np.array(matches)
                max_pos = np.max(matches_array[:, 1] + matches_array[:, 2])
                max_positions.append(max_pos)

        if not max_positions:
            return 0

        if average:
            mean_max_pos = int(np.mean(max_positions))
            return int(mean_max_pos * 1.1)
        else:
            return int(np.max(max_positions))

    def create_visualization_matrix_fast(self,
                                         compressed_data: List[List[Tuple]]) -> np.ndarray:
        """
        Быстрое создание визуализационной матрицы с использованием векторизации.
        """
        vis_width = self.calculate_visualization_width(compressed_data, True)
        if vis_width == 0:
            raise ValueError("No matches found in sequences")

        n_rows = len(compressed_data)
        vis_matrix = np.zeros((n_rows, vis_width, 2), dtype=np.int16)  # Используем int16 для экономии памяти

        # Статистика запросов
        query_counts = np.zeros(len(self.query_dict) + 1, dtype=np.int32)

        for row_idx, matches in enumerate(tqdm(compressed_data, desc="Создание матрицы")):
            if not matches:
                continue

            # Конвертируем в numpy массив для векторизации
            matches_array = np.array(matches)
            query_idxs = matches_array[:, 0].astype(int)
            positions = matches_array[:, 1].astype(int)
            lengths = matches_array[:, 2].astype(int)

            # Обновляем статистику
            for q_idx in query_idxs:
                query_counts[q_idx] += 1

            # Заполняем матрицу
            for q_idx, pos, length in zip(query_idxs, positions, lengths):
                if pos < vis_width:
                    end_pos = min(pos + length, vis_width)
                    vis_matrix[row_idx, pos:end_pos, 0] = q_idx
                    vis_matrix[row_idx, pos:end_pos, 1] = length

        # Выводим статистику
        print("\nQuery match statistics:")
        for i, (query, idx) in enumerate(self.query_to_idx.items(), 1):
            print(f"Query {query}: {query_counts[i]} matches")

        return vis_matrix

    def plot_heatmap_optimized(self,
                               vis_matrix: np.ndarray,
                               cluster_assignments: np.ndarray,
                               title: str = 'Sequence Match Heatmap',
                               figsize: Tuple[int, int] = (22, 12),
                               dpi: int = 100) -> plt.Figure:
        """
        Оптимизированная версия построения тепловой карты.
        Использует коллекции вместо множества отдельных Rectangle.
        """
        from matplotlib.collections import PatchCollection

        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = plt.GridSpec(1, 24)
        ax_main = plt.subplot(gs[0, :17])

        n_rows = vis_matrix.shape[0]
        max_x = vis_matrix.shape[1]

        ax_main.set_xlim(0, max_x)
        ax_main.set_ylim(0, n_rows)

        # Создаем коллекции для каждого типа запроса
        patches_by_query = defaultdict(list)

        for row_idx in range(n_rows):
            col_idx = 0
            while col_idx < max_x:
                query_idx = int(vis_matrix[row_idx, col_idx, 0])
                if query_idx > 0:
                    length = int(vis_matrix[row_idx, col_idx, 1])

                    rect = Rectangle(
                        (col_idx, row_idx),
                        length,
                        1,
                        linewidth=0.5,
                        edgecolor='gray'
                    )

                    patches_by_query[query_idx].append(rect)
                    col_idx += length
                else:
                    col_idx += 1

        # Добавляем коллекции на график
        for query_idx, patches in patches_by_query.items():
            if query_idx <= len(self.colors_rgb):
                collection = PatchCollection(
                    patches,
                    facecolor=self.colors_rgb[query_idx - 1][:3],
                    alpha=0.8,
                    edgecolor='gray',
                    linewidth=0.5
                )
                ax_main.add_collection(collection)

        # Настройка сетки
        ax_main.grid(True, which='major', color='gray', linestyle='-', alpha=0.2)
        ax_main.set_xticks(np.arange(0, max_x, 50))

        # Информация о кластерах
        ax_clusters = plt.subplot(gs[0, 19:])
        unique_clusters = np.unique(cluster_assignments)
        cluster_sizes = [np.sum(cluster_assignments == c) for c in unique_clusters]

        cluster_positions = np.arange(len(unique_clusters)) + np.min(unique_clusters)
        ax_clusters.barh(cluster_positions, cluster_sizes)
        ax_clusters.set_title('Sequences per Cluster')
        ax_clusters.set_xlabel('Number of Sequences')
        ax_clusters.set_ylabel('Cluster ID')

        # Легенда
        legend_elements = []
        query_keys = list(self.query_dict.keys())

        for i, query_key in enumerate(query_keys):
            color = self.colors_rgb[i][:3]
            legend_elements.append(
                Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='gray')
            )

        ax_main.legend(legend_elements,
                       query_keys,
                       bbox_to_anchor=(0.85, 1),
                       loc='upper left',
                       fontsize='small')

        # Заголовки и подписи
        ax_main.set_title(title, fontsize=14)
        ax_main.set_xlabel('Position in Sequence (bp)', fontsize=12)
        ax_main.set_ylabel('Sequence Clusters', fontsize=12)

        # Статистика
        stats_text = (
            f'Total Sequences: {sum(cluster_sizes)}\n'
            f'Number of Clusters: {len(unique_clusters)}\n'
            f'Average Cluster Size: {np.mean(cluster_sizes):.1f}\n'
            f'Max Cluster Size: {max(cluster_sizes)}'
        )
        plt.figtext(0.94, 0.15, stats_text,
                    bbox=dict(facecolor='white', alpha=0.8),
                    verticalalignment='center',
                    fontsize=10)

        # Выравнивание
        ax_clusters.set_ylim(np.min(cluster_assignments) - 0.5,
                             np.max(cluster_assignments) + 0.5)
        ax_clusters.set_yticks(range(np.min(cluster_assignments),
                                     np.max(cluster_assignments) + 1))

        plt.tight_layout()
        return fig

    def cleanup(self):
        """Очистка памяти."""
        self._cache.clear()
        gc.collect()


# Оптимизированная функция для запуска
def run_optimized_visualization(matches_path: str,
                                clusters_path: str,
                                query_dict: Dict[str, str],
                                compression_method: str = 'merge',
                                max_seq_per_cluster: int = 50,
                                output_image: str = 'cluster_visualization.png'):
    """
    Запуск оптимизированной визуализации.
    """
    start_time = time.time()

    # Создаем визуализатор
    visualizer = OptimizedSequenceVisualizer(query_dict)

    try:
        # Подготавливаем данные
        print("=" * 60)
        print("ОПТИМИЗИРОВАННАЯ ВИЗУАЛИЗАЦИЯ КЛАСТЕРОВ")
        print("=" * 60)

        compressed_data, cluster_ids = visualizer.prepare_visualization_data(
            matches_path=matches_path,
            clusters_path=clusters_path,
            compression_method=compression_method,
            max_seq_per_cluster=max_seq_per_cluster
        )

        print(f"\nПодготовлено {len(compressed_data)} последовательностей для визуализации")

        # Создаем матрицу
        vis_matrix = visualizer.create_visualization_matrix_fast(compressed_data)
        print(f"Размер матрицы: {vis_matrix.shape}")

        # Строим график
        fig = visualizer.plot_heatmap_optimized(
            vis_matrix=vis_matrix,
            cluster_assignments=cluster_ids,
            title=f'Sequence Clusters Visualization ({compression_method} compression)'
        )

        # Сохраняем
        plt.savefig(output_image, dpi=150, bbox_inches='tight')
        print(f"\nГрафик сохранен как {output_image}")

        # Показываем статистику
        elapsed = time.time() - start_time
        print(f"\nОбщее время выполнения: {elapsed:.2f} секунд")

        plt.show()

    finally:
        # Очищаем память
        visualizer.cleanup()

    return visualizer
#
#
# if __name__ == "__main__":
#     # Ваш query_dict
#     query_dict = {
#         "A1": "GATCAGTCCGATATC",
#         "A2": "TCGACATGCTAGTGC",
#         "A3": "GCTATCGGATACGTC",
#         "S1_L_S2": "ATGACTGCCATTTTTTTGGCAGTCAT",
#         "A1C": "GACGTATCCGATAGC",
#         "A2C": "GCACTAGCATGTCGA",
#         "A3C": "GATATCGGACTGATC"
#     }
#
#     # Запуск с разными методами сжатия
#     for method in ['merge', 'representative', 'sample']:
#         print(f"\n{'#' * 60}")
#         print(f"Метод сжатия: {method}")
#         print(f"{'#' * 60}")
#
#         run_optimized_visualization(
#             matches_path="matches.parquet",
#             clusters_path="clustering_results_optimized.parquet",
#             query_dict=query_dict,
#             compression_method=method,
#             max_seq_per_cluster=30,  # Для метода 'sample' показываем максимум 30 seq на кластер
#             output_image=f'cluster_viz_{method}.png'
#         )
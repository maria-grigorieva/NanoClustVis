import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import time
import gc
from tqdm import tqdm

# Bokeh imports
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, HoverTool
from bokeh.models import CategoricalColorMapper, FactorRange, Span, CustomJS
from bokeh.models import Slider, Select, CheckboxGroup
from bokeh.layouts import row, column, gridplot
from bokeh.palettes import Category20, Viridis256, Turbo256, Spectral10
from bokeh.transform import linear_cmap, factor_cmap
from bokeh.io import output_notebook
from bokeh.embed import file_html
from bokeh.resources import CDN

# Datashader imports
try:
    import datashader as ds
    from datashader import transfer_functions as tf
    from datashader.colors import colormap_select
    import colorcet as cc
    from datashader.bokeh_ext import InteractiveImage

    DATASHADER_AVAILABLE = True
except ImportError:
    DATASHADER_AVAILABLE = False
    print("Datashader не установлен. Установите: pip install datashader colorcet")


@dataclass
class SequenceMatch:
    """Класс для представления совпадения последовательности."""
    __slots__ = ['query_name', 'position', 'length', 'seq_id']
    query_name: str
    position: int
    length: int
    seq_id: str


class BokehSequenceVisualizer:
    """
    Сверхбыстрая визуализация последовательностей с использованием Bokeh и Datashader.
    """

    def __init__(self, query_dict: Dict[str, str], use_datashader: bool = True):
        self.query_dict = query_dict
        self.query_to_idx = {name: i for i, name in enumerate(query_dict.keys())}
        self.query_names = list(query_dict.keys())
        self.n_queries = len(query_dict)

        # Настройка цветов
        self.colors = Category20[20] if len(query_dict) <= 20 else Turbo256
        self.color_map = {name: self._get_color(i) for i, name in enumerate(self.query_names)}

        # Настройка Datashader
        self.use_datashader = use_datashader and DATASHADER_AVAILABLE
        if use_datashader and not DATASHADER_AVAILABLE:
            print("Datashader недоступен, используется стандартный рендеринг Bokeh")

    def _get_color(self, idx: int) -> str:
        """Получить цвет для индекса."""
        if hasattr(self.colors, '__getitem__'):
            if len(self.colors) > idx:
                return self.colors[idx]
            else:
                return self.colors[idx % len(self.colors)]
        else:
            # Для Turbo256
            return self.colors[idx * 10 % 256]

    def load_data_batched(self, matches_path: str, clusters_path: str,
                          batch_size: int = 500_000) -> Tuple[pd.DataFrame, Dict]:
        """
        Загрузка данных с оптимизацией памяти.
        """
        print(f"Загрузка данных из {matches_path} и {clusters_path}...")

        # Загружаем кластеры
        clusters_df = pd.read_parquet(clusters_path)
        seq_to_cluster = clusters_df.set_index('seq_id')['cluster'].to_dict()
        cluster_sizes = clusters_df['cluster'].value_counts().to_dict()

        # Освобождаем память
        del clusters_df
        gc.collect()

        # Загружаем matches чанками
        parquet_file = pq.ParquetFile(matches_path)

        all_data = []
        valid_seq_ids = set(seq_to_cluster.keys())

        for batch in tqdm(parquet_file.iter_batches(batch_size=batch_size),
                          desc="Загрузка чанков"):
            df = batch.to_pandas()

            # Фильтруем только нужные seq_id
            mask = df['seq_id'].isin(valid_seq_ids)
            df_filtered = df[mask].copy()

            if len(df_filtered) == 0:
                continue

            # Добавляем информацию о кластере
            df_filtered['cluster'] = df_filtered['seq_id'].map(seq_to_cluster)

            # Добавляем индекс запроса
            df_filtered['query_idx'] = df_filtered['query_name'].map(self.query_to_idx)

            # Удаляем строки с неизвестными запросами
            df_filtered = df_filtered.dropna(subset=['query_idx'])
            df_filtered['query_idx'] = df_filtered['query_idx'].astype(int)

            all_data.append(df_filtered)

        if not all_data:
            raise ValueError("Нет данных после фильтрации")

        # Объединяем все чанки
        final_df = pd.concat(all_data, ignore_index=True)

        print(f"Загружено {len(final_df)} совпадений")
        print(f"Уникальных кластеров: {final_df['cluster'].nunique()}")
        print(f"Уникальных seq_id: {final_df['seq_id'].nunique()}")

        return final_df, cluster_sizes

    def prepare_data_for_bokeh(self, df: pd.DataFrame) -> Tuple[ColumnDataSource, Dict]:
        """
        Подготовка данных для Bokeh с правильными колонками.
        """
        print("Подготовка данных для Bokeh...")

        # Сортируем для правильного отображения
        df_sorted = df.sort_values(['cluster', 'seq_id', 'position']).copy()

        # Создаем y-координаты для каждой последовательности
        unique_seqs = df_sorted['seq_id'].unique()
        seq_to_y = {seq: i for i, seq in enumerate(unique_seqs)}
        df_sorted['y'] = df_sorted['seq_id'].map(seq_to_y)

        # ВАЖНО: Создаем отдельные колонки для top и bottom
        df_sorted['y_bottom'] = df_sorted['y']
        df_sorted['y_top'] = df_sorted['y'] + 1

        # Создаем x-координаты для начала и конца
        df_sorted['x_start'] = df_sorted['position']
        df_sorted['x_end'] = df_sorted['position'] + df_sorted['length']

        # Добавляем цвет для каждого запроса
        df_sorted['color'] = df_sorted['query_idx'].apply(
            lambda x: self._get_color(x)
        )

        # Добавляем строковые представления для hover
        df_sorted['seq_id_short'] = df_sorted['seq_id'].apply(lambda x: f"...{str(x)[-8:]}")

        # Проверяем, что все колонки существуют
        required_columns = ['x_start', 'x_end', 'y_bottom', 'y_top', 'color',
                            'query_name', 'position', 'length', 'seq_id_short', 'cluster']
        for col in required_columns:
            if col not in df_sorted.columns:
                print(f"ВНИМАНИЕ: колонка {col} отсутствует!")

        print(f"Колонки в DataFrame: {list(df_sorted.columns)}")
        print(f"Количество строк: {len(df_sorted)}")

        # Создаем источник данных
        source = ColumnDataSource(df_sorted)

        return source, seq_to_y

    def create_bokeh_plot(self, df: pd.DataFrame,
                          title: str = 'Sequence Clusters Visualization',
                          width: int = 1200, height: int = 800) -> Tuple[figure, ColumnDataSource]:
        """
        Создание интерактивного графика с Bokeh.
        """
        # Подготавливаем данные
        source, seq_to_y = self.prepare_data_for_bokeh(df)

        # Создаем фигуру с WebGL для ускорения
        p = figure(
            title=title,
            width=width,
            height=height,
            tools='pan,wheel_zoom,box_zoom,reset,save,hover',
            active_scroll='wheel_zoom',
            output_backend='webgl',
            x_axis_label='Position (bp)',
            y_axis_label='Sequence Index'
        )

        # Добавляем прямоугольники с правильными именами колонок
        # ИСПОЛЬЗУЕМ ТОЛЬКО ИМЕНА КОЛОНОК, КОТОРЫЕ СУЩЕСТВУЮТ В source
        p.quad(
            left='x_start',
            right='x_end',
            bottom='y_bottom',
            top='y_top',
            source=source,
            color='color',
            line_color=None,
            alpha=0.8
        )

        # Настройка осей
        if len(seq_to_y) < 100:
            tick_positions = list(seq_to_y.values())[::max(1, len(seq_to_y) // 20)]
            tick_labels = {i: f"...{seq[-8:]}" for seq, i in seq_to_y.items()
                           if i in tick_positions}
            p.yaxis.ticker = tick_positions
            p.yaxis.major_label_overrides = tick_labels
        else:
            p.yaxis.visible = False

        # Добавляем hover инструмент
        hover = p.select(dict(type=HoverTool))
        hover.tooltips = [
            ('Query', '@query_name'),
            ('Position', '@position'),
            ('Length', '@length'),
            ('Seq ID', '@seq_id_short'),
            ('Cluster', '@cluster')
        ]

        return p, source

    def create_cluster_size_plot(self, cluster_sizes: Dict,
                                 width: int = 400, height: int = 800) -> figure:
        """
        Создает график размеров кластеров.
        """
        # Подготавливаем данные
        clusters = sorted(cluster_sizes.keys())
        sizes = [cluster_sizes[c] for c in clusters]

        # Создаем y-позиции, соответствующие основному графику
        y_positions = list(range(len(clusters)))

        source = ColumnDataSource(data={
            'clusters': clusters,
            'sizes': sizes,
            'y': y_positions
        })

        # Создаем фигуру
        p = figure(
            title='Sequences per Cluster',
            width=width,
            height=height,
            tools='pan,wheel_zoom,reset',
            y_range=[0, len(clusters)],
            x_axis_label='Count'
        )

        # Горизонтальные бары
        p.hbar(
            y='y',
            right='sizes',
            height=0.8,
            source=source,
            color='steelblue',
            alpha=0.7
        )

        # Добавляем подписи кластеров
        if len(clusters) < 100:
            p.yaxis.ticker = list(range(len(clusters)))
            p.yaxis.major_label_overrides = {i: str(c) for i, c in enumerate(clusters)}
        else:
            p.yaxis.visible = False

        return p

    def visualize(self, matches_path: str, clusters_path: str,
                  compression_method: str = 'merge',
                  use_datashader: bool = False,
                  output_html: str = 'cluster_viz.html',
                  max_sequences: int = 2000):
        """
        Полная визуализация с Bokeh.
        """
        start_time = time.time()

        print("=" * 70)
        print("BOKEH ВИЗУАЛИЗАЦИЯ КЛАСТЕРОВ")
        print("=" * 70)

        # Загружаем данные
        df, cluster_sizes = self.load_data_batched(matches_path, clusters_path)

        # Применяем сжатие
        if compression_method == 'merge':
            # Удаляем дубликаты
            df = df.drop_duplicates(subset=['seq_id', 'query_name', 'position', 'length'])
        elif compression_method == 'representative':
            # Берем только репрезентативные последовательности
            rep_seqs = df.groupby('cluster')['seq_id'].first().values
            df = df[df['seq_id'].isin(rep_seqs)]

        print(f"После сжатия: {len(df)} совпадений")
        print(f"Уникальных последовательностей: {df['seq_id'].nunique()}")

        # Ограничиваем количество данных для интерактива
        if df['seq_id'].nunique() > max_sequences:
            print(f"Слишком много последовательностей, берем {max_sequences}...")

            # Берем топ кластеры
            top_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)[:30]
            top_cluster_ids = [c[0] for c in top_clusters]
            df_filtered = df[df['cluster'].isin(top_cluster_ids)]

            # Сэмплируем последовательности из каждого кластера
            sampled_dfs = []
            for cluster_id in top_cluster_ids:
                cluster_df = df_filtered[df_filtered['cluster'] == cluster_id]
                unique_seqs_in_cluster = cluster_df['seq_id'].unique()

                # Берем не больше 50 последовательностей из кластера
                n_samples = min(50, len(unique_seqs_in_cluster))
                sampled_seqs = np.random.choice(unique_seqs_in_cluster, n_samples, replace=False)

                for seq in sampled_seqs:
                    sampled_dfs.append(cluster_df[cluster_df['seq_id'] == seq])

            if sampled_dfs:
                df = pd.concat(sampled_dfs, ignore_index=True)
            else:
                df = df_filtered.head(max_sequences * 10)

        print(f"Итоговое количество строк: {len(df)}")
        print(f"Итоговое количество последовательностей: {df['seq_id'].nunique()}")

        # Создаем основной график
        main_plot, source = self.create_bokeh_plot(df)

        # Создаем график размеров кластеров
        size_plot = self.create_cluster_size_plot(cluster_sizes)

        # Объединяем графики
        layout = row(main_plot, size_plot)

        # Сохраняем
        output_file(output_html)
        save(layout)

        print(f"\nГрафик сохранен как {output_html}")
        print(f"Время выполнения: {time.time() - start_time:.2f} сек")

        return layout

    def create_interactive_dashboard(self, matches_path: str, clusters_path: str,
                                     output_html: str = 'dashboard.html',
                                     max_points: int = 100000):
        """
        Создает интерактивный дашборд с фильтрами.
        """
        # Загружаем данные
        df, cluster_sizes = self.load_data_batched(matches_path, clusters_path)

        # Удаляем дубликаты
        df = df.drop_duplicates(subset=['seq_id', 'query_name', 'position', 'length']).copy()

        # Ограничиваем количество точек для производительности
        if len(df) > max_points:
            print(f"Слишком много точек ({len(df)}), сэмплируем до {max_points}...")
            df = df.sample(n=max_points, random_state=42)

        # Подготавливаем данные
        df['x_end'] = df['position'] + df['length']

        # Создаем y-координаты
        unique_seqs = df['seq_id'].unique()
        seq_to_y = {seq: i for i, seq in enumerate(unique_seqs)}
        df['y'] = df['seq_id'].map(seq_to_y)
        df['y_bottom'] = df['y']
        df['y_top'] = df['y'] + 1
        df['x_start'] = df['position']

        # Добавляем цвет
        df['color'] = df['query_idx'].apply(lambda x: self._get_color(x))

        # Короткий seq_id для отображения
        df['seq_id_short'] = df['seq_id'].apply(lambda x: f"...{str(x)[-8:]}")

        # Проверяем наличие всех колонок
        required_columns = ['x_start', 'x_end', 'y_bottom', 'y_top', 'color',
                            'query_name', 'position', 'length', 'seq_id_short', 'cluster', 'y']

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            print(f"ВНИМАНИЕ: отсутствуют колонки: {missing}")

        # Создаем источник данных
        source = ColumnDataSource(df)

        # Создаем виджеты
        cluster_options = ["all"] + sorted([str(c) for c in cluster_sizes.keys()])
        cluster_select = Select(
            title="Cluster",
            value="all",
            options=cluster_options
        )

        query_options = ["all"] + self.query_names
        query_select = Select(
            title="Query Type",
            value="all",
            options=query_options
        )

        width_slider = Slider(
            title="Max Width",
            start=1000,
            end=50000,
            value=20000,
            step=1000
        )

        # Создаем график
        p = figure(
            title="Interactive Sequence Visualization",
            width=1000,
            height=600,
            tools='pan,wheel_zoom,box_zoom,reset,hover',
            output_backend='webgl',
            x_axis_label='Position (bp)',
            y_axis_label='Sequence Index'
        )

        # Добавляем прямоугольники
        p.quad(
            left='x_start',
            right='x_end',
            bottom='y_bottom',
            top='y_top',
            source=source,
            color='color',
            line_color=None,
            alpha=0.8
        )

        # Hover tool
        hover = p.select(dict(type=HoverTool))
        hover.tooltips = [
            ('Query', '@query_name'),
            ('Position', '@position'),
            ('Length', '@length'),
            ('Seq ID', '@seq_id_short'),
            ('Cluster', '@cluster')
        ]

        # JavaScript для фильтрации
        callback = CustomJS(args=dict(
            source=source,
            cluster_select=cluster_select,
            query_select=query_select,
            width_slider=width_slider
        ), code="""
            // Получаем оригинальные данные (сохраняем их где-то)
            if (!window.original_data) {
                window.original_data = {};
                const data = source.data;
                for (let key in data) {
                    window.original_data[key] = data[key].slice();
                }
            }

            // Параметры фильтрации
            const cluster = cluster_select.value;
            const query = query_select.value;
            const max_width = width_slider.value;

            // Фильтруем
            const indices = [];
            const orig = window.original_data;

            for (let i = 0; i < orig['seq_id'].length; i++) {
                if (cluster !== 'all' && String(orig['cluster'][i]) !== cluster) continue;
                if (query !== 'all' && orig['query_name'][i] !== query) continue;
                if (orig['position'][i] + orig['length'][i] > max_width) continue;
                indices.push(i);
            }

            // Создаем новый источник данных
            const new_data = {};
            for (let key in orig) {
                new_data[key] = indices.map(i => orig[key][i]);
            }

            source.data = new_data;
            source.change.emit();
        """)

        # Привязываем callback
        cluster_select.js_on_change('value', callback)
        query_select.js_on_change('value', callback)
        width_slider.js_on_change('value', callback)

        # Компоновка
        controls = column(cluster_select, query_select, width_slider, width=200)
        layout = row(controls, p)

        # Сохраняем
        output_file(output_html)
        save(layout)
        print(f"Дашборд сохранен как {output_html}")

        return layout


# Функция для быстрого запуска
def quick_visualize(matches_path: str, clusters_path: str,
                    query_dict: Dict[str, str],
                    method: str = 'bokeh'):
    """
    Быстрая визуализация с выбором метода.
    """
    viz = BokehSequenceVisualizer(query_dict)

    if method == 'bokeh':
        return viz.visualize(
            matches_path,
            clusters_path,
            use_datashader=False,
            output_html='bokeh_viz.html',
            max_sequences=2000
        )
    elif method == 'dashboard':
        return viz.create_interactive_dashboard(
            matches_path,
            clusters_path,
            output_html='dashboard.html',
            max_points=100000
        )
    else:
        raise ValueError(f"Unknown method: {method}")


# Пример использования
if __name__ == "__main__":
    query_dict = {
        "A1": "GATCAGTCCGATATC",
        "A2": "TCGACATGCTAGTGC",
        "A3": "GCTATCGGATACGTC",
        "S1_L_S2": "ATGACTGCCATTTTTTTGGCAGTCAT",
        "A1C": "GACGTATCCGATAGC",
        "A2C": "GCACTAGCATGTCGA",
        "A3C": "GATATCGGACTGATC"
    }

    # Создаем визуализатор
    viz = BokehSequenceVisualizer(query_dict)

    # Запускаем визуализацию
    viz.visualize(
        matches_path="matches.parquet",
        clusters_path="clustering_results_optimized.parquet",
        output_html="sequence_viz.html",
        max_sequences=2000  # Ограничиваем количество последовательностей
    )

    # Или дашборд
    # viz.create_interactive_dashboard(
    #     matches_path="matches.parquet",
    #     clusters_path="clustering_results_optimized.parquet",
    #     output_html="sequence_dashboard.html",
    #     max_points=100000
    # )
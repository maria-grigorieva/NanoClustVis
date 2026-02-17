"""
Пример визуализации полного датасета (миллионы строк) с использованием Datashader.
Файл: full_dataset_viz.py - ИСПРАВЛЕННАЯ ВЕРСИЯ 3
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import time
import gc
from typing import Dict, Tuple, Optional
import datashader as ds
import datashader.transfer_functions as tf
import colorcet as cc
from datashader.utils import lnglat_to_meters
import holoviews as hv
from holoviews.operation.datashader import datashade, rasterize
import hvplot.pandas
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.layouts import column
import warnings
warnings.filterwarnings('ignore')

# Для сохранения изображений
from PIL import Image
import io


class FullDatasetVisualizer:
    """
    Визуализация полного датасета без уменьшения количества объектов.
    Использует Datashader для агрегации миллионов точек.
    """

    def __init__(self, query_dict: Dict[str, str]):
        self.query_dict = query_dict
        self.query_to_idx = {name: i for i, name in enumerate(query_dict.keys())}
        self.query_names = list(query_dict.keys())
        self.n_queries = len(query_dict)

        # Цветовая карта для Datashader
        self.cmap = cc.glasbey_light

    def load_full_dataset(self, matches_path: str, limit: int = None) -> pd.DataFrame:
        """
        Загружает полный датасет (без фильтрации по кластерам).
        """
        print(f"\n{'='*60}")
        print("ЗАГРУЗКА ПОЛНОГО ДАТАСЕТА")
        print('='*60)

        start_time = time.time()

        # Загружаем весь файл
        df = pd.read_parquet(matches_path)

        if limit:
            df = df.head(limit)

        # Добавляем индекс запроса
        df['query_idx'] = df['query_name'].map(self.query_to_idx)

        # Удаляем строки с неизвестными запросами
        df = df.dropna(subset=['query_idx'])
        df['query_idx'] = df['query_idx'].astype(int)

        # ПРЕОБРАЗУЕМ В КАТЕГОРИАЛЬНЫЙ ТИП ДЛЯ DATASHADER
        df['query_cat'] = df['query_idx'].astype('category')

        # Создаем y-координаты для каждой последовательности
        unique_seqs = df['seq_id'].unique()
        seq_to_y = {seq: i for i, seq in enumerate(unique_seqs)}
        df['y'] = df['seq_id'].map(seq_to_y)

        # Добавляем конечную позицию
        df['x_end'] = df['position'] + df['length']

        elapsed = time.time() - start_time

        print(f"Загружено: {len(df):,} строк")
        print(f"Уникальных seq_id: {len(unique_seqs):,}")
        print(f"Уникальных query_name: {df['query_name'].nunique()}")
        print(f"Время загрузки: {elapsed:.2f} сек")
        print(f"Память: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        return df

    def save_image(self, img, filename: str):
        """
        Сохраняет изображение из datashader.
        """
        # Конвертируем в PIL Image и сохраняем
        img_data = (img.data * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_data[::-1])  # Переворачиваем для правильной ориентации
        pil_img.save(filename)
        print(f"  ✓ Изображение сохранено: {filename}")

    def visualize_with_datashader_points(self, df: pd.DataFrame,
                                         plot_width: int = 1200,
                                         plot_height: int = 800,
                                         output_file: str = 'full_dataset_points.png'):
        """
        Визуализация с помощью Datashader (точечная агрегация).
        """
        print(f"\n{'='*60}")
        print("ВИЗУАЛИЗАЦИЯ С DATASHADER (ТОЧКИ)")
        print('='*60)

        start_time = time.time()

        # Определяем границы
        x_range = (df['position'].min(), df['x_end'].max())
        y_range = (df['y'].min(), df['y'].max() + 1)

        print(f"X range: {x_range}")
        print(f"Y range: {y_range}")
        print(f"Всего точек: {len(df):,}")

        # Создаем точки для каждого совпадения (центр прямоугольника)
        points_df = pd.DataFrame({
            'x': df['position'] + df['length']/2,
            'y': df['y'] + 0.5,
            'query_cat': df['query_cat']
        })

        # Создаем канвас Datashader
        cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height,
                        x_range=x_range, y_range=y_range)

        # Агрегируем точки с категориальным агрегатором
        agg = cvs.points(
            points_df,
            'x',
            'y',
            agg=ds.count_cat('query_cat')
        )

        # Применяем цветовую карту
        img = tf.shade(agg, color_key=self.cmap, how='eq_hist')

        # Добавляем границы
        img = tf.spread(img, px=1)

        # Сохраняем
        self.save_image(img, output_file)

        elapsed = time.time() - start_time

        print(f"Визуализация завершена за {elapsed:.2f} сек")

        return img

    def visualize_with_datashader_lines_alternative(self, df: pd.DataFrame,
                                                   plot_width: int = 1200,
                                                   plot_height: int = 800,
                                                   output_file: str = 'full_dataset_lines.png'):
        """
        Альтернативный метод для визуализации линий через точки.
        Каждая линия представляется множеством точек.
        """
        print(f"\n{'='*60}")
        print("ВИЗУАЛИЗАЦИЯ С DATASHADER (ЛИНИИ ЧЕРЕЗ ТОЧКИ)")
        print('='*60)

        start_time = time.time()

        # Определяем границы
        x_range = (df['position'].min(), df['x_end'].max())
        y_range = (df['y'].min(), df['y'].max() + 1)

        print(f"X range: {x_range}")
        print(f"Y range: {y_range}")

        # Создаем точки для каждой позиции вдоль линии
        # Для каждой линии создаем points_per_line точек
        points_per_line = 10  # Количество точек на линию

        expanded_data = []
        for _, row in df.iterrows():
            x_start = row['position']
            x_end = row['x_end']
            y = row['y'] + 0.5
            cat = row['query_cat']

            # Создаем точки вдоль линии
            for i in range(points_per_line):
                x = x_start + (x_end - x_start) * i / (points_per_line - 1)
                expanded_data.append({
                    'x': x,
                    'y': y,
                    'query_cat': cat
                })

        points_df = pd.DataFrame(expanded_data)

        print(f"Создано {len(points_df):,} точек для линий")

        # Создаем канвас
        cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height,
                        x_range=x_range, y_range=y_range)

        # Агрегируем точки
        agg = cvs.points(
            points_df,
            'x',
            'y',
            agg=ds.count_cat('query_cat')
        )

        # Применяем цветовую карту
        img = tf.shade(agg, color_key=self.cmap, how='eq_hist')

        # Добавляем границы для лучшей видимости
        img = tf.spread(img, px=1)

        # Сохраняем
        self.save_image(img, output_file)

        elapsed = time.time() - start_time

        print(f"Визуализация завершена за {elapsed:.2f} сек")

        return img

    def visualize_with_datashader_continuous(self, df: pd.DataFrame,
                                             plot_width: int = 1200,
                                             plot_height: int = 800,
                                             output_file: str = 'full_dataset_continuous.png'):
        """
        Визуализация с непрерывной цветовой шкалой (без категорий).
        САМЫЙ БЫСТРЫЙ МЕТОД.
        """
        print(f"\n{'='*60}")
        print("ВИЗУАЛИЗАЦИЯ С DATASHADER (НЕПРЕРЫВНАЯ) - САМЫЙ БЫСТРЫЙ")
        print('='*60)

        start_time = time.time()

        # Определяем границы
        x_range = (df['position'].min(), df['x_end'].max())
        y_range = (df['y'].min(), df['y'].max() + 1)

        print(f"X range: {x_range}")
        print(f"Y range: {y_range}")
        print(f"Всего точек: {len(df):,}")

        # Создаем точки
        points_df = pd.DataFrame({
            'x': df['position'] + df['length']/2,
            'y': df['y'] + 0.5
        })

        # Создаем канвас
        cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height,
                        x_range=x_range, y_range=y_range)

        # Используем обычный count (не категориальный) - САМЫЙ БЫСТРЫЙ
        agg = cvs.points(points_df, 'x', 'y', agg=ds.count())

        # Применяем непрерывную цветовую карту
        img = tf.shade(agg, cmap=cc.fire, how='log')

        # Сохраняем
        self.save_image(img, output_file)

        elapsed = time.time() - start_time

        print(f"Визуализация завершена за {elapsed:.2f} сек")
        print(f"  ▶ Скорость: {len(df)/elapsed:,.0f} точек/сек")

        return img

    def visualize_with_datashader_rectangles(self, df: pd.DataFrame,
                                            plot_width: int = 1200,
                                            plot_height: int = 800,
                                            output_file: str = 'full_dataset_rectangles.png'):
        """
        Визуализация с помощью rect (для версий datashader с поддержкой rect).
        """
        print(f"\n{'='*60}")
        print("ВИЗУАЛИЗАЦИЯ С DATASHADER (RECT)")
        print('='*60)

        start_time = time.time()

        # Определяем границы
        x_range = (df['position'].min(), df['x_end'].max())
        y_range = (df['y'].min(), df['y'].max() + 1)

        print(f"X range: {x_range}")
        print(f"Y range: {y_range}")

        # Создаем DataFrame с прямоугольниками
        rect_df = pd.DataFrame({
            'x_min': df['position'],
            'x_max': df['x_end'],
            'y_min': df['y'],
            'y_max': df['y'] + 1,
            'query_cat': df['query_cat']
        })

        # Создаем канвас
        cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height,
                        x_range=x_range, y_range=y_range)

        # Пытаемся использовать rect, если доступно
        try:
            agg = cvs.rect(
                rect_df,
                x='x_min',
                y='y_min',
                x_end='x_max',
                y_end='y_max',
                agg=ds.count_cat('query_cat')
            )
            method_used = "rect"
        except AttributeError:
            # Если rect не доступен, используем points
            print("  rect не доступен, используем точки")
            points_df = pd.DataFrame({
                'x': df['position'] + df['length']/2,
                'y': df['y'] + 0.5,
                'query_cat': df['query_cat']
            })
            agg = cvs.points(
                points_df,
                'x',
                'y',
                agg=ds.count_cat('query_cat')
            )
            method_used = "points"

        # Применяем цветовую карту
        img = tf.shade(agg, color_key=self.cmap, how='eq_hist')

        # Добавляем границы
        img = tf.spread(img, px=1)

        # Сохраняем
        self.save_image(img, output_file)

        elapsed = time.time() - start_time

        print(f"Визуализация завершена за {elapsed:.2f} сек (метод: {method_used})")

        return img

    def visualize_with_matplotlib(self, df: pd.DataFrame,
                                  sample_size: int = 100000,
                                  output_file: str = 'full_dataset_matplotlib.png'):
        """
        Визуализация с matplotlib (только для сравнения, медленно).
        """
        print(f"\n{'='*60}")
        print("ВИЗУАЛИЗАЦИЯ С MATPLOTLIB (ДЛЯ СРАВНЕНИЯ)")
        print('='*60)

        start_time = time.time()

        # Сэмплируем данные (matplotlib не справится с миллионами)
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
            print(f"Сэмплировано {sample_size:,} точек из {len(df):,}")
        else:
            df_sample = df

        # Создаем фигуру
        fig, ax = plt.subplots(figsize=(16, 10))

        # Группируем по цветам
        colors = plt.cm.tab10(np.linspace(0, 1, len(df_sample['query_name'].unique())))

        for i, (query_name, color) in enumerate(zip(df_sample['query_name'].unique(), colors)):
            query_df = df_sample[df_sample['query_name'] == query_name]
            ax.scatter(
                query_df['position'] + query_df['length']/2,
                query_df['y'] + 0.5,
                s=0.1,
                alpha=0.5,
                color=color,
                label=query_name,
                rasterized=True  # Растеризуем для экономии памяти
            )

        ax.set_xlabel('Position (bp)')
        ax.set_ylabel('Sequence Index')
        ax.set_title(f'Dataset Visualization (sampled to {len(df_sample):,} points)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight', rasterized=True)
        plt.close()

        elapsed = time.time() - start_time

        print(f"Визуализация завершена за {elapsed:.2f} сек")
        print(f"Сохранено в: {output_file}")

        return fig

    def visualize_with_holoviews(self, df: pd.DataFrame,
                                 plot_width: int = 1200,
                                 plot_height: int = 800,
                                 output_file: str = 'full_dataset_holoviews.html'):
        """
        Визуализация с HoloViews + Datashader (интерактивный).
        """
        print(f"\n{'='*60}")
        print("ВИЗУАЛИЗАЦИЯ С HOLOVIEWS + DATASHADER")
        print('='*60)

        start_time = time.time()

        # Создаем элементы HoloViews
        hv.extension('bokeh')

        # Создаем точки
        points = hv.Points(
            df,
            kdims=['position', 'y'],
            vdims=['query_cat', 'length', 'query_name', 'seq_id']
        )

        # Применяем datashader
        shaded = datashade(
            points,
            aggregator=ds.count_cat('query_cat'),
            color_key=self.cmap,
            width=plot_width,
            height=plot_height
        )

        # Настраиваем отображение
        plot = shaded.opts(
            title=f"Full Dataset - {len(df):,} matches",
            xlabel="Position (bp)",
            ylabel="Sequence Index",
            tools=['hover', 'box_zoom', 'reset', 'save'],
            colorbar=True,
            framewise=True,
            bgcolor='black',
            width=plot_width,
            height=plot_height
        )

        # Сохраняем
        hv.save(plot, output_file)

        elapsed = time.time() - start_time

        print(f"Визуализация завершена за {elapsed:.2f} сек")
        print(f"Сохранено в: {output_file}")

        return plot

    def compare_methods(self, df: pd.DataFrame):
        """
        Сравнивает производительность разных методов визуализации.
        """
        print(f"\n{'='*60}")
        print("СРАВНЕНИЕ МЕТОДОВ ВИЗУАЛИЗАЦИИ")
        print(f"Датасет: {len(df):,} строк")
        print('='*60)

        methods = [
            ('Datashader (непрерывный)', self.visualize_with_datashader_continuous),
            ('Datashader (точки)', self.visualize_with_datashader_points),
            ('Datashader (линии через точки)', self.visualize_with_datashader_lines_alternative),
            ('Datashader (rect)', self.visualize_with_datashader_rectangles),
            ('HoloViews + Datashader', self.visualize_with_holoviews),
            ('Matplotlib (сэмплинг)', self.visualize_with_matplotlib),
        ]

        results = []

        for name, method in methods:
            try:
                start = time.time()
                method(df)
                elapsed = time.time() - start
                results.append((name, elapsed))
                print(f"{name}: {elapsed:.2f} сек")
            except Exception as e:
                print(f"{name}: ОШИБКА - {e}")

        # Сортируем по времени
        results.sort(key=lambda x: x[1])

        print("\n" + "="*60)
        print("РЕЙТИНГ ПО СКОРОСТИ:")
        for i, (name, elapsed) in enumerate(results, 1):
            print(f"{i}. {name}: {elapsed:.2f} сек")

        return results


def create_sample_data(n_sequences: int = 10000, n_matches_per_seq: int = 10) -> pd.DataFrame:
    """
    Создает тестовые данные для демонстрации.
    """
    print(f"\nСоздание тестовых данных: {n_sequences * n_matches_per_seq:,} строк")

    np.random.seed(42)

    seq_ids = [f"seq_{i:08d}" for i in range(n_sequences)]
    query_names = ['A1', 'A2', 'A3', 'S1_L_S2', 'A1C', 'A2C', 'A3C']
    query_to_idx = {name: i for i, name in enumerate(query_names)}

    data = []
    for seq_id in seq_ids:
        n_matches = np.random.poisson(n_matches_per_seq)
        for _ in range(n_matches):
            query = np.random.choice(query_names)
            position = np.random.randint(0, 10000)
            length = np.random.randint(10, 100)
            data.append({
                'seq_id': seq_id,
                'query_name': query,
                'position': position,
                'length': length
            })

    df = pd.DataFrame(data)

    # Добавляем необходимые колонки
    df['query_idx'] = df['query_name'].map(query_to_idx)
    df['query_cat'] = df['query_idx'].astype('category')

    # Создаем y-координаты
    unique_seqs = df['seq_id'].unique()
    seq_to_y = {seq: i for i, seq in enumerate(unique_seqs)}
    df['y'] = df['seq_id'].map(seq_to_y)
    df['x_end'] = df['position'] + df['length']

    print(f"Создано {len(df):,} строк")

    return df


def minimal_example():
    """
    Минимальный рабочий пример для быстрой проверки.
    """
    print("\n" + "="*60)
    print("МИНИМАЛЬНЫЙ ПРИМЕР")
    print("="*60)

    # Создаем тестовые данные
    np.random.seed(42)
    n_points = 1_000_000

    print(f"Создание {n_points:,} тестовых точек...")
    df = pd.DataFrame({
        'x': np.random.randint(0, 100000, n_points),
        'y': np.random.randint(0, 10000, n_points)
    })

    # Создаем канвас
    cvs = ds.Canvas(plot_width=800, plot_height=600)

    # Агрегируем
    print("Агрегация...")
    agg = cvs.points(df, 'x', 'y', agg=ds.count())

    # Создаем изображение
    print("Создание изображения...")
    img = tf.shade(agg, cmap=cc.fire, how='log')

    # Сохраняем
    img_data = (img.data * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_data[::-1])
    pil_img.save('minimal_example.png')

    print("Готово! Сохранено как 'minimal_example.png'")


def main():
    """
    Основная функция демонстрирующая все методы.
    """

    # Параметры
    DATA_PATH = "matches.parquet"
    USE_REAL_DATA = True
    LIMIT_ROWS = 1_000_000  # Ограничиваем для тестирования

    # Словарь запросов
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
    viz = FullDatasetVisualizer(query_dict)

    # Загружаем или создаем данные
    if USE_REAL_DATA:
        df = viz.load_full_dataset(DATA_PATH, limit=LIMIT_ROWS)
    else:
        df = create_sample_data(n_sequences=20000, n_matches_per_seq=10)

    print(f"\nДатасет готов: {len(df):,} строк")
    print(f"Диапазон X: {df['position'].min():,} - {df['x_end'].max():,}")
    print(f"Диапазон Y: 0 - {df['y'].max():,}")

    # === МЕТОД 1: Datashader непрерывный (САМЫЙ БЫСТРЫЙ) ===
    print("\n" + "="*60)
    print("МЕТОД 1: DATASHADER НЕПРЕРЫВНЫЙ (САМЫЙ БЫСТРЫЙ)")
    print("="*60)
    print("▶ Показывает плотность распределения")
    print("▶ Не различает типы запросов")

    img1 = viz.visualize_with_datashader_continuous(
        df,
        plot_width=1600,
        plot_height=1000,
        output_file='full_dataset_continuous.png'
    )

    # === МЕТОД 2: Datashader точки (категориальный) ===
    print("\n" + "="*60)
    print("МЕТОД 2: DATASHADER ТОЧКИ (КАТЕГОРИАЛЬНЫЙ)")
    print("="*60)
    print("▶ Различает типы запросов по цвету")
    print("▶ Не показывает длину последовательностей")

    img2 = viz.visualize_with_datashader_points(
        df,
        plot_width=1600,
        plot_height=1000,
        output_file='full_dataset_points.png'
    )

    # === МЕТОД 3: Datashader линии через точки ===
    print("\n" + "="*60)
    print("МЕТОД 3: DATASHADER ЛИНИИ ЧЕРЕЗ ТОЧКИ")
    print("="*60)
    print("▶ Показывает длину последовательностей")
    print("▶ Различает типы запросов по цвету")

    img3 = viz.visualize_with_datashader_lines_alternative(
        df,
        plot_width=1600,
        plot_height=1000,
        output_file='full_dataset_lines.png'
    )

    # === МЕТОД 4: HoloViews интерактивный ===
    print("\n" + "="*60)
    print("МЕТОД 4: HOLOVIEWS ИНТЕРАКТИВНЫЙ")
    print("="*60)
    print("▶ Позволяет зумировать и панорамировать")
    print("▶ Сохраняется как HTML")

    plot = viz.visualize_with_holoviews(
        df,
        plot_width=1200,
        plot_height=800,
        output_file='full_dataset_holoviews.html'
    )

    # === Сравнение производительности ===
    print("\n" + "="*60)
    print("СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("="*60)

    viz.compare_methods(df)

    print("\n" + "="*60)
    print("РЕКОМЕНДАЦИИ")
    print("="*60)
    print("• Для максимальной скорости: непрерывный метод (точки)")
    print("• Для визуализации с категориями: точки с категориями")
    print("• Для визуализации длины: линии через точки")
    print("• Для интерактивности: HoloViews")
    print("• Для презентаций: сохраняйте PNG с высоким разрешением")


if __name__ == "__main__":
    # Запускаем основную функцию
    main()

    # Или минимальный пример
    # minimal_example()    # minimal_example()
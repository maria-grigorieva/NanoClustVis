import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
import time
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings('ignore')


class ClusteringMethod(ABC):
    @abstractmethod
    def reduce_samples(self, features: np.ndarray, n_representative: int = 10000) -> np.ndarray:
        pass

    @abstractmethod
    def fit_predict(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class OptimizedHierarchicalClustering(ClusteringMethod):
    """
    Оптимизированная версия иерархической кластеризации для больших данных.
    Использует несколько стратегий ускорения.
    """

    def __init__(self, reduction_strategy='kmeans', pca_components=None):
        """
        Args:
            reduction_strategy: 'kmeans', 'random', или 'none'
            pca_components: если указано, сначала применяется PCA
        """
        self.reduction_strategy = reduction_strategy
        self.pca_components = pca_components
        self.representative_indices = None  # Сохраняем индексы представительных точек

    def reduce_samples(self, features: np.ndarray, n_representative: int = 10000) -> np.ndarray:
        """Ускоренное уменьшение количества образцов."""

        if len(features) <= n_representative:
            self.representative_indices = np.arange(len(features))
            return features

        start_time = time.time()

        # Опционально: сначала уменьшаем размерность через PCA
        if self.pca_components and self.pca_components < features.shape[1]:
            print(f"Применяем PCA: {features.shape[1]} -> {self.pca_components}")
            pca = PCA(n_components=self.pca_components, random_state=42)
            features_reduced = pca.fit_transform(features)
        else:
            features_reduced = features

        if self.reduction_strategy == 'kmeans':
            # Используем MiniBatchKMeans для быстрого выбора представителей
            print(f"K-means reduction: {len(features)} -> {n_representative}")
            kmeans = MiniBatchKMeans(
                n_clusters=n_representative,
                random_state=42,
                batch_size=10000,  # Увеличенный batch_size для скорости
                n_init=1,  # Всего одна инициализация
                max_iter=50  # Меньше итераций
            )
            kmeans.fit(features_reduced)

            # Для каждого кластера находим ближайшую точку к центру
            self.representative_indices = []
            for i in range(n_representative):
                cluster_points = np.where(kmeans.labels_ == i)[0]
                if len(cluster_points) > 0:
                    # Находим точку, ближайшую к центру кластера
                    center = kmeans.cluster_centers_[i]
                    distances = np.linalg.norm(features_reduced[cluster_points] - center, axis=1)
                    self.representative_indices.append(cluster_points[np.argmin(distances)])

            self.representative_indices = np.array(self.representative_indices)
            result = features[self.representative_indices]

        elif self.reduction_strategy == 'random':
            # Самая быстрая стратегия - случайная выборка
            print(f"Random sampling: {len(features)} -> {n_representative}")
            self.representative_indices = np.random.choice(
                len(features), n_representative, replace=False
            )
            result = features[self.representative_indices]

        else:  # 'none'
            self.representative_indices = np.arange(len(features))
            result = features

        elapsed = time.time() - start_time
        print(f"Reduction completed in {elapsed:.2f} seconds")

        return result

    def fit_predict(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Быстрая кластеризация с использованием разных методов в зависимости от размера.
        """
        start_time = time.time()

        n_samples = len(features)

        if n_samples > 20000:
            # Для очень больших данных используем двухэтапный подход
            print(f"Large dataset ({n_samples} samples), using two-stage clustering")

            # Этап 1: MiniBatchKMeans для предварительной кластеризации
            print("Stage 1: MiniBatchKMeans pre-clustering...")
            pre_clusters = min(1000, n_samples // 100)  # Не более 1000 предварительных кластеров
            kmeans = MiniBatchKMeans(
                n_clusters=pre_clusters,
                random_state=42,
                batch_size=10000,
                n_init=1
            )
            pre_labels = kmeans.fit_predict(features)

            # Этап 2: Иерархическая кластеризация на центроидах
            print(f"Stage 2: Hierarchical clustering on {pre_clusters} centroids...")
            centroids = kmeans.cluster_centers_

            # Используем AgglomerativeClustering для центроидов
            agg_clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
            centroid_labels = agg_clustering.fit_predict(centroids)

            # Этап 3: Присваиваем метки исходным точкам
            labels = centroid_labels[pre_labels]

        elif n_samples > 5000:
            # Для средних данных используем AgglomerativeClustering
            print(f"Medium dataset ({n_samples} samples), using AgglomerativeClustering")
            agg_clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
            labels = agg_clustering.fit_predict(features)

        else:
            # Для малых данных используем классическую иерархию
            print(f"Small dataset ({n_samples} samples), using classical hierarchical")
            linkage_matrix = linkage(features, method='ward', optimal_ordering=False)
            labels = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust') - 1

        elapsed = time.time() - start_time
        print(f"Clustering completed in {elapsed:.2f} seconds")

        return labels

    def get_name(self) -> str:
        return f"OptimizedHierarchical_{self.reduction_strategy}"


class FastKMeansClustering(ClusteringMethod):
    """
    Альтернативный метод: MiniBatchKMeans для максимальной скорости.
    """

    def __init__(self, use_pca=False, pca_components=10):
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.pca = None

    def reduce_samples(self, features: np.ndarray, n_representative: int = 10000) -> np.ndarray:
        # Для KMeans не нужно уменьшать выборку
        return features

    def fit_predict(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        start_time = time.time()

        # Опционально: PCA для ускорения
        if self.use_pca and features.shape[1] > self.pca_components:
            print(f"Applying PCA: {features.shape[1]} -> {self.pca_components}")
            self.pca = PCA(n_components=self.pca_components, random_state=42)
            features_reduced = self.pca.fit_transform(features)
        else:
            features_reduced = features

        # MiniBatchKMeans для скорости
        print(f"Running MiniBatchKMeans with {n_clusters} clusters...")
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=10000,
            n_init=3,
            max_iter=100
        )

        labels = kmeans.fit_predict(features_reduced)

        elapsed = time.time() - start_time
        print(f"KMeans clustering completed in {elapsed:.2f} seconds")

        return labels

    def get_name(self) -> str:
        return "FastKMeans"


# Основная функция с оптимизированной загрузкой
def optimized_clustering_pipeline(
        embeddings_path: str,
        n_clusters: int = 5,
        method: str = 'optimized_hierarchical',
        reduction_size: int = 10000,  # Увеличенный размер репрезентативной выборки
        batch_size: int = 100000
):
    """
    Оптимизированный пайплайн кластеризации для больших данных.
    """

    print("=" * 60)
    print("ОПТИМИЗИРОВАННАЯ КЛАСТЕРИЗАЦИЯ")
    print("=" * 60)

    total_start = time.time()

    # 1. Загружаем данные с чанками (streaming)
    print("\n1. ЗАГРУЗКА ДАННЫХ")
    print("-" * 40)

    # Используем DuckDB для эффективной загрузки только нужных колонок
    import duckdb

    conn = duckdb.connect()

    # Загружаем все данные (они небольшие - 76MB)
    print(f"Загрузка {embeddings_path}...")
    df = conn.execute(f"SELECT * FROM '{embeddings_path}'").df()
    conn.close()

    # Отделяем признаки от идентификаторов
    seq_ids = df['seq_id'].values
    feature_columns = [col for col in df.columns if col != 'seq_id']
    features = df[feature_columns].values.astype(np.float32)  # Используем float32 для экономии памяти

    print(f"Загружено: {len(df)} записей")
    print(f"Признаки: {feature_columns}")
    print(f"Размер в памяти: {features.nbytes / 1024 ** 2:.2f} MB")

    # 2. ВЫБОР МЕТОДА КЛАСТЕРИЗАЦИИ
    print("\n2. НАСТРОЙКА МЕТОДА КЛАСТЕРИЗАЦИИ")
    print("-" * 40)

    if method == 'optimized_hierarchical':
        clusterer = OptimizedHierarchicalClustering(
            reduction_strategy='kmeans',  # или 'random' для максимальной скорости
            pca_components=10  # Уменьшаем размерность для ускорения
        )
    elif method == 'fast_kmeans':
        clusterer = FastKMeansClustering(use_pca=True, pca_components=5)
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"Метод: {clusterer.get_name()}")
    print(f"Количество кластеров: {n_clusters}")

    # 3. УМЕНЬШЕНИЕ ВЫБОРКИ (если нужно)
    print("\n3. ПОДГОТОВКА ДАННЫХ")
    print("-" * 40)

    if method == 'optimized_hierarchical':
        # Для иерархической кластеризации уменьшаем выборку
        representative_features = clusterer.reduce_samples(features, reduction_size)
        print(f"Репрезентативная выборка: {len(representative_features)} точек")
    else:
        # Для KMeans используем все данные
        representative_features = features

    # 4. КЛАСТЕРИЗАЦИЯ
    print("\n4. ВЫПОЛНЕНИЕ КЛАСТЕРИЗАЦИИ")
    print("-" * 40)

    labels = clusterer.fit_predict(representative_features, n_clusters)

    # 5. СОЗДАНИЕ РЕЗУЛЬТАТОВ
    print("\n5. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("-" * 40)

    # Для иерархической кластеризации нужно сопоставить метки с исходными данными
    if method == 'optimized_hierarchical' and hasattr(clusterer, 'representative_indices'):
        # Создаем массив меток для всех точек
        all_labels = np.zeros(len(features), dtype=int)

        # Для представительных точек используем полученные метки
        for idx, label in zip(clusterer.representative_indices, labels):
            all_labels[idx] = label

        # Для остальных точек находим ближайшую представительную
        if len(clusterer.representative_indices) < len(features):
            print("Присваиваем метки остальным точкам...")
            from sklearn.metrics.pairwise import euclidean_distances

            # Используем представительные точки как центры
            representative_points = features[clusterer.representative_indices]

            # Обрабатываем батчами для экономии памяти
            batch_size = 100000
            for i in range(0, len(features), batch_size):
                batch_end = min(i + batch_size, len(features))
                batch_features = features[i:batch_end]

                # Находим ближайшие представительные точки
                distances = euclidean_distances(batch_features, representative_points)
                nearest_indices = np.argmin(distances, axis=1)

                # Присваиваем метки
                all_labels[i:batch_end] = labels[nearest_indices]
    else:
        all_labels = labels

    # Создаем финальный DataFrame
    results_df = pd.DataFrame({
        'seq_id': seq_ids,
        'cluster': all_labels
    })

    # Добавляем исходные признаки для анализа
    for i, col in enumerate(feature_columns):
        results_df[f'feature_{col}'] = features[:, i]

    # Сохраняем результаты
    output_path = "clustering_results.parquet"
    results_df.to_parquet(output_path, index=False, compression='snappy')

    # Также сохраняем в CSV для быстрого просмотра
    results_df[['seq_id', 'cluster']].to_csv("clustering_results.csv", index=False)

    total_elapsed = time.time() - total_start

    print(f"\nРЕЗУЛЬТАТЫ СОХРАНЕНЫ:")
    print(f"- Parquet: {output_path}")
    print(f"- CSV: clustering_results.csv")
    print(f"\nСтатистика по кластерам:")
    print(results_df['cluster'].value_counts().sort_index())
    print(f"\nОбщее время выполнения: {total_elapsed:.2f} секунд")

    return results_df, clusterer


# Сравнение методов
def compare_methods(embeddings_path: str, n_clusters: int = 5):
    """
    Сравнивает разные методы кластеризации.
    """

    results = {}

    methods = [
        ('optimized_hierarchical', OptimizedHierarchicalClustering(
            reduction_strategy='kmeans', pca_components=5
        )),
        ('fast_kmeans', FastKMeansClustering(use_pca=True, pca_components=5)),
        ('random_hierarchical', OptimizedHierarchicalClustering(
            reduction_strategy='random', pca_components=5
        ))
    ]

    for name, clusterer in methods:
        print(f"\n{'=' * 60}")
        print(f"ТЕСТИРОВАНИЕ: {name}")
        print('=' * 60)

        start_time = time.time()

        # Загружаем данные
        df = pd.read_parquet(embeddings_path)
        features = df.drop('seq_id', axis=1).values.astype(np.float32)

        if 'hierarchical' in name:
            rep_features = clusterer.reduce_samples(features, 5000)
            labels = clusterer.fit_predict(rep_features, n_clusters)
        else:
            labels = clusterer.fit_predict(features, n_clusters)

        elapsed = time.time() - start_time
        results[name] = {
            'time': elapsed,
            'labels': labels,
            'clusterer': clusterer
        }

        print(f"Время: {elapsed:.2f} сек")

    return results


if __name__ == "__main__":
    # Запуск оптимизированной кластеризации
    results, clusterer = optimized_clustering_pipeline(
        embeddings_path="embeddings.parquet",
        n_clusters=5,  # Количество кластеров
        method='optimized_hierarchical',  # или 'fast_kmeans'
        reduction_size=10000  # Размер репрезентативной выборки
    )

    # Для сравнения методов (раскомментировать при необходимости)
    # comparison = compare_methods("embeddings.parquet", n_clusters=5)
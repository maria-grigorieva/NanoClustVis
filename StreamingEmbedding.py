from abc import ABC, abstractmethod

import duckdb
import numpy as np
from gensim.models import Word2Vec
import pyarrow as pa
import pyarrow.parquet as pq


class StreamingEmbedding(ABC):
    """
    Base class for ALL streaming embeddings.

    Designed for datasets 10GB+.
    """

    def __init__(self, vector_size: int = 64):
        self.vector_size = vector_size
        self.model = None

    # ---------- REQUIRED ----------
    @abstractmethod
    def train(self, parquet_path: str):
        """
        Train model using streaming parquet.
        """
        pass

    @abstractmethod
    def embed_to_parquet(
        self,
        parquet_path: str,
        output_path: str
    ):
        """
        Generate embeddings and write them to parquet.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    # ---------- OPTIONAL ----------
    def reduce_samples(self, features, n=5000):
        """
        Optional sample reduction.
        """
        from sklearn.cluster import MiniBatchKMeans

        if len(features) <= n:
            return features

        kmeans = MiniBatchKMeans(
            n_clusters=n,
            batch_size=4096,
            random_state=42
        )

        return kmeans.fit(features).cluster_centers_


class PositionedSequenceDataset:
    """
    Streaming dataset yielding:
    ["A1@0", "A3@48", "A1C@86"]
    """

    def __init__(self, parquet_path, position_bucket=None):
        self.parquet_path = parquet_path
        self.position_bucket = position_bucket  # например 10

    def _token(self, query, pos):

        if self.position_bucket:
            pos = pos // self.position_bucket

        return f"{query}@{pos}"

    def __iter__(self):

        con = duckdb.connect()

        try:
            reader = con.execute(f"""
                SELECT seq_id, query_name, position
                FROM '{self.parquet_path}'
                ORDER BY seq_id, position
            """).fetch_record_batch()

            current_seq = None
            buffer = []

            for batch in reader:

                seq_ids = batch.column(0).to_pylist()
                queries = batch.column(1).to_pylist()
                positions = batch.column(2).to_pylist()

                for seq_id, query, pos in zip(seq_ids, queries, positions):

                    if current_seq is None:
                        current_seq = seq_id

                    if seq_id != current_seq:
                        yield buffer
                        buffer = []
                        current_seq = seq_id

                    buffer.append(self._token(query, pos))

            if buffer:
                yield buffer

        finally:
            con.close()



class Word2VecStreamingEmbedding(StreamingEmbedding):

    def __init__(
        self,
        vector_size=64,
        window=6,
        workers=8,
        min_count=2,
        epochs=5,
        sg=1,
        position_bucket=10
    ):
        super().__init__(vector_size)

        self.window = window
        self.workers = workers
        self.min_count = min_count
        self.epochs = epochs
        self.sg = sg
        self.position_bucket = position_bucket

    # ---------- TRAIN ----------
    def train(self, parquet_path):

        dataset = PositionedSequenceDataset(
            parquet_path,
            position_bucket=self.position_bucket
        )

        self.model = Word2Vec(
            vector_size=self.vector_size,
            window=self.window,
            workers=self.workers,
            min_count=self.min_count,
            sg=self.sg
        )

        print("Building vocab...")
        self.model.build_vocab(dataset)

        print("Corpus size:", self.model.corpus_count)

        print("Training Word2Vec...")
        self.model.train(
            dataset,
            total_examples=self.model.corpus_count,
            epochs=self.epochs
        )

        print("✅ Training complete")

    # ---------- EMBEDD ----------
    def embed_to_parquet(
        self,
        parquet_path,
        output_path,
        write_batch_size=25_000
    ):

        if self.model is None:
            raise RuntimeError("Train model first.")

        dataset = PositionedSequenceDataset(
            parquet_path,
            position_bucket=self.position_bucket
        )

        writer = None
        seq_ids = []
        embeddings = []

        con = duckdb.connect()

        # получаем seq_id отдельно (чтобы не терять!)
        seq_reader = con.execute(f"""
            SELECT DISTINCT seq_id
            FROM '{parquet_path}'
            ORDER BY seq_id
        """).fetch_record_batch()

        seq_iter = (
            seq_id
            for batch in seq_reader
            for seq_id in batch.column(0).to_pylist()
        )

        for seq_id, sequence in zip(seq_iter, dataset):

            vectors = [
                self.model.wv[token]
                for token in sequence
                if token in self.model.wv
            ]

            emb = (
                np.mean(vectors, axis=0)
                if vectors
                else np.zeros(self.vector_size, dtype=np.float32)
            )

            seq_ids.append(seq_id)
            embeddings.append(emb)

            if len(seq_ids) >= write_batch_size:

                table = pa.Table.from_pydict({
                    "seq_id": seq_ids,
                    **{
                        f"e{i}": [vec[i] for vec in embeddings]
                        for i in range(self.vector_size)
                    }
                })

                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path,
                        table.schema,
                        compression="zstd"
                    )

                writer.write_table(table)

                seq_ids.clear()
                embeddings.clear()

        if seq_ids:

            table = pa.Table.from_pydict({
                "seq_id": seq_ids,
                **{
                    f"e{i}": [vec[i] for vec in embeddings]
                    for i in range(self.vector_size)
                }
            })

            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)

            writer.write_table(table)

        if writer:
            writer.close()

        con.close()

        print("✅ Embeddings saved →", output_path)

    def get_name(self):
        return "positioned_streaming_word2vec"


class DirectEmbedding(StreamingEmbedding):
    """
    Creates embeddings directly from position values for each query_name.
    For each seq_id, creates columns from query_name and fills with position values.
    """

    def __init__(self, vector_size: int = 64):
        super().__init__(vector_size)
        self.query_names = None

    def train(self, parquet_path: str):
        """
        Train the model by identifying all unique query_names.
        For DirectEmbedding, this just collects the query_names.
        """
        # Read only query_name column to get unique values
        conn = duckdb.connect()

        # Get unique query_names
        query = f"""
        SELECT DISTINCT query_name 
        FROM '{parquet_path}'
        ORDER BY query_name
        """

        self.query_names = conn.execute(query).df()['query_name'].tolist()
        conn.close()

        print(f"Found {len(self.query_names)} unique query_names: {self.query_names}")

        # For DirectEmbedding, we don't actually train a model
        # but we set model to a dummy value for compatibility
        self.model = "direct_embedding"

        return self

    def embed_to_parquet(self, parquet_path: str, output_path: str):
        """
        Generate embeddings by pivoting query_name to columns with position values.
        For each seq_id, creates one row with columns for each query_name.
        """
        if self.query_names is None:
            raise ValueError("Model not trained. Call train() first.")

        conn = duckdb.connect()

        # Create pivot table using DuckDB's PIVOT
        pivot_query = f"""
        PIVOT (
            SELECT seq_id, query_name, position
            FROM '{parquet_path}'
        )
        ON query_name IN ({','.join([f"'{q}'" for q in self.query_names])})
        USING FIRST(position)
        ORDER BY seq_id
        """

        # Execute pivot and get result
        result_df = conn.execute(pivot_query).df()
        conn.close()

        # Fill NaN values with 0 or any default value
        result_df = result_df.fillna(0)

        # Convert to integer type for position values
        for col in result_df.columns:
            if col != 'seq_id':
                result_df[col] = result_df[col].astype(int)

        # Write to parquet
        table = pa.Table.from_pandas(result_df)
        pq.write_table(table, output_path)

        print(f"Embeddings saved to {output_path}")
        print(f"Shape: {result_df.shape}")
        print(f"Columns: {result_df.columns.tolist()}")

        return result_df

    def get_name(self) -> str:
        return "DirectEmbedding"
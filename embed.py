from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm
import duckdb

class StreamingWord2VecEmbedding:

    def __init__(self,
                 vector_size=64,
                 window=5,
                 workers=8,
                 min_count=1):

        self.vector_size = vector_size
        self.window = window
        self.workers = workers
        self.min_count = min_count

        self.model = None

    # ---------- generator ----------
    def _sequence_generator(self, parquet_path, batch_size=200_000):

        con = duckdb.connect()

        offset = 0

        while True:
            df = con.execute(f"""
                SELECT seq_id, query_name, position
                FROM '{parquet_path}'
                ORDER BY seq_id, position
                LIMIT {batch_size}
                OFFSET {offset}
            """).fetch_df()

            if df.empty:
                break

            for _, g in df.groupby("seq_id"):
                yield g["query_name"].tolist()

            offset += batch_size

    # ---------- TRAIN ----------
    def train(self, parquet_path):

        print("Building vocab...")

        self.model = Word2Vec(
            vector_size=self.vector_size,
            window=self.window,
            workers=self.workers,
            min_count=self.min_count
        )

        generator = self._sequence_generator(parquet_path)

        self.model.build_vocab(generator)

        print("Training Word2Vec (streaming)...")

        generator = self._sequence_generator(parquet_path)

        self.model.train(
            generator,
            total_examples=self.model.corpus_count,
            epochs=5
        )

        print("Training finished.")

    # ---------- EMBEDDINGS ----------
    def embed_parquet(self,
                      parquet_path,
                      output_path="embeddings.parquet",
                      batch_size=100_000):

        if self.model is None:
            raise RuntimeError("Train model first!")

        con = duckdb.connect()

        offset = 0
        rows = []

        while True:
            df = con.execute(f"""
                SELECT seq_id, query_name
                FROM '{parquet_path}'
                ORDER BY seq_id
                LIMIT {batch_size}
                OFFSET {offset}
            """).fetch_df()

            if df.empty:
                break

            for seq_id, g in df.groupby("seq_id"):

                words = g["query_name"].tolist()

                vectors = [
                    self.model.wv[w]
                    for w in words
                    if w in self.model.wv
                ]

                if vectors:
                    emb = np.mean(vectors, axis=0)
                else:
                    emb = np.zeros(self.vector_size)

                rows.append([seq_id, *emb])

            offset += batch_size

            # flush to parquet periodically
            if len(rows) > 50_000:
                self._write_batch(rows, output_path)
                rows.clear()

        if rows:
            self._write_batch(rows, output_path)

    def _write_batch(self, rows, output_path):

        import pyarrow as pa
        import pyarrow.parquet as pq

        cols = ["seq_id"] + [f"e{i}" for i in range(self.vector_size)]

        table = pa.Table.from_arrays(
            list(zip(*rows)),
            names=cols
        )

        if not os.path.exists(output_path):
            pq.write_table(table, output_path)
        else:
            pq.write_table(
                table,
                output_path,
                append=True
            )

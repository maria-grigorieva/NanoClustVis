from gensim.models import Word2Vec
import pyarrow as pa
import pyarrow.parquet as pq
import os

from StreamingEmbedding import StreamingEmbedding
from parquet_reader import ParquetSequenceMixin


class Word2VecStreamingEmbedding(
    StreamingEmbedding,
    ParquetSequenceMixin
):

    def __init__(
        self,
        vector_size=64,
        window=8,
        workers=8,
        min_count=1
    ):

        super().__init__(vector_size)

        self.window = window
        self.workers = workers
        self.min_count = min_count

    # ---------- TRAIN ----------
    def train(self, parquet_path):

        self.model = Word2Vec(
            vector_size=self.vector_size,
            window=self.window,
            workers=self.workers,
            min_count=self.min_count
        )

        generator = self._sequence_generator(parquet_path)

        print("Building vocab...")
        self.model.build_vocab(generator)

        generator = self._sequence_generator(parquet_path)

        print("Training...")
        self.model.train(
            generator,
            total_examples=self.model.corpus_count,
            epochs=5
        )

    # ---------- EMBEDD ----------
    def embed_to_parquet(
        self,
        parquet_path,
        output_path,
        batch_size=100_000
    ):

        if self.model is None:
            raise RuntimeError("Train model first.")

        con = duckdb.connect()

        offset = 0
        first_write = True

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

            rows = []

            for seq_id, g in df.groupby("seq_id"):

                words = g["query_name"].tolist()

                vectors = [
                    self.model.wv[w]
                    for w in words
                    if w in self.model.wv
                ]

                emb = (
                    np.mean(vectors, axis=0)
                    if vectors
                    else np.zeros(self.vector_size)
                )

                rows.append([seq_id, *emb])

            table = pa.Table.from_arrays(
                list(zip(*rows)),
                names=["seq_id"] + [f"e{i}" for i in range(self.vector_size)]
            )

            if first_write:
                pq.write_table(table, output_path)
                first_write = False
            else:
                pq.write_table(table, output_path, append=True)

            offset += batch_size

    def get_name(self):
        return "streaming_word2vec"

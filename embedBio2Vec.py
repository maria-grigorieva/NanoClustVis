from StreamingEmbedding import StreamingEmbedding
from parquet_reader import ParquetSequenceMixin


class Bio2VecStreamingEmbedding(
    StreamingEmbedding,
    ParquetSequenceMixin
):

    def __init__(
        self,
        k=5,
        vector_size=64,
        window=5
    ):
        super().__init__(vector_size)
        self.k = k
        self.window = window

    def _kmers(self, sequence):
        return [
            sequence[i:i+self.k]
            for i in range(len(sequence)-self.k+1)
        ]

    def _sequence_generator(self, parquet_path):

        for seq in super()._sequence_generator(parquet_path):
            yield self._kmers("".join(seq))

    def train(self, parquet_path):

        self.model = Word2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=1,
            workers=8
        )

        gen = self._sequence_generator(parquet_path)

        self.model.build_vocab(gen)

        gen = self._sequence_generator(parquet_path)

        self.model.train(
            gen,
            total_examples=self.model.corpus_count,
            epochs=5
        )

    def embed_to_parquet(self, parquet_path, output_path):
        raise NotImplementedError(
            "Same as Word2Vec — reuse if needed"
        )

    def get_name(self):
        return f"bio2vec_k{self.k}"

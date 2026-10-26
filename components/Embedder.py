"""Fixed 50-dimensional GloVe word and TransE relation embeddings."""

from pathlib import Path
import numpy as np


class Embedder:
    def __init__(self, glove_path, relation_path, vocabulary=None, dim=50):
        self.dim = dim
        self.word_embeddings = {}
        glove_path = Path(glove_path)
        if not glove_path.is_file():
            raise FileNotFoundError("Provide pretrained GloVe 50d vectors with --glove: {}".format(glove_path))
        wanted = set(vocabulary) if vocabulary is not None else None
        with glove_path.open(encoding="utf-8") as handle:
            for line in handle:
                values = line.rstrip().split()
                if len(values) != dim + 1:
                    continue
                word = values[0]
                if wanted is None or word in wanted:
                    self.word_embeddings[word] = np.asarray(values[1:], dtype=np.float32)
        if not self.word_embeddings:
            raise ValueError("No matching {}d GloVe vectors found".format(dim))
        relation_path = Path(relation_path)
        if not relation_path.is_file():
            raise FileNotFoundError("Provide pretrained TransE relation vectors using --relation-embeddings: {}".format(relation_path))
        with np.load(relation_path, allow_pickle=False) as data:
            relations = data["relations"].astype(str).tolist()
            vectors = np.asarray(data["vectors"], dtype=np.float32)
        if vectors.shape != (len(relations), dim):
            raise ValueError("Relation embeddings must have shape (relations, {})".format(dim))
        self.relation_embedding = dict(zip(relations, vectors))
        self.missing_words = set()

    def embed_word(self, word):
        vector = self.word_embeddings.get(word.lower())
        if vector is None:
            self.missing_words.add(word.lower())
            return np.zeros(self.dim, dtype=np.float32)
        return vector

    def embed_relation(self, relation):
        return self.relation_embedding.get(relation)

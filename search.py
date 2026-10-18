import pickle, re
from collections import defaultdict

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def tokenize(text):
    """Lowercase words and numbers. 'KV-CAR' becomes 'kv', 'car'."""
    return re.findall(r"[a-z0-9]+", text.lower())


class Searcher:
    def __init__(self, words=200, overlap=50):
        tag = f"{words}_{overlap}"
        self.index = faiss.read_index(f"indexes/{tag}.faiss")
        self.chunks = pickle.load(open(f"indexes/{tag}.pkl", "rb"))
        self.model = SentenceTransformer(MODEL_NAME, device="cuda")
        self._bm25 = None

    @property
    def bm25(self):
        if self._bm25 is None:
            self._bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        return self._bm25

    def dense_ids(self, queries, k):
        q_emb = self.model.encode([QUERY_PREFIX + q for q in queries], normalize_embeddings=True)
        _, ids = self.index.search(q_emb, k)
        return [list(row) for row in ids]

    def bm25_ids(self, queries, k):
        out = []
        for q in queries:
            scores = self.bm25.get_scores(tokenize(q))
            out.append(list(np.argsort(-scores)[:k]))
        return out

    def search(self, queries, method="dense", k=10, pool=50):
        """Returns, for each query, a ranked list of chunk dicts."""
        if method == "dense":
            ids = self.dense_ids(queries, k)
        elif method == "bm25":
            ids = self.bm25_ids(queries, k)
        elif method == "hybrid":
            ids = []
            for d, b in zip(self.dense_ids(queries, pool), self.bm25_ids(queries, pool)):
                score = defaultdict(float)
                for lst in (d, b):
                    for rank, cid in enumerate(lst, 1):
                        score[cid] += 1 / (60 + rank)  # reciprocal rank fusion
                ids.append(sorted(score, key=score.get, reverse=True)[:k])
        else:
            raise ValueError(method)
        return [[self.chunks[i] for i in row] for row in ids]

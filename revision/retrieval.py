"""
Dense retrieval over the TRAIN split only (R1-C1, R1-C3, R1-C4).

The index is a plain normalized embedding matrix (cosine = dot product), so the
same code serves train-time retrieval, inference, kNN baselines and diagnostics.
"""

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from common import read_json, read_jsonl


class Retriever:
    def __init__(self, index_dir, device: Optional[str] = None):
        self.dir = Path(index_dir)
        self.info = read_json(self.dir / "info.json")
        self.emb = np.load(self.dir / "train_emb.npy")
        self.meta = read_jsonl(self.dir / "train_meta.jsonl")
        assert len(self.meta) == len(self.emb), "index/meta length mismatch"
        self.row_of = {m["id"]: i for i, m in enumerate(self.meta)}
        self.labels = np.array([m["label"] for m in self.meta])
        self.langs = np.array([m["lang"] for m in self.meta])
        self.clusters = np.array([m["cluster_id"] for m in self.meta])
        self.device = device
        self._encoder = None
        self._cache = {}  # id -> query embedding, from <split>_emb.npy files
        for f in self.dir.glob("*_ids.json"):
            split = f.name[: -len("_ids.json")]
            ids, emb = read_json(f), np.load(self.dir / f"{split}_emb.npy")
            self._cache.update(zip(ids, emb))
        for i, m in enumerate(self.meta):
            self._cache.setdefault(m["id"], self.emb[i])

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.info["model"], device=self.device)
        return self._encoder

    def encode(self, texts: List[str]) -> np.ndarray:
        pre = self.info.get("prefix", "")
        mc = self.info["max_chars"]
        return self.encoder.encode([pre + t[:mc] for t in texts], normalize_embeddings=True,
                                   batch_size=32, show_progress_bar=False).astype("float32")

    def query_vector(self, rec: dict) -> np.ndarray:
        v = self._cache.get(rec["id"])
        return v if v is not None else self.encode([rec["text"]])[0]

    def search(self, q: np.ndarray, k: int, query_lang: Optional[str] = None, exclude_ids: Iterable[str] = (),
               exclude_clusters: Iterable[str] = (), neighbor_lang: str = "any", mode: str = "topk",
               max_sim: Optional[float] = None, rng: Optional[np.random.Generator] = None) -> List[dict]:
        """Return k neighbors as dicts (id, label, lang, sim, text, signals, cluster_id).

        mode: topk     k most similar
              balanced ceil(k/2) most similar per label, interleaved
              random   k uniformly random training articles (R1-C3 control)
              flipped  topk, but every neighbor label is inverted (R1-C3 mismatched evidence)
        neighbor_lang: any | same | cross  (R1-C4 cross-lingual retrieval)
        max_sim: drop neighbors more similar than this (R1-C2 near-duplicate control)
        """
        sims = self.emb @ q
        allowed = np.ones(len(sims), dtype=bool)
        for i in exclude_ids:
            if i in self.row_of:
                allowed[self.row_of[i]] = False
        ex_c = set(exclude_clusters)
        if ex_c:
            allowed &= ~np.isin(self.clusters, list(ex_c))
        if neighbor_lang != "any" and query_lang:
            same = self.langs == query_lang
            allowed &= same if neighbor_lang == "same" else ~same
        if max_sim is not None:
            allowed &= sims <= max_sim
        idx_allowed = np.flatnonzero(allowed)
        if len(idx_allowed) == 0:
            return []

        if mode == "random":
            rng = rng or np.random.default_rng(0)
            chosen = rng.choice(idx_allowed, size=min(k, len(idx_allowed)), replace=False)
            chosen = chosen[np.argsort(-sims[chosen])]
        elif mode == "balanced":
            per = (k + 1) // 2
            picks = []
            for lab in ("native ads", "berita murni"):
                pool = idx_allowed[self.labels[idx_allowed] == lab]
                picks.append(pool[np.argsort(-sims[pool])[:per]])
            chosen = [x for pair in zip(*picks) for x in pair][:k]
        else:
            top = idx_allowed[np.argpartition(-sims[idx_allowed], min(k, len(idx_allowed)) - 1)[:k]]
            chosen = top[np.argsort(-sims[top])]

        out = []
        for i in chosen:
            m = self.meta[int(i)]
            lab = m["label"]
            if mode == "flipped":
                lab = "berita murni" if lab == "native ads" else "native ads"
            out.append({"id": m["id"], "label": lab, "true_label": m["label"], "lang": m["lang"],
                        "sim": float(sims[i]), "text": m["text"], "signals": m["signals"],
                        "cluster_id": m["cluster_id"]})
        return out

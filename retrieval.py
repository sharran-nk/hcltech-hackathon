"""Hybrid retrieval: BM25 and FAISS are searched independently, then merged by chunk_id.

No score fusion and no reranking: BM25 scores (unbounded, corpus-dependent) and FAISS cosine
scores (-1..1) are on different scales, so they are NEVER added or compared. Each candidate
carries both scores separately; the Phase 7 evidence check judges them with separate thresholds.
The union only decides WHICH passages are considered, not how good they are.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

import config
from chunking import Chunk
from embeddings import get_embedder
from index_store import IndexStore
from textutil import tokenize


@dataclass
class Candidate:
    chunk: Chunk
    bm25_score: float
    bm25_rank: Optional[int]    # 1-based rank within BM25 top-k, else None
    faiss_score: float          # cosine similarity (always filled in)
    faiss_rank: Optional[int]   # 1-based rank within FAISS top-k, else None

    @property
    def found_by(self) -> list:
        return [n for n, r in (("bm25", self.bm25_rank), ("faiss", self.faiss_rank)) if r is not None]


class Retriever:
    def __init__(self, store: IndexStore, embedder=None):
        self.store = store
        self.embedder = embedder or get_embedder()

    def search(self, question: str, bm25_k: int = None, faiss_k: int = None) -> list:
        bm25_k = config.BM25_TOP_K if bm25_k is None else bm25_k
        faiss_k = config.FAISS_TOP_K if faiss_k is None else faiss_k
        snap = self.store.snapshot  # one consistent version for the whole query
        if snap.chunk_count == 0 or not question.strip():
            return []

        # --- BM25 (keyword). Only chunks sharing at least one query term score > 0.
        bm25_scores = snap.bm25.get_scores(tokenize(question))
        order = np.argsort(-bm25_scores)[:bm25_k]
        bm25_rank = {int(i): r + 1 for r, i in enumerate(order) if bm25_scores[i] > 0}

        # --- FAISS (semantic)
        qvec = self.embedder.embed_query(question)
        k = min(faiss_k, snap.chunk_count)
        sims, ids = snap.faiss_index.search(qvec.reshape(1, -1), k)
        faiss_rank = {int(i): r + 1 for r, i in enumerate(ids[0]) if i >= 0}

        # --- merge + dedupe by chunk position (== chunk_id); fill each missing score so every
        #     candidate has both, each on its own scale
        positions = set(bm25_rank) | set(faiss_rank)
        candidates = [
            Candidate(
                chunk=snap.chunks[p],
                bm25_score=float(bm25_scores[p]),
                bm25_rank=bm25_rank.get(p),
                faiss_score=float(snap.vectors[p] @ qvec),
                faiss_rank=faiss_rank.get(p),
            )
            for p in positions
        ]
        # Display order only (not fusion): found by both first, then best rank in either list.
        candidates.sort(key=lambda c: (-len(c.found_by), min(r for r in (c.bm25_rank, c.faiss_rank) if r is not None)))
        return candidates

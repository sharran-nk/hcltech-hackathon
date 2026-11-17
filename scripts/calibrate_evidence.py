"""Calibrate the evidence thresholds from labelled questions (data/eval/questions.json).

For every threshold combination on a grid, the gate is applied to every labelled question:
  * answerable question   -> correct only if a SUPPORTING passage contains the expected fact
  * unanswerable question -> correct only if NO passage supports it (otherwise it is a "leak")
Cost = LEAK_COST * leaks + 1 * wrongly-refused answerable questions  (a wrong answer is worse than a refusal).
Among equally cheap combinations the one nearest their centre is chosen (robust, away from edges).
Leave-one-out validation re-runs the selection without each question and tests on it, to
estimate how the rule behaves on questions it was not tuned on.

    python scripts/calibrate_evidence.py            # report only
    python scripts/calibrate_evidence.py --write    # also save data/evidence_thresholds.json
BM25 scores depend on corpus size, so re-run this after the document set changes substantially.
"""
import argparse
import json
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chunking  # noqa: E402
import config  # noqa: E402
from embeddings import get_embedder  # noqa: E402
from evidence import term_coverage  # noqa: E402
from index_store import IndexStore  # noqa: E402
from retrieval import Retriever  # noqa: E402
from textutil import tokenize  # noqa: E402

LEAK_COST = 3
B_GRID = np.round(np.arange(0.0, 8.01, 0.25), 2)
F_GRID = np.round(np.arange(-0.10, 0.901, 0.02), 2)
C_GRID = np.round(np.arange(0.0, 1.001, 0.05), 2)


def build_features():
    emb = get_embedder()
    with tempfile.TemporaryDirectory() as tmp:
        store = IndexStore(Path(tmp) / "index")
        store.load()
        for d in chunking.load_seed_docs():
            cs = chunking.chunk_seed_doc(d)
            store.upsert_document(d["doc_id"], cs, emb.embed_texts([c.text for c in cs]))
        retriever = Retriever(store, emb)
        questions = json.loads(config.EVAL_QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
        feats = []
        for q in questions:
            terms = set(tokenize(q["question"]))
            cands = retriever.search(q["question"])
            fact = (q.get("expected_fact") or "").lower()
            feats.append({
                "q": q,
                "bm25": np.array([c.bm25_score for c in cands]),
                "faiss": np.array([c.faiss_score for c in cands]),
                "cov": np.array([term_coverage(terms, c.chunk.text) for c in cands]),
                "has_fact": np.array([bool(fact) and fact in c.chunk.text.lower() for c in cands]),
                "chunk_ids": [c.chunk.chunk_id for c in cands],
            })
    return feats


def outcome_grid(f):
    """Boolean grid (B,F,C): True where the gate behaves CORRECTLY for this question."""
    if len(f["bm25"]) == 0:
        supports = np.zeros((len(B_GRID), len(F_GRID), len(C_GRID), 0), dtype=bool)
    else:
        ret = (f["bm25"][None, None, None, :] >= B_GRID[:, None, None, None]) | \
              (f["faiss"][None, None, None, :] >= F_GRID[None, :, None, None])
        supports = ret & (f["cov"][None, None, None, :] >= C_GRID[None, None, :, None])
    if f["q"]["expected_doc_id"]:
        return (supports & f["has_fact"][None, None, None, :]).any(axis=-1)
    return ~supports.any(axis=-1)


SMOOTH_RADIUS = (2, 2, 2)  # grid steps: +-0.5 BM25, +-0.04 cosine, +-0.10 coverage


def smooth(cost):
    """Mean cost over a box neighbourhood. A threshold sitting on a knife-edge (one question
    just barely refused) has expensive neighbours, so it loses to a threshold inside a stable region."""
    pads = [(r, r) for r in SMOOTH_RADIUS]
    padded = np.pad(cost.astype(float), pads, mode="edge")
    out, n = np.zeros(cost.shape), 0
    for db in range(2 * SMOOTH_RADIUS[0] + 1):
        for df in range(2 * SMOOTH_RADIUS[1] + 1):
            for dc in range(2 * SMOOTH_RADIUS[2] + 1):
                out += padded[db:db + cost.shape[0], df:df + cost.shape[1], dc:dc + cost.shape[2]]
                n += 1
    return out / n


def choose(cost):
    cost = smooth(cost)
    best = cost.min()
    idx = np.argwhere(np.isclose(cost, best))
    centre = idx.mean(axis=0)
    scale = np.array([len(B_GRID), len(F_GRID), len(C_GRID)], dtype=float)
    pick = idx[np.argmin((((idx - centre) / scale) ** 2).sum(axis=1))]
    return tuple(int(i) for i in pick), float(best), len(idx)


def main():
    global LEAK_COST, C_GRID
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--leak-cost", type=float, default=LEAK_COST, help="cost of answering an unanswerable question (refusing an answerable one costs 1)")
    ap.add_argument("--coverage-floor", type=float, default=0.0, help="never consider min_coverage below this (principled lower bound)")
    ap.add_argument("--quiet", action="store_true", help="only print the summary lines")
    args = ap.parse_args()
    LEAK_COST = args.leak_cost
    C_GRID = C_GRID[C_GRID >= args.coverage_floor - 1e-9]

    feats = build_features()
    ok = np.stack([outcome_grid(f) for f in feats])                      # (Q,B,F,C)
    answerable = np.array([bool(f["q"]["expected_doc_id"]) for f in feats])
    cost_each = np.where(answerable[:, None, None, None], (~ok) * 1, (~ok) * LEAK_COST)
    (ib, jf, kc), best_cost, n_ties = choose(cost_each.sum(axis=0))
    bm, fa, co = float(B_GRID[ib]), float(F_GRID[jf]), float(C_GRID[kc])

    print(f"Questions: {len(feats)} ({int(answerable.sum())} answerable, {int((~answerable).sum())} unanswerable)")
    raw_cost = cost_each.sum(axis=0)[ib, jf, kc]
    print(f"Grid points: {len(B_GRID) * len(F_GRID) * len(C_GRID)}; chosen point raw cost {raw_cost:.0f} "
          f"(neighbourhood-average {best_cost:.2f}); {n_ties} tied")
    print(f"CHOSEN  bm25_min={bm}  faiss_min={fa}  min_coverage={co}\n")

    if not args.quiet:
        print(f"{'id':5} {'label':6} {'result':16} supporting passages")
    leaks = refused = 0
    for f, good in zip(feats, ok[:, ib, jf, kc]):
        q = f["q"]
        sup = (((f["bm25"] >= bm) | (f["faiss"] >= fa)) & (f["cov"] >= co))
        ids = [f["chunk_ids"][i] for i in np.flatnonzero(sup)]
        if q["expected_doc_id"]:
            result = "answered OK" if good else ("REFUSED (miss)" if not ids else "wrong passages")
            refused += not good
        else:
            result = "refused OK" if good else "LEAK -> LLM"
            leaks += not good
        if not args.quiet:
            print(f"{q['id']:5} {'answer' if q['expected_doc_id'] else 'refuse':6} {result:16} {ids}  | {q['question'][:50]}")
        elif result in ("REFUSED (miss)", "wrong passages", "LEAK -> LLM"):
            print(f"   {q['id']} {result}: {q['question'][:60]}")
    print(f"\nTraining result: {int(answerable.sum()) - refused}/{int(answerable.sum())} answerable pass, "
          f"{int((~answerable).sum()) - leaks}/{int((~answerable).sum())} unanswerable refused (leaks: {leaks})")

    # leave-one-out
    loo_miss = loo_leak = 0
    for i, f in enumerate(feats):
        mask = np.arange(len(feats)) != i
        (b, j, k), _, _ = choose(cost_each[mask].sum(axis=0))
        good = ok[i, b, j, k]
        if not good:
            if answerable[i]:
                loo_miss += 1
            else:
                loo_leak += 1
    print(f"Leave-one-out : {int(answerable.sum()) - loo_miss}/{int(answerable.sum())} answerable pass, "
          f"{int((~answerable).sum()) - loo_leak}/{int((~answerable).sum())} unanswerable refused (leaks: {loo_leak})")

    if args.write:
        payload = {"bm25_min": bm, "faiss_min": fa, "min_coverage": co,
                   "calibrated_on": date.today().isoformat(), "questions": len(feats),
                   "leak_cost": LEAK_COST, "coverage_floor": args.coverage_floor, "corpus_chunks": 13,
                   "note": "BM25 scale depends on corpus size: recalibrate after the document set changes."}
        config.EVIDENCE_THRESHOLDS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved -> {config.EVIDENCE_THRESHOLDS_PATH}")


if __name__ == "__main__":
    main()

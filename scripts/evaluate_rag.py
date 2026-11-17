"""End-to-end evaluation of the RAG pipeline with the REAL Ollama model.

Indexes the six seed articles, then asks every labelled question in data/eval/questions.json.
  answerable   -> pass if not refused AND a citation points at the expected document
                  (answer text is printed so YOU can judge accuracy; 'fact-in-answer' is only a hint)
  unanswerable -> pass only if the exact refusal is returned
    python scripts/evaluate_rag.py [--json out.json]
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chunking  # noqa: E402
import config  # noqa: E402
from embeddings import get_embedder  # noqa: E402
from index_store import IndexStore  # noqa: E402
from rag_pipeline import REFUSAL, RagPipeline  # noqa: E402
from retrieval import Retriever  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write raw results to this file")
    args = ap.parse_args()

    emb = get_embedder()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        store = IndexStore(Path(tmp) / "index")
        store.load()
        for d in chunking.load_seed_docs():
            cs = chunking.chunk_seed_doc(d)
            store.upsert_document(d["doc_id"], cs, emb.embed_texts([c.text for c in cs]))
        pipeline = RagPipeline(Retriever(store, emb))
        questions = json.loads(config.EVAL_QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
        for q in questions:
            t0 = time.time()
            r = pipeline.answer(q["question"])
            expected = q["expected_doc_id"]
            if expected:
                cited_ok = any(c.doc_id == expected for c in r.citations)
                passed = (not r.refused) and cited_ok and r.mode == "llm"
                fact_hint = q["expected_fact"].lower() in r.answer.lower()
            else:
                passed, cited_ok, fact_hint = r.refused and r.answer == REFUSAL, None, None
            rows.append({"id": q["id"], "question": q["question"], "expected_doc": expected, "answer": r.answer,
                         "mode": r.mode, "refused": r.refused, "passed": passed, "citation_ok": cited_ok,
                         "fact_in_answer": fact_hint, "cited": [c.chunk_id for c in r.citations],
                         "reason": r.reason, "seconds": round(time.time() - t0, 1)})
            flag = "PASS" if passed else "FAIL"
            print(f"\n[{flag}] {q['id']} ({r.mode}, {rows[-1]['seconds']}s) {q['question']}")
            print(f"       -> {r.answer}")
            if r.citations:
                print(f"       cited: {[c.chunk_id for c in r.citations]}")

    ans = [r for r in rows if r["expected_doc"]]
    una = [r for r in rows if not r["expected_doc"]]
    print("\n" + "=" * 70)
    print(f"Answerable   : {sum(r['passed'] for r in ans)}/{len(ans)} answered with the right citation "
          f"(fact-in-answer hint: {sum(bool(r['fact_in_answer']) for r in ans)}/{len(ans)})")
    print(f"Unanswerable : {sum(r['passed'] for r in una)}/{len(una)} refused with the exact message")
    by_evidence = sum(1 for r in una if r["refused"] and r["reason"] != "model found the passages insufficient")
    print(f"               ({by_evidence} refused by the evidence check, "
          f"{sum(r['passed'] for r in una) - by_evidence} by the model)")
    print(f"LLM calls    : {sum(r['mode'] == 'llm' or 'model' in r['reason'] for r in rows)}; "
          f"mean latency {sum(r['seconds'] for r in rows) / len(rows):.1f}s")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

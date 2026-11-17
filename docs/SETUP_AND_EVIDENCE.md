# Solstice Learning – Document Q&A Assistant

Ask plain-English questions about uploaded product documentation and get answers with source
citations. If the documents do not contain the answer, the assistant replies exactly:
**"I don't have an answer for that."**

> Status: built phase by phase. See "Implemented vs future work" at the bottom for what is
> actually done and tested. (Streamlit UI, document lifecycle and deployment come in later phases.)

## Setup (development)
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
winget install UB-Mannheim.TesseractOCR        # OCR (Ubuntu: sudo apt install tesseract-ocr)
ollama pull qwen2.5:7b                         # local LLM, ~4.7 GB
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m pytest -q
```

## Run the app locally (no AWS needed)
```powershell
.\.venv\Scripts\python.exe scripts\generate_admin_hash.py      # choose an admin password, copy the printed line into .env
# in .env set:  DEV_FAKE_S3=1     (documents live in an in-memory fake S3 and vanish on restart)
ollama serve                                                   # if Ollama is not already running
.\.venv\Scripts\streamlit.exe run app.py
```
Open http://localhost:8501. **Ask questions** is the user mode; **Admin** (sidebar) asks for the password, then
allows uploads and shows each document's processing status. With `DEV_FAKE_S3=0` and `S3_BUCKET_NAME` set,
the app uses the real private bucket through the EC2 IAM role (see `docs/AWS_SETUP.md`).

## How a question is answered
1. **Hybrid retrieval** – BM25 (keywords) and FAISS (semantic) are searched independently; results are
   merged and de-duplicated by chunk id. Scores are never added: BM25 scores are unbounded and corpus-dependent,
   FAISS cosine scores are in -1..1, so each keeps its own scale.
2. **Evidence check (no LLM involved)** – a retrieved passage *supports* the question only if
   * **retriever signal:** `bm25_score >= BM25_MIN_SCORE` **or** `faiss_score >= FAISS_MIN_SCORE`
     (each compared to its own threshold), **and**
   * **term overlap:** at least `MIN_TERM_OVERLAP` of the question's content words (stop-words removed,
     word forms folded: link/links, invite/inviting) appear in the passage.

   If no passage supports the question the assistant returns the exact refusal **without calling the LLM**.
3. **Grounded generation** – only supporting passages (max 4) go to Qwen2.5 at temperature 0 with instructions to
   answer only from them, cite `[n]`, or reply with the refusal. If the model refuses, we return exactly the refusal.
4. **Citations** – built from the passages that were supplied (title, chunk id, page, snippet).
5. **Ollama down** – a passage that already passed the evidence check is shown as an extractive answer, with a notice.

### Evidence thresholds (calibrated, not guessed)
Stored in `data/evidence_thresholds.json`, produced by `python scripts/calibrate_evidence.py --leak-cost 1 --coverage-floor 0.4 --write`
from 33 labelled questions in `data/eval/questions.json` (the 5 case-study questions + 28 **project-created** ones:
paraphrases, near-misses and out-of-scope questions). Override with env vars `BM25_MIN_SCORE`, `FAISS_MIN_SCORE`, `MIN_TERM_OVERLAP`.

| Setting | Value | Meaning |
|---|---|---|
| `bm25_min` | 3.5 | BM25 score a passage needs (6-article corpus) |
| `faiss_min` | 0.46 | cosine similarity a passage needs |
| `min_coverage` | 0.4 | share of question content words found in the passage |

How they were chosen: every combination on a grid was scored against the labelled questions (answering an
unanswerable question and refusing an answerable one both count as errors); the combination with the lowest
cost *averaged over its neighbourhood* wins, so it sits inside a stable region instead of on a knife edge.
`min_coverage` has a floor of 0.4 (the lowest coverage any answerable question needed), because with only
33 questions the optimiser otherwise ignores overlap and relies on score cut-offs alone, which is exactly what
the design warns against ("similarity alone is not proof").

**Trade-off (intended):** the gate may refuse some answerable questions; that is preferred to inventing an answer.
It also cannot catch every *near-miss* (e.g. "price of the Enterprise plan" shares words with the docs but the
price is not documented). Those pass to the LLM, whose prompt tells it to refuse unless a passage directly answers.
BM25 scores depend on corpus size, so **re-run the calibration after the document set changes substantially.**

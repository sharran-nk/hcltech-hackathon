"""Evidence-sufficiency check: decides, WITHOUT calling the LLM, whether retrieval found support.

A retrieved passage SUPPORTS the question only if BOTH hold:
  1. Retriever signal (each score judged on its OWN scale, never added or compared):
        bm25_score  >= bm25_min      OR      faiss_score >= faiss_min
  2. Term overlap: at least `min_coverage` of the question's content words (stopwords removed,
     plurals folded) appear in the passage. A similar-sounding passage about a different topic
     (e.g. "SSO" for a "Salesforce" question) fails this even when its similarity score is decent.
The answer is attempted only if at least one passage supports the question. Otherwise the caller
returns the fixed refusal. Thresholds come from data/evidence_thresholds.json, produced by
scripts/calibrate_evidence.py from labelled questions (see README).

Trade-off: this is deliberately conservative. Some answerable questions may be refused; that is
preferred over inventing an answer. It also cannot catch every near-miss (e.g. "price of the
Enterprise plan" shares words with the docs); the LLM prompt is the second line of defence.
"""
from dataclasses import dataclass, field

import config
from retrieval import Candidate
from textutil import tokenize


class EvidenceNotCalibrated(RuntimeError):
    pass


@dataclass(frozen=True)
class EvidenceThresholds:
    bm25_min: float
    faiss_min: float
    min_coverage: float
    max_passages: int = 4

    @classmethod
    def from_config(cls) -> "EvidenceThresholds":
        values = (config.BM25_MIN_SCORE, config.FAISS_MIN_SCORE, config.MIN_TERM_OVERLAP)
        if any(v is None for v in values):
            raise EvidenceNotCalibrated(
                "Evidence thresholds are not calibrated. Run: python scripts/calibrate_evidence.py")
        return cls(*values, max_passages=config.MAX_CONTEXT_PASSAGES)


@dataclass
class PassageEvidence:
    candidate: Candidate
    coverage: float
    retriever_ok: bool
    supports: bool


@dataclass
class EvidenceResult:
    sufficient: bool
    reason: str
    supporting: list = field(default_factory=list)   # Candidates, best first, capped at max_passages
    passages: list = field(default_factory=list)     # PassageEvidence for EVERY candidate (for debugging/eval)


def term_coverage(question_terms: set, passage_text: str) -> float:
    if not question_terms:
        return 0.0
    return len(question_terms & set(tokenize(passage_text))) / len(question_terms)


def check_evidence(question: str, candidates: list, thresholds: EvidenceThresholds = None) -> EvidenceResult:
    t = thresholds or EvidenceThresholds.from_config()
    terms = set(tokenize(question))
    if not terms:
        return EvidenceResult(False, "question has no content words")
    if not candidates:
        return EvidenceResult(False, "no passages retrieved")

    passages = []
    for c in candidates:
        coverage = term_coverage(terms, c.chunk.text)
        retriever_ok = c.bm25_score >= t.bm25_min or c.faiss_score >= t.faiss_min
        passages.append(PassageEvidence(c, coverage, retriever_ok, retriever_ok and coverage >= t.min_coverage))

    supporting = [p for p in passages if p.supports]
    if not supporting:
        best = max(p.coverage for p in passages)
        return EvidenceResult(False, f"no passage met both the score and term-overlap rules (best coverage {best:.2f})",
                              passages=passages)
    # Best first: most question words covered; ties broken by BM25 score (candidates compared with each
    # other on the SAME scale, never BM25 against cosine), then by cosine score.
    supporting.sort(key=lambda p: (-p.coverage, -p.candidate.bm25_score, -p.candidate.faiss_score))
    top = [p.candidate for p in supporting[: t.max_passages]]
    return EvidenceResult(True, f"{len(supporting)} supporting passage(s)", supporting=top, passages=passages)

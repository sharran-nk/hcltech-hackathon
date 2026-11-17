"""Sentence splitting and chunking: 3 consecutive sentences per chunk, 1-sentence overlap.

Guarantees (all covered by tests/test_chunking.py):
- No text is dropped: the sentences, re-joined, equal the (whitespace-normalised) input.
- The final sentences are always in the last chunk.
- A short document (even 1 sentence) yields at least one chunk.
- Every chunk carries doc_id, filename, chunk_id, page_number and text.

Chunks are built PER PAGE so each chunk has an accurate page number. A sentence that
runs across a page break is therefore split in two (nothing is lost).
For seed articles, `filename` holds the article title.
"""
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Optional

import config

# Boundary = sentence-ending punctuation (+ closing quotes/brackets), whitespace, then
# something that looks like a sentence start. A lowercase next word (e.g. "'Forgot password?' on
# the login screen") is NOT a boundary, so UI labels stay intact.
_BOUNDARY = re.compile(r"""([.!?]+["'’”)\]]*)\s+(?=["'‘“(\[]?[A-Z0-9])""")
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "vs", "etc", "inc", "ltd", "co", "no", "fig", "approx", "st", "jr", "sr"}
_ABBREV_LIKE = re.compile(r"(?:[a-z]\.)+[a-z]|[a-z]")  # e.g, i.e, u.s, or a single initial


@dataclass
class Chunk:
    doc_id: str
    filename: str
    chunk_id: str
    chunk_index: int
    page_number: Optional[int]
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ sentences
def _is_abbreviation(token: str) -> bool:
    core = token.lower().lstrip("\"'‘“([").rstrip(".!?\"'’”)]")
    return core in _ABBREVIATIONS or bool(_ABBREV_LIKE.fullmatch(core))


def _wrap_long(sentence: str, max_chars: int) -> list:
    """Split an over-long 'sentence' at word boundaries so no text is lost."""
    if len(sentence) <= max_chars:
        return [sentence]
    parts, current = [], ""
    for word in sentence.split():
        if current and len(current) + 1 + len(word) > max_chars:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        parts.append(current)
    return parts


def _split_line(line: str) -> list:
    sentences, start = [], 0
    for m in _BOUNDARY.finditer(line):
        terminator = m.group(1)
        if terminator.startswith(".") and not terminator.startswith("..."):
            prev_token = line[start:m.end(1)].split()[-1]
            if _is_abbreviation(prev_token):
                continue
        sentences.append(line[start:m.end(1)].strip())
        start = m.end()
    tail = line[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def split_sentences(text: str, newline_is_boundary: bool = False) -> list:
    """Split text into sentences.

    newline_is_boundary=True for DOCX (a newline there is a real paragraph/table-row break).
    For PDF/OCR text a newline is just a visual line wrap, so it is treated as a space.
    """
    if newline_is_boundary:
        lines = [ln for ln in text.splitlines() if ln.strip()]
    else:
        lines = [" ".join(text.split())]
    out = []
    for line in lines:
        line = " ".join(line.split())
        for sentence in _split_line(line):
            out.extend(_wrap_long(sentence, config.MAX_SENTENCE_CHARS))
    return [s for s in out if s]


# ------------------------------------------------------------------ chunks
def group_sentences(sentences: list, size: int = None, overlap: int = None) -> list:
    """Return lists of sentences: windows of `size` advancing by size-overlap.
    The window loop stops only once a window reaches the last sentence, so the tail is never lost."""
    size = config.SENTENCES_PER_CHUNK if size is None else size
    overlap = config.SENTENCE_OVERLAP if overlap is None else overlap
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("Need size >= 1 and 0 <= overlap < size.")
    if not sentences:
        return []
    step, groups, i = size - overlap, [], 0
    while True:
        groups.append(sentences[i:i + size])
        if i + size >= len(sentences):
            return groups
        i += step


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def chunk_document(doc_id: str, filename: str, pages: list) -> list:
    """pages: list of (page_number_or_None, text, newline_is_boundary) tuples.
    Returns Chunk objects. Exact-duplicate chunk texts within one document are dropped."""
    chunks, seen = [], set()
    for page_number, text, newline_is_boundary in pages:
        for group in group_sentences(split_sentences(text, newline_is_boundary)):
            chunk_text = " ".join(group)
            digest = hashlib.sha1(_normalise(chunk_text).encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            index = len(chunks)
            chunks.append(Chunk(doc_id, filename, f"{doc_id}-c{index}", index, page_number, chunk_text))
    return chunks


def chunk_extraction(doc_id: str, filename: str, extraction) -> list:
    """Chunk a document_processor.ExtractionResult (DOCX keeps its paragraph breaks)."""
    pages = [(p.page_number, p.text, p.method == "docx") for p in extraction.pages]
    return chunk_document(doc_id, filename, pages)


def load_seed_docs() -> list:
    with open(config.SEED_DOCS_PATH, encoding="utf-8") as f:
        return json.load(f)["docs"]


def chunk_seed_doc(doc: dict) -> list:
    return chunk_document(doc["doc_id"], doc["title"], [(None, doc["content"], False)])

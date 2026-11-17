import pytest

import chunking as ch
import config
from chunking import chunk_document, chunk_seed_doc, group_sentences, load_seed_docs, split_sentences


def norm(t):
    return " ".join(t.split())


def S(n):
    return [f"Sentence number {i} here." for i in range(1, n + 1)]


# ------------------------------------------------------------------ sentence splitting
def test_basic_split():
    assert split_sentences("One is here. Two is here! Is three here? Yes.") == \
        ["One is here.", "Two is here!", "Is three here?", "Yes."]


def test_ui_label_with_question_mark_not_split():
    s = split_sentences("Click 'Forgot password?' on the login screen and enter your email. We'll send a link.")
    assert s == ["Click 'Forgot password?' on the login screen and enter your email.", "We'll send a link."]


def test_numbers_versions_and_menu_paths_not_split():
    s = split_sentences("Enterprise plans support SAML 2.0. Go to Settings > Security > SSO. Limit is 10,000 rows.")
    assert s == ["Enterprise plans support SAML 2.0.", "Go to Settings > Security > SSO.", "Limit is 10,000 rows."]


def test_abbreviations_not_split():
    assert len(split_sentences("Use a format, e.g. CSV or XLSX. Dr. Smith agreed. Done.")) == 3


def test_text_with_no_punctuation_is_one_sentence():
    assert split_sentences("no punctuation at all here") == ["no punctuation at all here"]


def test_pdf_line_wraps_are_joined_but_docx_paragraphs_kept():
    wrapped = "This sentence was wrapped\nacross two lines. Next one."
    assert split_sentences(wrapped) == ["This sentence was wrapped across two lines.", "Next one."]
    rows = "Plan | Rate limit\nGrowth | 60 requests per minute"
    assert split_sentences(rows, newline_is_boundary=True) == ["Plan | Rate limit", "Growth | 60 requests per minute"]


def test_overlong_sentence_is_wrapped_without_loss(monkeypatch):
    monkeypatch.setattr(config, "MAX_SENTENCE_CHARS", 50)
    text = " ".join(f"word{i}" for i in range(100))
    parts = split_sentences(text)
    assert len(parts) > 1 and all(len(p) <= 50 for p in parts)
    assert " ".join(parts) == text


@pytest.mark.parametrize("doc", load_seed_docs(), ids=lambda d: d["doc_id"])
def test_no_text_lost_in_seed_docs(doc):
    assert " ".join(split_sentences(doc["content"])) == norm(doc["content"])


# ------------------------------------------------------------------ grouping
@pytest.mark.parametrize("n,expected", [
    (1, [[0]]),
    (2, [[0, 1]]),
    (3, [[0, 1, 2]]),
    (4, [[0, 1, 2], [2, 3]]),
    (5, [[0, 1, 2], [2, 3, 4]]),
    (6, [[0, 1, 2], [2, 3, 4], [4, 5]]),
    (7, [[0, 1, 2], [2, 3, 4], [4, 5, 6]]),
])
def test_window_boundaries(n, expected):
    sents = list(range(n))
    assert group_sentences(sents, 3, 1) == expected


@pytest.mark.parametrize("n", range(1, 40))
def test_every_sentence_covered_and_last_included(n):
    groups = group_sentences(list(range(n)), 3, 1)
    covered = {x for g in groups for x in g}
    assert covered == set(range(n))
    assert n - 1 in groups[-1]


def test_consecutive_chunks_overlap_by_exactly_one_sentence():
    g = group_sentences(list(range(9)), 3, 1)
    for a, b in zip(g, g[1:]):
        if len(b) > 1:
            assert a[-1] == b[0]


def test_empty_input_gives_no_chunks_and_bad_params_rejected():
    assert group_sentences([]) == []
    with pytest.raises(ValueError):
        group_sentences([1, 2, 3], 3, 3)


# ------------------------------------------------------------------ design example D-01
def test_d01_matches_design_example():
    d01 = next(d for d in load_seed_docs() if d["doc_id"] == "D-01")
    chunks = chunk_seed_doc(d01)
    sents = split_sentences(d01["content"])
    assert len(sents) == 7
    assert [c.chunk_id for c in chunks] == ["D-01-c0", "D-01-c1", "D-01-c2"]
    assert chunks[0].text == " ".join(sents[0:3])
    assert chunks[1].text == " ".join(sents[2:5])
    assert chunks[2].text == " ".join(sents[4:7])
    assert "15 minutes" in chunks[2].text and "brand new reset link" in chunks[2].text  # the tail is kept


def test_short_seed_docs_produce_one_chunk_and_all_text_is_covered():
    for d in load_seed_docs():
        chunks = chunk_seed_doc(d)
        assert chunks, d["doc_id"]
        assert norm(d["content"]).endswith(chunks[-1].text.split(". ")[-1].strip())
        covered = " ".join(c.text for c in chunks)
        for sentence in split_sentences(d["content"]):
            assert sentence in covered


def test_key_facts_are_in_expected_chunks():
    by_doc = {d["doc_id"]: chunk_seed_doc(d) for d in load_seed_docs()}
    assert any("10,000 rows" in c.text for c in by_doc["D-03"])
    assert any("60 requests per minute" in c.text for c in by_doc["D-06"])
    assert not any("salesforce" in c.text.lower() for cs in by_doc.values() for c in cs)


# ------------------------------------------------------------------ metadata / dedup
def test_metadata_and_page_numbers():
    pages = [(1, "A first page sentence. Another one here. A third one.", False),
             (2, "Second page only has one sentence.", False)]
    chunks = chunk_document("doc123", "manual.pdf", pages)
    assert [c.page_number for c in chunks] == [1, 2]
    assert all(c.doc_id == "doc123" and c.filename == "manual.pdf" and c.text for c in chunks)
    assert [c.chunk_id for c in chunks] == ["doc123-c0", "doc123-c1"]
    assert chunks[0].to_dict()["page_number"] == 1


def test_duplicate_chunks_within_a_document_are_dropped():
    pages = [(1, "Same text appears here. And here too. Third one.", False),
             (2, "Same text appears here. And here too. Third one.", False)]
    chunks = chunk_document("d", "f.pdf", pages)
    assert len(chunks) == 1


def test_chunk_extraction_uses_docx_paragraph_breaks():
    from document_processor import ExtractedPage, ExtractionResult
    r = ExtractionResult(kind="docx", pages=[ExtractedPage(None, "Plan | Rate\nGrowth | 60\nEnterprise | 300", "docx")])
    chunks = ch.chunk_extraction("d", "t.docx", r)
    assert len(chunks) == 1 and chunks[0].page_number is None
    assert "Growth | 60" in chunks[0].text and "Enterprise | 300" in chunks[0].text

"""Tokenizer shared by BM25 indexing/querying and the Phase 7 term-overlap evidence check."""
import re

_TOKEN = re.compile(r"[a-z0-9]+")

# Small, conservative list of question/filler words. Domain words (export, rows, limit,
# password, plan ...) are deliberately NOT here.
STOPWORDS = frozenset("""
a an the and or but if then so of to in on at by for from with about as into than that this these those
is are was were be been being am do does did done can could should would will shall may might must
i me my we our you your he she it its they them their there here
what which who whom whose when where why how
not no yes also any some all each per over under up down out off just only very
have has had get got use used using
solstice
""".split())  # "solstice" = the product name: it appears in almost every question, so it says nothing about the topic


def _fold(token: str) -> str:
    """Crude suffix folding so word forms match: link/links, invite/inviting/invited,
    export/exported, expire/expires. NOT a real stemmer (some stems look odd, e.g. 'statu');
    that is fine because the same function is applied to documents and questions."""
    if len(token) <= 3 or token.isdigit():
        return token
    if token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    for suffix in ("ing", "ed"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    if token.endswith("e") and len(token) > 3:
        token = token[:-1]
    return token


def tokenize(text: str) -> list:
    """Lowercase alphanumeric tokens: stopwords and 1-letter noise removed, word forms folded."""
    tokens = []
    for t in _TOKEN.findall(text.lower()):
        if t in STOPWORDS or (len(t) == 1 and not t.isdigit()):
            continue
        tokens.append(_fold(t))
    return tokens

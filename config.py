"""Central configuration. All settings come from environment variables
(optionally loaded from a local .env file) so no secrets live in code.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")  # no-op if the file does not exist

NO_ANSWER_MESSAGE = "I don't have an answer for that."


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


# --- AWS ---
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "")
S3_PREFIX = os.getenv("S3_PREFIX", "documents/")

# --- Admin auth ---
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", "")
SESSION_TIMEOUT_MINUTES = _int("SESSION_TIMEOUT_MINUTES", 30)
MAX_FAILED_LOGINS = _int("MAX_FAILED_LOGINS", 5)
LOGIN_DELAY_SECONDS = _float("LOGIN_DELAY_SECONDS", 2)
LOGIN_LOCKOUT_MINUTES = _int("LOGIN_LOCKOUT_MINUTES", 5)

# --- Development only: run S3 code against an in-memory fake (moto). NEVER set in production. ---
DEV_FAKE_S3 = os.getenv("DEV_FAKE_S3", "0") == "1"

# --- Uploads ---
MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 20)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_PDF_PAGES = _int("MAX_PDF_PAGES", 100)
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".json"}

# --- Extraction / OCR ---
def _find_tesseract() -> str:
    import shutil
    explicit = os.getenv("TESSERACT_CMD", "")
    if explicit:
        return explicit
    found = shutil.which("tesseract")
    if found:
        return found
    win_default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    return win_default if os.path.exists(win_default) else "tesseract"


TESSERACT_CMD = _find_tesseract()
OCR_LANG = os.getenv("OCR_LANG", "eng")
OCR_DPI = _int("OCR_DPI", 200)
OCR_TIMEOUT_SECONDS = _int("OCR_TIMEOUT_SECONDS", 60)       # per page / image
MIN_PAGE_TEXT_CHARS = _int("MIN_PAGE_TEXT_CHARS", 20)       # fewer chars => treat PDF page as scanned
MAX_IMAGE_PIXELS = _int("MAX_IMAGE_PIXELS", 50_000_000)     # guards against decompression bombs
MAX_DOCX_UNCOMPRESSED_MB = _int("MAX_DOCX_UNCOMPRESSED_MB", 100)  # guards against zip bombs

# --- Local runtime state (metadata, indexes). Originals live in S3 only. ---
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "runtime")).resolve()
INDEX_DIR = DATA_DIR / "index"          # FAISS + BM25 + chunk metadata (+ previous version backup)
TMP_DIR = DATA_DIR / "tmp"              # temporary processing files, always cleaned up
METADATA_DB_PATH = DATA_DIR / "documents.json"  # persistent document registry/status
SEED_DOCS_PATH = BASE_DIR / "data" / "seed" / "docs.json"
EXAMPLE_QUESTIONS_PATH = BASE_DIR / "data" / "seed" / "example_questions.json"

# --- Models ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TIMEOUT_SECONDS = _int("OLLAMA_TIMEOUT_SECONDS", 120)
# Which LLM backend answers questions: "ollama" (local) or "bedrock" (Amazon Bedrock, IAM-role credentials).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "qwen.qwen3-32b-v1:0")
BEDROCK_REGION = os.getenv("BEDROCK_REGION", AWS_REGION)

# --- Chunking / retrieval ---
SENTENCES_PER_CHUNK = _int("SENTENCES_PER_CHUNK", 3)
SENTENCE_OVERLAP = _int("SENTENCE_OVERLAP", 1)
MAX_SENTENCE_CHARS = _int("MAX_SENTENCE_CHARS", 600)   # longer "sentences" (tables, run-on OCR) are split at word boundaries
EMBEDDING_BATCH_SIZE = _int("EMBEDDING_BATCH_SIZE", 32)
BM25_TOP_K = _int("BM25_TOP_K", 5)
FAISS_TOP_K = _int("FAISS_TOP_K", 5)

# --- Evidence check ---
# Values are produced by scripts/calibrate_evidence.py (saved to data/evidence_thresholds.json)
# and can be overridden by environment variables. BM25 and FAISS get SEPARATE thresholds
# because their score scales differ. None = not calibrated yet.
EVIDENCE_THRESHOLDS_PATH = BASE_DIR / "data" / "evidence_thresholds.json"
EVAL_QUESTIONS_PATH = BASE_DIR / "data" / "eval" / "questions.json"


def _load_saved_thresholds() -> dict:
    import json
    try:
        return json.loads(EVIDENCE_THRESHOLDS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _opt_float(env_name: str, saved_key: str):
    raw = os.getenv(env_name)
    if raw not in (None, ""):
        return float(raw)
    return _load_saved_thresholds().get(saved_key)


BM25_MIN_SCORE = _opt_float("BM25_MIN_SCORE", "bm25_min")
FAISS_MIN_SCORE = _opt_float("FAISS_MIN_SCORE", "faiss_min")
MIN_TERM_OVERLAP = _opt_float("MIN_TERM_OVERLAP", "min_coverage")
MAX_CONTEXT_PASSAGES = _int("MAX_CONTEXT_PASSAGES", 4)


def ensure_runtime_dirs() -> None:
    """Create local runtime folders (idempotent)."""
    for d in (DATA_DIR, INDEX_DIR, TMP_DIR):
        d.mkdir(parents=True, exist_ok=True)

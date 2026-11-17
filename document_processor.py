"""Validate uploads and extract text (PDF, DOCX, PNG, JPG) with OCR where needed.

Security: uploads are only PARSED (PyMuPDF / python-docx / Pillow), never executed.
Validation checks extension, size, magic bytes, and format-specific sanity limits.

OCR note: Tesseract reads printed text from pixels. It does NOT understand diagrams,
charts or screenshots' meaning, and this module makes no such claim.
"""
import hashlib
import io
import json
import logging
import shutil
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pymupdf
import pytesseract
from PIL import Image, ImageOps

import config

logger = logging.getLogger(__name__)

pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

MAGIC = {
    "pdf": (b"%PDF-",),
    "png": (b"\x89PNG\r\n\x1a\n",),
    "jpeg": (b"\xff\xd8\xff",),
    "docx": (b"PK\x03\x04",),
}
EXT_TO_KIND = {".pdf": "pdf", ".docx": "docx", ".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".json": "json"}


class DocumentProcessingError(Exception):
    """Base class. str(exc) is safe to show to the admin."""


class ValidationError(DocumentProcessingError):
    """The upload is rejected before anything is stored."""


class ExtractionError(DocumentProcessingError):
    """The file is valid but no usable text could be extracted."""


@dataclass
class ExtractedPage:
    page_number: Optional[int]   # 1-based for PDF; None for DOCX / images
    text: str
    method: str                  # "text" | "ocr" | "docx"


@dataclass
class ExtractionResult:
    kind: str
    pages: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    page_count: int = 0


# ------------------------------------------------------------------ helpers
def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def detect_kind(data: bytes) -> Optional[str]:
    for kind, prefixes in MAGIC.items():
        if any(data.startswith(p) for p in prefixes):
            return kind
    if data.lstrip(b"\xef\xbb\xbf \t\r\n")[:1] in (b"{", b"["):  # JSON has no magic bytes: it must open with { or [
        return "json"
    return None


def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


@contextmanager
def temp_workspace():
    """A private temp dir under DATA_DIR/tmp, ALWAYS removed (even on errors).
    Kept separate from permanent storage (originals live only in S3)."""
    config.TMP_DIR.mkdir(parents=True, exist_ok=True)
    path = config.TMP_DIR / f"job-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


# ------------------------------------------------------------------ validation
def validate_upload(filename: str, data: bytes) -> str:
    """Raise ValidationError, or return the detected kind ('pdf','docx','png','jpeg')."""
    ext = Path(filename or "").suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise ValidationError(f"Unsupported file type '{ext or 'none'}'. Allowed: PDF, DOCX, PNG, JPG/JPEG.")
    if not data:
        raise ValidationError("The file is empty.")
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise ValidationError(f"File is larger than the {config.MAX_UPLOAD_MB} MB limit.")

    kind = detect_kind(data)
    if kind is None or kind != EXT_TO_KIND[ext]:
        raise ValidationError("File content does not match its extension (possible renamed or malicious file).")

    if kind == "pdf":
        _validate_pdf(data)
    elif kind == "docx":
        _validate_docx(data)
    elif kind == "json":
        _validate_json(data)
    else:
        _validate_image(data)
    return kind


def _json_texts(obj) -> list:
    """Text sections of a JSON document. Understands {"docs":[{"title","content"}]} (and a bare list of such
    objects): one section per article. Any other JSON falls back to every string value, one section."""
    items = obj.get("docs") if isinstance(obj, dict) and isinstance(obj.get("docs"), list) else obj
    if isinstance(items, list) and items and all(isinstance(i, dict) for i in items):
        sections = []
        for item in items:
            body = next((item[k] for k in ("content", "text", "body") if isinstance(item.get(k), str)), "")
            title = item.get("title") if isinstance(item.get("title"), str) else ""
            if body.strip():
                sections.append(f"{title.strip()}. {body.strip()}" if title.strip() else body.strip())
        if sections:
            return sections
    lines = []

    def walk(node, key=""):
        if isinstance(node, str):
            if node.strip():
                text = f"{key}: {node.strip()}" if key else node.strip()
                lines.append(text if text[-1] in ".!?" else text + ".")
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(v, str(k))
        elif isinstance(node, list):
            for v in node:
                walk(v, key)

    walk(obj)
    return ["\n".join(lines)] if lines else []


def _validate_json(data: bytes) -> None:
    try:
        parsed = json.loads(data.decode("utf-8-sig"))
        has_text = bool(_json_texts(parsed))
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise ValidationError("The JSON file is not valid UTF-8 JSON.") from None
    if not has_text:
        raise ValidationError("The JSON file contains no text content.")


def _validate_pdf(data: bytes) -> None:
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception:
        raise ValidationError("The PDF is corrupted or unreadable.") from None
    try:
        if doc.needs_pass or doc.is_encrypted:
            raise ValidationError("Password-protected PDFs are not supported.")
        if doc.page_count == 0:
            raise ValidationError("The PDF has no pages.")
        if doc.page_count > config.MAX_PDF_PAGES:
            raise ValidationError(f"The PDF has {doc.page_count} pages; the limit is {config.MAX_PDF_PAGES}.")
    finally:
        doc.close()


def _validate_docx(data: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = set(zf.namelist())
            if "word/document.xml" not in names or "[Content_Types].xml" not in names:
                raise ValidationError("The file is not a valid DOCX document.")
            total = sum(i.file_size for i in zf.infolist())
            if total > config.MAX_DOCX_UNCOMPRESSED_MB * 1024 * 1024 or len(names) > 5000:
                raise ValidationError("The DOCX expands to an unsafe size (possible zip bomb).")
    except zipfile.BadZipFile:
        raise ValidationError("The DOCX is corrupted.") from None


def _validate_image(data: bytes) -> None:
    try:
        with Image.open(io.BytesIO(data)) as img:
            w, h = img.size
            if w * h > config.MAX_IMAGE_PIXELS:
                raise ValidationError("The image dimensions are too large.")
            img.verify()
    except ValidationError:
        raise
    except Exception:
        raise ValidationError("The image is corrupted or unreadable.") from None


# ------------------------------------------------------------------ OCR
class OcrUnavailable(ExtractionError):
    pass


def _ocr_image(img: Image.Image) -> str:
    try:
        return pytesseract.image_to_string(img, lang=config.OCR_LANG, timeout=config.OCR_TIMEOUT_SECONDS)
    except pytesseract.TesseractNotFoundError:
        raise OcrUnavailable("Tesseract OCR is not installed or not found (set TESSERACT_CMD).") from None


# ------------------------------------------------------------------ extraction
def extract_text(path: Path, kind: str) -> ExtractionResult:
    """Extract text from a local (temp) file. Raises ExtractionError if nothing usable."""
    path = Path(path)
    if kind == "pdf":
        result = _extract_pdf(path)
    elif kind == "docx":
        result = _extract_docx(path)
    elif kind == "json":
        result = _extract_json(path)
    elif kind in ("png", "jpeg"):
        result = _extract_image(path)
    else:
        raise ExtractionError(f"Unsupported kind '{kind}'.")
    result.pages = [p for p in result.pages if p.text.strip()]  # never index empty results
    if not result.pages:
        detail = f" ({'; '.join(result.warnings)})" if result.warnings else ""
        raise ExtractionError("No readable text could be extracted from this document" + detail + ".")
    return result


def _extract_pdf(path: Path) -> ExtractionResult:
    result = ExtractionResult(kind="pdf")
    try:
        doc = pymupdf.open(str(path))
    except Exception:
        raise ExtractionError("The PDF could not be opened.") from None
    try:
        result.page_count = doc.page_count
        for index in range(doc.page_count):  # page by page: bounded memory
            number = index + 1
            try:
                page = doc[index]
                text = _clean(page.get_text("text"))
                if len(text) >= config.MIN_PAGE_TEXT_CHARS:
                    result.pages.append(ExtractedPage(number, text, "text"))
                    continue
                ocr_text = _clean(_ocr_image(_render_page(page)))  # scanned / image-only page
                if ocr_text:
                    result.pages.append(ExtractedPage(number, ocr_text, "ocr"))
                else:
                    result.warnings.append(f"page {number}: no text found")
            except OcrUnavailable:
                raise  # environment problem, not a bad page: fail the whole document so it can be retried
            except Exception as exc:
                logger.warning("PDF page %d failed: %s", number, type(exc).__name__)
                result.warnings.append(f"page {number}: extraction failed")
    finally:
        doc.close()
    return result


def _render_page(page) -> Image.Image:
    rect = page.rect
    dpi = config.OCR_DPI
    pixels = rect.width * rect.height * (dpi / 72) ** 2
    if pixels > config.MAX_IMAGE_PIXELS:  # huge page: lower the resolution
        dpi = max(50, int(dpi * (config.MAX_IMAGE_PIXELS / pixels) ** 0.5))
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _extract_docx(path: Path) -> ExtractionResult:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    result = ExtractionResult(kind="docx", page_count=1)
    try:
        document = docx.Document(str(path))
    except Exception:
        raise ExtractionError("The DOCX could not be opened.") from None

    lines = []
    for child in document.element.body.iterchildren():  # keeps paragraphs and tables in order
        if child.tag.endswith("}p"):
            lines.append(Paragraph(child, document).text)
        elif child.tag.endswith("}tbl"):
            for row in Table(child, document).rows:
                cells, seen = [], set()
                for cell in row.cells:
                    if id(cell._tc) in seen:  # merged cells repeat
                        continue
                    seen.add(id(cell._tc))
                    if cell.text.strip():
                        cells.append(" ".join(cell.text.split()))
                if cells:
                    lines.append(" | ".join(cells))
    text = _clean("\n".join(lines))
    if text:
        result.pages.append(ExtractedPage(None, text, "docx"))
    if document.inline_shapes:
        result.warnings.append("images inside DOCX are not OCR'd")
    return result


def _extract_json(path: Path) -> ExtractionResult:
    try:
        parsed = json.loads(path.read_bytes().decode("utf-8-sig"))
        sections = _json_texts(parsed)
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise ExtractionError("The JSON file could not be read.") from None
    result = ExtractionResult(kind="json", page_count=len(sections))
    result.pages = [ExtractedPage(None, _clean(s), "json") for s in sections]  # one section per article
    return result


def _extract_image(path: Path) -> ExtractionResult:
    result = ExtractionResult(kind="image", page_count=1)
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            text = _clean(_ocr_image(img))
    except OcrUnavailable:
        raise
    except Exception as exc:
        logger.warning("Image OCR failed: %s", type(exc).__name__)
        raise ExtractionError("OCR failed for this image.") from None
    if text:
        result.pages.append(ExtractedPage(None, text, "ocr"))
    return result


def extract_from_bytes(filename: str, data: bytes) -> ExtractionResult:
    """Validate, write to a private temp dir, extract, and always clean up."""
    kind = validate_upload(filename, data)
    with temp_workspace() as ws:
        tmp = ws / f"input{Path(filename).suffix.lower()}"  # fixed name: user filename never touches disk paths
        tmp.write_bytes(data)
        return extract_text(tmp, kind)

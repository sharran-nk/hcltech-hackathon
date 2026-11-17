import io

import docx
import pymupdf
import pytest
from PIL import Image, ImageDraw, ImageFont

import config
import document_processor as dp
from document_processor import ExtractionError, ValidationError

SENTENCE = "CSV exports are capped at 10,000 rows. Exports are kept for 7 days."


# ------------------------------------------------------------------ fixtures
def text_image(text: str, size=(1400, 300)) -> Image.Image:
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    d.text((30, 80), text, fill="black", font=ImageFont.load_default(size=48))
    return img


def img_bytes(img, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt)
    return buf.getvalue()


def digital_pdf(pages=("Resetting your password takes a few minutes. Check your spam folder.",)) -> bytes:
    doc = pymupdf.open()
    for t in pages:
        doc.new_page().insert_text((72, 100), t, fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


def scanned_pdf(text=SENTENCE, extra_digital: str = None) -> bytes:
    """A PDF whose page is just a picture (no embedded text) = a scan."""
    doc = pymupdf.open()
    if extra_digital:
        doc.new_page().insert_text((72, 100), extra_digital, fontsize=12)
    page = doc.new_page(width=842, height=300)
    page.insert_image(page.rect, stream=img_bytes(text_image(text, (1684, 600))))
    data = doc.tobytes()
    doc.close()
    return data


def docx_bytes() -> bytes:
    d = docx.Document()
    d.add_paragraph("Inviting team members is done from Settings.")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Plan", "Rate limit"
    t.cell(1, 0).text, t.cell(1, 1).text = "Growth", "60 requests per minute"
    d.add_paragraph("Final paragraph after the table.")
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def tmp_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TMP_DIR", tmp_path / "tmp")


def alltext(result):
    return " ".join(p.text for p in result.pages).lower()


# ------------------------------------------------------------------ extraction
def test_digital_pdf_uses_embedded_text_not_ocr():
    r = dp.extract_from_bytes("a.pdf", digital_pdf())
    assert r.pages[0].method == "text" and r.pages[0].page_number == 1
    assert "spam folder" in alltext(r)


def test_pdf_pages_keep_page_numbers():
    r = dp.extract_from_bytes("a.pdf", digital_pdf(("First page has enough text here.", "Second page also has enough text.")))
    assert [p.page_number for p in r.pages] == [1, 2]


def test_scanned_pdf_is_ocrd():
    r = dp.extract_from_bytes("scan.pdf", scanned_pdf())
    assert r.pages[0].method == "ocr"
    assert "10,000" in alltext(r) and "rows" in alltext(r)


def test_mixed_pdf_digital_and_scanned_pages():
    r = dp.extract_from_bytes("mix.pdf", scanned_pdf(extra_digital="This digital page has plenty of text."))
    assert [(p.page_number, p.method) for p in r.pages] == [(1, "text"), (2, "ocr")]


def test_png_and_jpg_ocr():
    img = text_image(SENTENCE)
    for name, fmt in (("a.png", "PNG"), ("a.jpg", "JPEG"), ("a.jpeg", "JPEG")):
        r = dp.extract_from_bytes(name, img_bytes(img, fmt))
        assert "10,000" in alltext(r) and r.pages[0].page_number is None


def test_docx_paragraphs_and_tables_in_order():
    r = dp.extract_from_bytes("a.docx", docx_bytes())
    t = r.pages[0].text
    assert t.index("Inviting") < t.index("Growth | 60 requests per minute") < t.index("Final paragraph")
    assert "Plan | Rate limit" in t


# ------------------------------------------------------------------ rejection
@pytest.mark.parametrize("name,data,msg", [
    ("virus.exe", b"MZ\x90\x00", "Unsupported"),
    ("notes.txt", b"hello", "Unsupported"),
    ("noext", b"hello", "Unsupported"),
    ("empty.pdf", b"", "empty"),
    ("fake.pdf", b"MZ\x90\x00 this is an exe", "does not match"),
    ("fake.png", b"%PDF-1.4 hi", "does not match"),
    ("png_as_jpg.jpg", None, "does not match"),
    ("corrupt.pdf", b"%PDF-1.4\ngarbage garbage", "corrupted"),
    ("corrupt.docx", b"PK\x03\x04 not really a zip", "corrupted"),
    ("zip_not_docx.docx", None, "not a valid DOCX"),
    ("corrupt.png", b"\x89PNG\r\n\x1a\ntruncated", "corrupted"),
])
def test_invalid_uploads_rejected(name, data, msg):
    if name == "png_as_jpg.jpg":
        data = img_bytes(text_image("x"), "PNG")
    if name == "zip_not_docx.docx":
        import zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("hello.txt", "x")
        data = buf.getvalue()
    with pytest.raises(ValidationError, match=msg):
        dp.validate_upload(name, data)


def test_oversized_file_rejected(monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 100)
    with pytest.raises(ValidationError, match="larger than"):
        dp.validate_upload("a.pdf", digital_pdf())


def test_pdf_page_limit_enforced(monkeypatch):
    monkeypatch.setattr(config, "MAX_PDF_PAGES", 2)
    with pytest.raises(ValidationError, match="limit is 2"):
        dp.validate_upload("a.pdf", digital_pdf(("p1 text here ok", "p2 text here ok", "p3 text here ok")))


def test_encrypted_pdf_rejected():
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 100), "secret secret secret")
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    with pytest.raises(ValidationError, match="Password-protected"):
        dp.validate_upload("enc.pdf", data)


def test_huge_image_dimensions_rejected(monkeypatch):
    monkeypatch.setattr(config, "MAX_IMAGE_PIXELS", 1000)
    with pytest.raises(ValidationError, match="too large"):
        dp.validate_upload("big.png", img_bytes(text_image("x")))


def test_zip_bomb_docx_rejected(monkeypatch):
    monkeypatch.setattr(config, "MAX_DOCX_UNCOMPRESSED_MB", 0)
    with pytest.raises(ValidationError, match="unsafe size"):
        dp.validate_upload("a.docx", docx_bytes())


# ------------------------------------------------------------------ empty / failure handling
def test_blank_image_yields_no_text_error():
    blank = img_bytes(Image.new("RGB", (400, 200), "white"))
    with pytest.raises(ExtractionError, match="No readable text"):
        dp.extract_from_bytes("blank.png", blank)


def test_empty_docx_yields_error():
    buf = io.BytesIO()
    docx.Document().save(buf)
    with pytest.raises(ExtractionError, match="No readable text"):
        dp.extract_from_bytes("empty.docx", buf.getvalue())


def test_ocr_failure_on_one_page_is_skipped_with_warning(monkeypatch):
    calls = {"n": 0}
    real = dp._ocr_image

    def flaky(img):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("tesseract crashed")
        return real(img)

    monkeypatch.setattr(dp, "_ocr_image", flaky)
    doc = pymupdf.open()
    for _ in range(2):  # two scanned pages
        p = doc.new_page(width=842, height=300)
        p.insert_image(p.rect, stream=img_bytes(text_image(SENTENCE, (1684, 600))))
    r = dp.extract_from_bytes("two.pdf", doc.tobytes())
    assert [p.page_number for p in r.pages] == [2]
    assert any("page 1" in w for w in r.warnings)


def test_all_pages_failing_ocr_raises(monkeypatch):
    monkeypatch.setattr(dp, "_ocr_image", lambda img: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(ExtractionError, match="No readable text"):
        dp.extract_from_bytes("scan.pdf", scanned_pdf())


def test_image_ocr_failure_is_graceful(monkeypatch):
    monkeypatch.setattr(dp, "_ocr_image", lambda img: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(ExtractionError, match="OCR failed"):
        dp.extract_from_bytes("a.png", img_bytes(text_image(SENTENCE)))


def test_tesseract_missing_gives_clear_error(monkeypatch):
    monkeypatch.setattr(dp.pytesseract.pytesseract, "tesseract_cmd", r"C:\nope\tesseract.exe")
    with pytest.raises(ExtractionError, match="not installed"):
        dp.extract_from_bytes("a.png", img_bytes(text_image(SENTENCE)))


def test_digital_pdf_works_without_tesseract(monkeypatch):
    monkeypatch.setattr(dp.pytesseract.pytesseract, "tesseract_cmd", r"C:\nope\tesseract.exe")
    assert dp.extract_from_bytes("a.pdf", digital_pdf()).pages


# ------------------------------------------------------------------ temp files / misc
def test_temp_dir_cleaned_after_success_and_failure():
    dp.extract_from_bytes("a.pdf", digital_pdf())
    with pytest.raises(ExtractionError):
        dp.extract_from_bytes("blank.png", img_bytes(Image.new("RGB", (400, 200), "white")))
    assert list(config.TMP_DIR.iterdir()) == []


def test_temp_workspace_cleaned_when_caller_raises():
    with pytest.raises(RuntimeError):
        with dp.temp_workspace() as ws:
            (ws / "x.txt").write_text("data")
            raise RuntimeError("crash mid-processing")
    assert list(config.TMP_DIR.iterdir()) == []


def test_user_filename_never_used_as_disk_path():
    # path-traversal style name is only inspected for its extension
    r = dp.extract_from_bytes("../../evil/name.pdf", digital_pdf())
    assert r.pages and list(config.TMP_DIR.iterdir()) == []


def test_content_hash_detects_duplicates():
    assert dp.content_hash(b"abc") == dp.content_hash(b"abc") != dp.content_hash(b"abd")

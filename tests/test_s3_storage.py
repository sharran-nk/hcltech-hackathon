import inspect

import boto3
import pytest
from moto import mock_aws

import s3_storage
from s3_storage import S3Storage, S3StorageError, new_document_id, sanitize_filename

BUCKET = "test-solstice-bucket"


@pytest.fixture
def aws(monkeypatch):
    # Fake creds ONLY for moto; real code never sets credentials.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        yield


@pytest.fixture
def store(aws):
    return S3Storage(bucket=BUCKET, prefix="documents/", region="us-east-1")


# ---- filename / key safety
@pytest.mark.parametrize("raw,expected", [
    ("report.pdf", "report.pdf"),
    ("../../etc/passwd", "passwd"),
    ("C:\\Users\\x\\evil.docx", "evil.docx"),
    ("my file (final) #2.pdf", "my_file_final_2.pdf"),
    ("", "file"),
    ("...", "file"),
    (".hidden", "hidden"),
    ("<script>.png", "script_.png"),
])
def test_sanitize_filename(raw, expected):
    assert sanitize_filename(raw) == expected


def test_sanitize_long_name_keeps_extension():
    out = sanitize_filename("a" * 300 + ".pdf")
    assert len(out) <= 100 and out.endswith(".pdf")


def test_same_filename_gets_unique_keys(store):
    k1 = store.make_key(new_document_id(), "a.pdf")
    k2 = store.make_key(new_document_id(), "a.pdf")
    assert k1 != k2 and k1.startswith("documents/") and k1.endswith("/a.pdf")


def test_key_cannot_escape_prefix(store):
    key = store.make_key(new_document_id(), "../../secret.pdf")
    assert key.startswith("documents/") and ".." not in key


# ---- operations
def test_upload_download_roundtrip(store, tmp_path):
    key = store.make_key(new_document_id(), "hello.pdf")
    store.upload_bytes(b"%PDF-1.4 hello", key)
    out = store.download_to_path(key, tmp_path / "t" / "hello.pdf")
    assert out.read_bytes() == b"%PDF-1.4 hello"


def test_upload_uses_server_side_encryption(store):
    key = store.make_key(new_document_id(), "x.pdf")
    store.upload_bytes(b"data", key)
    head = boto3.client("s3", region_name="us-east-1").head_object(Bucket=BUCKET, Key=key)
    assert head["ServerSideEncryption"] == "AES256"


def test_two_uploads_same_name_do_not_overwrite(store, tmp_path):
    k1 = store.make_key(new_document_id(), "same.pdf")
    k2 = store.make_key(new_document_id(), "same.pdf")
    store.upload_bytes(b"one", k1)
    store.upload_bytes(b"two", k2)
    assert store.download_to_path(k1, tmp_path / "1").read_bytes() == b"one"
    assert store.download_to_path(k2, tmp_path / "2").read_bytes() == b"two"


def test_delete_removes_object(store, tmp_path):
    key = store.make_key(new_document_id(), "gone.pdf")
    store.upload_bytes(b"bye", key)
    store.delete(key)
    dest = tmp_path / "gone.pdf"
    with pytest.raises(S3StorageError):
        store.download_to_path(key, dest)
    assert not dest.exists()  # no partial file left behind


def test_delete_missing_key_is_not_an_error(store):
    store.delete("documents/does-not-exist/x.pdf")


def test_download_missing_key_raises_clean_error(store, tmp_path):
    with pytest.raises(S3StorageError) as e:
        store.download_to_path("documents/nope/x.pdf", tmp_path / "x.pdf")
    assert "not found" in str(e.value)


# ---- failures
def test_missing_bucket_raises_clean_error(aws):
    bad = S3Storage(bucket="bucket-that-does-not-exist", region="us-east-1")
    with pytest.raises(S3StorageError) as e:
        bad.upload_bytes(b"x", "documents/a/x.pdf")
    assert "bucket does not exist" in str(e.value)


def test_unconfigured_bucket_rejected(monkeypatch):
    monkeypatch.setattr(s3_storage.config, "S3_BUCKET_NAME", "")
    with pytest.raises(S3StorageError):
        S3Storage()


def test_no_credentials_hardcoded():
    src = inspect.getsource(s3_storage)
    for needle in ("aws_access_key_id", "aws_secret_access_key", "AKIA"):
        assert needle not in src

import time

import boto3
import pytest
from moto import mock_aws

import auth
import chunking
import config
from document_manager import DocumentManager, DocumentRegistry
from embeddings import get_embedder
from index_store import IndexStore
from s3_storage import S3Storage

BUCKET = "test-docs-bucket"
TEST_PASSWORD = "correct horse battery"
TEST_HASH = auth.hash_password(TEST_PASSWORD, rounds=4)  # low cost only to keep tests fast


@pytest.fixture(scope="session")
def embedder():
    return get_embedder()


@pytest.fixture(scope="session")
def seed(embedder):
    """{doc_id: (chunks, vectors)} for the six seed articles (embedded once per test session)."""
    out = {}
    for d in chunking.load_seed_docs():
        chunks = chunking.chunk_seed_doc(d)
        out[d["doc_id"]] = (chunks, embedder.embed_texts([c.text for c in chunks]))
    return out


@pytest.fixture
def fake_s3(monkeypatch):
    """In-memory S3 (moto). Yields a boto3 client for inspecting what the app stored."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def make_auth(clock=None):
    return auth.AuthManager(password_hash=TEST_HASH, session_timeout_minutes=30, max_failed_logins=3,
                            login_delay_seconds=0, lockout_minutes=5, clock=clock or time.monotonic, sleep=lambda s: None)


@pytest.fixture
def make_manager(tmp_path, fake_s3, embedder, monkeypatch):
    """Factory for a started DocumentManager (real extraction/embedding/index, fake S3)."""
    monkeypatch.setattr(config, "TMP_DIR", tmp_path / "tmp")
    created = []

    def factory(clock=None, registry_path=None, index_dir=None, auth_manager=None):
        mgr = DocumentManager(
            S3Storage(bucket=BUCKET, prefix="documents/", region="us-east-1"),
            IndexStore(index_dir or tmp_path / "index"), embedder,
            auth_manager or make_auth(clock), DocumentRegistry(registry_path or tmp_path / "documents.json"))
        mgr.start()
        created.append(mgr)
        return mgr

    yield factory
    for m in created:
        m.stop()

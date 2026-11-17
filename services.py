"""Builds the long-lived objects the Streamlit app shares across all browser sessions."""
import logging
import os
import threading
from dataclasses import dataclass

import boto3

import config
from auth import AuthManager
from document_manager import DocumentManager
from embeddings import get_embedder
from index_store import IndexStore
from rag_pipeline import BedrockClient, OllamaClient, RagPipeline
from retrieval import Retriever
from s3_storage import S3Storage

logger = logging.getLogger(__name__)


@dataclass
class Services:
    auth: AuthManager
    manager: DocumentManager
    pipeline: RagPipeline
    store: IndexStore
    dev_fake_s3: bool = False
    _mock: object = None  # keeps the fake S3 alive


_fake_s3_mock = None


def stop_fake_s3() -> None:
    """Stop the dev fake S3 (used by tests so its global state cannot leak into other tests)."""
    global _fake_s3_mock
    if _fake_s3_mock is not None:
        _fake_s3_mock.stop()
        _fake_s3_mock = None


def _start_fake_s3() -> tuple:
    """DEV ONLY: in-memory S3 via moto, so the real S3Storage code runs without AWS."""
    global _fake_s3_mock
    from moto import mock_aws
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        os.environ.setdefault(k, "dev-fake")
    stop_fake_s3()
    mock = _fake_s3_mock = mock_aws()
    mock.start()
    bucket = config.S3_BUCKET_NAME or "dev-fake-bucket"
    client = boto3.client("s3", region_name=config.AWS_REGION)
    kwargs = {} if config.AWS_REGION == "us-east-1" else {"CreateBucketConfiguration": {"LocationConstraint": config.AWS_REGION}}
    client.create_bucket(Bucket=bucket, **kwargs)
    return mock, bucket


def build_services() -> Services:
    config.ensure_runtime_dirs()
    mock, bucket = (None, config.S3_BUCKET_NAME)
    if config.DEV_FAKE_S3:
        logger.warning("DEV_FAKE_S3 is on: documents are stored in an in-memory fake S3 and vanish on restart.")
        mock, bucket = _start_fake_s3()

    store = IndexStore()
    embedder = get_embedder()
    llm = BedrockClient() if config.LLM_PROVIDER == "bedrock" else OllamaClient()
    manager = DocumentManager(S3Storage(bucket=bucket), store, embedder, AuthManager())
    manager.start()
    embedder.embed_query("warm up")                                    # load the embedding model now, not on the first question
    threading.Thread(target=llm.warm_up, daemon=True).start()         # load Qwen in the background
    return Services(manager.auth, manager, RagPipeline(Retriever(store, embedder), llm), store, config.DEV_FAKE_S3, mock)

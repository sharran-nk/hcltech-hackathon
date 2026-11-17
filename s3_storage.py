"""S3 storage for ORIGINAL uploaded documents.

- No credentials in code: boto3's default credential chain is used. On EC2 that is the
  instance's IAM role; locally it is your `aws configure` profile / env vars.
- Keys are unique (uuid per document) so an upload can never overwrite another file.
- The IAM policy in docs/AWS_SETUP.md grants only Put/Get/DeleteObject (no ListBucket),
  so this module deliberately avoids HEAD/list calls: without ListBucket, S3 answers
  403 (not 404) for missing keys.
"""
import logging
import re
import uuid
from pathlib import Path
from typing import Optional

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

import config

logger = logging.getLogger(__name__)

_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
MAX_FILENAME_LEN = 100


class S3StorageError(Exception):
    """Any S3 failure. The message is safe to show to an admin (no secrets)."""


def sanitize_filename(filename: str) -> str:
    """Strip directories and unsafe characters. Never returns an empty string or a dotfile."""
    name = (filename or "").replace("\\", "/").split("/")[-1]  # drop any path (../ tricks)
    name = _SAFE_CHARS.sub("_", name).strip("._")
    if not name:
        return "file"
    if len(name) > MAX_FILENAME_LEN:
        stem, dot, ext = name.rpartition(".")
        if dot and len(ext) <= 8:
            name = stem[: MAX_FILENAME_LEN - len(ext) - 1] + "." + ext
        else:
            name = name[:MAX_FILENAME_LEN]
    return name


def new_document_id() -> str:
    return uuid.uuid4().hex


class S3Storage:
    def __init__(self, bucket: Optional[str] = None, prefix: Optional[str] = None,
                 region: Optional[str] = None, client=None):
        self.bucket = config.S3_BUCKET_NAME if bucket is None else bucket
        self.prefix = config.S3_PREFIX if prefix is None else prefix
        if not self.bucket:
            raise S3StorageError("S3_BUCKET_NAME is not configured.")
        self._client = client or boto3.client(
            "s3",
            region_name=region or config.AWS_REGION,
            config=Config(retries={"max_attempts": 3, "mode": "standard"},
                          connect_timeout=5, read_timeout=30),
        )

    # ---- keys
    def make_key(self, doc_id: str, filename: str) -> str:
        """documents/<doc_id>/<safe_filename>; doc_id is a fresh uuid, so keys never collide."""
        return f"{self.prefix}{doc_id}/{sanitize_filename(filename)}"

    # ---- operations
    def upload_bytes(self, data: bytes, key: str) -> None:
        try:
            self._client.put_object(Bucket=self.bucket, Key=key, Body=data,
                                    ServerSideEncryption="AES256")
        except (ClientError, BotoCoreError) as exc:
            raise self._wrap("upload", exc) from exc
        logger.info("Uploaded %s (%d bytes)", key, len(data))

    def download_to_path(self, key: str, dest: Path) -> Path:
        """Download to a local TEMP path chosen by the caller (caller must clean it up)."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._client.download_file(self.bucket, key, str(dest))
        except (ClientError, BotoCoreError) as exc:
            dest.unlink(missing_ok=True)  # never leave a partial file behind
            raise self._wrap("download", exc) from exc
        return dest

    def delete(self, key: str) -> None:
        """Delete is idempotent in S3: deleting a missing key is not an error."""
        try:
            self._client.delete_object(Bucket=self.bucket, Key=key)
        except (ClientError, BotoCoreError) as exc:
            raise self._wrap("delete", exc) from exc
        logger.info("Deleted %s", key)

    # ---- errors
    @staticmethod
    def _wrap(action: str, exc: Exception) -> S3StorageError:
        code = exc.response.get("Error", {}).get("Code", "") if isinstance(exc, ClientError) else type(exc).__name__
        logger.error("S3 %s failed: %s", action, code)
        if code in ("404", "NoSuchKey", "403", "AccessDenied"):
            return S3StorageError(f"S3 {action} failed: object not found or access denied ({code}).")
        if code == "NoSuchBucket":
            return S3StorageError(f"S3 {action} failed: bucket does not exist.")
        return S3StorageError(f"S3 {action} failed ({code}). Check network and IAM permissions.")

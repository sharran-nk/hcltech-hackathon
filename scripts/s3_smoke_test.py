"""Real-AWS smoke test: upload -> download -> delete one tiny object.

Run on the EC2 instance (IAM role) or locally with `aws configure`:
    python scripts/s3_smoke_test.py
Requires S3_BUCKET_NAME (and AWS_REGION) in .env. Cleans up after itself.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from s3_storage import S3Storage, S3StorageError, new_document_id  # noqa: E402


def main() -> int:
    try:
        s3 = S3Storage()
        key = s3.make_key(new_document_id(), "smoke-test.txt")
        s3.upload_bytes(b"solstice smoke test", key)
        print("upload OK  ->", key)
        with tempfile.TemporaryDirectory() as tmp:
            data = s3.download_to_path(key, Path(tmp) / "smoke.txt").read_bytes()
        print("download OK, content matches:", data == b"solstice smoke test")
        s3.delete(key)
        print("delete OK")
        return 0
    except S3StorageError as exc:
        print("FAILED:", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

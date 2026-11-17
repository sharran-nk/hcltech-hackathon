"""Generate a bcrypt hash for the admin password.

Run from the project root:
    python scripts/generate_admin_hash.py

The password is typed with hidden input and is never written to disk or printed.
Copy the printed hash line into your .env (or Streamlit secrets / systemd env file).
"""
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auth import MAX_PASSWORD_BYTES, MIN_PASSWORD_LENGTH, hash_password  # noqa: E402


def main() -> int:
    pw = getpass.getpass("Choose admin password: ")
    if len(pw) < MIN_PASSWORD_LENGTH:
        print(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
        return 1
    if len(pw.encode("utf-8")) > MAX_PASSWORD_BYTES:
        print(f"Password must be at most {MAX_PASSWORD_BYTES} bytes.")
        return 1
    if pw != getpass.getpass("Repeat password: "):
        print("Passwords do not match.")
        return 1
    print("\nAdd this line to your .env (keep the single quotes, the hash contains '$'):\n")
    print(f"ADMIN_PASSWORD_HASH='{hash_password(pw)}'")
    print("\nDo NOT commit .env. The password itself is not stored anywhere.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

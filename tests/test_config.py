import json

import config


def test_refusal_message_is_exact():
    assert config.NO_ANSWER_MESSAGE == "I don't have an answer for that."


def test_upload_limit_default_is_20mb():
    assert config.MAX_UPLOAD_BYTES == 20 * 1024 * 1024


def test_seed_data_present():
    docs = json.loads(config.SEED_DOCS_PATH.read_text(encoding="utf-8"))["docs"]
    qs = json.loads(config.EXAMPLE_QUESTIONS_PATH.read_text(encoding="utf-8"))["example_questions"]
    assert len(docs) == 6
    assert len(qs) == 5


def test_no_secrets_in_defaults():
    assert config.ADMIN_PASSWORD_HASH == "" or config.ADMIN_PASSWORD_HASH.startswith("$2")


def test_runtime_dirs_created(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "rt")
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "rt" / "index")
    monkeypatch.setattr(config, "TMP_DIR", tmp_path / "rt" / "tmp")
    config.ensure_runtime_dirs()
    assert (tmp_path / "rt" / "index").is_dir() and (tmp_path / "rt" / "tmp").is_dir()

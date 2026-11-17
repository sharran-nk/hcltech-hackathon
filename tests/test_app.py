"""UI tests using Streamlit's AppTest: the real app.py script runs headlessly and we click/type into it."""
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import config
import services
from auth import AuthManager
from document_manager import DocumentManager, DocumentRegistry
from index_store import IndexStore
from rag_pipeline import REFUSAL, RagPipeline
from retrieval import Retriever
from s3_storage import S3Storage
from tests.conftest import BUCKET, TEST_PASSWORD, make_auth

APP = str(config.BASE_DIR / "app.py")


class FakeLLM:
    def __init__(self, reply="Password reset links expire after 15 minutes. [1]", fail=False):
        self.reply, self.fail = reply, fail

    def chat(self, system, user):
        return self.reply


class Clock:
    t = 1000.0
    def __call__(self): return self.t


@pytest.fixture
def app(tmp_path, fake_s3, embedder, monkeypatch):
    """Patch services.build_services so the real app.py runs against fake S3 + a fake LLM."""
    monkeypatch.setattr(config, "TMP_DIR", tmp_path / "tmp")
    st.cache_resource.clear()
    holder = {"clock": Clock(), "llm": FakeLLM()}

    def build():
        store = IndexStore(tmp_path / "index")
        mgr = DocumentManager(S3Storage(bucket=BUCKET, prefix="documents/", region="us-east-1"), store, embedder,
                              make_auth(holder["clock"]), DocumentRegistry(tmp_path / "documents.json"))
        mgr.start()
        holder["svc"] = services.Services(mgr.auth, mgr, RagPipeline(Retriever(store, embedder), holder["llm"]), store)
        return holder["svc"]

    monkeypatch.setattr(services, "build_services", build)
    at = AppTest.from_file(APP, default_timeout=60)
    at.holder = holder
    yield at
    if "svc" in holder:
        holder["svc"].manager.stop()
    st.cache_resource.clear()


def all_text(at):
    return "\n".join([m.value for m in at.markdown] + [t.value for t in at.text] + [e.value for e in at.error]
                     + [w.value for w in at.warning] + [i.value for i in at.info] + [c.value for c in at.caption]
                     + [s.value for s in at.success])


def ask(at, q):
    at.chat_input[0].set_value(q).run()
    return at


def go_admin(at):
    at.sidebar.radio[0].set_value("Admin").run()
    return at


def is_password_field(el) -> bool:
    return el.proto.type == 1  # TextInput.Type.PASSWORD (AppTest's .type is just the element kind)


def login(at, password=TEST_PASSWORD):
    at.text_input[0].set_value(password)
    at.button[0].click().run()
    return at


# ------------------------------------------------------------------ user mode
def test_user_asks_supported_question_and_sees_answer_and_sources(app):
    at = ask(app.run(), "How long do I have to use a password reset link before it expires?")
    assert not at.exception
    assert "Password reset links expire after 15 minutes." in all_text(at)
    assert [e.label for e in at.expander] == ["Sources (1)"]
    assert "Resetting Your Password" in all_text(at) and "D-01-c2" in all_text(at)


def test_unsupported_question_shows_exact_refusal_and_no_sources(app):
    at = ask(app.run(), "Can I connect Solstice to Salesforce?")
    assert REFUSAL == "I don't have an answer for that."
    assert any(m.value == REFUSAL for m in at.markdown)
    assert len(at.expander) == 0


def test_example_question_buttons_ask_the_question_and_then_disappear(app):
    at = app.run()
    examples = [b for b in at.button if b.key and b.key.startswith("example-")]
    assert len(examples) == 4                                   # shown on the empty state
    examples[0].click().run()
    assert not at.exception
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    assert "How do I reset my password?" in all_text(at)
    assert not [b for b in at.button if b.key and b.key.startswith("example-")]   # gone once a chat has started


def test_chat_history_is_kept_during_the_session(app):
    at = app.run()
    ask(at, "How do I reset my password?")
    ask(at, "Can I connect Solstice to Salesforce?")
    assert [m.name for m in at.chat_message] == ["user", "assistant", "user", "assistant"]
    assert len(at.session_state["messages"]) == 4


def test_pipeline_crash_shows_friendly_error_not_a_traceback(app, monkeypatch):
    at = app.run()
    monkeypatch.setattr(at.holder["svc"].pipeline, "answer", lambda q: (_ for _ in ()).throw(RuntimeError("boom internal")))
    ask(at, "How do I reset my password?")
    assert not at.exception
    assert "something went wrong" in all_text(at).lower() and "boom internal" not in all_text(at)


def test_question_length_is_capped_in_the_input(app):
    assert app.run().chat_input[0].max_chars == 500


def test_empty_index_tells_the_user(app, tmp_path):
    at = app.run()
    at.holder["svc"].store._snapshot = IndexStore(tmp_path / "nothing")._snapshot
    at.run()
    assert "No documents are indexed yet" in all_text(at)


def test_user_mode_exposes_no_admin_controls(app):
    at = app.run()
    assert len(at.get("file_uploader")) == 0 and not at.text_input


# ------------------------------------------------------------------ admin mode
def test_admin_mode_shows_login_only_when_logged_out(app):
    at = go_admin(app.run())
    assert is_password_field(at.text_input[0])  # typed characters are masked
    assert len(at.get("file_uploader")) == 0 and len(at.dataframe) == 0


def test_wrong_password_rejected_and_nothing_revealed(app):
    at = login(go_admin(app.run()), "not the password!")
    assert "Incorrect password." in all_text(at)
    assert len(at.get("file_uploader")) == 0 and "admin_token" not in at.session_state


def test_correct_password_shows_upload_and_document_list(app):
    at = login(go_admin(app.run()))
    assert not at.exception and len(at.get("file_uploader")) == 1
    df = at.dataframe[0].value
    assert len(df) == 6 and set(df["Status"]) == {"✅ Indexed"}
    assert TEST_PASSWORD not in str(at.session_state["admin_token"])


def test_lockout_message_after_repeated_failures(app):
    at = go_admin(app.run())
    for _ in range(3):
        login(at, "wrong password!!")
    assert "Too many failed attempts" in all_text(login(at, TEST_PASSWORD))


def test_logout_hides_admin_controls(app):
    at = login(go_admin(app.run()))
    next(b for b in at.button if b.label == "Log out").click().run()
    assert len(at.get("file_uploader")) == 0 and is_password_field(at.text_input[0])


def test_session_expiry_returns_to_login(app):
    at = login(go_admin(app.run()))
    at.holder["clock"].t += 31 * 60
    at.run()
    assert "session expired" in all_text(at).lower()
    assert len(at.get("file_uploader")) == 0 and "admin_token" not in at.session_state


def test_forged_token_in_session_state_is_not_accepted(app):
    at = go_admin(app.run())
    at.session_state["admin_token"] = "forged"
    at.run()
    assert len(at.get("file_uploader")) == 0 and is_password_field(at.text_input[0])


def delete_button(at):
    return next(b for b in at.button if b.label == "Delete document")


def test_delete_button_is_disabled_until_confirmed(app):
    at = login(go_admin(app.run()))
    assert delete_button(at).disabled
    at.checkbox[0].check().run()
    assert not delete_button(at).disabled


def test_admin_deletes_a_document_and_it_stops_being_answerable(app):
    at = login(go_admin(app.run()))
    assert len(at.dataframe[0].value) == 6
    at.selectbox[0].select("D-03").run()
    at.checkbox[0].check().run()
    delete_button(at).click().run()
    assert not at.exception
    assert "Deleted Exporting Reports to CSV." in all_text(at)             # confirmation message survives the rerun
    assert len(at.dataframe[0].value) == 5 and "Exporting Reports to CSV" not in set(at.dataframe[0].value["Document"])
    assert at.holder["svc"].manager.registry.get("D-03") is None           # really gone server-side
    assert not at.checkbox[0].value                                        # confirmation must be given again

    at.sidebar.radio[0].set_value("Ask questions").run()
    ask(at, "What's the maximum number of rows I can export to CSV?")
    assert any(m.value == REFUSAL for m in at.markdown)                    # the exact refusal, not a stale answer


def test_delete_controls_are_not_visible_to_users_or_logged_out_admins(app):
    at = app.run()
    assert not at.selectbox and not at.checkbox
    at = go_admin(at)
    assert not at.selectbox and not at.checkbox and all(b.label != "Delete document" for b in at.button)


def test_delete_after_session_expiry_is_rejected(app):
    at = login(go_admin(app.run()))
    at.selectbox[0].select("D-03").run()
    at.checkbox[0].check().run()
    at.holder["clock"].t += 31 * 60                                         # session times out before the click
    delete_button(at).click().run()
    assert "session expired" in all_text(at).lower()
    assert at.holder["svc"].manager.registry.get("D-03") is not None       # nothing was deleted


# ------------------------------------------------------------------ real services (real embeddings + real Qwen)
def _ollama_ready():
    import requests
    try:
        return any(m["name"] == config.OLLAMA_MODEL for m in requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=2).json()["models"])
    except Exception:
        return False


@pytest.mark.llm
@pytest.mark.skipif(not _ollama_ready(), reason="Ollama / qwen2.5:7b not available")
def test_full_app_with_real_services_and_dev_fake_s3(tmp_path, monkeypatch):
    for name, value in (("DATA_DIR", tmp_path / "rt"), ("INDEX_DIR", tmp_path / "rt" / "index"), ("TMP_DIR", tmp_path / "rt" / "tmp"),
                        ("METADATA_DB_PATH", tmp_path / "rt" / "documents.json"), ("DEV_FAKE_S3", True)):
        monkeypatch.setattr(config, name, value)
    monkeypatch.setattr(config, "ADMIN_PASSWORD_HASH", __import__("tests.conftest", fromlist=["x"]).TEST_HASH)
    st.cache_resource.clear()
    try:
        at = AppTest.from_file(APP, default_timeout=180).run()
        assert not at.exception and "DEV MODE" not in all_text(at)  # the notice was removed from the UI
        ask(at, "How long do I have to use a password reset link before it expires?")
        assert "15 minutes" in all_text(at) and [e.label for e in at.expander] == ["Sources (1)"]
        ask(at, "Can I connect Solstice to Salesforce?")
        assert any(m.value == REFUSAL for m in at.markdown)
    finally:
        st.cache_resource.clear()
        services.stop_fake_s3()  # otherwise moto's global state leaks into later tests

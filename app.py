"""Solstice Learning - Document Q&A Assistant (single Streamlit site: User mode + Admin mode).

Security note: hiding admin widgets is only cosmetic. Every admin action is re-checked on the
server inside DocumentManager (auth.require_admin), using a server-side session token.
"""
import logging

import pandas as pd
import streamlit as st

import config
import services
from auth import AuthError, NotAuthenticatedError
from document_manager import DeleteIncompleteError, DocumentManagerError
from document_processor import DocumentProcessingError
from index_store import IndexUpdateError
from s3_storage import S3StorageError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

st.set_page_config(page_title="Solstice Docs Assistant", page_icon="📘", layout="centered")

STATUS_LABEL = {"queued": "⏳ Queued", "processing": "⚙️ Processing", "indexed": "✅ Indexed", "failed": "❌ Failed",
                "delete_failed": "⚠️ Delete incomplete"}


@st.cache_resource(show_spinner="Starting up: loading indexes and models…")
def get_services() -> services.Services:
    return services.build_services()


def human_size(n: int) -> str:
    return f"{n / 1024:.0f} KB" if n < 1024 * 1024 else f"{n / 1024 / 1024:.1f} MB"


# ------------------------------------------------------------------ User mode
def render_citations(citations) -> None:
    with st.expander(f"Sources ({len(citations)})"):
        for c in citations:
            page = f" · page {c.page_number}" if c.page_number else ""
            st.markdown(f"**[{c.number}] {c.title}**{page} · `{c.chunk_id}`")
            st.text(c.snippet)  # st.text: document content is shown literally, never interpreted as markdown/HTML


def render_assistant_message(msg: dict) -> None:
    st.markdown(msg["content"])
    if msg.get("citations"):
        render_citations(msg["citations"])


def user_mode(svc: services.Services) -> None:
    st.subheader("Ask a question")
    st.caption("Answers come only from the uploaded documentation, with sources. If the documents do not cover it, "
               "the assistant will say so.")
    if svc.store.snapshot.chunk_count == 0:
        st.info("No documents are indexed yet. An admin needs to upload documentation first.")

    history = st.session_state.setdefault("messages", [])
    for msg in history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_assistant_message(msg)
            else:
                st.markdown(msg["content"])

    question = st.chat_input("Ask about the documentation…", max_chars=500)
    if not question:
        return
    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the documentation…"):
                result = svc.pipeline.answer(question)
            msg = {"role": "assistant", "content": result.answer, "citations": result.citations, "notice": result.notice}
        except Exception:
            logger.exception("Answering failed")
            msg = {"role": "assistant", "content": "Sorry, something went wrong while answering. Please try again.",
                   "citations": [], "notice": ""}
        render_assistant_message(msg)
    history.append(msg)


# ------------------------------------------------------------------ Admin mode
def login_form(svc: services.Services) -> None:
    st.subheader("Admin login")
    with st.form("login"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
    if submitted:
        try:
            with st.spinner("Checking…"):
                st.session_state["admin_token"] = svc.auth.login(password)
            st.rerun()
        except AuthError as exc:
            st.error(str(exc))


@st.fragment(run_every="3s")
def documents_table(svc: services.Services) -> None:
    """Polls every 3 s so processing status updates live. Polling must NOT extend the admin session."""
    token = st.session_state.get("admin_token")
    try:
        docs = svc.manager.list_documents(token, refresh=False)
    except NotAuthenticatedError:
        st.session_state.pop("admin_token", None)
        st.session_state["session_expired"] = True
        st.rerun()
        return
    if not docs:
        st.caption("No documents yet.")
        return
    st.dataframe(pd.DataFrame([{
        "Document": d["filename"], "Status": STATUS_LABEL.get(d["status"], d["status"]),
        "Chunks": d["chunk_count"], "Pages": d["page_count"], "Size": human_size(d["size"]),
        "Added": d["created_at"].replace("T", " ").replace("+00:00", " UTC"),
        "Details": d["error"] or ("; ".join(d["warnings"]) if d["warnings"] else ""),
    } for d in docs]), hide_index=True, width="stretch")


def upload_section(svc: services.Services, token: str) -> None:
    st.markdown("#### Upload documents")
    st.caption(f"PDF (digital or scanned), DOCX, PNG, JPG, JSON. Max {config.MAX_UPLOAD_MB} MB each. "
               "OCR reads printed text only; it does not interpret diagrams or charts.")
    key = st.session_state.setdefault("uploader_key", 0)
    files = st.file_uploader("Choose files", type=["pdf", "docx", "png", "jpg", "jpeg", "json"],
                             accept_multiple_files=True, key=f"uploader-{key}")
    if st.button("Upload and process", disabled=not files, type="primary"):
        for f in files:
            try:
                with st.spinner(f"Uploading {f.name}…"):
                    svc.manager.submit_upload(token, f.name, f.getvalue())
                st.success(f"{f.name}: uploaded. Processing in the background.")
            except NotAuthenticatedError:
                st.session_state.pop("admin_token", None)
                st.session_state["session_expired"] = True
                st.rerun()
            except (DocumentProcessingError, DocumentManagerError, S3StorageError) as exc:
                st.error(f"{f.name}: {exc}")
            except Exception:
                logger.exception("Upload failed")
                st.error(f"{f.name}: unexpected error. Please try again.")
        st.session_state["uploader_key"] = key + 1  # clears the uploader so a rerun cannot re-submit the files


def delete_section(svc: services.Services, token: str) -> None:
    st.markdown("#### Delete a document")
    docs = svc.manager.list_documents(token)
    if not docs:
        st.caption("Nothing to delete.")
        return
    names = {d["doc_id"]: f"{d['filename']} ({d['status']})" for d in docs}
    doc_id = st.selectbox("Document", list(names), format_func=names.get, key="delete_doc")
    confirm = st.checkbox("I understand this removes the document and its stored file permanently.", key="delete_confirm")
    if st.button("Delete document", disabled=not confirm):
        try:
            with st.spinner("Deleting…"):
                svc.manager.delete_document(token, doc_id)
            st.session_state["flash"] = ("success", f"Deleted {names[doc_id].rsplit(' (', 1)[0]}.")
            st.session_state.pop("delete_doc", None)       # the selected document no longer exists
            st.session_state.pop("delete_confirm", None)   # the confirmation must be given again next time
            st.rerun()
        except NotAuthenticatedError:
            st.session_state.pop("admin_token", None)
            st.session_state["session_expired"] = True
            st.rerun()
        except DeleteIncompleteError as exc:
            st.warning(str(exc))
        except (DocumentManagerError, IndexUpdateError, S3StorageError) as exc:
            st.error(str(exc))
        except Exception:
            logger.exception("Delete failed")
            st.error("Unexpected error while deleting. Please try again.")


def admin_mode(svc: services.Services) -> None:
    token = st.session_state.get("admin_token")
    if token and not svc.auth.is_authenticated(token):  # server-side expiry check on every interaction
        st.session_state.pop("admin_token", None)
        st.session_state["session_expired"] = True
        token = None
    if not token:
        if st.session_state.pop("session_expired", False):
            st.warning("Your admin session expired. Please log in again.")
        login_form(svc)
        return

    top = st.columns([4, 1])
    top[0].subheader("Admin")
    if top[1].button("Log out"):
        svc.auth.logout(token)
        st.session_state.pop("admin_token", None)
        st.rerun()
    flash = st.session_state.pop("flash", None)  # a message that must survive st.rerun()
    if flash:
        getattr(st, flash[0])(flash[1])
    upload_section(svc, token)
    st.markdown("#### Documents")
    documents_table(svc)
    delete_section(svc, token)


# ------------------------------------------------------------------ main
def main() -> None:
    svc = get_services()
    st.title("📘 Solstice Learning Docs Assistant")
    mode = st.sidebar.radio("Mode", ["Ask questions", "Admin"], key="mode")
    st.sidebar.caption("Admin access is password protected.")
    if mode == "Ask questions":
        user_mode(svc)
    else:
        admin_mode(svc)


main()

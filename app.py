"""Solstice Learning - Document Q&A Assistant (single Streamlit site: User mode + Admin mode).

Security note: hiding admin widgets is only cosmetic. Every admin action is re-checked on the
server inside DocumentManager (auth.require_admin), using a server-side session token.
"""
import html
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

EXAMPLE_QUESTIONS = [
    ("🔑", "How do I reset my password?"),
    ("📊", "What's the maximum number of rows I can export to CSV?"),
    ("⚡", "What are the API rate limits for the Growth plan?"),
    ("🧾", "When do I receive my invoice before renewal?"),
]

# Static stylesheet written by us (no user content is ever placed in it).
CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.6rem; max-width: 860px;}

/* ---------- hero ---------- */
.hero {background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 55%, #06B6D4 100%);
       border-radius: 22px; padding: 30px 32px 26px; color: #fff; margin-bottom: 1.1rem;
       box-shadow: 0 14px 34px rgba(79,70,229,.28); position: relative; overflow: hidden;}
.hero:after {content: ""; position: absolute; right: -60px; top: -60px; width: 220px; height: 220px;
             border-radius: 50%; background: rgba(255,255,255,.10);}
.hero h1 {margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: -.02em; color: #fff;}
.hero p {margin: .45rem 0 1.05rem; font-size: 1.02rem; opacity: .93; max-width: 560px;}
.chip {display: inline-block; background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.28);
       color: #fff; border-radius: 999px; padding: 5px 13px; font-size: .82rem; margin: 0 8px 6px 0;
       backdrop-filter: blur(4px);}

/* ---------- chat ---------- */
[data-testid="stChatMessage"] {background: #fff; border: 1px solid #E7E9F3; border-radius: 18px;
       padding: 14px 18px; box-shadow: 0 2px 10px rgba(27,32,51,.05); margin-bottom: .7rem;}
.badge {display: inline-block; border-radius: 999px; padding: 3px 11px; font-size: .76rem; font-weight: 600;}
.badge-ok {background: #E7F8EF; color: #137A47; border: 1px solid #BDEBD2;}
.badge-no {background: #FFF4E0; color: #9A5B00; border: 1px solid #F6DBA6;}
.section-label {font-size: .8rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
                color: #6B7194; margin: .6rem 0 .4rem;}

/* ---------- source cards ---------- */
.src-title {font-weight: 700; color: #1B2033;}
.pill {display: inline-block; background: #EEF2FF; color: #4338CA; border-radius: 999px;
       padding: 2px 10px; font-size: .75rem; font-weight: 600; margin-left: 6px;}

/* ---------- controls ---------- */
.stButton > button {border-radius: 12px; border: 1px solid #DADDF0; font-weight: 600; transition: all .15s;}
.stButton > button:hover {border-color: #4F46E5; color: #4F46E5; transform: translateY(-1px);
                          box-shadow: 0 6px 16px rgba(79,70,229,.16);}
.stButton > button[kind="primary"] {background: linear-gradient(135deg, #4F46E5, #7C3AED); color: #fff; border: 0;}
[data-testid="stMetric"] {background: #fff; border: 1px solid #E7E9F3; border-radius: 16px; padding: 14px 16px;
                          box-shadow: 0 2px 10px rgba(27,32,51,.05);}
[data-testid="stMetricLabel"] {color: #6B7194;}
[data-testid="stTabs"] button[role="tab"] {font-weight: 600;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 16px;}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {background: linear-gradient(180deg, #ECEBFF 0%, #F6F7FB 60%);}
.brand {display: flex; align-items: center; gap: 10px; font-weight: 800; font-size: 1.1rem; margin-bottom: .2rem;}
.brand .logo {background: linear-gradient(135deg, #4F46E5, #06B6D4); color: #fff; border-radius: 12px;
              width: 38px; height: 38px; display: flex; align-items: center; justify-content: center; font-size: 1.25rem;}
.step {display: flex; gap: 10px; margin: 8px 0; font-size: .88rem; color: #39406A;}
.step b {background: #4F46E5; color: #fff; border-radius: 50%; min-width: 22px; height: 22px;
         display: flex; align-items: center; justify-content: center; font-size: .75rem;}
</style>
"""


@st.cache_resource(show_spinner="Starting up: loading indexes and models…")
def get_services() -> services.Services:
    return services.build_services()


def human_size(n: int) -> str:
    return f"{n / 1024:.0f} KB" if n < 1024 * 1024 else f"{n / 1024 / 1024:.1f} MB"


def hero(svc: services.Services, title: str, subtitle: str) -> None:
    snap = svc.store.snapshot
    st.markdown(
        f'<div class="hero"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p>'
        f'<span class="chip">📚 {len(snap.doc_ids())} documents</span>'
        f'<span class="chip">🧩 {snap.chunk_count} searchable passages</span>'
        f'<span class="chip">🔎 Every answer cites its source</span></div>',
        unsafe_allow_html=True)


# ------------------------------------------------------------------ User mode
def render_citations(citations) -> None:
    with st.expander(f"Sources ({len(citations)})"):
        for c in citations:
            page = f" · page {c.page_number}" if c.page_number else ""
            st.markdown(
                f'<span class="src-title">[{c.number}] {html.escape(c.title)}</span>{html.escape(page)}'
                f'<span class="pill">{html.escape(c.chunk_id)}</span>', unsafe_allow_html=True)
            st.text(c.snippet)  # st.text: document content is shown literally, never interpreted as markdown/HTML


def render_assistant_message(msg: dict) -> None:
    st.markdown(msg["content"])
    if msg["content"] == config.NO_ANSWER_MESSAGE:
        st.markdown('<span class="badge badge-no">Not covered by the documentation</span>', unsafe_allow_html=True)
    elif msg.get("citations"):
        st.markdown('<span class="badge badge-ok">Answered from the documentation</span>', unsafe_allow_html=True)
    if msg.get("citations"):
        render_citations(msg["citations"])


def user_mode(svc: services.Services) -> None:
    hero(svc, "Ask the docs", "Plain-English questions, answers taken only from your documentation, "
                              "with sources you can check.")
    if svc.store.snapshot.chunk_count == 0:
        st.info("No documents are indexed yet. An admin needs to upload documentation first.")

    history = st.session_state.setdefault("messages", [])
    question = st.chat_input("Ask about the documentation…", max_chars=500) or st.session_state.pop("pending_question", None)

    if not history and not question:  # empty state: one-click example questions
        st.markdown('<div class="section-label">Try asking</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, (icon, text) in enumerate(EXAMPLE_QUESTIONS):
            if cols[i % 2].button(f"{icon}  {text}", key=f"example-{i}", width="stretch"):
                st.session_state["pending_question"] = text
                st.rerun()

    for msg in history:
        with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "📘"):
            if msg["role"] == "assistant":
                render_assistant_message(msg)
            else:
                st.markdown(msg["content"])

    if not question:
        return
    history.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="📘"):
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
    hero(svc, "Admin console", "Sign in to upload, review and remove documentation.")
    with st.container(border=True):
        st.markdown("#### 🔐 Admin login")
        with st.form("login"):
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", type="primary")
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
    count = lambda *s: sum(1 for d in docs if d["status"] in s)  # noqa: E731
    cols = st.columns(4)
    cols[0].metric("Documents", len(docs))
    cols[1].metric("Searchable", count("indexed"))
    cols[2].metric("In progress", count("queued", "processing"))
    cols[3].metric("Needs attention", count("failed", "delete_failed"))
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
    st.markdown("##### Add documents")
    st.caption(f"PDF (digital or scanned), DOCX, PNG, JPG, JSON. Max {config.MAX_UPLOAD_MB} MB each. "
               "OCR reads printed text only; it does not interpret diagrams or charts.")
    key = st.session_state.setdefault("uploader_key", 0)
    files = st.file_uploader("Drop files here or browse", type=["pdf", "docx", "png", "jpg", "jpeg", "json"],
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
    st.markdown("##### Remove a document")
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

    hero(svc, "Admin console", "Upload documentation, watch it get indexed, and remove what is out of date.")
    if st.button("Log out"):
        svc.auth.logout(token)
        st.session_state.pop("admin_token", None)
        st.rerun()
    flash = st.session_state.pop("flash", None)  # a message that must survive st.rerun()
    if flash:
        getattr(st, flash[0])(flash[1])

    tab_docs, tab_upload, tab_manage = st.tabs(["📚 Documents", "📤 Upload", "🗑️ Manage"])
    with tab_docs:
        documents_table(svc)
    with tab_upload:
        with st.container(border=True):
            upload_section(svc, token)
    with tab_manage:
        with st.container(border=True):
            delete_section(svc, token)


# ------------------------------------------------------------------ main
def main() -> None:
    svc = get_services()
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('<div class="brand"><div class="logo">📘</div>Solstice Docs</div>', unsafe_allow_html=True)
        st.caption("Documentation Q&A assistant")
        mode = st.radio("Mode", ["Ask questions", "Admin"], key="mode")
        st.markdown(
            '<div class="section-label">How it works</div>'
            '<div class="step"><b>1</b><span>You ask a question in plain English.</span></div>'
            '<div class="step"><b>2</b><span>We search the documents and check there is real evidence.</span></div>'
            '<div class="step"><b>3</b><span>You get an answer with its sources, or an honest "I don\'t know".</span></div>',
            unsafe_allow_html=True)
        st.caption("Admin access is password protected.")
    if mode == "Ask questions":
        user_mode(svc)
    else:
        admin_mode(svc)


main()

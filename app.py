"""Streamlit UI — a thin client of the FastAPI service.

Every action is an HTTP call to the API (API_BASE_URL), so the UI holds no
index, model or database of its own and can run in its own container.

    streamlit run app.py
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API = os.getenv("API_BASE_URL", "http://localhost:8080").rstrip("/")
CHAT_TIMEOUT_S = float(os.getenv("CHAT_TIMEOUT_S", "180"))

st.set_page_config(page_title="Personal AI Knowledge Assistant", page_icon="🧠", layout="wide")


# --------------------------------------------------------------------------
# API client
# --------------------------------------------------------------------------

def api(method: str, path: str, *, timeout: float = 30, **kwargs):
    """Returns (json_or_None, error_or_None). Never raises into the page."""
    headers = {}
    if st.session_state.get("token"):
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    try:
        resp = httpx.request(method, f"{API}{path}", headers=headers, timeout=timeout, **kwargs)
    except httpx.HTTPError as exc:
        return None, f"Cannot reach the API at {API} ({type(exc).__name__})."
    if resp.status_code == 401:
        st.session_state.pop("token", None)
        return None, "login required"
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        return None, f"{resp.status_code}: {detail}"
    if not resp.content:
        return None, None
    try:
        return resp.json(), None
    except ValueError:
        return resp.text, None


def show_error(error: str | None) -> bool:
    if error:
        st.error(error)
        return True
    return False


# --------------------------------------------------------------------------
# Session / auth
# --------------------------------------------------------------------------

st.session_state.setdefault("conversation_id", None)
st.session_state.setdefault("results", {})        # message_id -> full /chat result
st.session_state.setdefault("feedback_done", set())
st.session_state.setdefault("undo", None)

me, me_error = api("GET", "/auth/me")
if me_error == "login required":
    st.title("🧠 Personal AI Knowledge Assistant")
    with st.form("login"):
        st.subheader("Sign in")
        tenant = st.text_input("Workspace", value="default")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary"):
            token, error = api(
                "POST", "/auth/login", json={"tenant_id": tenant, "email": email, "password": password}
            )
            if not show_error(error):
                st.session_state.token = token["access_token"]
                st.rerun()
    st.stop()
if show_error(me_error):
    st.info("Start the API first:  `uvicorn aka.api.main:app --port 8080`")
    st.stop()

is_reviewer = me["role"] in ("reviewer", "admin")
is_admin = me["role"] == "admin"

with st.sidebar:
    st.title("🧠 Knowledge Assistant")
    pages = ["Chat", "Documents"] + (["Review"] if is_reviewer else []) + ["System"]
    page = st.radio("Page", pages, label_visibility="collapsed")
    st.caption(f"{me['email']} · {me['role']} · workspace `{me['tenant_id']}`")
    if me["auth_enabled"] and st.button("Sign out"):
        st.session_state.pop("token", None)
        st.rerun()
    st.divider()


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------

def answer_badges(meta: dict) -> None:
    """One line telling the user what kind of answer this is."""
    parts = []
    mode = meta.get("mode")
    if mode and mode != "generated":
        parts.append(f"mode: **{mode}**")
    if meta.get("retrieval_mode") == "lexical":
        parts.append("keyword retrieval")
    if meta.get("needs_review"):
        parts.append("⚠️ weakly supported — queued for review")
    if meta.get("review"):
        parts.append(f"reviewed: {meta['review']}")
    if meta.get("degraded") and meta.get("degrade_reason"):
        parts.append(f"degraded: {meta['degrade_reason']}")
    if meta.get("grounding") is not None and mode in ("generated", "cached"):
        parts.append(f"grounding {meta['grounding']:.2f}")
    if parts:
        st.caption(" · ".join(parts))


def why_this_answer(result: dict) -> None:
    with st.expander("Why this answer"):
        policy = result.get("policy") or {}
        st.write(
            f"Policy: `{policy.get('rule')}` → **{policy.get('verdict')}** · "
            f"model `{result.get('model')}` (tier {result.get('model_tier')}) · "
            f"{result.get('latency_s', 0):.1f}s"
        )
        if result.get("review_reason"):
            st.write(result["review_reason"])
        for ev in result.get("evidence", []):
            st.markdown(f"**[{ev['n']}] {ev['source']}** · score {ev['score']:.2f}")
            st.text(ev["excerpt"])


def feedback_buttons(message_id: str, meta: dict) -> None:
    verdict = (meta.get("policy") or {}).get("verdict")
    if verdict not in ("allow", "review") or meta.get("mode") in ("extractive", "refused"):
        return
    if message_id in st.session_state.feedback_done or meta.get("review"):
        return
    up, down, _ = st.columns([1, 1, 10])
    for col, helpful, label in ((up, True, "👍"), (down, False, "👎")):
        if col.button(label, key=f"fb-{helpful}-{message_id}"):
            _, error = api("POST", "/chat/feedback", json={"message_id": message_id, "helpful": helpful})
            if not show_error(error):
                st.session_state.feedback_done.add(message_id)
                st.toast("Thanks — recorded.")
                st.rerun()


def chat_page() -> None:
    with st.sidebar:
        if st.button("+ New chat"):
            st.session_state.conversation_id = None
            st.rerun()
        convos, error = api("GET", "/conversations")
        if not error:
            for convo in convos or []:
                label = convo["title"][:40] or "Untitled"
                if st.button(label, key=f"c-{convo['id']}"):
                    st.session_state.conversation_id = convo["id"]
                    st.rerun()

    st.header("Ask your documents")
    cid = st.session_state.conversation_id
    if cid:
        convo, error = api("GET", f"/conversations/{cid}")
        if error:
            st.session_state.conversation_id = None
        else:
            if st.button("Archive this chat"):
                api("DELETE", f"/conversations/{cid}")
                st.session_state.conversation_id = None
                st.rerun()
            for msg in convo["messages"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant":
                        answer_badges(msg["meta"])
                        result = st.session_state.results.get(msg["id"])
                        if result:
                            why_this_answer(result)
                        feedback_buttons(msg["id"], msg["meta"])

    if question := st.chat_input("Ask a question..."):
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"), st.spinner("Thinking..."):
            result, error = api(
                "POST",
                "/chat",
                json={"question": question, "conversation_id": st.session_state.conversation_id},
                timeout=CHAT_TIMEOUT_S,
            )
        if not show_error(error):
            st.session_state.conversation_id = result["conversation_id"]
            st.session_state.results[result["message_id"]] = result
            st.rerun()


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------

def documents_page() -> None:
    st.header("Documents")
    st.caption(
        "Uploads and deletes queue a background reindex. The current index keeps "
        "answering until the new one has been built and checked."
    )

    docs, error = api("GET", "/documents")
    if not show_error(error):
        if docs:
            st.dataframe(
                [{"file": d["rel_path"], "status": d["status"], "KB": round(d["size_bytes"] / 1024, 1)} for d in docs],
                hide_index=True,
            )
        else:
            st.info("No documents yet.")

    upload_col, delete_col = st.columns(2)
    with upload_col:
        st.subheader("Upload")
        files = st.file_uploader(
            "PDF, TXT, MD, DOCX", type=["pdf", "txt", "md", "docx"], accept_multiple_files=True
        )
        subfolder = st.text_input("Subfolder (optional)", placeholder="e.g. research_papers")
        if st.button("Upload", type="primary", disabled=not files):
            payload = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in files]
            result, error = api("POST", "/documents", files=payload, data={"subfolder": subfolder}, timeout=120)
            if not show_error(error):
                for item in result["rejected"]:
                    st.warning(f"Skipped {item['filename']}: {item['reason']}")
                if result["stored"]:
                    st.success(f"Stored {len(result['stored'])} file(s); reindex queued.")

    with delete_col:
        st.subheader("Delete")
        indexed = [d for d in docs or [] if d["id"]]
        if indexed:
            choice = st.selectbox("Document", indexed, format_func=lambda d: d["rel_path"])
            confirm = st.checkbox("Move it to the trash (undo available)")
            if st.button("Delete", disabled=not confirm):
                result, error = api("DELETE", f"/documents/{choice['id']}")
                if not show_error(error):
                    st.session_state.undo = result
                    st.rerun()
        else:
            st.caption("Nothing indexed yet.")
        undo = st.session_state.undo
        if undo:
            st.info(f"Deleted `{undo['rel_path']}`.")
            if st.button("Undo delete"):
                _, error = api("POST", f"/documents/undo/{undo['undo_token']}")
                if not show_error(error):
                    st.session_state.undo = None
                    st.success("Restored; reindex queued.")

    st.subheader("Index")
    status, error = api("GET", "/index/status")
    if not show_error(error):
        active = status["active"]
        if active:
            st.write(
                f"Active version `{active['id'][:10]}` · {active['doc_count']} documents · "
                f"{active['chunk_count']} chunks · {active['embed_model']}"
            )
        else:
            st.write("No active index yet.")
        if status["building"]:
            st.write(f"Building {len(status['building'])} new version(s)...")
        if status["jobs"]:
            st.dataframe(
                [
                    {
                        "job": j["id"][:10],
                        "status": j["status"],
                        "attempts": j["attempts"],
                        "embedded": j["result"].get("chunks_embedded"),
                        "reused": j["result"].get("chunks_reused"),
                        "seconds": j["result"].get("duration_s"),
                        "error": j["last_error"],
                    }
                    for j in status["jobs"]
                ],
                hide_index=True,
            )
    st.button("Refresh")


# --------------------------------------------------------------------------
# Review
# --------------------------------------------------------------------------

def review_page() -> None:
    st.header("Review queue")
    stats, error = api("GET", "/stats/overrides")
    if not error:
        a, b, c = st.columns(3)
        a.metric("Override rate (30 days)", f"{stats['override_rate']:.1%}")
        b.metric("Human decisions", stats["human_decisions"])
        c.metric("Overrides", stats["overrides"])
        if stats["per_rule"]:
            st.dataframe(
                [{"rule": rule, **row} for rule, row in stats["per_rule"].items()], hide_index=True
            )

    pending, error = api("GET", "/approvals")
    if show_error(error):
        return
    if not pending:
        st.success("Nothing waiting for review.")
    for item in pending:
        with st.container(border=True):
            st.markdown(f"**{item['kind']}** · `{item['rule']}`")
            st.write(item["title"])
            st.caption(item["reason"])
            ctx = item["context"]
            if ctx.get("answer"):
                st.markdown(f"> {ctx['answer']}")
                if ctx.get("sources"):
                    st.caption("sources: " + ", ".join(ctx["sources"]))
            if ctx.get("evidence"):
                st.caption(f"evidence: {ctx['evidence']}")
            note = st.text_input("Note (optional)", key=f"note-{item['id']}")
            ok, no, _ = st.columns([1, 1, 6])
            for col, approved, label in ((ok, True, "Approve"), (no, False, "Reject")):
                if col.button(label, key=f"{label}-{item['id']}"):
                    _, error = api(
                        "POST", f"/approvals/{item['id']}/resolve", json={"approved": approved, "note": note}
                    )
                    if not show_error(error):
                        st.rerun()

    st.subheader("What the assistant remembers about you")
    memories, error = api("GET", "/memory")
    if not show_error(error):
        if not memories:
            st.caption("Nothing yet.")
        for m in memories or []:
            left, right = st.columns([8, 1])
            left.write(f"{m['content']}  \n`{m['status']}` · confidence {m['confidence']:.2f}")
            if right.button("Forget", key=f"forget-{m['id']}"):
                _, error = api("POST", f"/memory/{m['id']}/feedback", json={"keep": False})
                if not show_error(error):
                    st.rerun()


# --------------------------------------------------------------------------
# System
# --------------------------------------------------------------------------

def system_page() -> None:
    st.header("System")
    ready, error = api("GET", "/readyz")
    if ready:
        st.write({name: "✅" if ok else "❌" for name, ok in ready["checks"].items()})
    if is_reviewer:
        health, error = api("GET", "/health/llm")
        if not show_error(error):
            st.write("Model chain: " + " → ".join(health["chain"]))
            st.dataframe(health["breakers"], hide_index=True)
    if is_admin:
        st.subheader("Index maintenance")
        force = st.checkbox("Full rebuild (re-embed everything)")
        if st.button("Rebuild index"):
            job, error = api("POST", "/index/rebuild", json={"force_full": force})
            if not show_error(error):
                st.success(f"Queued job {job['id'][:10]}.")
        if st.button("Roll back to previous version"):
            result, error = api("POST", "/index/rollback")
            if not show_error(error):
                st.success(f"Active version is now {result['active_version'][:10]}.")


{"Chat": chat_page, "Documents": documents_page, "Review": review_page, "System": system_page}[page]()

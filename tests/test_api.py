"""HTTP surface, end to end: upload -> worker builds the index -> chat ->
feedback -> review queue -> override stats. Runs against the fake backend."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aka.llm.fake import FakeOllama

DOC = (
    "The Hermes deployment runs on three nodes in the Frankfurt region. "
    "Failover is automatic and takes about ninety seconds. "
    "The on-call rotation is weekly and starts on Mondays."
)


@pytest.fixture()
def backend():
    return FakeOllama()


def _make_client(settings, backend, *, auth: bool):
    from aka.api.main import create_app
    from aka.ingest import vectorstore

    settings.AUTH_ENABLED = auth
    settings.MEMORY_EXTRACTION_ENABLED = False
    settings.LLM_MODEL = "primary"
    settings.LLM_FALLBACK_MODELS = ["small"]
    # Bag-of-words fake embeddings give a short question and a whole document
    # a low cosine; the production floor is tuned for real embeddings.
    settings.RETRIEVAL_MIN_SCORE = 0.05
    vectorstore.reset_chroma_for_tests()
    app = create_app(settings, llm_client=backend, start_worker=False)
    return TestClient(app)


@pytest.fixture()
def client(settings, db, backend):
    with _make_client(settings, backend, auth=False) as c:
        yield c


def run_worker(settings, backend) -> None:
    from aka.ingest.worker import Worker

    worker = Worker(settings, client=backend)
    while worker.run_once():
        pass


def upload(client, name="hermes.txt", body=DOC, subfolder=""):
    return client.post(
        "/documents",
        files=[("files", (name, body.encode(), "text/plain"))],
        data={"subfolder": subfolder},
    )


# --------------------------------------------------------------------------


def test_health_endpoints(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz").json()
    assert ready["checks"]["database"] is True
    assert "aka_chat_requests_total" in client.get("/metrics").text


def test_upload_index_chat_flow(client, settings, backend):
    resp = upload(client)
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["stored"][0]["rel_path"] == "hermes.txt"
    assert body["job"]["status"] == "queued"

    # Before the worker runs there is no index: the API still answers honestly.
    early = client.post("/chat", json={"question": "How long does Hermes failover take?"}).json()
    assert early["mode"] == "refused"

    run_worker(settings, backend)
    status = client.get("/index/status").json()
    assert status["active"]["chunk_count"] > 0
    assert status["jobs"][0]["status"] == "succeeded"

    answer = client.post("/chat", json={"question": "How long does Hermes failover take?"}).json()
    assert answer["mode"] == "generated", answer
    assert answer["outcome"] == "answered"
    assert answer["policy"]["verdict"] == "allow"
    assert answer["evidence"][0]["source"] == "hermes.txt"

    convos = client.get("/conversations").json()
    assert any(c["id"] == answer["conversation_id"] for c in convos)
    history = client.get(f"/conversations/{answer['conversation_id']}").json()
    assert [m["role"] for m in history["messages"]][-2:] == ["user", "assistant"]


def test_duplicate_uploads_share_one_queued_reindex(client):
    first = upload(client, "a.txt").json()["job"]["id"]
    second = upload(client, "b.txt").json()["job"]["id"]
    assert first == second


def test_upload_rejects_traversal_and_bad_types(client):
    assert upload(client, subfolder="../../etc").status_code == 400
    bad = upload(client, name="run.exe").json()
    assert bad["stored"] == [] and bad["rejected"][0]["reason"].startswith("unsupported")


def test_feedback_on_released_answer_counts_as_override(client, settings, backend):
    upload(client)
    run_worker(settings, backend)
    answer = client.post("/chat", json={"question": "How long does Hermes failover take?"}).json()
    assert answer["policy"]["verdict"] == "allow"

    resp = client.post("/chat/feedback", json={"message_id": answer["message_id"], "helpful": False})
    assert resp.status_code == 200, resp.text
    assert resp.json()["was_override"] is True

    # One verdict per answer keeps the denominator honest.
    again = client.post("/chat/feedback", json={"message_id": answer["message_id"], "helpful": True})
    assert again.status_code == 409

    stats = client.get("/stats/overrides").json()
    assert stats["human_decisions"] == 1 and stats["overrides"] == 1
    assert stats["per_rule"]["answer_release.grounded"]["override_rate"] == 1.0


def test_low_grounding_answer_is_queued_and_resolvable(client, settings, backend):
    upload(client)
    run_worker(settings, backend)
    # Half the answer's words are in the evidence: shown, but flagged.
    backend.answer_override = "Failover takes ninety seconds, says the quarterly penguin census [1]."
    answer = client.post("/chat", json={"question": "How long does Hermes failover take?"}).json()
    assert answer["needs_review"] is True, answer
    assert answer["policy"]["rule"] == "answer_release.low_grounding"

    queue_ = client.get("/approvals").json()
    assert [a["id"] for a in queue_] == [answer["approval_id"]]

    resolved = client.post(f"/approvals/{answer['approval_id']}/resolve", json={"approved": True})
    assert resolved.status_code == 200, resolved.text
    assert resolved.json()["was_override"] is True
    assert client.get("/approvals").json() == []


def test_delete_and_undo(client, settings, backend):
    upload(client)
    run_worker(settings, backend)
    doc = client.get("/documents").json()[0]

    deleted = client.delete(f"/documents/{doc['id']}").json()
    assert not (settings.DATA_DIR / "hermes.txt").exists()
    run_worker(settings, backend)
    assert client.get("/documents").json() == []

    restored = client.post(f"/documents/undo/{deleted['undo_token']}")
    assert restored.status_code == 200, restored.text
    assert (settings.DATA_DIR / "hermes.txt").exists()
    assert client.post(f"/documents/undo/{deleted['undo_token']}").status_code == 409

    run_worker(settings, backend)
    assert client.get("/documents").json()[0]["status"] == "indexed"


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------


@pytest.fixture()
def secured(settings, db, backend):
    with _make_client(settings, backend, auth=True) as c:
        yield c


def _user_token(session, settings, email, role):
    from aka.security.auth import create_user, issue_token, principal_from_user

    user = create_user(session, tenant_id="default", email=email, password="correct-horse", role=role)
    session.commit()
    return issue_token(principal_from_user(user), settings)


def test_auth_required_and_roles_enforced(secured, session, settings):
    assert secured.get("/documents").status_code == 401
    assert secured.get("/healthz").status_code == 200

    user = _user_token(session, settings, "u@example.com", "user")
    reviewer = _user_token(session, settings, "r@example.com", "reviewer")

    assert secured.get("/documents", headers={"Authorization": f"Bearer {user}"}).status_code == 200
    assert secured.get("/approvals", headers={"Authorization": f"Bearer {user}"}).status_code == 403
    assert secured.get("/approvals", headers={"Authorization": f"Bearer {reviewer}"}).status_code == 200


def test_login_issues_working_token(secured, session, settings):
    _user_token(session, settings, "u@example.com", "user")
    bad = secured.post("/auth/login", json={"email": "u@example.com", "password": "wrong"})
    assert bad.status_code == 401
    good = secured.post("/auth/login", json={"email": "u@example.com", "password": "correct-horse"})
    token = good.json()["access_token"]
    me = secured.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).json()
    assert me["email"] == "u@example.com" and me["auth_enabled"] is True

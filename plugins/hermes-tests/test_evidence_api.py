"""Tests for the citation evidence lookup in plugins/hermes/dashboard/plugin_api.py.

The model writes the id of a citation, so the id is untrusted input. These tests pin what keeps it harmless: it only
ever becomes one `--scope=` argument, what the card shows and opens comes from the index's answer, and an id the
index does not know is reported as such.

Run: uv run --with pytest --with fastapi --with httpx pytest plugins/hermes-tests -q
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

API = Path(__file__).resolve().parents[1] / "hermes" / "dashboard" / "plugin_api.py"


@pytest.fixture()
def api():
    spec = importlib.util.spec_from_file_location("autorag_plugin_api_evidence_under_test", API)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.clear_evidence_cache()
    return module


ENV = {"AUTORAG_LIBRARIAN_CLI": "/lib/cli.js"}
NODE = lambda name: f"/bin/{name}"  # noqa: E731


def hit(**over):
    base = {
        "number": 1,
        "source": "/slack/default/chunks/C1-1",
        "metadata": {"datasourceId": "slack", "title": "general", "senderName": "Mina Park",
                     "publishedAt": 1793000000000, "hierarchy": "workspaces/T/channels/general"},
        "content": "unit price is 11,500",
    }
    base.update(over)
    return base


class FakeRun:
    def __init__(self, payload=None, returncode=0, stderr="", stdout=None, raises=None):
        self.calls = []
        self.payload = {"ok": True, "results": [hit()]} if payload is None else payload
        self.returncode, self.stderr, self.stdout, self.raises = returncode, stderr, stdout, raises

    def __call__(self, argv, **kw):
        self.calls.append((list(argv), kw))
        if self.raises:
            raise self.raises
        out = self.stdout if self.stdout is not None else json.dumps(self.payload)
        return types.SimpleNamespace(returncode=self.returncode, stdout=out, stderr=self.stderr)


def lookup(api, source_id="/slack/default/chunks/C1-1", query="price", run=None, **kw):
    run = run or FakeRun()
    return api.fetch_evidence(source_id, query, env=ENV, run=run, which=NODE, **kw), run


# ---- the id ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "docs/a.pdf", "--scope=/x", "/a/../etc/passwd", "/a\nb", "/a\x00b", "/" + "x" * 1100, None, 5])
def test_ids_that_are_not_a_rooted_clean_path_are_rejected(api, bad):
    with pytest.raises(ValueError):
        api.validate_source_id(bad)


def test_the_id_is_one_scope_argument_and_the_query_never_looks_like_a_flag(api):
    argv = api.build_evidence_argv("/docs/한글 계약서 \"final\".pdf", "--help", env=ENV, which=NODE)
    assert argv == ["/bin/node", "/lib/cli.js", "lite", "retrieve", "help", "--scope=/docs/한글 계약서 \"final\".pdf",
                    "--top-k", "1", "--json"]
    assert api.validate_cite_query("-- -x") == "x"
    assert api.validate_cite_query("   ") == "evidence"
    assert len(api.validate_cite_query("q" * 999)) == api.MAX_CITE_QUERY_CHARS


def test_the_child_gets_no_shell_and_a_reduced_environment(api):
    _, run = lookup(api)
    _argv, kw = run.calls[0]
    assert kw["stdin"] == subprocess.DEVNULL
    assert "shell" not in kw
    assert set(kw["env"]) <= set(ENV) | {"PATH", "HOME"}


# ---- what the card is told ------------------------------------------------------------------------------------


def test_a_chat_message_is_shaped_from_the_index_not_from_the_model(api):
    ev, _ = lookup(api)
    assert ev["kind"] == "message"
    assert (ev["title"], ev["author"], ev["datasource"]) == ("general", "Mina Park", "slack")
    assert ev["location"] == "workspaces/T/channels/general"
    assert ev["timestamp"] == 1793000000000
    assert ev["path"] is None and ev["missing"] is False


def test_a_file_hit_drops_the_index_label_line_and_offers_its_real_path(api, tmp_path):
    real = tmp_path / "계약서.pdf"
    real.write_text("x")
    payload = {"results": [hit(source=str(real), metadata={"virtualPath": "/docs/계약서.pdf"},
                               content="files/docs/계약서.pdf.md\n# Agreement\nArticle 4")]}
    ev, _ = lookup(api, "/docs/계약서.pdf", run=FakeRun(payload))
    assert ev["kind"] == "file"
    assert ev["content"] == "# Agreement\nArticle 4"
    assert ev["path"] == str(real)
    assert ev["title"] == "계약서.pdf"


def test_a_file_that_was_moved_is_reported_missing_instead_of_openable(api, tmp_path):
    payload = {"results": [hit(source=str(tmp_path / "gone.pdf"), metadata={"virtualPath": "/docs/gone.pdf"}, content="files/x\nbody")]}
    ev, _ = lookup(api, "/docs/gone.pdf", run=FakeRun(payload))
    assert ev["path"] is None and ev["missing"] is True


def test_only_http_links_from_the_index_become_a_link(api):
    for url, expected in [("https://example.com/a", "https://example.com/a"), ("javascript:alert(1)", None), ("file:///etc/passwd", None)]:
        payload = {"results": [hit(metadata={"datasourceId": "notion", "url": url})]}
        api.clear_evidence_cache()
        ev, _ = lookup(api, run=FakeRun(payload))
        assert ev["url"] == expected


def test_long_evidence_is_cut_and_says_so(api):
    ev, _ = lookup(api, run=FakeRun({"results": [hit(content="a" * 5000)]}))
    assert len(ev["content"]) == api.MAX_EVIDENCE_CHARS and ev["truncated"] is True


def test_a_malformed_timestamp_is_not_passed_through(api):
    ev, _ = lookup(api, run=FakeRun({"results": [hit(metadata={"publishedAt": "yesterday"})]}))
    assert ev["timestamp"] is None


# ---- not in the index, and failures ---------------------------------------------------------------------------


def test_an_id_the_index_does_not_know_is_not_indexed(api):
    with pytest.raises(api.SourceNotIndexed):
        lookup(api, "/slack/default/chunks/invented", run=FakeRun({"ok": True, "results": []}))


def test_a_failing_librarian_is_an_error_with_its_stderr_not_a_missing_source(api):
    with pytest.raises(RuntimeError, match="boom"):
        lookup(api, run=FakeRun(returncode=1, stderr="boom"))
    with pytest.raises(RuntimeError, match="JSON"):
        lookup(api, run=FakeRun(stdout="not json"))


def test_a_stuck_librarian_times_out(api):
    with pytest.raises(api.FinderUnavailable, match="timed out"):
        lookup(api, run=FakeRun(raises=subprocess.TimeoutExpired("x", 1)))


def test_repeat_lookups_do_not_start_another_librarian_and_failures_are_not_cached(api):
    run = FakeRun()
    lookup(api, run=run)
    lookup(api, run=run)
    assert len(run.calls) == 1
    bad = FakeRun(returncode=1, stderr="x")
    with pytest.raises(RuntimeError):
        lookup(api, "/slack/default/chunks/other", run=bad)
    with pytest.raises(RuntimeError):
        lookup(api, "/slack/default/chunks/other", run=bad)
    assert len(bad.calls) == 2


# ---- the route ------------------------------------------------------------------------------------------------


@pytest.fixture()
def client(api):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(api.router, prefix="/api/plugins/autorag")
    return TestClient(app)


def test_route_returns_the_evidence(api, client, monkeypatch):
    monkeypatch.setattr(api, "fetch_evidence", lambda sid, q: {"id": sid, "content": "hi", "q": q})
    r = client.get("/api/plugins/autorag/evidence", params={"id": "/docs/a.pdf", "q": "price"})
    assert r.status_code == 200 and r.json() == {"id": "/docs/a.pdf", "content": "hi", "q": "price"}


def test_route_rejects_a_bad_id_before_any_process_starts(api, client, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("must not start a librarian")

    monkeypatch.setattr(api, "fetch_evidence", boom)
    assert client.get("/api/plugins/autorag/evidence", params={"id": "../../etc/passwd"}).status_code == 422
    assert client.get("/api/plugins/autorag/evidence", params={"id": "/a/../b"}).status_code == 422


@pytest.mark.parametrize("error,status,detail", [
    ("not_indexed", 404, "not-in-index"),
    ("unavailable", 503, None),
    ("failed", 502, None),
])
def test_route_maps_errors_to_distinct_statuses(api, client, monkeypatch, error, status, detail):
    def fail(*_a, **_k):
        raise {"not_indexed": api.SourceNotIndexed("x"), "unavailable": api.FinderUnavailable("no node"),
               "failed": RuntimeError("librarian crashed")}[error]

    monkeypatch.setattr(api, "fetch_evidence", fail)
    r = client.get("/api/plugins/autorag/evidence", params={"id": "/docs/a.pdf"})
    assert r.status_code == status
    if detail:
        assert r.json()["detail"] == detail


def test_route_while_the_librarian_installs_says_so_and_starts_the_install(api, client, monkeypatch):
    warmed = []
    monkeypatch.setattr(api, "fetch_evidence", lambda *_a, **_k: (_ for _ in ()).throw(api.FinderNotReady("installing")))
    monkeypatch.setattr(api, "get_runtime", lambda: types.SimpleNamespace(warm=lambda: warmed.append(1)))
    r = client.get("/api/plugins/autorag/evidence", params={"id": "/docs/a.pdf"})
    assert r.status_code == 503 and r.json()["detail"]["state"] == "installing" and warmed

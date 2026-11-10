"""Tests for plugins/hermes/dashboard/plugin_api.py (the AutoRAG finder backend).

Run: uv run --with pytest --with fastapi --with httpx pytest plugins/hermes-tests -q
"""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "hermes"
API = ROOT / "dashboard" / "plugin_api.py"


def load_api():
    spec = importlib.util.spec_from_file_location("autorag_plugin_api_under_test", API)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def api():
    return load_api()


class FakeStream:
    def __init__(self, lines):
        self._lines = list(lines)

    def __iter__(self):
        return iter(self._lines)

    def read(self):
        return "".join(self._lines)


class FakeProc:
    """Popen stand-in: stdout yields the given lines, then the process 'exits' with `code`."""

    def __init__(self, lines, code=0, stderr="", hold=None):
        self.stdout = FakeStream(lines)
        self.stderr = FakeStream([stderr] if stderr else [])
        self.returncode = None
        self._code = code
        self._hold = hold
        self.terminated = False
        self.killed = False

    def wait(self, timeout=None):
        if self._hold is not None:
            self._hold.wait(timeout)
        self.returncode = -15 if self.terminated else self._code
        return self.returncode

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        if self._hold is not None:
            self._hold.set()

    def kill(self):
        self.killed = True
        self.terminate()


def nd(*objs):
    return [json.dumps(o) + "\n" for o in objs]


def wait_for(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


GROUPS = {"type": "groups", "groups": [{"key": "files"}, {"key": "content"}]}
FILES = {"type": "group", "key": "files", "status": "ready", "hits": [], "error": None, "elapsedMs": 5}
DONE = {"type": "done", "elapsedMs": 9}


# ---- argv ------------------------------------------------------------------------------------


def test_argv_uses_the_version_pinned_in_mcp_json(api):
    pin = json.loads((ROOT / "mcp.json").read_text())["mcpServers"]["autorag"]["args"]
    spec = next(a for a in pin if a.startswith("@autorag/librarian@"))
    argv = api.build_find_argv("계약서", None, env={"PATH": "/bin"}, which=lambda n: f"/x/{n}")
    assert argv[:5] == ["/x/npx", "-y", "-p", spec, "autorag"]
    assert argv[5:7] == ["lite", "find"]
    assert "--json" in argv


def test_argv_passes_a_dash_query_as_one_value_flag(api):
    argv = api.build_find_argv("--help -v2", None, env={}, which=lambda n: f"/x/{n}")
    assert "--query=--help -v2" in argv
    assert "--help -v2" not in argv  # never a bare positional


def test_argv_narrows_groups(api):
    argv = api.build_find_argv("x", ["files", "source:kakao"], env={}, which=lambda n: f"/x/{n}")
    assert "--groups=files,source:kakao" in argv


def test_argv_local_build_override(api):
    env = {"AUTORAG_LIBRARIAN_CLI": "/tmp/dist/cli/index.js"}
    argv = api.build_find_argv("x", None, env=env, which=lambda n: f"/x/{n}")
    assert argv[:2] == ["/x/node", "/tmp/dist/cli/index.js"]
    assert argv[2:4] == ["lite", "find"]


@pytest.mark.parametrize("bad", ["", "   ", "x" * 501, "a\x00b"])
def test_argv_rejects_bad_queries(api, bad):
    with pytest.raises(ValueError):
        api.build_find_argv(bad, None, env={}, which=lambda n: f"/x/{n}")


@pytest.mark.parametrize("bad", ["../etc", "source:", "source:a b", "FILES", "files;rm", "source:" + "a" * 80])
def test_argv_rejects_bad_groups(api, bad):
    with pytest.raises(ValueError):
        api.build_find_argv("x", [bad], env={}, which=lambda n: f"/x/{n}")


def test_missing_node_is_a_clear_error(api):
    with pytest.raises(api.FinderUnavailable, match="Node.js"):
        api.build_find_argv("x", None, env={}, which=lambda n: None)


# ---- child environment -------------------------------------------------------------------------


def test_child_env_drops_secrets_and_keeps_runtime_basics(api):
    env = api.child_env(
        {
            "PATH": "/bin",
            "HOME": "/h",
            "LANG": "ko_KR.UTF-8",
            "AUTORAG_HOME": "/a",
            "AUTORAG_WORKSPACE": "/w",
            "OPENAI_API_KEY": "sk-secret",
            "FIREWORKS_API_KEY": "fw-secret",
            "GITHUB_TOKEN": "ghp_x",
            "HTTPS_PROXY": "http://p",
        }
    )
    assert env["PATH"] == "/bin" and env["HOME"] == "/h" and env["AUTORAG_HOME"] == "/a"
    assert env["HTTPS_PROXY"] == "http://p"
    assert "OPENAI_API_KEY" not in env and "FIREWORKS_API_KEY" not in env and "GITHUB_TOKEN" not in env


# ---- registry ------------------------------------------------------------------------------------


def make_registry(api, procs, **kw):
    it = iter(procs)
    spawned = []

    def spawn(argv, env):
        spawned.append(argv)
        return next(it)

    reg = api.FindRegistry(spawn=spawn, argv_builder=lambda q, g: ["fake", q], **kw)
    return reg, spawned


def test_lines_stream_in_order_with_a_cursor(api):
    reg, _ = make_registry(api, [FakeProc(nd(GROUPS, FILES, DONE))])
    job = reg.start("계약", None)
    assert wait_for(lambda: reg.get(job)["state"] == "done")
    first = reg.get(job, after=0)
    assert [l["type"] for l in first["lines"]] == ["groups", "group", "done"]
    assert first["next"] == 3
    again = reg.get(job, after=first["next"])
    assert again["lines"] == [] and again["state"] == "done"
    tail = reg.get(job, after=2)
    assert [l["type"] for l in tail["lines"]] == ["done"]


def test_non_json_stdout_is_ignored(api):
    reg, _ = make_registry(api, [FakeProc(["npm WARN banner\n"] + nd(GROUPS, DONE))])
    job = reg.start("x", None)
    assert wait_for(lambda: reg.get(job)["state"] == "done")
    assert [l["type"] for l in reg.get(job)["lines"]] == ["groups", "done"]


def test_nonzero_exit_without_done_is_an_error_with_stderr(api):
    reg, _ = make_registry(api, [FakeProc(nd(GROUPS), code=2, stderr="Unknown lite subcommand: find\n")])
    job = reg.start("x", None)
    assert wait_for(lambda: reg.get(job)["state"] == "error")
    out = reg.get(job)
    assert "Unknown lite subcommand: find" in out["error"]


def test_clean_exit_without_done_still_ends_the_job(api):
    reg, _ = make_registry(api, [FakeProc(nd(GROUPS), code=0)])
    job = reg.start("x", None)
    assert wait_for(lambda: reg.get(job)["state"] in {"done", "error"})
    assert reg.get(job)["state"] == "error"  # a find that never said done did not finish


def test_cancel_terminates_and_drops_later_output(api):
    hold = threading.Event()
    proc = FakeProc(nd(GROUPS), hold=hold)
    reg, _ = make_registry(api, [proc])
    job = reg.start("x", None)
    assert wait_for(lambda: len(reg.get(job)["lines"]) == 1)
    assert reg.cancel(job) is True
    assert wait_for(lambda: reg.get(job)["state"] == "cancelled")
    assert proc.terminated
    assert reg.cancel("nope") is False


def test_timeout_kills_a_stuck_job(api):
    hold = threading.Event()
    proc = FakeProc([], hold=hold)
    reg, _ = make_registry(api, [proc], timeout_s=0.05)
    job = reg.start("x", None)
    assert wait_for(lambda: reg.get(job)["state"] == "error")
    assert "timed out" in reg.get(job)["error"]
    assert proc.terminated


def test_starting_beyond_the_cap_evicts_the_oldest_finished_job(api):
    procs = [FakeProc(nd(GROUPS, DONE)) for _ in range(4)]
    reg, _ = make_registry(api, procs, max_jobs=3)
    ids = []
    for i in range(4):
        ids.append(reg.start(f"q{i}", None))
        assert wait_for(lambda i=i: reg.get(ids[i])["state"] == "done")
    assert reg.get(ids[0]) is None  # evicted
    assert reg.get(ids[3])["state"] == "done"


def test_cap_cancels_the_oldest_running_job_when_nothing_is_finished(api):
    holds = [threading.Event() for _ in range(3)]
    procs = [FakeProc([], hold=h) for h in holds]
    reg, _ = make_registry(api, procs, max_jobs=2)
    a = reg.start("a", None)
    reg.start("b", None)
    reg.start("c", None)
    assert wait_for(lambda: procs[0].terminated)
    assert reg.get(a) is None or reg.get(a)["state"] == "cancelled"
    for h in holds:
        h.set()


def test_unknown_job_is_none(api):
    reg, _ = make_registry(api, [])
    assert reg.get("missing") is None


def test_a_real_subprocess_streams_and_can_be_cancelled(api):
    script = (
        "import json,sys,time\n"
        "print(json.dumps({'type':'groups','groups':[{'key':'files'}]}),flush=True)\n"
        "time.sleep(30)\n"
    )

    def spawn(argv, env):
        return api.spawn_process([sys.executable, "-c", script], env)

    reg = api.FindRegistry(spawn=spawn, argv_builder=lambda q, g: ["x"])
    job = reg.start("x", None)
    assert wait_for(lambda: len(reg.get(job)["lines"]) == 1, timeout=10)
    t0 = time.time()
    assert reg.cancel(job)
    assert wait_for(lambda: reg.get(job)["state"] == "cancelled", timeout=10)
    assert time.time() - t0 < 8


# ---- HTTP routes ---------------------------------------------------------------------------------


@pytest.fixture()
def client(api):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(api.router, prefix="/api/plugins/autorag")
    reg, spawned = make_registry(api, [FakeProc(nd(GROUPS, FILES, DONE)) for _ in range(5)])
    api.set_registry(reg)
    c = TestClient(app)
    c.spawned = spawned
    return c


def test_post_find_then_poll(client):
    r = client.post("/api/plugins/autorag/find", json={"query": "계약서"})
    assert r.status_code == 200
    job = r.json()["id"]
    assert wait_for(lambda: client.get(f"/api/plugins/autorag/find/{job}").json()["state"] == "done")
    body = client.get(f"/api/plugins/autorag/find/{job}?after=1").json()
    assert [l["type"] for l in body["lines"]] == ["group", "done"]
    assert body["next"] == 3


def test_post_find_rejects_empty_query(client):
    assert client.post("/api/plugins/autorag/find", json={"query": "  "}).status_code == 422


def test_post_find_rejects_bad_group(client):
    r = client.post("/api/plugins/autorag/find", json={"query": "x", "groups": ["../x"]})
    assert r.status_code == 422


def test_get_unknown_job_404(client):
    assert client.get("/api/plugins/autorag/find/nope").status_code == 404


def test_delete_cancels(client, api):
    hold = threading.Event()
    reg, _ = make_registry(api, [FakeProc([], hold=hold)])
    api.set_registry(reg)
    job = client.post("/api/plugins/autorag/find", json={"query": "x"}).json()["id"]
    assert client.delete(f"/api/plugins/autorag/find/{job}").json() == {"cancelled": True}
    assert wait_for(lambda: client.get(f"/api/plugins/autorag/find/{job}").json()["state"] == "cancelled")
    assert client.delete("/api/plugins/autorag/find/nope").status_code == 404


def test_unavailable_node_is_503(client, api):
    def boom(q, g):
        raise api.FinderUnavailable("Node.js 24+ was not found on PATH")

    api.set_registry(api.FindRegistry(spawn=lambda a, e: None, argv_builder=boom))
    r = client.post("/api/plugins/autorag/find", json={"query": "x"})
    assert r.status_code == 503 and "Node.js" in r.json()["detail"]


def test_manifest_declares_a_hidden_backend_only_plugin(api):
    m = json.loads((ROOT / "dashboard" / "manifest.json").read_text())
    assert m["name"] == "autorag" and m["api"] == "plugin_api.py" and m["tab"]["hidden"] is True

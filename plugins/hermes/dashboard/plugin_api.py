"""AutoRAG finder backend, mounted at /api/plugins/autorag/.

The desktop half (desktop/plugin.js) asks for a "whole search": file names, local document
content and every connected datasource, filtered for relevance. The work is done by the pinned
`autorag lite find --json` CLI from @autorag/librarian, which prints one NDJSON line per result
group as each group finishes. This module only runs that child process, keeps its lines, and lets
the UI poll them with a cursor.

Nothing here reads documents or credentials. The child gets a reduced environment: PATH, HOME,
locale, proxy and AUTORAG_* variables; provider API keys in the gateway's environment are not
passed on (the CLI resolves its own decision-model credentials from ~/.autorag and Pi's auth.json).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
MAX_QUERY_CHARS = 500
JOB_TIMEOUT_S = 120.0
MAX_JOBS = 8
MAX_LINES_PER_JOB = 200
MAX_STDERR_CHARS = 2000

# `lite find` shipped in 2.13.9; older releases lack it. A librarian without it rejects the flag (or the subcommand) and then prints its
# whole usage text; show one sentence the user can act on instead.
_OLD_LIBRARIAN_RE = re.compile(r"^(?:Unknown flag: --query\b|Unknown lite subcommand: find\b)")

_GROUP_RE = re.compile(r"^(?:files|content|source:[a-z0-9][a-z0-9_.-]{0,63})$")
_ENV_KEEP = re.compile(
    r"^(?:PATH|HOME|USER|LOGNAME|SHELL|TMPDIR|TERM|LANG|LANGUAGE|LC_[A-Z]+|TZ|SSL_CERT_FILE|SSL_CERT_DIR|"
    r"NODE_EXTRA_CA_CERTS|(?:HTTPS?|NO|ALL)_PROXY|(?:https?|no|all)_proxy|XDG_[A-Z_]+|AUTORAG_[A-Z_]+|"
    r"PI_CODING_AGENT_DIR|SystemRoot|USERPROFILE|APPDATA|LOCALAPPDATA)$"
)


class FinderUnavailable(RuntimeError):
    """The finder cannot start on this machine (for example Node.js is missing)."""


class FinderNotReady(FinderUnavailable):
    """The librarian is still being installed; a search started now would only wait on the install."""

    def __init__(self, state: str, message: str = "") -> None:
        super().__init__(message or f"the AutoRAG finder is {state}")
        self.state = state

def pinned_librarian_spec() -> str:
    """The package `mcp.json` runs (the value after `-p`), so the finder and the MCP tools always run the same
    code: normally `@autorag/librarian@x.y.z`, or a tarball or path when a build is under test."""
    mcp = json.loads((PLUGIN_ROOT / "mcp.json").read_text(encoding="utf-8"))
    args = mcp["mcpServers"]["autorag"]["args"]
    for flag, value in zip(args, args[1:]):
        if flag == "-p" and isinstance(value, str) and value:
            return value
    raise FinderUnavailable("mcp.json does not name the librarian package to run (-p)")


def child_env(source: Optional[dict[str, str]] = None) -> dict[str, str]:
    env = os.environ if source is None else source
    return {k: v for k, v in env.items() if _ENV_KEEP.match(k)}


def validate_query(query: str) -> str:
    if not isinstance(query, str) or "\x00" in query:
        raise ValueError("query must be text")
    text = query.strip()
    if not text:
        raise ValueError("query is empty")
    if len(text) > MAX_QUERY_CHARS:
        raise ValueError(f"query is longer than {MAX_QUERY_CHARS} characters")
    return text


def validate_groups(groups: Optional[Sequence[str]]) -> Optional[list[str]]:
    if groups is None:
        return None
    out: list[str] = []
    for g in groups:
        if not isinstance(g, str) or not _GROUP_RE.match(g):
            raise ValueError(f"invalid group {g!r}")
        out.append(g)
    return out


RUNTIME_DIR = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))) / "plugin-data" / "autorag" / "runtime"
_CLI_REL = Path("node_modules") / "@autorag" / "librarian" / "dist" / "cli" / "index.js"
INSTALL_TIMEOUT_S = 600.0


def _npm_major(npm: str, env: dict[str, str], run: Callable[..., Any]) -> int:
    """npm's major version, or 0 when it cannot be read (treated as an old npm, which is the safe direction)."""
    try:
        out = run([npm, "--version"], cwd=None, env=env, capture_output=True, text=True, timeout=30)
        return int(str(out.stdout).strip().split(".")[0])
    except Exception:
        return 0


def default_installer(
    spec: str,
    runtime_dir: Path,
    env: dict[str, str],
    *,
    run: Callable[..., Any] = subprocess.run,
    which: Callable[..., Optional[str]] = shutil.which,
) -> None:
    """`npm install` the pinned librarian into `runtime_dir`, once, so a search never installs anything.

    The librarian depends on `xlsx` through an https tarball URL (cdn.sheetjs.com). npm 12 refuses remote tarballs
    unless `--allow-remote` says otherwise, so on npm 12+ the flag is passed; older npm does not know it and would
    reject it. Dependency lifecycle scripts stay off: the finder needs no native build step."""
    runtime_dir.mkdir(parents=True, exist_ok=True)
    pkg = runtime_dir / "package.json"
    if not pkg.exists():
        pkg.write_text(json.dumps({"name": "autorag-finder-runtime", "private": True}), encoding="utf-8")
    npm = which("npm", path=env.get("PATH"))
    if not npm:
        raise FinderUnavailable("npm was not found on PATH (the AutoRAG finder installs its librarian with it)")
    argv = [npm, "install", "--no-audit", "--no-fund", "--no-progress", "--no-update-notifier", "--ignore-scripts", "--save-exact"]
    if _npm_major(npm, env, run) >= 12:
        argv.append("--allow-remote=all")
    argv.append(spec)
    done = run(argv, cwd=str(runtime_dir), env=env, capture_output=True, text=True, timeout=INSTALL_TIMEOUT_S)
    if done.returncode != 0:
        text = (done.stderr or done.stdout or "npm install failed").strip()
        # npm prints the cause first and a log hint last; keep the cause.
        raise RuntimeError(text[:600])


class Runtime:
    """The resident librarian install: where it lives, whether it is ready, and the one place it gets installed.

    `ensure()` is idempotent and thread-safe (concurrent callers share one install); `warm()` runs it in the
    background. A search only runs against a ready install, so it never waits on npm or the registry."""

    def __init__(
        self,
        runtime_dir: Path,
        *,
        spec: Optional[str] = None,
        installer: Callable[[str, Path, dict[str, str]], None] = default_installer,
    ) -> None:
        self.dir = Path(runtime_dir)
        self._spec = spec
        self._installer = installer
        self._lock = threading.Lock()
        self._state = "idle"
        self._error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def spec(self) -> str:
        return self._spec or pinned_librarian_spec()

    @property
    def cli(self) -> Path:
        return self.dir / _CLI_REL

    def _recorded(self) -> Optional[str]:
        try:
            return json.loads((self.dir / ".installed").read_text(encoding="utf-8")).get("spec")
        except (OSError, ValueError):
            return None

    def is_ready(self) -> bool:
        return self.cli.is_file() and self._recorded() == self.spec

    def status(self) -> dict[str, Any]:
        with self._lock:
            state = "ready" if self._state != "installing" and self.is_ready() else self._state
            return {"state": state, "error": self._error, "spec": self.spec}

    def ensure(self) -> None:
        with self._lock:
            if self._state != "installing" and self.is_ready():
                self._state = "ready"
                return
            if self._state == "installing":
                waiter = self._thread
            else:
                waiter = None
                self._state, self._error = "installing", None
                self._thread = threading.current_thread()
        if waiter is not None:
            waiter.join(INSTALL_TIMEOUT_S)
            self._raise_if_failed()
            return
        try:
            self._installer(self.spec, self.dir, child_env())
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / ".installed").write_text(json.dumps({"spec": self.spec}), encoding="utf-8")
            with self._lock:
                self._state = "ready"
        except Exception as exc:
            with self._lock:
                self._state, self._error = "failed", str(exc)
            raise FinderUnavailable(f"could not install the AutoRAG finder: {exc}") from exc

    def _raise_if_failed(self) -> None:
        if self._state == "failed":
            raise FinderUnavailable(f"could not install the AutoRAG finder: {self._error}")

    def warm(self) -> None:
        """Install in the background (if needed). Never raises; `status()` reports the outcome."""
        with self._lock:
            if self._state == "installing" or (self._state != "failed" and self.is_ready()):
                return

        def run() -> None:
            try:
                self.ensure()
            except FinderUnavailable:
                pass

        threading.Thread(target=run, name="autorag-find-warm", daemon=True).start()


_runtime: Optional[Runtime] = None
_runtime_lock = threading.Lock()


def get_runtime() -> Runtime:
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = Runtime(RUNTIME_DIR)
        return _runtime


def set_runtime(runtime: Optional[Runtime]) -> None:
    global _runtime
    with _runtime_lock:
        _runtime = runtime


def build_find_argv(
    query: str,
    groups: Optional[Sequence[str]],
    *,
    env: Optional[dict[str, str]] = None,
    which: Callable[[str], Optional[str]] = shutil.which,
    runtime_dir: Optional[Path] = None,
    runtime: Optional[Runtime] = None,
) -> list[str]:
    """argv for `autorag lite find`, run with `node` against the resident install (never npx, so a search cannot
    wait on npm). The query always travels as ONE `--query=` token so text that starts with `-` is searched, never
    parsed as a flag. No shell is involved."""
    text = validate_query(query)
    wanted = validate_groups(groups)
    environment = os.environ if env is None else env
    tail = ["lite", "find", f"--query={text}", "--json"]
    if wanted:
        tail.append("--groups=" + ",".join(wanted))

    node = which("node")
    if not node:
        raise FinderUnavailable("Node.js 24+ was not found on PATH (the AutoRAG finder runs on it)")

    local_cli = environment.get("AUTORAG_LIBRARIAN_CLI", "").strip()
    if local_cli:
        return [node, local_cli, *tail]

    rt = runtime or (Runtime(runtime_dir) if runtime_dir is not None else get_runtime())
    if not rt.is_ready():
        state = rt.status()["state"]
        raise FinderNotReady("installing" if state == "installing" else state)
    return [node, str(rt.cli), *tail]


def spawn_process(argv: Sequence[str], env: dict[str, str]) -> "subprocess.Popen[str]":
    return subprocess.Popen(
        list(argv),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        start_new_session=(os.name != "nt"),
    )


def _terminate(proc: Any) -> None:
    try:
        if os.name != "nt" and isinstance(getattr(proc, "pid", None), int):
            os.killpg(os.getpgid(proc.pid), 15)
        else:
            proc.terminate()
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except Exception:
            pass


def _kill(proc: Any) -> None:
    try:
        if os.name != "nt" and isinstance(getattr(proc, "pid", None), int):
            os.killpg(os.getpgid(proc.pid), 9)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except Exception:
            pass


class _Job:
    __slots__ = ("id", "query", "proc", "lines", "state", "error", "created", "lock", "saw_done", "cancelled")

    def __init__(self, job_id: str, query: str) -> None:
        self.id = job_id
        self.query = query
        self.proc: Any = None
        self.lines: list[dict[str, Any]] = []
        self.state = "running"
        self.error: Optional[str] = None
        self.created = time.time()
        self.lock = threading.Lock()
        self.saw_done = False
        self.cancelled = False


class FindRegistry:
    """Runs `lite find` children and keeps their parsed NDJSON lines for cursor polling."""

    def __init__(
        self,
        *,
        spawn: Callable[[Sequence[str], dict[str, str]], Any] = spawn_process,
        argv_builder: Optional[Callable[[str, Optional[Sequence[str]]], list[str]]] = None,
        runtime: Optional[Runtime] = None,
        timeout_s: float = JOB_TIMEOUT_S,
        max_jobs: int = MAX_JOBS,
    ) -> None:
        self._spawn = spawn
        self._argv_builder = argv_builder or (lambda q, g: build_find_argv(q, g, runtime=runtime))
        self._timeout_s = timeout_s
        self._max_jobs = max_jobs
        self._jobs: "OrderedDict[str, _Job]" = OrderedDict()
        self._lock = threading.Lock()

    def start(self, query: str, groups: Optional[Sequence[str]]) -> str:
        argv = self._argv_builder(query, groups)  # validates; raises before anything is spawned
        job = _Job(uuid.uuid4().hex[:16], query)
        with self._lock:
            self._make_room()
            self._jobs[job.id] = job
        try:
            job.proc = self._spawn(argv, child_env())
        except OSError as exc:
            self._finish(job, "error", f"could not start the AutoRAG finder: {exc}")
            return job.id
        threading.Thread(target=self._pump, args=(job,), name=f"autorag-find-{job.id}", daemon=True).start()
        threading.Thread(target=self._watchdog, args=(job,), name=f"autorag-find-wd-{job.id}", daemon=True).start()
        return job.id

    def _make_room(self) -> None:
        while len(self._jobs) >= self._max_jobs:
            finished = next((j for j in self._jobs.values() if j.state != "running"), None)
            victim = finished or next(iter(self._jobs.values()))
            if victim.state == "running":
                self._cancel_job(victim)
            self._jobs.pop(victim.id, None)

    def _pump(self, job: _Job) -> None:
        proc = job.proc
        stderr_box: list[str] = []

        def drain_stderr() -> None:
            try:
                stderr_box.append(proc.stderr.read()[:MAX_STDERR_CHARS * 4])
            except Exception:
                pass

        err_thread = threading.Thread(target=drain_stderr, daemon=True)
        err_thread.start()
        try:
            for raw in proc.stdout:
                if job.cancelled:
                    break
                text = raw.strip()
                if not text.startswith("{"):
                    continue
                try:
                    obj = json.loads(text)
                except ValueError:
                    continue
                if not isinstance(obj, dict) or "type" not in obj:
                    continue
                with job.lock:
                    if len(job.lines) < MAX_LINES_PER_JOB:
                        job.lines.append(obj)
                    if obj.get("type") == "done":
                        job.saw_done = True
        except Exception as exc:  # reader failure ends the job visibly instead of hanging it
            self._finish(job, "error", f"finder output could not be read: {exc}")
            return
        try:
            code = proc.wait(timeout=10)
        except Exception:
            _kill(proc)
            code = -9
        err_thread.join(timeout=2)
        if job.cancelled:
            self._finish(job, "cancelled", None)
        elif job.saw_done and code == 0:
            self._finish(job, "done", None)
        else:
            raw = (stderr_box[0].strip() if stderr_box else "")[:MAX_STDERR_CHARS]
            if _OLD_LIBRARIAN_RE.match(raw):
                detail = (
                    "The installed @autorag/librarian has no `lite find` yet. Update the plugin to a release that "
                    "pins a newer @autorag/librarian, or set AUTORAG_LIBRARIAN_CLI to a build that has it."
                )
            else:
                detail = raw
            if not detail:
                detail = "finder exited without finishing" if code == 0 else f"finder exited with code {code}"
            self._finish(job, "error", detail)

    def _watchdog(self, job: _Job) -> None:
        deadline = time.time() + self._timeout_s
        while time.time() < deadline:
            if job.state != "running":
                return
            time.sleep(min(0.05, max(0.0, deadline - time.time())))
        if job.state == "running":
            self._finish(job, "error", f"finder timed out after {int(self._timeout_s)}s")
            if job.proc is not None:
                _terminate(job.proc)
                threading.Timer(3.0, _kill, args=(job.proc,)).start()

    def _finish(self, job: _Job, state: str, error: Optional[str]) -> None:
        with job.lock:
            if job.state != "running":
                return
            job.state = state
            job.error = error

    def _cancel_job(self, job: _Job) -> None:
        job.cancelled = True
        self._finish(job, "cancelled", None)
        if job.proc is not None:
            _terminate(job.proc)
            threading.Timer(3.0, _kill, args=(job.proc,)).start()

    def get(self, job_id: str, after: int = 0) -> Optional[dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return None
        with job.lock:
            start = max(0, after)
            return {
                "id": job.id,
                "state": job.state,
                "error": job.error,
                "lines": job.lines[start:],
                "next": len(job.lines),
            }

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return False
        if job.state == "running":
            self._cancel_job(job)
        return True


_registry: Optional[FindRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> FindRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = FindRegistry()
        return _registry


def set_registry(registry: Optional[FindRegistry]) -> None:
    global _registry
    with _registry_lock:
        _registry = registry


class FindRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS)
    groups: Optional[list[str]] = Field(default=None, max_length=80)


def _unprocessable(exc: Exception) -> HTTPException:
    return HTTPException(status_code=422, detail=str(exc))


@router.get("/status")
async def finder_status() -> dict[str, Any]:
    """Whether the librarian install is ready. The pane shows "preparing" from this instead of spinning on a search."""
    return get_runtime().status()


@router.post("/prepare")
async def prepare_finder() -> dict[str, Any]:
    """Start installing in the background (idempotent) and return immediately."""
    get_runtime().warm()
    return get_runtime().status()


@router.post("/find")
async def start_find(body: FindRequest) -> dict[str, str]:
    try:
        validate_query(body.query)
        validate_groups(body.groups)
    except ValueError as exc:
        raise _unprocessable(exc)
    try:
        return {"id": get_registry().start(body.query.strip(), body.groups)}
    except ValueError as exc:
        raise _unprocessable(exc)
    except FinderNotReady as exc:
        get_runtime().warm()
        raise HTTPException(status_code=503, detail={"state": exc.state, "message": str(exc)})
    except FinderUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/find/{job_id}")
async def poll_find(job_id: str, after: int = 0) -> dict[str, Any]:
    out = get_registry().get(job_id, after)
    if out is None:
        raise HTTPException(status_code=404, detail="unknown find job")
    return out


@router.delete("/find/{job_id}")
async def cancel_find(job_id: str) -> dict[str, bool]:
    if not get_registry().cancel(job_id):
        raise HTTPException(status_code=404, detail="unknown find job")
    return {"cancelled": True}

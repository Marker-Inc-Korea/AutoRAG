#!/usr/bin/env python3
"""Set up Hermes cron jobs that keep the AutoRAG index fresh and watch for duplicate files. Standard library only
(Python 3.9+, macOS and Linux).

Both jobs are Hermes *no-agent* cron jobs: the script is the job, no model runs, and a script that prints nothing
delivers nothing. This helper writes the script under `$HERMES_HOME/scripts/` and creates or updates the cron job
with the `hermes` CLI.

    index       a job that refreshes the AutoRAG index on a schedule, honouring power and time windows
    duplicates  a job that scans for duplicate documents and reports only groups it has not reported before
    list        show the jobs this helper made and the last result of each
    remove      delete one job, its script and its state

The index job calls `autorag watch --once`, so the power/window/day rules are AutoRAG's own (`refresh.schedule`),
not a copy. Nothing here moves, edits or deletes the user's files.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,40}$")
WINDOW_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)-([01]\d|2[0-3]):([0-5]\d)$")
DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
KINDS = ("index", "duplicates")
MAX_LISTED = 10
INDEX_TIMEOUT_S = 3300  # under Hermes' default 1 h script timeout, so the job reports its own timeout
SCAN_TIMEOUT_S = 1800
# Environment a cron script gets from the job; only what AutoRAG needs, never the gateway's provider keys.
KEEP_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "AUTORAG_HOME", "AUTORAG_CONFIG", "AUTORAG_WORKSPACE",
            "AUTORAG_SEARCH_PATHS")


class SetupError(Exception):
    pass


# The scripts below are rendered with their settings as one JSON document, read back with json.loads. Nothing the
# caller passes is ever spliced into code.
_COMMON = r'''
import json, os, subprocess, sys, time

CONFIG = json.loads(__CONFIG__)
STATE = CONFIG["state"]
os.makedirs(STATE, exist_ok=True)


def child_env():
    env = {k: v for k, v in os.environ.items() if k in CONFIG["keep_env"]}
    env.update(CONFIG["env"])
    return env


def run_cli(args, timeout):
    """(exit code, stdout, stderr); a CLI that cannot start or hangs is reported, never raised."""
    try:
        done = subprocess.run(CONFIG["cli"] + args, env=child_env(), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "", "timed out after %d s" % timeout
    except OSError as exc:
        return None, "", str(exc)
    return done.returncode, done.stdout, done.stderr


def tail(text, limit=400):
    text = (text or "").strip()
    return text[-limit:]


def explain(code, out, err):
    """Why a CLI call failed, in words: the `error` of a JSON body (AutoRAG prints one to stderr, some paths to
    stdout), else stderr, else stdout, else the exit code."""
    for stream in (out, err):
        try:
            body = json.loads(stream)
        except (TypeError, ValueError):
            continue
        if isinstance(body, dict) and isinstance(body.get("error"), str) and body["error"].strip():
            return tail(body["error"])
    return tail(err) or tail(out) or ("exit %s" % code)


def write_json(name, value):
    tmp = os.path.join(STATE, name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False)
    os.replace(tmp, os.path.join(STATE, name))


def read_json(name, default):
    try:
        with open(os.path.join(STATE, name), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default
'''

INDEX_TEMPLATE = _COMMON + r'''

args = ["watch", "--once", "--json"]
if CONFIG.get("power"):
    args += ["--power", CONFIG["power"]]
for window in CONFIG.get("windows") or []:
    args += ["--window", window]
if CONFIG.get("days"):
    args += ["--days", ",".join(CONFIG["days"])]

code, out, err = run_cli(args, CONFIG.get("timeout_s", __TIMEOUT__))
now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
if code == 0:
    try:
        body = json.loads(out)
    except ValueError:
        body = {}
    if body.get("skipped"):
        write_json("last.json", {"at": now, "outcome": "skipped", "reason": body.get("reason")})
    else:
        write_json("last.json", {"at": now, "outcome": "refreshed"})
    # Silent on purpose: an index that is up to date, or a window that is closed, is not news.
else:
    reason = explain(code, out, err)
    write_json("last.json", {"at": now, "outcome": "failed", "reason": reason})
    print("AutoRAG index refresh failed: " + reason)
'''

DUPLICATES_TEMPLATE = _COMMON + r'''

LIMIT = __LIMIT__
roots = CONFIG.get("paths") or [None]
groups = {}  # key -> (label, files)
failure = None
for root in roots:
    args = ["duplicates", "--json"] + ([root] if root else [])
    code, out, err = run_cli(args, CONFIG.get("timeout_s", __TIMEOUT__))
    if code != 0:
        failure = explain(code, out, err)
        break
    try:
        scan = json.loads(out)
    except ValueError:
        failure = "the scan printed no JSON"
        break
    exact_sets = set()
    for group in scan.get("exactGroups") or []:
        files = sorted(group.get("files") or [])
        exact_sets.add(tuple(files))
        groups["exact:" + str(group.get("hash"))] = ("exact", files)
    for family in scan.get("families") or []:
        files = sorted(family.get("files") or [])
        if family.get("relation") == "exact" and tuple(files) in exact_sets:
            continue  # the same files, already listed as an exact group
        groups["%s:%s" % (family.get("relation"), "|".join(files))] = (str(family.get("relation")), files)

if failure is not None:
    # Keep what was reported before, so the next good run does not repeat it.
    print("AutoRAG duplicate scan failed: " + failure)
else:
    seen = read_json("reported.json", {})
    fresh = {key: value for key, value in groups.items() if seen.get(key) != value[1]}
    write_json("reported.json", {key: value[1] for key, value in groups.items()})
    write_json("last.json", {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "outcome": "scanned",
                             "groups": len(groups), "new": len(fresh)})
    if fresh:
        lines = ["AutoRAG found %d new duplicate group(s). Nothing was moved or deleted; review them yourself." % len(fresh)]
        for index, (label, files) in enumerate(list(fresh.values())[:LIMIT], 1):
            lines.append("%d. %s" % (index, label))
            lines.extend("   - " + path for path in files)
        if len(fresh) > LIMIT:
            lines.append("... and %d more group(s)." % (len(fresh) - LIMIT))
        print("\n".join(lines))
'''


def _render(template: str, config: dict, **tokens: object) -> str:
    full = {"keep_env": list(KEEP_ENV), "env": {}, "state": "", "cli": [], **config}
    body = template.replace("__CONFIG__", repr(json.dumps(full, ensure_ascii=False)))
    for key, value in tokens.items():
        body = body.replace(f"__{key.upper()}__", str(value))
    return "#!/usr/bin/env python3\n" + body.lstrip("\n")


def render_index_script(config: dict) -> str:
    return _render(INDEX_TEMPLATE, config, timeout=INDEX_TIMEOUT_S)


def render_duplicates_script(config: dict) -> str:
    return _render(DUPLICATES_TEMPLATE, config, timeout=SCAN_TIMEOUT_S, limit=MAX_LISTED)


# ---- validation --------------------------------------------------------------------------------------------------


def valid_id(value: str) -> str:
    if not ID_RE.match(value or ""):
        raise SetupError(f"--id must match {ID_RE.pattern} (letters, digits, '-' and '_')")
    return value


def valid_windows(values: list[str] | None) -> list[str]:
    out = []
    for value in values or []:
        match = WINDOW_RE.match(value)
        if not match:
            raise SetupError(f"--window {value!r} must look like HH:MM-HH:MM (24 h, e.g. 22:00-07:00)")
        if match.group(1, 2) == match.group(3, 4):
            raise SetupError(f"--window {value!r} has equal start and end")
        out.append(value)
    return out


def valid_days(value: str | None) -> list[str]:
    if not value:
        return []
    wanted = [d.strip().lower() for d in value.split(",") if d.strip()]
    bad = [d for d in wanted if d not in DAYS]
    if bad or not wanted:
        raise SetupError(f"--days must be a comma list of {', '.join(DAYS)} (got {value!r})")
    return [d for d in DAYS if d in wanted]


def valid_power(value: str | None) -> str | None:
    if value is None:
        return None
    if value not in ("any", "ac"):
        raise SetupError("--power must be 'any' or 'ac' (ac = index only while plugged in)")
    return value


# ---- paths and the cron CLI ----------------------------------------------------------------------------------------


def hermes_home(explicit: str | None) -> Path:
    return Path(explicit or os.environ.get("HERMES_HOME") or Path.home() / ".hermes").expanduser().resolve()


def autorag_home(explicit: str | None) -> Path:
    return Path(explicit or os.environ.get("AUTORAG_HOME") or Path.home() / ".autorag").expanduser().resolve()


def state_dir(home: Path, kind: str, ident: str) -> Path:
    return home / "plugin-data" / "autorag" / "schedules" / f"{kind}-{ident}"


def job_name(kind: str, ident: str) -> str:
    return f"autorag-{kind}-{ident}"


def script_path(home: Path, kind: str, ident: str) -> Path:
    return home / "scripts" / f"{job_name(kind, ident)}.py"


def autorag_command(home: Path, explicit: list[str] | None) -> list[str]:
    """argv prefix of the AutoRAG CLI: an explicit one, the resident install the finder keeps, or `autorag`."""
    if explicit:
        return list(explicit)
    resident = (home / "plugin-data" / "autorag" / "runtime" / "node_modules" / "@autorag" / "librarian"
                / "dist" / "cli" / "index.js")
    node = shutil.which("node")
    if resident.is_file() and node:
        return [node, str(resident)]
    found = shutil.which("autorag")
    if found:
        return [found]
    raise SetupError("no AutoRAG CLI found: open the AutoRAG search pane once so it installs one, or pass --autorag-cli")


def hermes_argv(explicit: str | None) -> list[str]:
    found = explicit or shutil.which("hermes")
    if not found:
        raise SetupError("no `hermes` on PATH: pass --hermes with the launcher of the install that serves this profile")
    return [found]


def hermes_env(home: Path) -> dict[str, str]:
    return {**os.environ, "HERMES_HOME": str(home)}


def read_jobs(home: Path) -> list[dict]:
    path = home / "cron" / "jobs.json"
    if not path.exists():
        return []
    try:
        return list(json.loads(path.read_text(encoding="utf-8")).get("jobs", []))
    except (ValueError, AttributeError):
        return []


def cron(run: Callable, hermes: list[str], home: Path, *argv: str) -> str:
    done = run(hermes + ["cron", *argv], env=hermes_env(home), capture_output=True, text=True, timeout=120)
    if done.returncode != 0:
        raise SetupError(f"`hermes cron {argv[0]}` failed: {(done.stderr or done.stdout or '').strip()[-300:]}")
    return done.stdout or ""


def upsert_job(run: Callable, hermes: list[str], home: Path, name: str, script: str, schedule: str,
               deliver: str) -> str:
    existing = next((j for j in read_jobs(home) if j.get("name") == name), None)
    if existing:
        cron(run, hermes, home, "edit", "--schedule", schedule, "--deliver", deliver, existing["id"])
        return "updated"
    cron(run, hermes, home, "create", schedule, "--name", name, "--no-agent", "--script", script,
         "--deliver", deliver)
    return "created"


# ---- commands -----------------------------------------------------------------------------------------------------


def _write_script(home: Path, kind: str, ident: str, body: str) -> Path:
    path = script_path(home, kind, ident)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _install(args: argparse.Namespace, run: Callable, kind: str, body: str, extra: dict) -> dict:
    home = hermes_home(args.hermes_home)
    hermes = hermes_argv(args.hermes)
    path = _write_script(home, kind, args.id, body)
    existed_before = any(j.get("name") == job_name(kind, args.id) for j in read_jobs(home))
    try:
        action = upsert_job(run, hermes, home, job_name(kind, args.id), path.name, args.schedule, args.deliver)
    except SetupError:
        if not existed_before:
            path.unlink(missing_ok=True)
        raise
    return {"ok": True, "job": job_name(kind, args.id), "action": action, "schedule": args.schedule,
            "deliver": args.deliver, "script": str(path), "no_agent": True, **extra}


def cmd_index(args: argparse.Namespace, run: Callable = subprocess.run) -> dict:
    valid_id(args.id)
    power = valid_power(args.power)
    windows = valid_windows(args.window)
    days = valid_days(args.days)
    home = hermes_home(args.hermes_home)
    arhome = autorag_home(args.autorag_home)
    cli = autorag_command(home, args.autorag_cli)
    config = {"cli": cli, "env": {"AUTORAG_HOME": str(arhome)}, "state": str(state_dir(home, "index", args.id)),
              "power": power, "windows": windows, "days": days}
    return _install(args, run, "index", render_index_script(config),
                    {"gate": {"power": power, "windows": windows, "days": days},
                     "note": "Silent unless a refresh fails. A closed window or battery skip is recorded, not delivered."})


def cmd_duplicates(args: argparse.Namespace, run: Callable = subprocess.run) -> dict:
    valid_id(args.id)
    paths = [str(Path(p).expanduser().resolve()) for p in args.path or []]
    for p in paths:
        if not Path(p).is_dir():
            raise SetupError(f"scan folder does not exist: {p}")
    home = hermes_home(args.hermes_home)
    arhome = autorag_home(args.autorag_home)
    cli = autorag_command(home, args.autorag_cli)
    config = {"cli": cli, "env": {"AUTORAG_HOME": str(arhome)}, "state": str(state_dir(home, "duplicates", args.id)),
              "paths": paths}
    return _install(args, run, "duplicates", render_duplicates_script(config),
                    {"paths": paths or "the configured search paths",
                     "note": "Silent unless a duplicate group is new. It never moves or deletes files."})


def cmd_list(args: argparse.Namespace, run: Callable = subprocess.run) -> dict:
    home = hermes_home(args.hermes_home)
    out = []
    for job in read_jobs(home):
        name = str(job.get("name", ""))
        for kind in KINDS:
            prefix = f"autorag-{kind}-"
            if name.startswith(prefix):
                ident = name[len(prefix):]
                last = {}
                try:
                    last = json.loads((state_dir(home, kind, ident) / "last.json").read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
                out.append({"job": name, "kind": kind, "id": job.get("id"), "schedule": job.get("schedule_display"),
                            "last": last})
    return {"ok": True, "jobs": out}


def cmd_remove(args: argparse.Namespace, run: Callable = subprocess.run) -> dict:
    valid_id(args.id)
    home = hermes_home(args.hermes_home)
    name = job_name(args.kind, args.id)
    job = next((j for j in read_jobs(home) if j.get("name") == name), None)
    if job:
        cron(run, hermes_argv(args.hermes), home, "remove", job["id"])
    script = script_path(home, args.kind, args.id)
    removed_script = script.exists()
    script.unlink(missing_ok=True)
    state = state_dir(home, args.kind, args.id)
    removed_state = state.exists()
    shutil.rmtree(state, ignore_errors=True)
    return {"ok": True, "job": name, "job_removed": bool(job), "script_removed": removed_script,
            "state_removed": removed_state}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hermes-home", help="the Hermes profile directory (default: $HERMES_HOME or ~/.hermes)")
    p.add_argument("--autorag-home", help="AutoRAG home (default: $AUTORAG_HOME or ~/.autorag)")
    p.add_argument("--autorag-cli", nargs="+", help="argv prefix of the AutoRAG CLI (default: the resident install)")
    p.add_argument("--hermes", help="the `hermes` launcher that serves this profile (default: the one on PATH)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s: argparse.ArgumentParser, schedule: str) -> None:
        s.add_argument("--id", required=True)
        s.add_argument("--schedule", default=schedule, help=f"Hermes schedule: '30m', 'every 2h' or a cron "
                                                            f"expression (default: {schedule!r})")
        s.add_argument("--deliver", default="local", help="where a message goes: local, telegram, discord, ... "
                                                          "(default: local)")

    i = sub.add_parser("index", help="refresh the index on a schedule")
    common(i, "every 1h")
    i.add_argument("--power", help="any | ac: with ac, index only while plugged in")
    i.add_argument("--window", action="append", help="HH:MM-HH:MM local time, repeatable; 22:00-07:00 wraps midnight")
    i.add_argument("--days", help="comma list of mon..sun")
    i.set_defaults(func=cmd_index)

    d = sub.add_parser("duplicates", help="scan for duplicate documents on a schedule")
    common(d, "0 9 * * *")
    d.add_argument("--path", action="append", help="folder to scan (repeatable; default: the configured paths)")
    d.set_defaults(func=cmd_duplicates)

    ls = sub.add_parser("list", help="list the jobs made by this helper")
    ls.set_defaults(func=cmd_list)

    r = sub.add_parser("remove", help="remove one job, its script and its state")
    r.add_argument("--kind", required=True, choices=KINDS)
    r.add_argument("--id", required=True)
    r.set_defaults(func=cmd_remove)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = args.func(args)
    except SetupError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

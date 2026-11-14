"""Tests for plugins/hermes/skills/schedule: Hermes cron jobs that index and scan for duplicates on their own.

The jobs are no-agent cron scripts: no model runs, and a script that has nothing to say prints nothing.

Run: uv run --with pytest pytest plugins/hermes-tests -q
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "hermes" / "skills" / "schedule" / "scripts" / "autorag_schedule.py"


def load_helper():
    spec = importlib.util.spec_from_file_location("autorag_schedule_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def helper():
    return load_helper()


FAKE_CLI = '''
import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "calls.log"), "a") as fh:
    fh.write(json.dumps(sys.argv[1:]) + "\\n")
out = os.path.join(here, "stdout.txt")
sys.stdout.write(open(out).read() if os.path.exists(out) else "")
err = os.path.join(here, "stderr.txt")
sys.stderr.write(open(err).read() if os.path.exists(err) else "")
code = os.path.join(here, "code.txt")
sys.exit(int(open(code).read()) if os.path.exists(code) else 0)
'''


@pytest.fixture()
def fake(tmp_path):
    """A stand-in AutoRAG CLI whose output and exit code the test sets."""
    cli_dir = tmp_path / "cli"
    cli_dir.mkdir()
    (cli_dir / "cli.py").write_text(FAKE_CLI, encoding="utf-8")

    class Fake:
        argv = [sys.executable, str(cli_dir / "cli.py")]

        def say(self, text: str = "", err: str = "", code: int = 0) -> None:
            (cli_dir / "stdout.txt").write_text(text, encoding="utf-8")
            (cli_dir / "stderr.txt").write_text(err, encoding="utf-8")
            (cli_dir / "code.txt").write_text(str(code), encoding="utf-8")

        def calls(self) -> list[list[str]]:
            log = cli_dir / "calls.log"
            return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    return Fake()


def run_script(path: Path) -> tuple[int, str]:
    done = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=60)
    return done.returncode, done.stdout


# ---- the index job -------------------------------------------------------------------------------------------------


@pytest.fixture()
def index_script(helper, fake, tmp_path):
    state = tmp_path / "state"

    def build(**overrides):
        config = {"cli": fake.argv, "env": {"AUTORAG_HOME": str(tmp_path / "ar")}, "state": str(state),
                  "power": "ac", "windows": ["22:00-07:00"], "days": ["mon", "tue"]}
        config.update(overrides)
        script = tmp_path / "index.py"
        script.write_text(helper.render_index_script(config), encoding="utf-8")
        return script

    return build, state


def test_the_index_job_hands_the_schedule_to_autorags_own_gate(index_script, fake):
    build, _ = index_script
    fake.say(json.dumps({"ok": True, "skipped": True, "reason": "battery"}))
    run_script(build())
    assert fake.calls() == [["watch", "--once", "--json", "--power", "ac", "--window", "22:00-07:00", "--days", "mon,tue"]]


def test_the_index_job_omits_flags_it_was_not_given(index_script, fake):
    build, _ = index_script
    fake.say(json.dumps({"ok": True}))
    run_script(build(power=None, windows=[], days=[]))
    assert fake.calls() == [["watch", "--once", "--json"]]


def test_each_window_becomes_its_own_flag(index_script, fake):
    build, _ = index_script
    run_script(build(windows=["22:00-07:00", "12:00-13:00"], days=[], power=None))
    assert fake.calls()[0].count("--window") == 2


def test_a_successful_or_skipped_index_run_is_silent(index_script, fake):
    build, _ = index_script
    for body in (json.dumps({"ok": True}), json.dumps({"ok": True, "skipped": True, "reason": "outside-window"})):
        fake.say(body)
        assert run_script(build()) == (0, "")


def test_a_failed_index_run_says_why_and_still_exits_cleanly(index_script, fake):
    build, _ = index_script
    fake.say("", err="refresh exploded: disk full", code=1)
    code, out = run_script(build())
    assert code == 0  # a cron script that exits non-zero would hide its own message
    assert "AutoRAG index refresh failed" in out and "disk full" in out


def test_a_json_error_body_is_shown_as_its_message_not_as_raw_json(index_script, fake):
    build, state = index_script
    fake.say(json.dumps({"ok": False, "error": "Failed to parse config file: line 1"}), code=1)
    _, out = run_script(build())
    assert "Failed to parse config file: line 1" in out
    assert '"ok"' not in out and "{" not in out
    assert json.loads((state / "last.json").read_text())["reason"] == "Failed to parse config file: line 1"


def test_the_real_clis_json_error_on_stderr_is_also_shown_as_its_message(index_script, fake):
    """`autorag` prints its error JSON (multi-line, indented) to stderr, not stdout."""
    build, state = index_script
    body = json.dumps({"ok": False, "error": "Failed to parse config file: line 1"}, indent=2)
    fake.say("", err=body, code=1)
    _, out = run_script(build())
    assert out.strip() == "AutoRAG index refresh failed: Failed to parse config file: line 1"
    assert json.loads((state / "last.json").read_text())["reason"] == "Failed to parse config file: line 1"


def test_a_scan_failure_on_stderr_json_is_also_plain_text(dupes_script, fake):
    build, _ = dupes_script
    fake.say("", err=json.dumps({"ok": False, "error": "root not readable"}, indent=2), code=1)
    assert run_script(build())[1].strip() == "AutoRAG duplicate scan failed: root not readable"


def test_a_missing_cli_is_reported_not_raised(index_script):
    build, _ = index_script
    code, out = run_script(build(cli=["/nonexistent/autorag"]))
    assert code == 0 and "AutoRAG index refresh failed" in out


def test_the_index_job_records_its_last_result(index_script, fake):
    build, state = index_script
    fake.say(json.dumps({"ok": True, "skipped": True, "reason": "battery"}))
    run_script(build())
    last = json.loads((state / "last.json").read_text())
    assert last["outcome"] == "skipped" and last["reason"] == "battery" and "at" in last


def test_a_refresh_that_hangs_is_cut_off(helper, fake, tmp_path):
    config = {"cli": fake.argv, "env": {}, "state": str(tmp_path / "s"), "power": None, "windows": [], "days": [],
              "timeout_s": 1}
    (tmp_path / "cli" / "cli.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    script = tmp_path / "index.py"
    script.write_text(helper.render_index_script(config), encoding="utf-8")
    code, out = run_script(script)
    assert code == 0 and "timed out" in out


# ---- the duplicates job --------------------------------------------------------------------------------------------


def scan(exact=(), families=(), errors=()):
    return json.dumps({
        "ok": True, "roots": ["/r"],
        "exactGroups": [{"hash": h, "files": list(files)} for h, files in exact],
        "families": [{"id": i, "relation": rel, "files": list(files)} for i, (rel, files) in enumerate(families)],
        "extractionErrors": list(errors), "action": "review",
    })


@pytest.fixture()
def dupes_script(helper, fake, tmp_path):
    state = tmp_path / "dstate"

    def build(**overrides):
        config = {"cli": fake.argv, "env": {}, "state": str(state), "paths": []}
        config.update(overrides)
        script = tmp_path / "dupes.py"
        script.write_text(helper.render_duplicates_script(config), encoding="utf-8")
        return script

    return build, state


def test_no_duplicates_is_silent(dupes_script, fake):
    build, _ = dupes_script
    fake.say(scan())
    assert run_script(build()) == (0, "")


def test_new_exact_duplicates_are_reported_with_their_files(dupes_script, fake):
    build, _ = dupes_script
    fake.say(scan(exact=[("h1", ["/d/a.pdf", "/d/b.pdf"])]))
    code, out = run_script(build())
    assert code == 0
    assert "/d/a.pdf" in out and "/d/b.pdf" in out and "1 new" in out


def test_the_same_duplicates_are_not_reported_twice(dupes_script, fake):
    build, _ = dupes_script
    fake.say(scan(exact=[("h1", ["/d/a.pdf", "/d/b.pdf"])]))
    script = build()
    assert run_script(script)[1] != ""
    assert run_script(script) == (0, "")


def test_only_the_new_group_is_reported_on_a_later_run(dupes_script, fake):
    build, _ = dupes_script
    script = build()
    fake.say(scan(exact=[("h1", ["/d/a.pdf", "/d/b.pdf"])]))
    run_script(script)
    fake.say(scan(exact=[("h1", ["/d/a.pdf", "/d/b.pdf"]), ("h2", ["/d/x.md", "/d/y.md"])]))
    out = run_script(script)[1]
    assert "/d/x.md" in out and "/d/a.pdf" not in out


def test_a_group_that_grows_is_reported_again(dupes_script, fake):
    build, _ = dupes_script
    script = build()
    fake.say(scan(exact=[("h1", ["/d/a.pdf", "/d/b.pdf"])]))
    run_script(script)
    fake.say(scan(exact=[("h1", ["/d/a.pdf", "/d/b.pdf", "/d/c.pdf"])]))
    assert "/d/c.pdf" in run_script(script)[1]


def test_a_group_that_went_away_and_came_back_is_reported_again(dupes_script, fake):
    build, _ = dupes_script
    script = build()
    group = scan(exact=[("h1", ["/d/a.pdf", "/d/b.pdf"])])
    fake.say(group)
    run_script(script)
    fake.say(scan())
    assert run_script(script) == (0, "")
    fake.say(group)
    assert "/d/a.pdf" in run_script(script)[1]


def test_near_duplicate_families_are_reported_but_exact_ones_are_not_counted_twice(dupes_script, fake):
    build, _ = dupes_script
    fake.say(scan(exact=[("h1", ["/d/a.docx", "/d/b.docx"])],
                  families=[("exact", ["/d/a.docx", "/d/b.docx"]), ("near", ["/d/r1.hwp", "/d/r2.hwp"])]))
    out = run_script(build())[1]
    assert "2 new" in out and "near" in out and "/d/r1.hwp" in out


def test_the_report_never_suggests_deleting_and_says_nothing_was_changed(dupes_script, fake):
    build, _ = dupes_script
    fake.say(scan(exact=[("h1", ["/d/a.pdf", "/d/b.pdf"])]))
    out = run_script(build())[1].lower()
    assert "nothing was moved or deleted" in out
    assert "rm " not in out and "delete these" not in out


def test_a_long_list_is_capped_and_the_rest_counted(dupes_script, fake):
    build, _ = dupes_script
    fake.say(scan(exact=[(f"h{i}", [f"/d/{i}a", f"/d/{i}b"]) for i in range(30)]))
    out = run_script(build())[1]
    assert "30 new" in out and "20 more" in out and "/d/0a" in out and "/d/29a" not in out


def test_extraction_errors_alone_do_not_wake_the_user(dupes_script, fake):
    build, _ = dupes_script
    fake.say(scan(errors=[{"path": "/d/broken.pdf", "message": "extract failed"}]))
    assert run_script(build()) == (0, "")


def test_a_failed_scan_says_so_and_keeps_the_previous_state(dupes_script, fake):
    build, state = dupes_script
    script = build()
    fake.say(scan(exact=[("h1", ["/d/a.pdf", "/d/b.pdf"])]))
    run_script(script)
    fake.say("", err="dupey exploded", code=1)
    code, out = run_script(script)
    assert code == 0 and "AutoRAG duplicate scan failed" in out and "dupey exploded" in out
    fake.say(scan(exact=[("h1", ["/d/a.pdf", "/d/b.pdf"])]))
    assert run_script(script) == (0, "")  # the earlier report was remembered through the failure


def test_each_configured_path_is_scanned(dupes_script, fake):
    build, _ = dupes_script
    fake.say(scan())
    run_script(build(paths=["/p1", "/p2"]))
    assert fake.calls() == [["duplicates", "--json", "/p1"], ["duplicates", "--json", "/p2"]]


def test_without_paths_the_configured_roots_are_scanned(dupes_script, fake):
    build, _ = dupes_script
    fake.say(scan())
    run_script(build())
    assert fake.calls() == [["duplicates", "--json"]]


# ---- validation ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "-x", "a b", "a/b", "x" * 60, "../up"])
def test_ids_that_could_escape_the_scripts_dir_are_rejected(helper, bad):
    with pytest.raises(helper.SetupError):
        helper.valid_id(bad)


@pytest.mark.parametrize("bad", ["25:00-07:00", "22:00", "22:00-22:00", "7:00-8:00", "22:00-07:60", "evening"])
def test_bad_windows_are_rejected(helper, bad):
    with pytest.raises(helper.SetupError):
        helper.valid_windows([bad])


def test_a_window_that_wraps_midnight_is_accepted(helper):
    assert helper.valid_windows(["22:00-07:00"]) == ["22:00-07:00"]


@pytest.mark.parametrize("bad", ["funday", "mon,xyz", ","])
def test_bad_days_are_rejected(helper, bad):
    with pytest.raises(helper.SetupError):
        helper.valid_days(bad)


def test_days_are_normalised_and_deduplicated_in_week_order(helper):
    assert helper.valid_days("Fri, mon,mon") == ["mon", "fri"]


def test_power_is_any_or_ac(helper):
    assert helper.valid_power("ac") == "ac" and helper.valid_power(None) is None
    with pytest.raises(helper.SetupError):
        helper.valid_power("battery")


# ---- creating and removing the cron jobs ---------------------------------------------------------------------------


class FakeHermes:
    """Records `hermes cron ...` calls and keeps a jobs.json like Hermes does."""

    def __init__(self, home: Path):
        self.home = home
        self.calls: list[list[str]] = []
        (home / "cron").mkdir(parents=True, exist_ok=True)

    def jobs(self) -> list[dict]:
        path = self.home / "cron" / "jobs.json"
        return json.loads(path.read_text())["jobs"] if path.exists() else []

    def __call__(self, argv, **_):
        self.calls.append(list(argv))
        jobs = self.jobs()
        sub = argv[argv.index("cron") + 1]
        if sub == "create":
            name = argv[argv.index("--name") + 1]
            jobs.append({"id": f"job{len(jobs) + 1}", "name": name, "script": argv[argv.index("--script") + 1]})
        elif sub in ("remove", "rm"):
            jobs = [j for j in jobs if j["id"] != argv[-1]]
        (self.home / "cron" / "jobs.json").write_text(json.dumps({"jobs": jobs}))
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")


@pytest.fixture()
def env(helper, fake, tmp_path):
    hermes_home = tmp_path / "hh"
    (hermes_home / "scripts").mkdir(parents=True)
    runner = FakeHermes(hermes_home)

    def call(*argv):
        args = helper.parser().parse_args(["--hermes-home", str(hermes_home), "--autorag-home", str(tmp_path / "ar"),
                                           "--autorag-cli", *fake.argv, "--hermes", "hermes", *argv])
        return args.func(args, run=runner)

    return call, runner, hermes_home


def test_setup_index_writes_the_script_and_creates_a_no_agent_job(env):
    call, hermes, home = env
    out = call("index", "--id", "main", "--schedule", "every 30m", "--power", "ac", "--window", "22:00-07:00")
    assert out["ok"] and out["job"] == "autorag-index-main"
    assert (home / "scripts" / "autorag-index-main.py").is_file()
    create = next(c for c in hermes.calls if "create" in c)
    assert create[create.index("--name") + 1] == "autorag-index-main"
    assert create[create.index("--script") + 1] == "autorag-index-main.py"
    assert "--no-agent" in create and create[create.index("--deliver") + 1] == "local"
    assert "every 30m" in create


def test_the_index_script_that_was_written_runs_and_carries_the_schedule(env, fake):
    call, _, home = env
    call("index", "--id", "main", "--power", "ac", "--days", "sat,sun")
    fake.say(json.dumps({"ok": True}))
    run_script(home / "scripts" / "autorag-index-main.py")
    assert fake.calls() == [["watch", "--once", "--json", "--power", "ac", "--days", "sat,sun"]]


def test_setup_is_idempotent_it_edits_the_job_instead_of_duplicating_it(env):
    call, hermes, _ = env
    call("index", "--id", "main", "--schedule", "every 30m")
    out = call("index", "--id", "main", "--schedule", "every 2h", "--power", "ac")
    assert [j["name"] for j in hermes.jobs()] == ["autorag-index-main"]
    edit = next(c for c in hermes.calls if "edit" in c)
    assert "every 2h" in edit and out["action"] == "updated"


def test_setup_duplicates_creates_its_own_job(env):
    call, hermes, home = env
    out = call("duplicates", "--id", "docs", "--schedule", "0 9 * * *", "--path", str(home))
    assert out["job"] == "autorag-duplicates-docs"
    assert (home / "scripts" / "autorag-duplicates-docs.py").is_file()
    assert any("autorag-duplicates-docs" in c for c in hermes.calls)


def test_a_missing_scan_folder_is_refused_before_anything_is_created(env):
    call, hermes, home = env
    with pytest.raises(Exception, match="does not exist"):
        call("duplicates", "--id", "docs", "--path", "/no/such/folder/at/all")
    assert hermes.calls == [] and list((home / "scripts").iterdir()) == []


def test_bad_window_is_refused_before_anything_is_created(env):
    call, hermes, home = env
    with pytest.raises(Exception, match="window"):
        call("index", "--id", "main", "--window", "25:00-07:00")
    assert hermes.calls == [] and list((home / "scripts").iterdir()) == []


def test_a_failing_cron_create_removes_the_script_it_wrote(helper, env):
    call, hermes, home = env

    def broken(argv, **_):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="cron exploded")

    args = helper.parser().parse_args(["--hermes-home", str(home), "--autorag-cli", "x", "--hermes", "hermes",
                                       "index", "--id", "main"])
    with pytest.raises(helper.SetupError, match="cron exploded"):
        args.func(args, run=broken)
    assert list((home / "scripts").iterdir()) == []


def test_list_shows_the_jobs_with_their_last_result(env, fake):
    call, _, home = env
    call("index", "--id", "main", "--power", "ac")
    fake.say(json.dumps({"ok": True, "skipped": True, "reason": "battery"}))
    run_script(home / "scripts" / "autorag-index-main.py")
    jobs = call("list")["jobs"]
    assert jobs[0]["job"] == "autorag-index-main" and jobs[0]["last"]["reason"] == "battery"


def test_remove_deletes_the_job_the_script_and_the_state(env):
    call, hermes, home = env
    call("index", "--id", "main")
    out = call("remove", "--kind", "index", "--id", "main")
    assert out["job_removed"] and hermes.jobs() == []
    assert list((home / "scripts").iterdir()) == []


def test_remove_leaves_other_jobs_alone(env):
    call, hermes, home = env
    call("index", "--id", "main")
    call("duplicates", "--id", "docs")
    call("remove", "--kind", "index", "--id", "main")
    assert [j["name"] for j in hermes.jobs()] == ["autorag-duplicates-docs"]
    assert (home / "scripts" / "autorag-duplicates-docs.py").is_file()


def test_remove_of_something_that_does_not_exist_is_not_an_error(env):
    call, _, _ = env
    out = call("remove", "--kind", "index", "--id", "ghost")
    assert out["ok"] and not out["job_removed"]


def test_the_generated_scripts_embed_values_as_data_not_code(helper, tmp_path):
    hostile = "x'); import os; os.system('touch " + str(tmp_path / "pwned") + "'); ('"
    script = helper.render_index_script({"cli": [sys.executable, "-c", "pass"], "env": {"AUTORAG_HOME": hostile},
                                         "state": str(tmp_path / "s"), "power": None, "windows": [], "days": []})
    path = tmp_path / "s.py"
    path.write_text(script, encoding="utf-8")
    run_script(path)
    assert not (tmp_path / "pwned").exists()

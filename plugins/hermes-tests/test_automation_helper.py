"""Tests for the generated gate script of plugins/hermes/skills/automate (the part Hermes cron runs).

Run: uv run --with pytest pytest plugins/hermes-tests -q
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "hermes" / "skills" / "automate" / "scripts" / "autorag_automation.py"


def load_helper():
    spec = importlib.util.spec_from_file_location("autorag_automation_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def gate(tmp_path):
    helper = load_helper()
    state = tmp_path / "state"
    (state / "queue").mkdir(parents=True)
    script = tmp_path / "gate.py"
    script.write_text(helper.GATE_TEMPLATE.format(state=str(state), limit=3), encoding="utf-8")

    def enqueue(name: str, events: list[dict]) -> None:
        (state / "queue" / name).write_text(json.dumps(events), encoding="utf-8")

    def run() -> list[str]:
        done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=True)
        return done.stdout.strip().splitlines()

    return state, enqueue, run


def test_empty_queue_skips_the_agent(gate):
    _, _, run = gate
    assert [json.loads(line) for line in run()] == [{"wakeAgent": False}]


def test_files_that_vanished_before_the_gate_ran_are_dropped(gate, tmp_path):
    _, enqueue, run = gate
    gone = tmp_path / "gone.pdf"
    enqueue("1.json", [{"type": "create", "path": str(gone)}])
    assert [json.loads(line) for line in run()] == [{"wakeAgent": False}]


def test_delete_events_survive_even_though_the_file_is_gone(gate, tmp_path):
    _, enqueue, run = gate
    gone = tmp_path / "removed.pdf"
    enqueue("1.json", [{"type": "delete", "path": str(gone)}])
    out = run()
    assert json.loads(out[-1]) == {"event": "delete", "path": str(gone)}


def test_newest_event_for_a_path_wins_across_batches(gate, tmp_path):
    _, enqueue, run = gate
    file = tmp_path / "a.pdf"
    file.write_text("x")
    enqueue("1.json", [{"type": "create", "path": str(file)}])
    enqueue("2.json", [{"type": "update", "path": str(file)}])
    events = [json.loads(line) for line in run() if line.startswith("{")]
    assert events == [{"event": "update", "path": str(file)}]


def test_a_batch_is_handed_over_once(gate, tmp_path):
    state, enqueue, run = gate
    file = tmp_path / "a.pdf"
    file.write_text("x")
    enqueue("1.json", [{"type": "create", "path": str(file)}])
    assert any(line.startswith("{") for line in run())
    assert [json.loads(line) for line in run()] == [{"wakeAgent": False}]
    assert list((state / "queue").iterdir()) == []


def test_listing_is_capped_and_the_overflow_is_counted(gate, tmp_path):
    _, enqueue, run = gate
    paths = []
    for i in range(5):
        file = tmp_path / f"f{i}.pdf"
        file.write_text("x")
        paths.append(str(file))
    enqueue("1.json", [{"type": "create", "path": p} for p in paths])
    lines = [json.loads(line) for line in run() if line.startswith("{")]
    assert len([x for x in lines if "event" in x]) == 3
    assert {"not_listed": 2} in lines


def test_a_corrupt_batch_does_not_block_later_ones(gate, tmp_path):
    state, enqueue, run = gate
    file = tmp_path / "ok.pdf"
    file.write_text("x")
    (state / "queue" / "1.json").write_text("{not json", encoding="utf-8")
    enqueue("2.json", [{"type": "create", "path": str(file)}])
    assert {"event": "create", "path": str(file)} in [json.loads(line) for line in run() if line.startswith("{")]


@pytest.mark.parametrize("bad", ["", "-x", "a b", "a/b", "x" * 60, "../up"])
def test_rule_ids_that_could_escape_the_state_dir_are_rejected(bad):
    helper = load_helper()
    with pytest.raises(helper.SetupError):
        helper.valid_id(bad)


def test_fire_script_queues_stdin_and_invokes_hermes_without_shell(tmp_path):
    helper = load_helper()
    state = tmp_path / "state"
    hermes_home = tmp_path / "hh"
    hermes_home.mkdir()
    recorded = tmp_path / "calls.json"
    fake_hermes = tmp_path / "fake_hermes.py"
    fake_hermes.write_text(
        "import json, os, sys\n"
        f"open({str(recorded)!r}, 'w', encoding='utf-8').write(json.dumps({{'argv': sys.argv[1:], 'hermes_home': os.environ.get('HERMES_HOME')}}))\n",
        encoding="utf-8",
    )
    fire = state / "fire.py"
    state.mkdir(parents=True)
    fire.write_text(
        helper.render_fire_script(state, hermes_home, [sys.executable, str(fake_hermes)], fake_hermes, "autorag-inv"),
        encoding="utf-8",
    )
    payload = json.dumps([{"type": "create", "path": "/tmp/a.pdf"}])
    done = subprocess.run([sys.executable, str(fire)], input=payload, capture_output=True, text=True, check=True)
    assert done.returncode == 0
    queued = list((state / "queue").glob("*.json"))
    assert len(queued) == 1
    assert json.loads(queued[0].read_text(encoding="utf-8")) == [{"type": "create", "path": "/tmp/a.pdf"}]
    call = json.loads(recorded.read_text(encoding="utf-8"))
    assert call["argv"] == ["cron", "run", "autorag-inv"]
    assert call["hermes_home"] == str(hermes_home)


def test_hermes_environment_pythons_discovers_both_posix_and_windows_venvs(tmp_path):
    helper = load_helper()
    posix_py = tmp_path / "installs" / "a" / "environments" / "1" / "venv" / "bin" / "python"
    win_py = tmp_path / "installs" / "b" / "environments" / "2" / "venv" / "Scripts" / "python.exe"
    for p in (posix_py, win_py):
        p.parent.mkdir(parents=True)
        p.write_text("", encoding="utf-8")
    found = helper.hermes_environment_pythons(tmp_path)
    assert str(posix_py) in found and str(win_py) in found

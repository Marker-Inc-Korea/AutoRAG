"""Tests for plugins/hermes/skills/setup/scripts/autorag_inventory.py: the read-only look at a folder before indexing.

Run: uv run --with pytest pytest plugins/hermes-tests -q
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "hermes" / "skills" / "setup" / "scripts" / "autorag_inventory.py"


def load():
    spec = importlib.util.spec_from_file_location("autorag_inventory_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def inv():
    return load()


def touch(path: Path, text: str = "x", size: int | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size if size is not None else text.encode())
    return path


@pytest.fixture()
def tree(tmp_path):
    root = tmp_path / "Documents"
    touch(root / "계약" / "lease.pdf")
    touch(root / "계약" / "memo.md")
    touch(root / "report.docx")
    touch(root / "sheet.xlsx")
    touch(root / "notes.txt")
    for i in range(30):
        touch(root / "app" / "node_modules" / "lodash" / f"f{i}.js")
    for i in range(40):
        touch(root / "app" / "src" / f"c{i}.ts")
    touch(root / "app" / ".git" / "HEAD")
    for i in range(25):
        touch(root / "Downloads" / f"IMG_{i}.png")
    touch(root / "movie.mp4")
    touch(root / "archive.zip")
    return root


def test_counts_what_would_be_indexed_by_kind(inv, tree):
    out = inv.inventory(tree)
    assert out["indexable"]["pdf"] == 1
    assert out["indexable"]["md"] == 1
    assert out["indexable"]["docx"] == 1
    assert out["indexable"]["xlsx"] == 1
    assert out["indexable"]["txt"] == 1


def test_files_autorag_skips_anyway_are_not_counted_as_indexable(inv, tree):
    out = inv.inventory(tree)
    assert "js" not in out["indexable"] or out["indexable"]["js"] == 0
    assert out["indexable_total"] == sum(out["indexable"].values())
    assert "mp4" not in out["indexable"] and "zip" not in out["indexable"]
    assert out["unsupported_total"] >= 2


def test_autorags_own_skip_dirs_are_reported_as_skipped_not_scanned(inv, tree):
    out = inv.inventory(tree)
    names = {s["dir"] for s in out["skipped_dirs"]}
    assert "node_modules" in names and ".git" in names


def test_a_folder_that_is_mostly_images_is_flagged_as_noise_with_a_glob(inv, tree):
    out = inv.inventory(tree)
    noise = {n["folder"]: n for n in out["noise"]}
    assert "Downloads" in noise
    assert noise["Downloads"]["files"] == 25
    assert noise["Downloads"]["suggested_exclude"] == f"{tree.resolve().as_posix()}/Downloads/**"


def test_source_code_folders_are_flagged_so_a_project_does_not_flood_the_index(inv, tree):
    out = inv.inventory(tree)
    code = [n for n in out["noise"] if n["reason"] == "source-code"]
    assert any(n["folder"] == "app/src" or n["folder"] == "app" for n in code)


def test_a_suggested_glob_is_anchored_to_the_real_folder_not_a_bare_name(inv, tmp_path):
    """`**/src/**` would also exclude an unrelated documents folder that happens to be called src."""
    root = tmp_path / "Documents"
    for i in range(40):
        touch(root / "app" / "src" / f"c{i}.ts")
    touch(root / "other" / "src" / "contract.pdf")
    out = inv.inventory(root)
    globs = out["recommended_excludes"]
    assert f"{root.resolve().as_posix()}/app/src/**" in globs
    assert "**/src/**" not in globs


def test_a_folder_of_many_small_same_type_files_is_reported_as_bulk_even_if_the_type_is_indexable(inv, tmp_path):
    """Receipts, logs and exports are .txt/.csv, which AutoRAG can parse, but 180 of them is a pile, not a library."""
    root = tmp_path / "Documents"
    for i in range(60):
        touch(root / "Downloads" / f"receipt_{i}.txt", text="영수증")
    touch(root / "계약" / "lease.pdf")
    out = inv.inventory(root)
    bulk = {n["folder"]: n for n in out["noise"] if n["reason"] == "bulk"}
    assert bulk["Downloads"]["files"] == 60
    assert out["keep"] == ["계약"] or "계약" in out["keep"]


def test_a_large_folder_of_varied_documents_is_not_called_bulk(inv, tmp_path):
    root = tmp_path / "Documents"
    kinds = [".pdf", ".docx", ".xlsx", ".md", ".pptx"]
    for i in range(60):
        touch(root / "reports" / f"r{i}{kinds[i % len(kinds)]}")
    out = inv.inventory(root)
    assert not [n for n in out["noise"] if n["folder"] == "reports"]


def test_the_recommendation_leads_with_the_document_folders(inv, tree):
    out = inv.inventory(tree)
    assert "계약" in " ".join(out["keep"]) or out["keep"] == ["."]
    assert out["recommended_excludes"], "noise folders become exclude globs"
    assert all(g.startswith("/") and g.endswith("/**") for g in out["recommended_excludes"])


def test_the_scan_is_read_only(inv, tree):
    before = sorted(str(p) for p in tree.rglob("*"))
    mtimes = {p: p.stat().st_mtime_ns for p in tree.rglob("*") if p.is_file()}
    inv.inventory(tree)
    assert sorted(str(p) for p in tree.rglob("*")) == before
    assert {p: p.stat().st_mtime_ns for p in tree.rglob("*") if p.is_file()} == mtimes


def test_symlinks_are_not_followed_out_of_the_root(inv, tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    touch(outside / "secret.md")
    touch(root / "a.md")
    os.symlink(outside, root / "link")
    out = inv.inventory(root)
    assert out["indexable"].get("md") == 1


def test_a_huge_tree_is_capped_and_says_so(inv, tmp_path):
    root = tmp_path / "big"
    for i in range(60):
        touch(root / f"f{i}.md")
    out = inv.inventory(root, max_entries=20)
    assert out["truncated"] is True
    assert out["scanned"] <= 20


def test_a_file_over_the_size_cap_is_counted_as_too_large(inv, tmp_path):
    root = tmp_path / "r"
    touch(root / "small.pdf")
    touch(root / "huge.pdf", size=inv.MAX_SOURCE_BYTES + 1)
    out = inv.inventory(root)
    assert out["indexable"]["pdf"] == 1
    assert out["too_large"] == 1


def test_a_missing_folder_is_an_error_not_a_crash(inv, tmp_path):
    with pytest.raises(inv.InventoryError, match="does not exist"):
        inv.inventory(tmp_path / "nope")


def test_a_permission_hang_is_reported_not_waited_on(inv, tmp_path, monkeypatch):
    root = tmp_path / "locked"
    touch(root / "a.md")

    def hang(*_a, **_k):
        raise TimeoutError

    monkeypatch.setattr(inv, "_list_dir", hang)
    out = inv.inventory(root, per_dir_timeout_s=0.01)
    assert out["unreadable"] and out["indexable_total"] == 0


def test_the_cli_prints_json_and_exits_zero(tree):
    done = subprocess.run([sys.executable, str(SCRIPT), str(tree)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0
    body = json.loads(done.stdout)
    assert body["ok"] is True and body["indexable_total"] >= 5


def test_the_cli_reports_a_bad_folder_as_json_with_exit_one(tmp_path):
    done = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / "nope")], capture_output=True, text=True, timeout=60)
    assert done.returncode == 1
    assert json.loads(done.stdout)["ok"] is False


def test_each_recommended_glob_covers_its_noise_folder_and_nothing_else(inv, tree):
    """`<abs folder>/**` must exclude every file under the noise folder and leave documents alone. Checked with
    Node's own matchesGlob, the matcher AutoRAG applies to `excludePaths`."""
    out = inv.inventory(tree)
    assert out["recommended_excludes"]
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    files = [p.resolve().as_posix() for p in tree.rglob("*") if p.is_file()]
    script = (
        "const {matchesGlob}=require('node:path');const [globs,files]=JSON.parse(process.argv[1]);"
        "console.log(JSON.stringify(files.filter(f=>globs.some(g=>matchesGlob(f,g)))))"
    )
    done = subprocess.run([node, "-e", script, json.dumps([out["recommended_excludes"], files])],
                          capture_output=True, text=True, timeout=30)
    excluded = set(json.loads(done.stdout))
    root = tree.resolve().as_posix()
    assert f"{root}/계약/lease.pdf" not in excluded and f"{root}/report.docx" not in excluded
    assert all("/Downloads/" in f or "/app/" in f for f in excluded)
    assert any("/Downloads/" in f for f in excluded)

#!/usr/bin/env python3
"""Look at a folder before AutoRAG indexes it: what would be indexed, what is noise, what to exclude.

Read-only and standard library only (Python 3.9+). It lists directories and stats files; it never opens a file's
contents and never writes anything. Output is one JSON object.

    autorag_inventory.py FOLDER [--max-entries N]

The point is to index the documents that matter. A Documents folder usually also holds a project with
`node_modules`, a Downloads pile of screenshots, and exported media; indexing those costs minutes and fills search
results with noise. The recommended excludes use AutoRAG's own `excludePaths` glob form (`**/name/**`).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Any

# Files AutoRAG can parse (src/parser: plain text, csv, spreadsheet, office, kordoc, eml). Images need OCR, which
# is off unless the user enables it, so they are counted as not indexable.
INDEXABLE = {
    ".txt", ".text", ".md", ".markdown", ".csv", ".tsv", ".xlsx", ".xlsm", ".xlsb", ".xls", ".ods", ".pptx", ".eml",
    ".hwp", ".hwpx", ".hml", ".hwpml", ".pdf", ".docx",
}
# Directories AutoRAG's mirror never descends into (src/mirror/sync.ts SKIP_DIR_NAMES).
AUTORAG_SKIPS = {
    ".autorag", ".git", ".jikji", ".cache", ".venv", "__pycache__", "node_modules", "dist", "build", "target",
    ".Trash", ".Spotlight-V100", ".DocumentRevisions-V100", ".fseventsd", "CacheStorage", "Service Worker",
    "Code Cache", "GPUCache", "GrShaderCache", "ShaderCache",
}
MAX_SOURCE_BYTES = 50 * 1024 * 1024  # src/mirror/sync.ts DEFAULT_MAX_SOURCE_BYTES
CODE_EXTENSIONS = {
    ".js", ".jsx", ".ts", ".tsx", ".py", ".go", ".rs", ".java", ".c", ".h", ".cpp", ".hpp", ".cs", ".rb", ".php",
    ".swift", ".kt", ".json", ".yaml", ".yml", ".toml", ".lock", ".map", ".css", ".scss",
}
MEDIA_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".heic", ".bmp", ".tiff", ".mp4", ".mov", ".mkv", ".mp3",
                    ".wav", ".m4a", ".avi"}
DEFAULT_MAX_ENTRIES = 200_000
# A folder is noise when it holds at least this many files and this share of them is not a document.
NOISE_MIN_FILES = 20
NOISE_SHARE = 0.8
# How many file names per folder are kept to judge naming patterns, and how many patterns still count as one pile.
NAME_SAMPLE = 200
PILE_MAX_PATTERNS = 2
# Plain-text-like types that pile up as exports, logs and receipts. PDF/DOCX/XLSX are not here: a folder of 60
# PDFs is a real library.
BULK_PRONE = {".txt", ".text", ".csv", ".tsv", ".eml", ".md", ".markdown"}


class InventoryError(Exception):
    pass


def _list_dir(path: Path) -> list[os.DirEntry]:
    with os.scandir(path) as it:
        return list(it)


def _bounded_list(path: Path, timeout_s: float) -> list[os.DirEntry]:
    """List a directory, giving up after `timeout_s`. A protected macOS folder can block on a privacy prompt
    that nobody sees; that must read as `unreadable`, not as a hang."""
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["entries"] = _list_dir(path)
        except BaseException as exc:  # noqa: BLE001 - reported to the caller as unreadable
            box["error"] = exc

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise TimeoutError(str(path))
    if "error" in box:
        raise box["error"]
    return box["entries"]


def _name_patterns(names: list[str]) -> set[str]:
    """The shapes of a set of file names: digits collapse to `N`, so `IMG_0001.txt` and `IMG_0002.txt` are one
    pattern and `회의록 1.md` and `여행 계획.md` are two."""
    return {re.sub(r"\d+", "N", os.path.splitext(n)[0]) for n in names}


def _classify_folder(files: Counter, total: int, names: list[str]) -> str | None:
    """`source-code` or `media` (sure), `ask` (cannot tell from the files), or None (keep)."""
    if total < NOISE_MIN_FILES:
        return None
    code = sum(n for ext, n in files.items() if ext in CODE_EXTENSIONS)
    media = sum(n for ext, n in files.items() if ext in MEDIA_EXTENSIONS)
    if code / total >= NOISE_SHARE:
        return "source-code"
    if media / total >= NOISE_SHARE:
        return "media"
    # A pile of one small text type whose names all follow one or two patterns looks generated; but it may also be
    # a person's own numbered reports, so it is a question, not a verdict. Distinct names (notes) never qualify.
    ext, top = files.most_common(1)[0]
    if top / total >= NOISE_SHARE and ext in BULK_PRONE and names and len(_name_patterns(names)) <= PILE_MAX_PATTERNS:
        return "ask"
    return None


def inventory(root: Path, max_entries: int = DEFAULT_MAX_ENTRIES, per_dir_timeout_s: float = 5.0) -> dict:
    root = Path(root).expanduser()
    if not root.is_dir():
        raise InventoryError(f"folder does not exist or is not a directory: {root}")
    root = root.resolve()

    indexable: Counter = Counter()
    unsupported = 0
    too_large = 0
    scanned = 0
    truncated = False
    skipped: dict[str, int] = {}
    unreadable: list[str] = []
    folder_files: dict[str, Counter] = {}
    folder_totals: Counter = Counter()
    folder_names: dict[str, list[str]] = {}

    stack = [root]
    while stack and not truncated:
        current = stack.pop()
        rel = current.relative_to(root).as_posix() if current != root else "."
        try:
            entries = _bounded_list(current, per_dir_timeout_s)
        except (TimeoutError, OSError):
            unreadable.append(rel)
            continue
        for entry in entries:
            if scanned >= max_entries:
                truncated = True
                break
            try:
                if entry.is_symlink():
                    continue  # never leave the root through a link
                if entry.is_dir(follow_symlinks=False):
                    if entry.name in AUTORAG_SKIPS:
                        skipped[entry.name] = skipped.get(entry.name, 0) + 1
                    else:
                        stack.append(Path(entry.path))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                scanned += 1
                ext = os.path.splitext(entry.name)[1].lower()
                folder_totals[rel] += 1
                folder_files.setdefault(rel, Counter())[ext] += 1
                names_here = folder_names.setdefault(rel, [])
                if len(names_here) < NAME_SAMPLE:
                    names_here.append(entry.name)
                if ext in INDEXABLE:
                    if entry.stat(follow_symlinks=False).st_size > MAX_SOURCE_BYTES:
                        too_large += 1
                    else:
                        indexable[ext[1:]] += 1
                else:
                    unsupported += 1
            except OSError:
                unreadable.append(f"{rel}/{entry.name}")

    # A folder is judged by everything under it, and reported at the shallowest folder that is flagged, so
    # `app/src` does not repeat for each of its subfolders. Two outcomes:
    #   noise: the file types say so (source code, media): excluded without asking, `confidence: high`.
    #   ask:   the types cannot say (a pile of same-pattern .txt/.csv/.md: receipts, exports, logs, or one
    #          person's numbered reports). The agent puts these to the user; nothing is excluded for them.
    noise: list[dict] = []
    ask: list[dict] = []
    flagged: list[str] = []
    for folder in sorted(folder_totals, key=lambda f: (f.count("/"), f)):
        if folder == "." or any(folder == n or folder.startswith(n + "/") for n in flagged):
            continue
        counts: Counter = Counter()
        names: list[str] = []
        total = 0
        for other, c in folder_files.items():
            if other == folder or other.startswith(folder + "/"):
                counts.update(c)
                total += folder_totals[other]
                names.extend(folder_names.get(other, []))
        verdict = _classify_folder(counts, total, names)
        if verdict is None:
            continue
        flagged.append(folder)
        entry = {
            "folder": folder,
            "files": total,
            # Anchored to this folder: a bare `**/src/**` would also drop an unrelated `src` folder.
            "suggested_exclude": f"{(root / folder).as_posix()}/**",
        }
        if verdict == "ask":
            patterns = _name_patterns(names)
            ask.append({**entry, "name_patterns": len(patterns), "example": names[0] if names else ""})
        else:
            noise.append({**entry, "reason": verdict, "confidence": "high"})

    keep = sorted(
        {
            f.split("/")[0]
            for f, c in folder_files.items()
            if f != "."
            and not any(f == n or f.startswith(n + "/") for n in [x["folder"] for x in noise])
            and any(e in INDEXABLE for e in c)
        }
    )
    if any(e in INDEXABLE for e in folder_files.get(".", Counter())):
        keep.insert(0, ".")
    excludes = sorted({n["suggested_exclude"] for n in noise})

    return {
        "ok": True,
        "root": str(root),
        "scanned": scanned,
        "truncated": truncated,
        "indexable": dict(indexable.most_common()),
        "indexable_total": sum(indexable.values()),
        "unsupported_total": unsupported,
        "too_large": too_large,
        "skipped_dirs": [{"dir": name, "count": count} for name, count in sorted(skipped.items())],
        "noise": noise,
        "ask": ask,
        "keep": keep or ["."],
        "recommended_excludes": excludes,
        "unreadable": unreadable[:20],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder")
    p.add_argument("--max-entries", type=int, default=DEFAULT_MAX_ENTRIES)
    args = p.parse_args(argv)
    try:
        result = inventory(Path(args.folder), max_entries=args.max_entries)
    except InventoryError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

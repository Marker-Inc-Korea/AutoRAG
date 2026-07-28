#!/usr/bin/env python3
"""Verify legacy/uv.lock pins against open Dependabot alerts.

Reads a JSON array of Dependabot alerts (GitHub REST shape) and the uv lockfile,
then reports, per alert, whether the currently locked version still falls inside
the advisory's vulnerable range.

Usage:
    check-lock-versions.py <alerts.json> <uv.lock> [--json out.json]

Exit code 1 when at least one alert is still VULNERABLE, else 0.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


def parse_version(raw: str) -> tuple:
    """PEP440-lite ordering key.

    Release segment compares numerically; a pre-release sorts before its release
    (1.0.0rc3 < 1.0.0) by appending a trailing marker.
    """
    text = raw.strip().lower()
    text = re.sub(r"^v", "", text)
    # Split off local/build metadata, which never participates in ordering.
    text = text.split("+", 1)[0]

    match = re.match(
        r"^(?P<release>\d+(?:\.\d+)*)"
        r"\.?(?P<pre>(?:a|b|rc|alpha|beta|pre|dev|post)\.?\d*)?",
        text,
    )
    if not match:
        return ((0,), 1, 0)

    release = tuple(int(part) for part in match.group("release").split("."))
    pre = match.group("pre")
    if not pre:
        # No pre-release marker: sorts after any pre-release of the same release.
        return (release, 1, 0)

    kind_match = re.match(r"^[a-z]+", pre)
    kind = kind_match.group(0) if kind_match else ""
    number_text = re.sub(r"^[a-z]+\.?", "", pre)
    number = int(number_text) if number_text else 0
    if kind == "post":
        return (release, 2, number)
    return (release, 0, number)


def compare(left: str, right: str) -> int:
    a, b = parse_version(left), parse_version(right)
    # Zero-pad the release segments so 3.14 and 3.14.0 compare equal (PEP 440).
    width = max(len(a[0]), len(b[0]))
    a = (a[0] + (0,) * (width - len(a[0])),) + a[1:]
    b = (b[0] + (0,) * (width - len(b[0])),) + b[1:]
    return (a > b) - (a < b)


@dataclass(frozen=True)
class Clause:
    operator: str
    version: str

    def matches(self, candidate: str) -> bool:
        result = compare(candidate, self.version)
        if self.operator == "<":
            return result < 0
        if self.operator == "<=":
            return result <= 0
        if self.operator == ">":
            return result > 0
        if self.operator == ">=":
            return result >= 0
        if self.operator in ("=", "=="):
            return result == 0
        raise ValueError(f"unsupported operator: {self.operator!r}")


def parse_range(raw: str) -> list[Clause]:
    clauses: list[Clause] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        match = re.match(r"^(<=|>=|==|<|>|=)\s*([0-9][0-9A-Za-z.!+-]*)$", part)
        if not match:
            raise ValueError(f"unsupported version constraint: {part!r}")
        clauses.append(Clause(operator=match.group(1), version=match.group(2).strip()))
    return clauses


def in_vulnerable_range(locked: str, raw_range: str) -> bool:
    return all(clause.matches(locked) for clause in parse_range(raw_range))


def load_locked_versions(lock_path: Path) -> dict[str, list[str]]:
    data = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    locked: dict[str, list[str]] = {}
    for package in data.get("package", []):
        name = package.get("name")
        version = package.get("version")
        if name and version:
            key = name.lower().replace("_", "-")
            versions = locked.setdefault(key, [])
            if version not in versions:
                versions.append(version)
    return locked


@dataclass
class Row:
    number: int
    package: str
    severity: str
    locked: str | None
    vulnerable_range: str
    first_patched: str | None
    manifest: str
    status: str = field(default="")
    affected_versions: list[str] = field(default_factory=list)


def classify(alert: dict, locked_versions: dict[str, list[str]]) -> Row:
    dependency = alert["dependency"]
    package = dependency["package"]["name"].lower().replace("_", "-")
    vulnerability = alert["security_vulnerability"]
    patched_info = vulnerability.get("first_patched_version") or {}
    vulnerable_range = vulnerability["vulnerable_version_range"]
    versions = locked_versions.get(package) or []
    affected = [v for v in versions if in_vulnerable_range(v, vulnerable_range)]

    row = Row(
        number=alert["number"],
        package=package,
        severity=alert["security_advisory"]["severity"],
        locked=",".join(versions) if versions else None,
        vulnerable_range=vulnerable_range,
        first_patched=patched_info.get("identifier"),
        manifest=dependency.get("manifest_path", ""),
    )
    row.affected_versions = affected

    if not versions:
        row.status = "NOT_IN_LOCK"
    elif affected:
        row.status = "VULNERABLE" if row.first_patched else "VULNERABLE_NO_PATCH"
    else:
        row.status = "ALREADY_FIXED"
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("alerts", type=Path)
    parser.add_argument("lock", type=Path)
    parser.add_argument("--json", dest="json_out", type=Path)
    args = parser.parse_args()

    alerts = json.loads(args.alerts.read_text(encoding="utf-8"))
    locked_versions = load_locked_versions(args.lock)
    rows = [classify(alert, locked_versions) for alert in alerts]

    by_status: dict[str, list[Row]] = {}
    for row in rows:
        by_status.setdefault(row.status, []).append(row)

    for status in sorted(by_status):
        entries = by_status[status]
        print(f"\n=== {status} ({len(entries)}) ===")
        for row in sorted(entries, key=lambda item: (item.package, item.number)):
            affected = ",".join(row.affected_versions) or "-"
            print(
                f"  #{row.number:<4} {row.package:<16} locked={row.locked or '-':<20}"
                f" affected={affected:<12}"
                f" range='{row.vulnerable_range}' patched={row.first_patched or 'NONE'}"
            )

    print("\n--- summary ---")
    for status in sorted(by_status):
        print(f"{status}: {len(by_status[status])}")
    print(f"TOTAL: {len(rows)}")

    if args.json_out:
        args.json_out.write_text(
            json.dumps([row.__dict__ for row in rows], indent=2), encoding="utf-8"
        )

    vulnerable = len(by_status.get("VULNERABLE", []))
    return 1 if vulnerable else 0


if __name__ == "__main__":
    sys.exit(main())

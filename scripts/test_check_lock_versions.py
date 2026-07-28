import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "check_lock_versions", Path(__file__).parent / "check-lock-versions.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


clv = _load_module()


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ("1.0.0", "1.0.0", 0),
        ("1.0.1", "1.0.0", 1),
        ("1.0.0", "1.0.1", -1),
        ("2.8.4", "2.8.3", 1),
        ("3.14.0", "3.14", 0),
        ("46.0.7", "46.0.10", -1),
        ("5.0.0rc3", "5.0.0", -1),
        ("5.0.0", "5.0.0rc3", 1),
        ("0.24.0", "0.9.0", 1),
        ("1.6.12", "1.6.9", 1),
        ("2.20.0", "2.9.0", 1),
        ("1.0.0.post1", "1.0.0", 1),
        ("1.0.0", "1.0.0.post1", -1),
        ("2.9.0.post0", "2.9.0", 1),
    ],
)
def test_compare_orders_versions(left, right, expected):
    assert clv.compare(left, right) == expected


@pytest.mark.parametrize("operator", ["!=", "~="])
def test_parse_range_rejects_unsupported_operators(operator):
    with pytest.raises(ValueError):
        clv.in_vulnerable_range("1.0.0", f"{operator} 1.2.3")


def test_parse_range_rejects_wildcard_versions():
    with pytest.raises(ValueError):
        clv.in_vulnerable_range("1.0.0", "== 1.*")


@pytest.mark.parametrize(
    "locked,vulnerable_range,expected",
    [
        ("3.13.5", "<= 3.14.0", True),
        ("3.14.1", "<= 3.14.0", False),
        ("0.20.0", ">= 0.3.0, < 0.22.0", True),
        ("0.24.0", ">= 0.3.0, < 0.22.0", False),
        ("0.22.0", ">= 0.3.0, < 0.22.0", False),
        ("0.3.0", ">= 0.3.0, < 0.22.0", True),
        ("3.1.0", "= 3.1.0", True),
        ("3.1.1", "= 3.1.0", False),
        ("1.6.5", ">= 1.0.0, <= 1.6.5", True),
        ("1.6.6", ">= 1.0.0, <= 1.6.5", False),
        ("4.51.3", "< 5.0.0rc3", True),
        ("5.0.0", "< 5.0.0rc3", False),
    ],
)
def test_in_vulnerable_range(locked, vulnerable_range, expected):
    assert clv.in_vulnerable_range(locked, vulnerable_range) is expected


def test_load_locked_versions_normalizes_names(tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text(
        '\n'.join(
            [
                "version = 1",
                "",
                "[[package]]",
                'name = "PyPDF2"',
                'version = "3.0.1"',
                "",
                "[[package]]",
                'name = "rouge_score"',
                'version = "0.1.2"',
            ]
        ),
        encoding="utf-8",
    )
    locked = clv.load_locked_versions(lock)
    assert locked["pypdf2"] == ["3.0.1"]
    assert locked["rouge-score"] == ["0.1.2"]


def test_load_locked_versions_keeps_every_version_of_a_package(tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text(
        '\n'.join(
            [
                "version = 1",
                "",
                "[[package]]",
                'name = "cryptography"',
                'version = "46.0.0"',
                "",
                "[[package]]",
                'name = "cryptography"',
                'version = "49.0.0"',
            ]
        ),
        encoding="utf-8",
    )
    locked = clv.load_locked_versions(lock)
    assert sorted(locked["cryptography"]) == ["46.0.0", "49.0.0"]


def test_classify_flags_vulnerable_when_any_locked_version_is_affected():
    locked = {"cryptography": ["46.0.0", "49.0.0"], "starlette": ["0.46.2", "1.3.1"]}

    crypto = clv.classify(_alert(1, "cryptography", "<= 46.0.4", "46.0.5"), locked)
    assert crypto.status == "VULNERABLE"
    assert "46.0.0" in (crypto.locked or "")

    starlette = clv.classify(_alert(2, "starlette", "< 0.47.2", "0.47.2"), locked)
    assert starlette.status == "VULNERABLE"


def _alert(number, package, vulnerable_range, patched, severity="high"):
    return {
        "number": number,
        "security_advisory": {"severity": severity},
        "security_vulnerability": {
            "vulnerable_version_range": vulnerable_range,
            "first_patched_version": {"identifier": patched} if patched else None,
        },
        "dependency": {
            "package": {"name": package},
            "manifest_path": "legacy/uv.lock",
        },
    }


def test_classify_distinguishes_all_four_states():
    locked = {"aiohttp": ["3.13.5"], "vllm": ["0.24.0"], "chromadb": ["1.5.9"]}

    still_vulnerable = clv.classify(_alert(1, "aiohttp", "<= 3.14.0", "3.14.1"), locked)
    assert still_vulnerable.status == "VULNERABLE"

    already_fixed = clv.classify(_alert(2, "vllm", "< 0.22.0", "0.22.0"), locked)
    assert already_fixed.status == "ALREADY_FIXED"

    no_patch = clv.classify(_alert(3, "chromadb", ">= 1.0.0, <= 1.5.9", None), locked)
    assert no_patch.status == "VULNERABLE_NO_PATCH"

    absent = clv.classify(_alert(4, "not-installed", "< 9.9.9", "9.9.9"), locked)
    assert absent.status == "NOT_IN_LOCK"

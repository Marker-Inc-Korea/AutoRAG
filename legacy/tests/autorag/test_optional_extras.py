import re
from pathlib import Path

LEGACY_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]


def _extra_items(name: str) -> list[str]:
	text = (LEGACY_ROOT / "pyproject.toml").read_text()
	match = re.search(rf"^{name} = \[", text, re.M)
	assert match is not None, f"missing optional extra {name}"
	index = match.end()
	depth = 1
	cursor = index
	while cursor < len(text) and depth:
		char = text[cursor]
		if char == "[":
			depth += 1
		elif char == "]":
			depth -= 1
		cursor += 1
	body = text[index : cursor - 1]
	return [
		item.strip().strip("\"'")
		for item in body.split(",")
		if item.strip()
	]


def test_gpu_extra_does_not_include_vllm():
	items = _extra_items("gpu")
	assert not any(item.startswith("vllm") for item in items)


def test_vllm_is_its_own_optional_extra():
	assert any(item.startswith("vllm") for item in _extra_items("vllm"))
	assert any("vllm" in item for item in _extra_items("all"))


def test_legacy_unit_test_job_skips_vllm_install_and_bounds_runtime():
	text = (REPO_ROOT / ".github/workflows/test.yml").read_text()
	assert re.search(r"timeout-minutes:\s*[1-9]\d*", text)
	sync_lines = [
		line.strip() for line in text.splitlines() if "uv sync" in line
	]
	assert sync_lines, "legacy-test must install with uv sync"
	assert all("--all-extras" not in line for line in sync_lines)
	assert all("--extra vllm" not in line for line in sync_lines)
	assert any("--extra gpu" in line for line in sync_lines)

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


def test_legacy_unit_test_job_bounds_the_pytest_step():
	# A stalled test reached 98% and then sat silent for 35 minutes until the
	# 40-minute job wall killed it. The job bound alone cannot surface that:
	# the step must carry its own bound so a stall fails fast and loudly.
	text = (REPO_ROOT / ".github/workflows/test.yml").read_text()
	steps = text.split("- name:")
	pytest_step = next((step for step in steps if step.lstrip().startswith("Run AutoRAG tests")), None)
	assert pytest_step is not None, "legacy-test must run pytest"
	assert "pytest" in pytest_step, "Run AutoRAG tests must invoke pytest"
	assert re.search(r"timeout-minutes:\s*[1-9]\d*", pytest_step), "pytest step must bound its runtime"


def _extra_vllm_specifier() -> str:
	# _extra_items() splits on commas, so read the specifier straight off the line.
	match = re.search(r'^vllm = \["vllm([^"]*)"\]', (LEGACY_ROOT / "pyproject.toml").read_text(), re.M)
	assert match is not None, "missing optional extra vllm"
	return match.group(1)


def _locked_vllm_specifier() -> str:
	for line in (LEGACY_ROOT / "uv.lock").read_text().splitlines():
		if 'name = "vllm"' in line and "extra == 'vllm'" in line:
			match = re.search(r'specifier = "([^"]*)"', line)
			assert match is not None, f"vllm lock entry has no specifier: {line.strip()}"
			return match.group(1)
	raise AssertionError("legacy/uv.lock has no vllm entry for the 'vllm' extra")


def test_vllm_extra_specifier_matches_the_lock():
	# Dependabot rewrites the pyproject extra but never uv.lock, so the specifier
	# recorded in the lock drifts and CI silently re-locks. Keep them equal.
	assert _locked_vllm_specifier() == _extra_vllm_specifier()

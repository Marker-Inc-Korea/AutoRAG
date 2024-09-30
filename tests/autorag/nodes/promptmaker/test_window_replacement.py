import os.path
import tempfile

import pytest

from autorag.nodes.promptmaker import WindowReplacement
from tests.autorag.nodes.promptmaker.test_prompt_maker_base import (
	prompt,
	queries,
	retrieved_contents,
	retrieved_metadata,
	previous_result,
	corpus_df,
)


@pytest.fixture
def pseudo_project_dir():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as project_dir:
		data_dir = os.path.join(project_dir, "data")
		os.makedirs(data_dir)
		corpus_df.to_parquet(os.path.join(data_dir, "corpus.parquet"))
		yield project_dir


@pytest.fixture
def window_replacement_instance(pseudo_project_dir):
	return WindowReplacement(project_dir=pseudo_project_dir)


def test_window_replacement(window_replacement_instance):
	result_prompts = window_replacement_instance._pure(
		prompt, queries, retrieved_contents, retrieved_metadata
	)
	assert len(result_prompts) == 2
	assert isinstance(result_prompts, list)
	assert (
		result_prompts[0]
		== "Answer this question: What is the capital of Japan? \n\n havertz arsenal doosan minji naeun gaeun lets go\n\nhavertz arsenal doosan minji naeun gaeun lets go"
	)
	assert (
		result_prompts[1]
		== "Answer this question: What is the capital of China? \n\n havertz arsenal doosan minji naeun gaeun lets go\n\nhavertz arsenal doosan minji naeun gaeun lets go"
	)


def test_window_replacement_node(pseudo_project_dir):
	result = WindowReplacement.run_evaluator(
		project_dir=pseudo_project_dir, previous_result=previous_result, prompt=prompt
	)
	assert len(result) == 2
	assert result.columns == ["prompts"]
	assert (
		result["prompts"][0]
		== "Answer this question: What is the capital of Japan? \n\n havertz arsenal doosan minji naeun gaeun lets go\n\nhavertz arsenal doosan minji naeun gaeun lets go"
	)
	assert (
		result["prompts"][1]
		== "Answer this question: What is the capital of China? \n\n havertz arsenal doosan minji naeun gaeun lets go\n\nhavertz arsenal doosan minji naeun gaeun lets go"
	)

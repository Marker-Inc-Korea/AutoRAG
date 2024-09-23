import pytest

from autorag.nodes.promptmaker import LongContextReorder
from tests.autorag.nodes.promptmaker.test_prompt_maker_base import (
	prompt,
	queries,
	retrieved_contents,
	retrieve_scores,
	previous_result,
)


@pytest.fixture
def long_context_reorder_instance():
	return LongContextReorder(project_dir="pseudo_project_dir")


def test_long_context_reorder(long_context_reorder_instance):
	result_prompts = long_context_reorder_instance._pure(
		prompt, queries, retrieved_contents, retrieve_scores
	)
	assert len(result_prompts) == 2
	assert isinstance(result_prompts, list)
	assert (
		result_prompts[0]
		== "Answer this question: What is the capital of Japan? \n\n Tokyo is the capital of Japan.\n\nTokyo, the capital of Japan, is a huge metropolitan city.\n\nTokyo is the capital of Japan."
	)
	assert (
		result_prompts[1]
		== "Answer this question: What is the capital of China? \n\n Beijing is the capital of China.\n\nBeijing, the capital of China, is a huge metropolitan city.\n\nBeijing is the capital of China."
	)


def test_long_context_reorder_node():
	result = LongContextReorder.run_evaluator(
		project_dir="pseudo_project_dir", previous_result=previous_result, prompt=prompt
	)
	assert len(result) == 2
	assert result.columns == ["prompts"]
	assert (
		result["prompts"][0]
		== "Answer this question: What is the capital of Japan? \n\n Tokyo is the capital of Japan.\n\nTokyo, the capital of Japan, is a huge metropolitan city.\n\nTokyo is the capital of Japan."
	)
	assert (
		result["prompts"][1]
		== "Answer this question: What is the capital of China? \n\n Beijing is the capital of China.\n\nBeijing, the capital of China, is a huge metropolitan city.\n\nBeijing is the capital of China."
	)

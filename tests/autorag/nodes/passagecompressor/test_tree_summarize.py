from unittest.mock import patch

import pandas as pd
import pytest
from llama_index.llms.openai import OpenAI

from autorag import generator_models
from autorag.nodes.passagecompressor import TreeSummarize
from tests.autorag.nodes.passagecompressor.test_base_passage_compressor import (
	queries,
	retrieved_contents,
	df,
	check_result,
)
from tests.mock import MockLLM


@pytest.fixture
def tree_summarize_instance():
	return TreeSummarize("project_dir", llm="mock")


@pytest.fixture
def tree_summarize_instance_chat():
	return TreeSummarize("project_dir", llm="openai", model="gpt-4o-mini")


async def mock_openai_apredict(self, prompt, *args, **kwargs):
	return "What is the capital of France?"


def test_tree_summarize_default(tree_summarize_instance):
	result = tree_summarize_instance._pure(queries, retrieved_contents)
	check_result(result)


@patch.object(
	OpenAI,
	"apredict",
	mock_openai_apredict,
)
def test_tree_summarize_chat(tree_summarize_instance_chat):
	result = tree_summarize_instance_chat._pure(queries, retrieved_contents)
	check_result(result)


def test_tree_summarize_custom_prompt(tree_summarize_instance):
	prompt = "This is a custom prompt. {context_str} {query_str}"
	result = tree_summarize_instance._pure(
		queries,
		retrieved_contents,
		prompt=prompt,
	)
	check_result(result)
	assert bool(result[0]) is True


@patch.object(
	OpenAI,
	"apredict",
	mock_openai_apredict,
)
def test_tree_summarize_custom_prompt_chat(tree_summarize_instance_chat):
	prompt = "Query: {query_str} Passages: {context_str}. Repeat the query."
	result = tree_summarize_instance_chat._pure(
		queries, retrieved_contents, chat_prompt=prompt
	)
	check_result(result)
	assert "What is the capital of France?" in result[0]


def test_tree_summarize_node():
	generator_models["mock"] = MockLLM
	result = TreeSummarize.run_evaluator(
		"project_dir",
		df,
		llm="mock",
		prompt="This is a custom prompt. {context_str} {query_str}",
		max_tokens=64,
	)
	assert isinstance(result, pd.DataFrame)
	contents = result["retrieved_contents"].tolist()
	assert isinstance(contents, list)
	assert len(contents) == len(queries)
	assert isinstance(contents[0], list)
	assert len(contents[0]) == 1
	assert isinstance(contents[0][0], str)
	assert bool(contents[0][0]) is True

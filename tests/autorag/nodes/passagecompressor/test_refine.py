from unittest.mock import patch

import pandas as pd
import pytest
from llama_index.llms.openai import OpenAI

from autorag import generator_models
from autorag.nodes.passagecompressor.refine import Refine
from tests.autorag.nodes.passagecompressor.test_base_passage_compressor import (
	queries,
	retrieved_contents,
	check_result,
	df,
)
from tests.mock import MockLLM


@pytest.fixture
def refine_instance():
	return Refine("project_dir", llm="mock")


@pytest.fixture
def refine_instance_chat():
	return Refine("project_dir", llm="openai", model="gpt-4o-mini")


async def mock_openai_apredict(self, prompt, *args, **kwargs):
	return "What is the capital of France?"


def test_refine_default(refine_instance):
	result = refine_instance._pure(queries, retrieved_contents)
	check_result(result)


@patch.object(
	OpenAI,
	"apredict",
	mock_openai_apredict,
)
def test_refine_chat(refine_instance_chat):
	result = refine_instance_chat._pure(queries, retrieved_contents)
	check_result(result)


def test_refine_custom_prompt(refine_instance):
	prompt = "This is a custom prompt. {context_msg} {query_str}"
	result = refine_instance._pure(queries, retrieved_contents, prompt=prompt)
	check_result(result)
	assert bool(result[0]) is True


@patch.object(
	OpenAI,
	"apredict",
	mock_openai_apredict,
)
def test_refine_custom_prompt_chat(refine_instance_chat):
	prompt = "Query: {query_str} Passages: {context_msg}. Repeat the query."
	result = refine_instance_chat._pure(queries, retrieved_contents, chat_prompt=prompt)
	check_result(result)
	assert "What is the capital of France?" in result[0]


def test_refine_node():
	generator_models["mock"] = MockLLM
	result = Refine.run_evaluator(
		"project_dir",
		df,
		llm="mock",
		prompt="This is a custom prompt. {context_msg} {query_str}",
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

import logging
import os
from unittest.mock import patch

import pandas as pd
import pytest
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel

from autorag import generator_models
from autorag.nodes.generator import LlamaIndexLLM
from tests.autorag.nodes.generator.test_generator_base import (
	prompts,
	check_generated_texts,
	check_generated_tokens,
	check_generated_log_probs,
)
from tests.delete_tests import is_github_action
from tests.mock import MockLLM

logger = logging.getLogger("AutoRAG")


@pytest.fixture
def llama_index_llm_instance():
	generator_models["mock"] = MockLLM
	return LlamaIndexLLM(project_dir=".", llm="mock", temperature=0.5, top_p=0.9)


def test_llama_index_llm(llama_index_llm_instance):
	answers, tokens, log_probs = llama_index_llm_instance._pure(prompts)
	check_generated_texts(answers)
	check_generated_tokens(tokens)
	check_generated_log_probs(log_probs)
	assert all(
		all(log_prob == 0.5 for log_prob in log_prob_list)
		for log_prob_list in log_probs
	)
	assert all(len(tokens[i]) == len(log_probs[i]) for i in range(len(tokens)))


def test_llama_index_llm_node():
	generator_models["mock"] = MockLLM
	previous_result = pd.DataFrame(
		{"prompts": prompts, "qid": ["id-1", "id-2", "id-3"]}
	)
	result_df = LlamaIndexLLM.run_evaluator(
		project_dir=".",
		previous_result=previous_result,
		llm="mock",
		temperature=0.5,
		top_p=0.9,
	)
	check_generated_texts(result_df["generated_texts"].tolist())
	check_generated_tokens(result_df["generated_tokens"].tolist())
	check_generated_log_probs(result_df["generated_log_probs"].tolist())


async def mock_openai_acomplete(self, messages, **kwargs):
	return CompletionResponse(
		text="""'```json
{
  "name": "John Doe",
  "phone_number": "1234567890",
  "age": 30,
  "is_dead": false
}
```'"""
	)


@patch.object(
	OpenAI,
	"acomplete",
	mock_openai_acomplete,
)
def test_llama_index_llm_structured_output():
	class TestResponse(BaseModel):
		name: str
		phone_number: str
		age: int
		is_dead: bool

	prompt = """You must transform the user introduction to json format. You have to extract four information: name, phone number, age, and is_dead.
Hello, my name is John Doe. My phone number is 1234567890. I am 30 years old. I am alive. I am good at soccer."""

	response = LlamaIndexLLM(
		"project_dir", llm="openai", model="gpt-4o-mini"
	).structured_output([prompt], TestResponse)
	output = response[0]
	assert isinstance(output, TestResponse)
	assert output.name == "John Doe"
	assert output.phone_number == "1234567890"
	assert output.age == 30
	assert output.is_dead is False


@pytest.mark.skipif(
	is_github_action(),
	reason="Skipping this test on GitHub Actions because it uses the real OpenAI API.",
)
@pytest.mark.asyncio()
async def test_llama_index_llm_astream():
	import asyncstdlib as a

	llm_instance = LlamaIndexLLM(
		project_dir=".",
		llm="openai",
		model="gpt-4o-mini",
		api_key=os.getenv("OPENAI_API_KEY"),
	)
	result = []
	async for i, s in a.enumerate(
		llm_instance.astream("Hello. Tell me about who is Kai Havertz")
	):
		assert isinstance(s, str)
		result.append(s)
		if i >= 1:
			assert len(result[i]) >= len(result[i - 1])

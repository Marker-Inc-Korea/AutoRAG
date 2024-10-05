import os
import pathlib
import re
from unittest.mock import patch

import pandas as pd
import pytest
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.llms import MockLLM

from autorag.data.legacy.qacreation.llama_index import (
	async_qa_gen_llama_index,
	generate_qa_llama_index_by_ratio,
	generate_qa_llama_index,
	generate_basic_answer,
	generate_answers,
)
from tests.delete_tests import is_github_action

root_dir = pathlib.PurePath(
	os.path.dirname(os.path.realpath(__file__))
).parent.parent.parent.parent
resource_dir = os.path.join(root_dir, "resources")

content = """Fire and brimstone ( or , alternatively , brimstone and fire ) is an idiomatic expression of referring to God 's wrath in the Hebrew Bible ( Old Testament ) and the New Testament .
In the Bible , it often appears in reference to the fate of the unfaithful .
Brimstone , an archaic term synonymous with sulfur , evokes the acrid odor of sulphur dioxide given off by lightning strikes .
Lightning was understood as divine punishment by many ancient religions ; the association of sulphur with God 's retribution is common in the Bible .
The English phrase `` fire and brimstone '' originates in the King James Bible ."""

prompt_dir = os.path.join(resource_dir, "qa_gen_prompts")
sample_prompt = open(os.path.join(prompt_dir, "prompt1.txt")).read()


@pytest.fixture
def contents():
	df = pd.read_csv(os.path.join(resource_dir, "sample_contents_nqa.csv"))
	yield df["passage"].tolist()[:6]


@pytest.fixture
def questions():
	df = pd.read_csv(os.path.join(resource_dir, "sample_contents_nqa.csv"))
	yield df["question"].tolist()[:6]


async def acomplete_qa_creation(self, messages, **kwargs):
	pattern = r"Number of questions to generate: (\d+)"
	matches = re.findall(pattern, messages)
	num_questions = int(matches[-1])
	return CompletionResponse(
		text="[Q]: Is this the test question?\n[A]: Yes, this is the test answer."
		* num_questions
	)


@patch.object(MockLLM, "acomplete", acomplete_qa_creation)
@pytest.mark.asyncio()
@pytest.mark.skipif(
	is_github_action(),
	reason="Skipping this test on GitHub Actions because it will be deprecated.",
)
async def test_async_qa_gen_llama_index():
	result = await async_qa_gen_llama_index(
		content, llm=MockLLM(), prompt=sample_prompt, question_num=3
	)
	assert len(result) == 3
	for res in result:
		assert "query" in res
		assert "generation_gt" in res
		assert bool(res["query"])
		assert bool(res["generation_gt"])


@patch.object(MockLLM, "acomplete", acomplete_qa_creation)
@pytest.mark.skipif(
	is_github_action(),
	reason="Skipping this test on GitHub Actions because it will be deprecated.",
)
def test_qa_gen_llama_index(contents):
	llm = MockLLM()
	result = generate_qa_llama_index(llm, contents, sample_prompt)
	check_multi_qa_gen(result)


@patch.object(MockLLM, "acomplete", acomplete_qa_creation)
@pytest.mark.skipif(
	is_github_action(),
	reason="Skipping this test on GitHub Actions because it will be deprecated.",
)
def test_qa_gen_llama_index_by_ratio(contents):
	ratio_dict = {
		str(os.path.join(prompt_dir, "prompt1.txt")): 1,
		str(os.path.join(prompt_dir, "prompt2.txt")): 2,
		str(os.path.join(prompt_dir, "prompt3.txt")): 3,
	}
	llm = MockLLM()
	result = generate_qa_llama_index_by_ratio(llm, contents, ratio_dict, batch=8)
	check_multi_qa_gen(result)


@pytest.mark.skipif(
	is_github_action(),
	reason="Skipping this test on GitHub Actions because it will be deprecated.",
)
def test_generate_answers(contents, questions):
	llm = MockLLM()
	result = generate_answers(llm, contents, questions)
	assert len(result) == len(contents) == len(questions)
	assert all(isinstance(res, str) for res in result)


@pytest.mark.asyncio()
@pytest.mark.skipif(
	is_github_action(),
	reason="Skipping this test on GitHub Actions because it will be deprecated.",
)
async def test_generate_basic_answer():
	llm = MockLLM()
	query = "What is Fire and brimstone?"
	response = await generate_basic_answer(llm, content, query)
	assert isinstance(response, str)
	assert bool(response) is True


def check_multi_qa_gen(result):
	assert len(result) == 6
	for res in result:
		assert all("query" in r for r in res)
		assert all("generation_gt" in r for r in res)
		assert all(bool(r["query"]) for r in res)
		assert all(bool(r["generation_gt"]) for r in res)

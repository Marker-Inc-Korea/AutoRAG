import asyncio
import os
import pathlib

import pandas as pd
import pytest
from llama_index.llms.openai import OpenAI

from autorag.data.qacreation.llama_index import async_qa_gen_llama_index, generate_qa_llama_index_by_ratio, \
    generate_qa_llama_index

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
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
    yield df['passage'].tolist()[:6]


def test_async_qa_gen_llama_index():
    result = asyncio.run(async_qa_gen_llama_index(content, llm=OpenAI(model="gpt-3.5-turbo",
                                                                      temperature=1.0),
                                                  prompt=sample_prompt, question_num=3))
    assert len(result) == 3
    for res in result:
        assert "query" in res
        assert "generation_gt" in res
        assert bool(res['query'])
        assert bool(res['generation_gt'])


def test_qa_gen_llama_index(contents):
    llm = OpenAI(model='gpt-3.5-turbo', temperature=1.0)
    result = generate_qa_llama_index(llm, contents, sample_prompt)
    check_multi_qa_gen(result)


def test_qa_gen_llama_index_by_ratio(contents):
    ratio_dict = {
        str(os.path.join(prompt_dir, "prompt1.txt")): 1,
        str(os.path.join(prompt_dir, "prompt2.txt")): 2,
        str(os.path.join(prompt_dir, "prompt3.txt")): 3,
    }
    llm = OpenAI(model='gpt-3.5-turbo', temperature=1.0)
    result = generate_qa_llama_index_by_ratio(llm, contents, ratio_dict, batch=8)
    check_multi_qa_gen(result)


def check_multi_qa_gen(result):
    assert len(result) == 6
    for res in result:
        assert all("query" in r for r in res)
        assert all("generation_gt" in r for r in res)
        assert all(bool(r['query']) for r in res)
        assert all(bool(r['generation_gt']) for r in res)

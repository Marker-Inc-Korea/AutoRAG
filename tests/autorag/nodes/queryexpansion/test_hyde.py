from unittest.mock import patch, AsyncMock

from llama_index.llms.openai import OpenAI

from autorag.nodes.queryexpansion import hyde
from tests.autorag.nodes.queryexpansion.test_query_expansion_base import project_dir, previous_result, \
    base_query_expansion_node_test, ingested_vectordb_node
from tests.mock import mock_openai_acomplete

sample_query = ["How many members are in Newjeans?", "What is visconde structure?"]


@patch.object(OpenAI, 'acomplete', new_callable=AsyncMock)
def test_hyde(mock_openai):
    mock_openai.side_effect = mock_openai_acomplete
    llm = OpenAI(max_tokens=64)
    original_hyde = hyde.__wrapped__
    result = original_hyde(sample_query, llm, prompt="")
    assert len(result[0]) == 1
    assert len(result) == 2


@patch.object(OpenAI, 'acomplete', new_callable=AsyncMock)
def test_hyde_node(mock_openai, ingested_vectordb_node):
    mock_openai.side_effect = mock_openai_acomplete
    result_df = hyde(project_dir=project_dir, previous_result=previous_result,
                     llm="openai", max_tokens=64)
    base_query_expansion_node_test(result_df)

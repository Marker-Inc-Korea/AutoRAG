from unittest.mock import patch

import pytest
from llama_index.embeddings.openai import OpenAIEmbedding

from autorag.nodes.passagefilter import SimilarityPercentileCutoff
from tests.autorag.nodes.passagefilter.test_passage_filter_base import (
	queries_example,
	contents_example,
	scores_example,
	ids_example,
	base_passage_filter_test,
	project_dir,
	previous_result,
	base_passage_filter_node_test,
)
from tests.mock import mock_get_text_embedding_batch


@pytest.fixture
def similarity_percentile_cutoff_instance():
	return SimilarityPercentileCutoff(
		project_dir=project_dir,
		previous_result=previous_result,
		embedding_model="openai_embed_3_large",
	)


@patch.object(
	OpenAIEmbedding,
	"get_text_embedding_batch",
	mock_get_text_embedding_batch,
)
def test_similarity_percentile_cutoff(similarity_percentile_cutoff_instance):
	assert (
		similarity_percentile_cutoff_instance.embedding_model.model_name
		== "text-embedding-3-large"
	)
	contents, ids, scores = similarity_percentile_cutoff_instance._pure(
		queries_example,
		contents_example,
		scores_example,
		ids_example,
		percentile=0.85,
		batch=64,
	)
	num_top_k = int(len(contents_example[0]) * 0.85)
	assert len(contents[0]) == len(contents[1]) == num_top_k
	base_passage_filter_test(contents, ids, scores)


@patch.object(
	OpenAIEmbedding,
	"get_text_embedding_batch",
	mock_get_text_embedding_batch,
)
def test_similarity_percentile_cutoff_node():
	result_df = SimilarityPercentileCutoff.run_evaluator(
		project_dir=project_dir, previous_result=previous_result, percentile=0.9
	)
	base_passage_filter_node_test(result_df)

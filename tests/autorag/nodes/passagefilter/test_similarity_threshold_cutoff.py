from unittest.mock import patch

import pytest
from llama_index.embeddings.openai import OpenAIEmbedding

from autorag.nodes.passagefilter import SimilarityThresholdCutoff
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
def similarity_threshold_cutoff_instance():
	return SimilarityThresholdCutoff(
		project_dir,
		embedding_model="openai_embed_3_large",
	)


@patch.object(
	OpenAIEmbedding,
	"get_text_embedding_batch",
	mock_get_text_embedding_batch,
)
def test_similarity_threshold_cutoff(similarity_threshold_cutoff_instance):
	contents, ids, scores = similarity_threshold_cutoff_instance._pure(
		queries_example,
		contents_example,
		scores_example,
		ids_example,
		threshold=0.85,
		batch=64,
	)
	base_passage_filter_test(contents, ids, scores)


@patch.object(
	OpenAIEmbedding,
	"get_text_embedding_batch",
	mock_get_text_embedding_batch,
)
def test_similarity_threshold_cutoff_node():
	result_df = SimilarityThresholdCutoff.run_evaluator(
		project_dir=project_dir, previous_result=previous_result, threshold=0.9
	)
	base_passage_filter_node_test(result_df)

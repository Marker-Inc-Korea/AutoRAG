from datetime import datetime

import pandas as pd
from llama_index.core.llms import MockLLM

from autorag.data.beta.filter.dontknow import dontknow_filter_rule_based
from autorag.data.beta.generation_gt.llama_index_gen_gt import (
	make_basic_gen_gt,
	make_concise_gen_gt,
)
from autorag.data.beta.query.llama_gen_query import factoid_query_gen
from autorag.data.beta.sample import random_single_hop
from autorag.data.beta.schema import Raw

initial_raw = Raw(
	pd.DataFrame(
		{
			"texts": [
				"The Kia Tigers lost the Korean Series this year and failed to win the championship. jeffrey went to gwangju to the Korean Series, but they lost there. I love this story."
				* 10,
				"minsing's Real Madrid were crushed by Ulsan Hyundai of Korea's BOBB. minsing's Man United beat estdside_gunn's Chelsea. estdside_gunn always loses. I love this story."
				* 10,
				"Bobb wanted to eat Korean food, but he couldn't do that because he had a date with his girlfriend. And Minsing could eat Korean food, because he is solo. But why minsing is so sad? He ate Korean BBQ and Katsu. I love this story."
				* 10,
			],
			"path": [
				"jax/sad_story.pdf",
				"jax/sad_story.pdf",
				"jax/havertz_story.pdf",
			],
			"page": [1, 2, -1],
			"last_modified_datetime": [
				datetime.strptime("2021-08-01", "%Y-%m-%d"),
				datetime.strptime("2021-08-02", "%Y-%m-%d"),
				datetime.strptime("2021-08-03", "%Y-%m-%d"),
			],
		}
	)
)


def test_make_dataset_from_raw():
	initial_corpus = initial_raw.chunk(
		"llama_index_chunk", chunk_method="token", chunk_size=128, chunk_overlap=5
	)
	llm = MockLLM()
	initial_qa = (
		initial_corpus.sample(random_single_hop, n=3)
		.map(
			lambda df: df.reset_index(drop=True),
		)
		.make_retrieval_gt_contents()
		.batch_apply(
			factoid_query_gen,
			llm=llm,
		)
		.batch_apply(
			make_basic_gen_gt,
			llm=llm,
		)
		.batch_apply(
			make_concise_gen_gt,
			llm=llm,
		)
		.filter(
			dontknow_filter_rule_based,
			lang="en",
		)
	)
	assert len(initial_qa.data) == 3
	assert set(initial_qa.data.columns) == {
		"qid",
		"retrieval_gt",
		"generation_gt",
		"query",
		"retrieval_gt_contents",
	}
	assert all(len(gen_gt) == 2 for gen_gt in initial_qa.data["generation_gt"].tolist())
	assert all(
		len(retrieval_gt[0]) == 1
		for retrieval_gt in initial_qa.data["retrieval_gt"].tolist()
	)

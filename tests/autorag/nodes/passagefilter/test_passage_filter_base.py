import os
import pathlib
import tempfile
from datetime import datetime
from uuid import uuid4

import pandas as pd
import pytest

root_dir = pathlib.PurePath(
	os.path.dirname(os.path.realpath(__file__))
).parent.parent.parent
project_dir = os.path.join(root_dir, "resources", "sample_project")
qa_data = pd.read_parquet(os.path.join(project_dir, "data", "qa.parquet"))
corpus_data = pd.read_parquet(os.path.join(project_dir, "data", "corpus.parquet"))
previous_result = qa_data.sample(2)

queries_example = [
	"What is the capital of France?",
	"How many members are in Newjeans?",
]
contents_example = [
	[
		"NomaDamas is Great Team",
		"Paris is the capital of France.",
		"havertz is suck at soccer",
		"Paris is one of the capital from France. Isn't it?",
	],
	[
		"i am hungry",
		"LA is a country in the United States.",
		"Newjeans has 5 members.",
		"Danielle is one of the members of Newjeans.",
	],
]
time_list = [
	[
		datetime(2015, 1, 1),
		datetime(2021, 9, 3),
		datetime(2022, 3, 5, 12, 30),
		datetime(2022, 3, 5, 12, 45, 00),
	],
	[
		datetime(2015, 1, 1),
		datetime(2021, 1, 1),
		datetime(2022, 3, 5, 12, 40),
		datetime(2022, 3, 5, 12, 45, 00),
	],
]
ids_example = [
	[str(uuid4()) for _ in range(len(contents_example[0]))],
	[str(uuid4()) for _ in range(len(contents_example[1]))],
]
scores_example = [[0.1, 0.8, 0.1, 0.5], [0.1, 0.2, 0.7, 0.3]]
f1_example = [0.4, 1.0]
recall_example = [1.0, 0.3]

previous_result["query"] = queries_example
previous_result["retrieved_contents"] = contents_example
previous_result["retrieved_ids"] = ids_example
previous_result["retrieve_scores"] = scores_example
previous_result["retrieval_f1"] = f1_example
previous_result["retrieval_recall"] = recall_example


@pytest.fixture
def project_dir_with_corpus():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
		data_dir = os.path.join(temp_dir, "data")
		os.makedirs(data_dir, exist_ok=True)
		qa_data.to_parquet(os.path.join(data_dir, "qa.parquet"), index=False)
		new_rows = pd.DataFrame(
			{
				"doc_id": ids_example[0] + ids_example[1],
				"contents": contents_example[0] + contents_example[1],
				"metadata": list(
					map(lambda x: {"last_modified_datetime": x}, time_list[0])
				)
				+ list(map(lambda x: {"last_modified_datetime": x}, time_list[1])),
			}
		)
		new_corpus = pd.concat([corpus_data, new_rows], ignore_index=True, axis=0)
		new_corpus.to_parquet(os.path.join(data_dir, "corpus.parquet"), index=False)
		yield temp_dir


def base_passage_filter_test(contents, ids, scores):
	assert len(contents) == len(ids) == len(scores) == 2
	for content_list, id_list, score_list in zip(contents, ids, scores):
		assert isinstance(content_list, list)
		assert isinstance(id_list, list)
		assert isinstance(score_list, list)
		for content, _id, score in zip(content_list, id_list, score_list):
			assert isinstance(content, str)
			assert isinstance(_id, str)
			assert isinstance(score, float)

	assert all([len(c) > 0 for c in contents])
	assert all([len(i) > 0 for i in ids])
	assert all([len(s) > 0 for s in scores])


def base_passage_filter_node_test(result_df):
	assert all(
		[
			column_name in result_df.columns
			for column_name in [
				"retrieved_contents",
				"retrieved_ids",
				"retrieve_scores",
			]
		]
	)
	base_passage_filter_test(
		result_df["retrieved_contents"].tolist(),
		result_df["retrieved_ids"].tolist(),
		result_df["retrieve_scores"].tolist(),
	)

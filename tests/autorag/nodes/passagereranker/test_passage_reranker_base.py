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
	],
	["i am hungry", "Newjeans has 5 members.", "LA is a country in the United States."],
]
ko_queries_example = ["프랑스의 수도는 어디인가요?", "뉴진스의 멤버는 몇 명인가요?"]
ko_contents_example = [
	[
		"마커AI는 멋진 회사입니다.",
		"프랑스의 수도는 파리 입니다.",
		"아스날은 축구를 못합니다.",
	],
	["배고파요", "뉴진스의 멤버는 5명 입니다.", "LA는 미국의 도시입니다."],
]
time_list = [
	[datetime(2021, 1, 1), datetime(2021, 1, 20), datetime(2021, 1, 3)],
	[datetime(2021, 3, 1), datetime(2030, 3, 1), datetime(1999, 1, 1)],
]
ids_example = [
	[str(uuid4()) for _ in range(len(contents_example[0]))],
	[str(uuid4()) for _ in range(len(contents_example[1]))],
]
scores_example = [[0.1, 0.8, 0.1], [0.1, 0.2, 0.7]]
f1_example = [0.4, 0.4]
recall_example = [1.0, 1.0]

previous_result["query"] = queries_example
previous_result["retrieved_contents"] = contents_example
previous_result["retrieved_ids"] = ids_example
previous_result["retrieve_scores"] = scores_example
previous_result["retrieval_f1"] = f1_example
previous_result["retrieval_recall"] = recall_example

ko_previous_result = previous_result.copy(deep=True)
ko_previous_result["query"] = ko_queries_example
ko_previous_result["retrieved_contents"] = ko_contents_example


def base_reranker_test(contents, ids, scores, top_k, use_ko=False, descending=True):
	assert len(contents) == len(ids) == len(scores) == 2
	assert len(contents[0]) == len(ids[0]) == len(scores[0]) == top_k
	for content_list, id_list, score_list in zip(contents, ids, scores):
		assert isinstance(content_list, list)
		assert isinstance(id_list, list)
		assert isinstance(score_list, list)
		for content, _id, score in zip(content_list, id_list, score_list):
			assert isinstance(content, str)
			assert isinstance(_id, str)
			assert isinstance(score, float)
		if descending is True:
			for i in range(1, len(score_list)):
				assert score_list[i - 1] >= score_list[i]
		else:
			for i in range(1, len(score_list)):
				assert score_list[i - 1] <= score_list[i]
	if use_ko is True:
		assert contents[0][0] == "프랑스의 수도는 파리 입니다."
		assert ids[0][0] in ids_example[0][1]
		assert contents[1][0] == "뉴진스의 멤버는 5명 입니다."
		assert ids[1][0] in ids_example[1][1]
	else:
		assert contents[0][0] == "Paris is the capital of France."
		assert ids[0][0] in ids_example[0][1]
		assert contents[1][0] == "Newjeans has 5 members."
		assert ids[1][0] in ids_example[1][1]


def base_reranker_node_test(result_df, top_k, use_ko=False, descending=True):
	contents = result_df["retrieved_contents"].tolist()
	ids = result_df["retrieved_ids"].tolist()
	scores = result_df["retrieve_scores"].tolist()
	base_reranker_test(contents, ids, scores, top_k, use_ko, descending=descending)


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

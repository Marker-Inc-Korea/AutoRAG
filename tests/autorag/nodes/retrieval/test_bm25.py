import pickle
import tempfile
from datetime import datetime

import pandas as pd
import pytest

from autorag.nodes.retrieval import bm25
from autorag.nodes.retrieval.bm25 import (
	bm25_ingest,
	tokenize_ko_kiwi,
	tokenize_porter_stemmer,
	tokenize_space,
	tokenize_ko_kkma,
	tokenize_ko_okt,
)
from autorag.utils.util import to_list
from tests.autorag.nodes.retrieval.test_retrieval_base import (
	queries,
	project_dir,
	corpus_df,
	previous_result,
	base_retrieval_test,
	base_retrieval_node_test,
	searchable_input_ids,
)

ko_texts = [
	"안녕? 나는 혜인이야. 내가 비눗방울 만드는 방법을 알려줄께.",
	"너 정말 잘한다. 넌 정말 짱이야. 우리 친구할래?",
	"내 생일 파티에 너만 못 온 그날, 혜진이가 엄청 혼났던 그날, 지원이가 여친이랑 헤어진 그날",
]


@pytest.fixture
def ingested_bm25_path():
	with tempfile.NamedTemporaryFile(suffix=".pkl", mode="w+b") as path:
		bm25_ingest(path.name, corpus_df)
		yield path.name


def test_bm25_retrieval(ingested_bm25_path):
	with open(ingested_bm25_path, "rb") as r:
		bm25_corpus = pickle.load(r)
	top_k = 3
	original_bm25 = bm25.__wrapped__
	id_result, score_result = original_bm25(
		queries, top_k=top_k, bm25_corpus=bm25_corpus
	)
	base_retrieval_test(id_result, score_result, top_k)


def test_bm25_retrieval_ids(ingested_bm25_path):
	with open(ingested_bm25_path, "rb") as r:
		bm25_corpus = pickle.load(r)
	original_bm25 = bm25.__wrapped__
	input_ids = [["doc2", "doc3"], ["doc1"], ["doc3", "doc4"]]
	id_result, score_result = original_bm25(
		queries, top_k=3, bm25_corpus=bm25_corpus, ids=input_ids
	)
	assert id_result == input_ids
	assert len(score_result) == 3
	assert len(score_result[0]) == 2
	assert len(score_result[1]) == 1
	assert len(score_result[2]) == 2


def test_bm25_retrieval_ids_empty(ingested_bm25_path):
	with open(ingested_bm25_path, "rb") as r:
		bm25_corpus = pickle.load(r)
	original_bm25 = bm25.__wrapped__
	input_ids = [["doc2", "doc3"], [], ["doc3"]]
	id_result, score_result = original_bm25(
		queries, top_k=3, bm25_corpus=bm25_corpus, ids=input_ids
	)
	assert id_result == input_ids
	assert len(score_result) == 3
	assert len(score_result[0]) == 2
	assert len(score_result[1]) == 0
	assert len(score_result[2]) == 1


def test_bm25_node():
	result_df = bm25(
		project_dir=project_dir,
		previous_result=previous_result,
		top_k=4,
		bm25_tokenizer="gpt2",
	)
	base_retrieval_node_test(result_df)


def test_bm25_node_ids():
	result_df = bm25(
		project_dir=project_dir,
		previous_result=previous_result,
		top_k=4,
		bm25_tokenizer="gpt2",
		ids=searchable_input_ids,
	)
	assert to_list(result_df["retrieved_ids"].tolist()) == searchable_input_ids
	score_result = to_list(result_df["retrieve_scores"].tolist())
	assert len(score_result) == 5
	assert len(score_result[0]) == 2


def test_bm25_ingest(ingested_bm25_path):
	with open(ingested_bm25_path, "rb") as r:
		corpus = pickle.load(r)
	assert set(corpus.keys()) == {"tokens", "passage_id", "tokenizer_name"}
	assert isinstance(corpus["tokens"], list)
	assert isinstance(corpus["passage_id"], list)
	assert isinstance(corpus["tokenizer_name"], str)
	assert corpus["tokenizer_name"] == "porter_stemmer"
	assert len(corpus["tokens"]) == len(corpus["passage_id"]) == 5
	assert set(corpus["passage_id"]) == {"doc1", "doc2", "doc3", "doc4", "doc5"}

	bm25_origin = bm25.__wrapped__
	top_k = 2
	id_result, score_result = bm25_origin(
		[["What is test document?"], ["What is test document number 2?"]],
		top_k=top_k,
		bm25_corpus=corpus,
	)
	assert len(id_result) == len(score_result) == 2
	for id_list, score_list in zip(id_result, score_result):
		assert isinstance(id_list, list)
		assert isinstance(score_list, list)
		for _id in id_list:
			assert isinstance(_id, str)
			assert _id in ["doc1", "doc2", "doc3", "doc4", "doc5"]


def test_duplicate_id_bm25_ingest(ingested_bm25_path):
	new_doc_id = ["doc4", "doc5", "doc6", "doc7", "doc8"]
	new_contents = [
		"This is a test document 4.",
		"This is a test document 5.",
		"This is a test document 6.",
		"This is a test document 7.",
		"This is a test document 8.",
	]
	new_metadata = [{"datetime": datetime.now()} for _ in range(5)]
	new_corpus_df = pd.DataFrame(
		{"doc_id": new_doc_id, "contents": new_contents, "metadata": new_metadata}
	)
	bm25_ingest(ingested_bm25_path, new_corpus_df)
	with open(ingested_bm25_path, "rb") as r:
		corpus = pickle.load(r)
	assert len(corpus["tokens"]) == len(corpus["passage_id"]) == 8


def test_other_method_bm25(ingested_bm25_path):
	with open(ingested_bm25_path, "rb") as r:
		corpus = pickle.load(r)
	bm25_origin = bm25.__wrapped__
	with pytest.raises(AssertionError):
		_, _ = bm25_origin(
			[["What is test document?"], ["What is test document number 2?"]],
			top_k=1,
			bm25_corpus=corpus,
			bm25_tokenizer="gpt2",
		)


def test_tokenize_ko_wiki():
	tokenized_list = tokenize_ko_kiwi(ko_texts)
	assert len(tokenized_list) == len(ko_texts)
	assert isinstance(tokenized_list[0], list)
	assert all(isinstance(x, str) for x in tokenized_list[0])


def test_tokenize_ko_kkma():
	tokenized_list = tokenize_ko_kkma(ko_texts)
	assert len(tokenized_list) == len(ko_texts)
	assert isinstance(tokenized_list[0], list)
	assert all(isinstance(x, str) for x in tokenized_list[0])


def test_tokenize_ko_okt():
	tokenized_list = tokenize_ko_okt(ko_texts)
	assert len(tokenized_list) == len(ko_texts)
	assert isinstance(tokenized_list[0], list)
	assert all(isinstance(x, str) for x in tokenized_list[0])


def test_tokenize_porter_stemmer():
	texts = [
		"The best baseball team in the world is Kia Tigers.",
		"And for a fortnight there, we were forever. Run into you sometimes, ask about the weather",
		"I walked through the door with you. The air was cold.",
	]
	tokenized_list = tokenize_porter_stemmer(texts)
	assert len(tokenized_list) == len(texts)
	assert isinstance(tokenized_list[0], list)
	assert all(isinstance(x, str) for x in tokenized_list[0])


def test_tokenize_space():
	texts = [
		"내 생일 파티에 너만 못 온 그날, 혜진이가 엄청 혼났던 그날, 지원이가 여친이랑 헤어진 그날,,,, What's your ETA?  ",
		"  The best baseball team in the world is Kia Tigers. 최강 기아 타이거즈 최형우! 최형우!!! 기아의 해결사~   ",
		" You're my chemical hype boy (ah-ah) \n 내 지난날들은 눈 뜨면 잊는 꿈\nHype boy 너만 원해\nHype boy 내가 전해;",
	]
	tokenized_list = tokenize_space(texts)
	assert len(tokenized_list) == len(texts)
	assert isinstance(tokenized_list[0], list)
	assert all(isinstance(x, str) for x in tokenized_list[0])

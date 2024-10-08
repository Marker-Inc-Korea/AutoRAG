import os
import tempfile

import pandas as pd

from autorag.data.qa.schema import Raw, Corpus, QA
from tests.autorag.data.qa.test_data_creation_piepline import initial_raw


def test_raw_add():
	raw_1 = Raw(
		pd.DataFrame(
			{
				"texts": ["hello", "world"],
				"path": ["path1", "path1"],
				"page": [1, 2],
				"last_modified_datetime": ["2021-08-01", "2021-08-02"],
			}
		)
	)
	raw_2 = Raw(
		pd.DataFrame(
			{
				"texts": ["foo", "bar"],
				"path": ["path2", "path2"],
				"page": [3, 4],
				"last_modified_datetime": ["2021-08-03", "2021-08-04"],
			}
		)
	)
	expected_dataframe = pd.DataFrame(
		{
			"texts": ["hello", "world", "foo", "bar"],
			"path": ["path1", "path1", "path2", "path2"],
			"page": [1, 2, 3, 4],
			"last_modified_datetime": [
				"2021-08-01",
				"2021-08-02",
				"2021-08-03",
				"2021-08-04",
			],
		}
	)
	pd.testing.assert_frame_equal((raw_1 + raw_2).data, expected_dataframe)


def test_raw_chunk():
	corpus = initial_raw.chunk(
		"llama_index_chunk", chunk_method="token", chunk_size=128, chunk_overlap=5
	)
	assert isinstance(corpus, Corpus)
	pd.testing.assert_frame_equal(corpus.linked_raw.data, initial_raw.data)
	assert set(corpus.data.columns) == {
		"doc_id",
		"contents",
		"path",
		"start_end_idx",
		"metadata",
	}
	assert corpus.data["doc_id"].nunique() == len(corpus.data)
	assert all(
		origin_path in initial_raw.data["path"].tolist()
		for origin_path in corpus.data["path"].tolist()
	)


def test_update_corpus():
	raw = Raw(
		pd.DataFrame(
			{
				"texts": ["hello", "world", "jax"],
				"path": ["path1", "path1", "path2"],
				"page": [1, 2, -1],
				"last_modified_datetime": [
					"2021-08-01",
					"2021-08-02",
					"2021-08-03",
				],
			}
		)
	)
	original_corpus = Corpus(
		pd.DataFrame(
			{
				"doc_id": ["id1", "id2", "id3", "id4", "id5", "id6"],
				"contents": ["hello", "world", "foo", "bar", "baz", "jax"],
				"path": ["path1", "path1", "path1", "path1", "path2", "path2"],
				"start_end_idx": [
					(0, 120),
					(90, 200),
					(0, 40),
					(35, 75),
					(0, 100),
					(150, 200),
				],
				"metadata": [
					{"page": 1, "last_modified_datetime": "2021-08-01"},
					{"page": 1, "last_modified_datetime": "2021-08-01"},
					{"page": 2, "last_modified_datetime": "2021-08-02"},
					{"page": 2, "last_modified_datetime": "2021-08-02"},
					{"last_modified_datetime": "2021-08-01"},
					{"last_modified_datetime": "2021-08-01"},
				],
			}
		),
		raw,
	)

	qa = QA(
		pd.DataFrame(
			{
				"qid": ["qid1", "qid2", "qid3", "qid4"],
				"query": ["hello", "world", "foo", "bar"],
				"retrieval_gt": [
					[["id1"]],
					[["id1"], ["id2"]],
					[["id3", "id4"]],
					[["id6", "id2"], ["id5"]],
				],
				"generation_gt": ["world", "foo", "bar", "jax"],
			}
		),
		original_corpus,
	)

	new_corpus = Corpus(
		pd.DataFrame(
			{
				"doc_id": [
					"new_id1",
					"new_id2",
					"new_id3",
					"new_id4",
					"new_id5",
					"new_id6",
				],
				"contents": ["hello", "world", "foo", "bar", "baz", "jax"],
				"path": ["path1", "path1", "path1", "path1", "path2", "path2"],
				"start_end_idx": [
					(0, 80),
					(80, 150),
					(15, 50),
					(50, 80),
					(0, 200),
					(201, 400),
				],
				"metadata": [
					{"page": 1, "last_modified_datetime": "2021-08-01"},
					{"page": 1, "last_modified_datetime": "2021-08-01"},
					{"page": 2, "last_modified_datetime": "2021-08-02"},
					{"page": 2, "last_modified_datetime": "2021-08-02"},
					{"last_modified_datetime": "2021-08-01"},
					{"last_modified_datetime": "2021-08-01"},
				],
			}
		),
		raw,
	)

	new_qa = qa.update_corpus(new_corpus)

	expected_dataframe = pd.DataFrame(
		{
			"qid": ["qid1", "qid2", "qid3", "qid4"],
			"retrieval_gt": [
				[["new_id1", "new_id2"]],
				[["new_id1", "new_id2"], ["new_id2"]],
				[["new_id3", "new_id3", "new_id4"]],
				[["new_id5", "new_id2"], ["new_id5"]],
			],
		}
	)
	pd.testing.assert_frame_equal(
		new_qa.data[["qid", "retrieval_gt"]], expected_dataframe
	)
	with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as qa_path:
		with tempfile.NamedTemporaryFile(
			suffix=".parquet", delete=False
		) as corpus_path:
			new_qa.to_parquet(qa_path.name, corpus_path.name)
			loaded_qa = pd.read_parquet(qa_path.name, engine="pyarrow")
			assert set(loaded_qa.columns) == {
				"qid",
				"query",
				"retrieval_gt",
				"generation_gt",
			}
			loaded_corpus = pd.read_parquet(corpus_path.name, engine="pyarrow")
			assert set(loaded_corpus.columns) == {
				"doc_id",
				"contents",
				"metadata",
				"path",
				"start_end_idx",
			}
			corpus_path.close()
			os.unlink(corpus_path.name)
		qa_path.close()
		os.unlink(qa_path.name)

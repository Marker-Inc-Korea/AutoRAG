import pandas as pd

from autorag.data.beta.schema import Raw, Corpus
from tests.autorag.data.beta.test_data_creation_piepline import initial_raw


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

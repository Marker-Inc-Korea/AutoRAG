import pandas as pd

from autorag.data.beta.schema import Raw, Corpus, QA


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


# def test_update_corpus():
# 	raw = Raw(
# 		pd.DataFrame(
# 			{
# 				{
# 					"texts": ["hello", "world", "jax"],
# 					"path": ["path1", "path1", "path2"],
# 					"page": [1, 2, -1],
# 					"last_modified_datetime": [
# 						"2021-08-01",
# 						"2021-08-02",
# 						"2021-08-03",
# 					],
# 				}
# 			}
# 		)
# 	)
# original_corpus = Corpus(
# 	pd.DataFrame(
# 		{
# 			"doc_id": ["id1", "id2", "id3", "id4", "id5", "id6"],
# 			"contents": ["hello", "world", "foo", "bar", "baz", "jax"],
# 			"path": ["path1", "path1", "path1", "path1", "path2", "path2"],
# 			"start_end_idx": [
# 				(0, 120),
# 				(90, 200),
# 				(4, 5),
# 				(6, 7),
# 				(8, 9),
# 				(10, 11),
# 			],
# 		}
# 	),
# 	raw,
# )

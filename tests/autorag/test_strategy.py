import time

import pandas as pd
import pytest

from autorag.strategy import (
	measure_speed,
	filter_by_threshold,
	select_best_average,
	select_best_rr,
	select_normalize_mean,
)

sample_dfs = [
	pd.DataFrame(
		{
			"content": ["a", "b", "c"],
			"retrieval_f1": [0.1, 0.2, 0.3],
			"retrieval_recall": [0.1, 0.2, 0.3],
		}
	),
	pd.DataFrame(
		{
			"content": ["d", "e", "f"],
			"retrieval_f1": [0.2, 0.3, 0.4],
			"retrieval_recall": [0.2, 0.3, 0.4],
		}
	),
	pd.DataFrame(
		{
			"content": ["g", "h", "i"],
			"retrieval_f1": [0.3, 0.4, 0.5],
			"retrieval_recall": [0.3, 0.4, 0.5],
		}
	),
]
sample_metadatas = ["a", "b", "c"]


def test_measure_speed():
	empty_result, five_seconds = measure_speed(time.sleep, 2)
	assert empty_result is None
	assert pytest.approx(2, 0.1) == five_seconds


def test_filter_by_threshold():
	results = [1, 2, 3, 4]
	values = [1, 2, 3, 4]
	threshold = 3
	filename = ["a", "b", "c", "d"]
	filtered_results, filtered_filenames = filter_by_threshold(
		results, values, threshold, filename
	)
	assert filtered_results == [1, 2, 3]
	assert filtered_filenames == ["a", "b", "c"]

	filtered_results, _ = filter_by_threshold(results, values, threshold)
	assert filtered_results == [1, 2, 3]


def test_avoid_empty_result():
	results = [1, 2, 3, 4]
	values = [1, 2, 3, 4]
	threshold = 0.5
	filenames = ["a", "b", "c", "d"]
	filtered_results, filtered_filenames = filter_by_threshold(
		results, values, threshold, filenames
	)
	assert filtered_results == [1, 2, 3, 4]
	assert filtered_filenames == ["a", "b", "c", "d"]


def test_select_best_average():
	best_df, best_filename = select_best_average(
		sample_dfs, ["retrieval_f1", "retrieval_recall"], sample_metadatas
	)
	assert best_df["content"].tolist() == ["g", "h", "i"]
	assert best_df["retrieval_f1"].tolist() == [0.3, 0.4, 0.5]
	assert best_df["retrieval_recall"].tolist() == [0.3, 0.4, 0.5]
	assert best_filename == "c"

	best_df, _ = select_best_average(sample_dfs, ["retrieval_f1", "retrieval_recall"])
	assert best_df["content"].tolist() == ["g", "h", "i"]


def test_select_best_rr():
	best_df, best_filename = select_best_rr(
		sample_dfs, ["retrieval_f1", "retrieval_recall"], sample_metadatas
	)
	assert best_df["content"].tolist() == ["g", "h", "i"]
	assert best_df["retrieval_f1"].tolist() == [0.3, 0.4, 0.5]
	assert best_df["retrieval_recall"].tolist() == [0.3, 0.4, 0.5]
	assert best_filename == "c"

	best_df, _ = select_best_average(sample_dfs, ["retrieval_f1", "retrieval_recall"])
	assert best_df["content"].tolist() == ["g", "h", "i"]


def test_select_normalize_mean():
	best_df, best_filename = select_normalize_mean(
		sample_dfs, ["retrieval_f1", "retrieval_recall"], sample_metadatas
	)
	assert best_df["content"].tolist() == ["g", "h", "i"]
	assert best_df["retrieval_f1"].tolist() == [0.3, 0.4, 0.5]
	assert best_df["retrieval_recall"].tolist() == [0.3, 0.4, 0.5]
	assert best_filename == "c"

	best_df, _ = select_best_average(sample_dfs, ["retrieval_f1", "retrieval_recall"])
	assert best_df["content"].tolist() == ["g", "h", "i"]

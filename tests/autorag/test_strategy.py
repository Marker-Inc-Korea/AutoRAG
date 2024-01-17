import time

import pandas as pd
import pytest

from autorag.strategy import measure_speed, filter_by_threshold, select_best_average


def test_measure_speed():
    empty_result, five_seconds = measure_speed(time.sleep, 2)
    assert empty_result is None
    assert pytest.approx(2, 0.1) == five_seconds


def test_filter_by_threshold():
    results = [1, 2, 3, 4]
    values = [1, 2, 3, 4]
    threshold = 3
    filtered_results = filter_by_threshold(results, values, threshold)
    assert filtered_results == [1, 2, 3]


def test_avoid_empty_result():
    results = [1, 2, 3, 4]
    values = [1, 2, 3, 4]
    threshold = 5
    filtered_results = filter_by_threshold(results, values, threshold)
    assert filtered_results == [1, 2, 3, 4]


def test_select_best_average():
    sample_dfs = [
        pd.DataFrame({'content': ['a', 'b', 'c'], 'retrieval_f1': [0.1, 0.2, 0.3], 'retrieval_recall': [0.1, 0.2, 0.3]}),
        pd.DataFrame({'content': ['d', 'e', 'f'], 'retrieval_f1': [0.2, 0.3, 0.4], 'retrieval_recall': [0.2, 0.3, 0.4]}),
        pd.DataFrame({'content': ['g', 'h', 'i'], 'retrieval_f1': [0.3, 0.4, 0.5], 'retrieval_recall': [0.3, 0.4, 0.5]}),
    ]
    best_df = select_best_average(sample_dfs, ['retrieval_f1', 'retrieval_recall'])
    assert best_df['content'].tolist() == ['g', 'h', 'i']
    assert best_df['retrieval_f1'].tolist() == [0.3, 0.4, 0.5]
    assert best_df['retrieval_recall'].tolist() == [0.3, 0.4, 0.5]

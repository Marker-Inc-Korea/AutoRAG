import functools
import time
from typing import List, Iterable

import pandas as pd


def measure_speed(func, *args, **kwargs):
    """
    Method for measuring execution speed of the function.
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


def avoid_empty_result(func):
    """
    Decorator for avoiding empty results from the function.
    When the func returns an empty result, it will return the origin results.
    It keeps the first input parameter of the function as the origin results.
    :param func:
    :return:
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> List:
        func_result = func(*args, **kwargs)
        if len(func_result) == 0:
            return args[0]
        else:
            return func_result

    return wrapper


@avoid_empty_result
def filter_by_threshold(results, value, threshold) -> List:
    """
    Filter results by value's threshold.
    :param results: The result list to be filtered.
    :param value: The value list to be filtered.
        It must have the same length with results.
    :param threshold: The threshold value.
    :return: Filtered list of results.
    """
    assert len(results) == len(value), "results and value must have the same length."
    filtered_results, _ = zip(*filter(lambda x: x[1] <= threshold, zip(results, value)))
    return list(filtered_results)


def select_best_average(results: List[pd.DataFrame], columns=Iterable[str]) -> pd.DataFrame:
    """
    Select the best result by average value among given columns.
    :param results: The list of results.
        Each result must be pd.DataFrame.
    :param columns: Column names to be averaged.
        Standard to select the best result.
    :return: The best result.
    """
    assert all([isinstance(result, pd.DataFrame) for result in results]), \
        "results must be pd.DataFrame."
    assert all([column in result.columns for result in results for column in columns]), \
        "columns must be in the columns of results."
    each_average = [df[columns].mean(axis=1).mean() for df in results]
    best_index = each_average.index(max(each_average))
    return results[best_index]

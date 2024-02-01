import functools
import time
from typing import List, Iterable, Tuple, Any, Optional, Callable

import pandas as pd


def measure_speed(func, *args, **kwargs):
    """
    Method for measuring execution speed of the function.
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


def avoid_empty_result(return_index: List[int]):
    """
    Decorator for avoiding empty results from the function.
    When the func returns an empty result, it will return the origin results.
    When the func returns a None, it will return the origin results.
    When the return value is a tuple, it will check all the value or list is empty.
    If so, it will return the origin results.
    It keeps parameters at return_index of the function as the origin results.

    :param return_index: The index of the result to be returned when there is no result.
    :return: The origin results or the results from the function.
    """

    def decorator_avoid_empty_result(func: Callable):

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> List:
            func_result = func(*args, **kwargs)
            if isinstance(func_result, tuple):
                # if all the results are empty, return the origin results.
                if all([not bool(result) for result in func_result]):
                    return [args[index] for index in return_index]
            if not bool(func_result):
                return [args[index] for index in return_index]
            else:
                return func_result

        return wrapper

    return decorator_avoid_empty_result


@avoid_empty_result([0, 3])
def filter_by_threshold(results, value, threshold, metadatas=None) -> Tuple[List, List]:
    """
    Filter results by value's threshold.

    :param results: The result list to be filtered.
    :param value: The value list to be filtered.
        It must have the same length with results.
    :param threshold: The threshold value.
    :param metadatas: The metadata of each result.
    :return: Filtered list of results and filtered list of metadatas.
        Metadatas will be returned even if you did not give input metadatas.
    :rtype: Tuple[List, List]
    """
    if metadatas is None:
        metadatas = [None] * len(results)
    assert len(results) == len(value), "results and value must have the same length."
    try:
        filtered_results, _, filtered_metadatas = zip(
            *filter(lambda x: x[1] <= threshold, zip(results, value, metadatas)))
    except ValueError:
        return [], []
    return list(filtered_results), list(filtered_metadatas)


def select_best_average(results: List[pd.DataFrame], columns: Iterable[str],
                        metadatas: Optional[List[Any]] = None) -> Tuple[pd.DataFrame, Any]:
    """
    Select the best result by average value among given columns.

    :param results: The list of results.
        Each result must be pd.DataFrame.
    :param columns: Column names to be averaged.
        Standard to select the best result.
    :param metadatas: The metadata of each result. 
        It will select one metadata with the best result.
    :return: The best result and the best metadata.
        The metadata will be returned even if you did not give input 'metadatas' parameter.
    :rtype: Tuple[pd.DataFrame, Any]
    """
    if metadatas is None:
        metadatas = [None] * len(results)
    assert len(results) == len(metadatas), "results and module_filename must have the same length."
    assert all([isinstance(result, pd.DataFrame) for result in results]), \
        "results must be pd.DataFrame."
    assert all([column in result.columns for result in results for column in columns]), \
        "columns must be in the columns of results."
    each_average = [df[columns].mean(axis=1).mean() for df in results]
    best_index = each_average.index(max(each_average))
    return results[best_index], metadatas[best_index]

import ast
import asyncio
import functools
import itertools
import os
from copy import deepcopy
import re
import string
from typing import List, Callable, Dict, Optional, Any, Collection
import ast

import pandas as pd
import swifter

import logging

logger = logging.getLogger("AutoRAG")


def fetch_contents(corpus_data: pd.DataFrame, ids: List[List[str]]) -> List[List[str]]:
    flat_ids = itertools.chain.from_iterable(ids)
    contents = list(map(lambda x: corpus_data.loc[lambda row: row['doc_id'] == x]['contents'].values[0], flat_ids))

    result = []
    idx = 0
    for sublist in ids:
        result.append(contents[idx:idx + len(sublist)])
        idx += len(sublist)

    return result


def result_to_dataframe(column_names: List[str]):
    """
    Decorator for converting results to pd.DataFrame.
    """

    def decorator_result_to_dataframe(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> pd.DataFrame:
            results = func(*args, **kwargs)
            if len(column_names) == 1:
                df_input = {column_names[0]: results}
            else:
                df_input = {column_name: result for result, column_name in zip(results, column_names)}
            result_df = pd.DataFrame(df_input)
            return result_df

        return wrapper

    return decorator_result_to_dataframe


def find_best_result_path(node_dir: str) -> str:
    """
    Find the best result filepath from node directory.
    :param node_dir: The directory of the node.
    :return: The filepath of the best result.
    """
    return list(filter(lambda x: x.endswith(".parquet") and x.startswith("best_"), os.listdir(node_dir)))[0]


def load_summary_file(summary_path: str,
                      dict_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load summary file from summary_path.

    :param summary_path: The path of the summary file.
    :param dict_columns: The columns that are dictionary type.
        You must fill this parameter if you want to load summary file properly.
        Default is ['module_params'].
    :return: The summary dataframe.
    """
    if not os.path.exists(summary_path):
        raise ValueError(f"summary.csv does not exist in {summary_path}.")
    summary_df = pd.read_csv(summary_path)
    if dict_columns is None:
        dict_columns = ['module_params']

    if any([col not in summary_df.columns for col in dict_columns]):
        raise ValueError(f"{dict_columns} must be in summary_df.columns.")

    def convert_dict(elem):
        return ast.literal_eval(elem)

    summary_df[dict_columns] = summary_df[dict_columns].applymap(convert_dict)
    return summary_df


def make_combinations(target_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Make combinations from target_dict.
    The target_dict key value must be a string,
    and the value can be list of values or single value.
    If generates all combinations of values from target_dict,
    which means generated dictionaries that contain only one value for each key,
    and all dictionaries will be different from each other.

    :param target_dict: The target dictionary.
    :return: The list of generated dictionaries.
    """
    dict_with_lists = dict(map(lambda x: (x[0], x[1] if isinstance(x[1], list) else [x[1]]),
                               target_dict.items()))

    def delete_duplicate(x):
        def is_hashable(obj):
            try:
                hash(obj)
                return True
            except TypeError:
                return False

        if any([not is_hashable(elem) for elem in x]):
            # TODO: add duplication check for unhashable objects
            return x
        else:
            return list(set(x))

    dict_with_lists = dict(map(lambda x: (x[0], delete_duplicate(x[1])), dict_with_lists.items()))
    combination = list(itertools.product(*dict_with_lists.values()))
    combination_dicts = [dict(zip(dict_with_lists.keys(), combo)) for combo in combination]
    return combination_dicts


def explode(index_values: Collection[Any], explode_values: Collection[Collection[Any]]):
    """
    Explode index_values and explode_values.
    The index_values and explode_values must have the same length.
    It will flatten explode_values and keep index_values as a pair.

    :param index_values: The index values.
    :param explode_values: The exploded values.
    :return: Tuple of exploded index_values and exploded explode_values.
    """
    assert len(index_values) == len(explode_values), "Index values and explode values must have same length"
    df = pd.DataFrame({
        'index_values': index_values,
        'explode_values': explode_values
    })
    df = df.explode('explode_values')
    return df['index_values'].tolist(), df['explode_values'].tolist()


def replace_value_in_dict(target_dict: Dict, key: str,
                          replace_value: Any) -> Dict:
    """
    Replace the value of the certain key in target_dict.
    If there is not targeted key in target_dict, it will return target_dict.

    :param target_dict: The target dictionary.
    :param key: The key to replace.
    :param replace_value: The value to replace.
    :return: The replaced dictionary.
    """
    replaced_dict = deepcopy(target_dict)
    if key not in replaced_dict:
        return replaced_dict
    replaced_dict[key] = replace_value
    return replaced_dict


def normalize_string(s: str) -> str:
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def convert_string_to_tuple_in_dict(d):
    """Recursively converts strings that start with '(' and end with ')' to tuples in a dictionary."""
    for key, value in d.items():
        # If the value is a dictionary, recurse
        if isinstance(value, dict):
            convert_string_to_tuple_in_dict(value)
        # If the value is a list, iterate through its elements
        elif isinstance(value, list):
            for i, item in enumerate(value):
                # If an item in the list is a dictionary, recurse
                if isinstance(item, dict):
                    convert_string_to_tuple_in_dict(item)
                # If an item in the list is a string matching the criteria, convert it to a tuple
                elif isinstance(item, str) and item.startswith('(') and item.endswith(')'):
                    value[i] = ast.literal_eval(item)
        # If the value is a string matching the criteria, convert it to a tuple
        elif isinstance(value, str) and value.startswith('(') and value.endswith(')'):
            d[key] = ast.literal_eval(value)

    return d


async def process_batch(tasks, batch_size: int = 64) -> List[Any]:
    """
    Processes tasks in batches asynchronously.

    :param tasks: A list of no-argument functions or coroutines to be executed.
    :param batch_size: The number of tasks to process in a single batch.
        Default is 64.
    :return: A list of results from the processed tasks.
    """
    results = []

    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)

    return results

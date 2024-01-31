import functools
import itertools
import os
from copy import deepcopy
import re
import string
from typing import List, Callable, Dict, Optional, Any, Collection

import pandas as pd
import swifter

import logging

logger = logging.getLogger("AutoRAG")


def fetch_contents(corpus_data: pd.DataFrame, ids: List[List[str]]) -> List[List[str]]:
    assert isinstance(ids[0], list), "ids must be a list of list of ids."
    id_df = pd.DataFrame(ids, columns=[f'id_{i}' for i in range(len(ids[0]))])
    try:
        contents_df = id_df.swifter.applymap(
            lambda x: corpus_data.loc[lambda row: row['doc_id'] == x]['contents'].values[0])
    except IndexError:
        logger.error(f"doc_id does not exist in corpus_data.")
        raise IndexError("doc_id does not exist in corpus_data.")
    return contents_df.values.tolist()


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


def make_module_file_name(module_name: str, module_params: Dict) -> str:
    """
    Make module parquet file name for saving results dataframe.

    :param module_name: Module name.
        It can be module function's name.
    :param module_params: Parameters of the module function.
    :return: Module parquet file name
    """
    module_params_str = "-".join(list(map(lambda x: f"{x[0]}_{x[1]}", module_params.items())))
    if len(module_params_str) <= 0:
        return f"{module_name}.parquet"
    return f"{module_name}=>{module_params_str}.parquet"


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
        You must fill this parameter if you want to load summary file properly.l
        Default is None.
    :return: The summary dataframe.
    """
    if not os.path.exists(summary_path):
        raise ValueError(f"summary.parquet does not exist in {summary_path}.")
    summary_df = pd.read_parquet(summary_path)
    if dict_columns is None:
        logger.warning("dict_columns is None."
                       "If your input summary_df has dictionary type columns, you must fill dict_columns.")
        return summary_df

    def delete_none_at_dict(elem):
        return dict(filter(lambda item: item[1] is not None, elem.items()))

    summary_df[dict_columns] = summary_df[dict_columns].applymap(delete_none_at_dict)
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
    dict_with_lists = dict(map(lambda x: (x[0], list(set(x[1]))), dict_with_lists.items()))
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

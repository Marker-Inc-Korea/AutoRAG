import functools
from typing import List, Callable, Dict

import pandas as pd
import swifter


def fetch_contents(corpus_data: pd.DataFrame, ids: List[List[str]]) -> List[List[str]]:
    assert isinstance(ids[0], list), "ids must be a list of list of ids."
    id_df = pd.DataFrame(ids, columns=[f'id_{i}' for i in range(len(ids[0]))])
    contents_df = id_df.swifter.applymap(
        lambda x: corpus_data.loc[lambda row: row['doc_id'] == x]['contents'].values[0])
    return contents_df.values.tolist()


def result_to_dataframe(column_names: List[str]):
    """
    Decorator for converting results to pd.DataFrame.
    """

    def decorator_result_to_dataframe(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> pd.DataFrame:
            results = func(*args, **kwargs)
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

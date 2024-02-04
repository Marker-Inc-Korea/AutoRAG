import functools
from pathlib import Path
from typing import List, Union, Dict

import pandas as pd
from llama_index.llms.base import BaseLLM

from autorag import generator_models
from autorag.utils import result_to_dataframe


def passage_compressor_node(func):
    @functools.wraps(func)
    @result_to_dataframe(['retrieved_contents'])
    def wrapper(
            project_dir: Union[str, Path],
            previous_result: pd.DataFrame,
            *args, **kwargs) -> List[List[str]]:
        assert all([column in previous_result.columns for column in
                    ['query', 'retrieved_contents', 'retrieved_ids', 'retrieve_scores']]), \
            "previous_result must have retrieved_contents, retrieved_ids, and retrieve_scores columns."
        assert len(previous_result) > 0, "previous_result must have at least one row."

        queries = previous_result['query'].tolist()
        retrieved_contents = previous_result['retrieved_contents'].tolist()
        retrieved_ids = previous_result['retrieved_ids'].tolist()
        retrieve_scores = previous_result['retrieve_scores'].tolist()

        if func.__name__ == 'tree_summarize':
            param_list = ['prompt', 'chat_prompt', 'context_window', 'num_output', 'batch']
            param_dict = dict(filter(lambda x: x[0] in param_list, kwargs.items()))
            kwargs_dict = dict(filter(lambda x: x[0] not in param_list, kwargs.items()))
            llm_name = kwargs_dict.pop('llm')
            llm = make_llm(llm_name, kwargs_dict)
            result = func(
                queries=queries,
                contents=retrieved_contents,
                scores=retrieve_scores,
                ids=retrieved_ids,
                llm=llm,
                **param_dict
            )
        else:
            raise ValueError(f"{func.__name__} is not supported in passage compressor node.")

        return list(map(lambda x: [x], result))

    return wrapper


def make_llm(llm_name: str, kwargs: Dict) -> BaseLLM:
    if llm_name not in generator_models:
        raise KeyError(f"{llm_name} is not supported. "
                       "You can add it manually by calling autorag.generator_models.")
    return generator_models[llm_name](**kwargs)

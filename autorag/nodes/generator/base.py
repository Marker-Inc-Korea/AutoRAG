import functools
from pathlib import Path
from typing import Union, Tuple, List

import pandas as pd

from autorag import generator_models
from autorag.utils import result_to_dataframe


def generator_node(func):
    @functools.wraps(func)
    @result_to_dataframe(["generated_texts", "generated_tokens", "generated_log_probs"])
    def wrapper(
            project_dir: Union[str, Path],
            previous_result: pd.DataFrame,
            llm: str,
            **kwargs) -> Tuple[List[str], List[List[int]], List[List[float]]]:
        """
        This decorator makes a generator module to be a node.
        It automatically extracts prompts from previous_result and runs the generator function.
        Plus, it retrieves the llm instance from autorag.generator_models.
        
        :param project_dir: The project directory.
        :param previous_result: The previous result that contains prompts,
        :param llm: The llm name that you want to use.
        :param kwargs: The extra parameters for initializing the llm instance.
        :return: Pandas dataframe that contains generated texts, generated tokens, and generated log probs.
            Each column is "generated_texts", "generated_tokens", and "generated_log_probs".
        """
        assert 'prompts' in previous_result.columns, "previous_result must contain prompts column."
        prompts = previous_result['prompts'].tolist()
        if func.__name__ == 'llama_index_llm':
            if llm not in generator_models:
                raise ValueError(f"{llm} is not a valid llm name. Please check the llm name."
                                 "You can check valid llm names from autorag.generator_models.")
            batch = kwargs.pop('batch', 16)
            llm_instance = generator_models[llm](**kwargs)
            return func(prompts=prompts, llm=llm_instance, batch=batch)
    return wrapper

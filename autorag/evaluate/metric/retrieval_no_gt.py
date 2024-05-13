from typing import List

import numpy as np
import pandas as pd
import ragas
from datasets import Dataset
from langchain_openai.chat_models import ChatOpenAI
from ragas.metrics import context_precision


def ragas_context_precision(queries: List[str], retrieved_contents: List[List[str]], generation_gt: List[List[str]],
                            openai_model_name: str = "gpt-4-turbo") -> List[float]:
    generation_gt = cast_generation_gt(generation_gt)
    data_samples = {
        'question': queries,
        'contexts': retrieved_contents,
        'ground_truth': generation_gt,
    }
    llm = ChatOpenAI(model_name=openai_model_name)
    dataset = Dataset.from_dict(data_samples)
    score = ragas.evaluate(dataset, metrics=[context_precision], llm=llm, is_async=True)
    score_df = score.to_pandas()
    return score_df['context_precision'].tolist()


def cast_generation_gt(generation_gt: List[List[str]]) -> List[str]:
    result = []
    for gt in generation_gt:
        if isinstance(gt, str):
            result.append(gt)
        elif isinstance(gt, list) or isinstance(gt, np.ndarray):
            result.append(gt[0])
        elif isinstance(gt, pd.Series):
            result.append(gt.iloc[0])
        else:
            raise ValueError(f"Unexpected type of generation gt elements : {type(gt)}")
    return result

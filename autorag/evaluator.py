from typing import Callable, List, Dict

import pandas as pd

from autorag.strategy import measure_time


# 이건 별로 필요 없을 듯
def run_modules(modules: List[Callable], module_params: List[Dict]) -> List[pd.DataFrame]:
    results = []
    for module, module_param in zip(modules, module_params):
        result, spent_time = measure_time(module, **module_param)
        average_spend_time = spent_time / len(result)
        results.append(result)
    return results


def filter_by_time(results, spend_times, threshold):
    filtered_results, _ = zip(*filter(lambda x: x[1] <= threshold, zip(results, spend_times)))
    return filtered_results

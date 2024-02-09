from typing import Union, List, Dict, Tuple, Any

from autorag import embedding_models


def cast_metrics(metrics: Union[List[str], List[Dict]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
     Turn metrics to list of metric names and parameter list.

    :param metrics: List of string or dictionary.
    :return: The list of metric names and dictionary list of metric parameters.
    """
    if not isinstance(metrics, list):
        raise ValueError("metrics must be a list of string or dictionary.")
    if isinstance(metrics[0], str):
        return metrics, [{} for _ in metrics]
    elif isinstance(metrics[0], dict):
        # pop 'metric_name' key from dictionary
        metric_names = list(map(lambda x: x.pop('metric_name'), metrics))
        metric_params = [dict(map(lambda x, y: cast_embedding_model(x, y), metric.keys(), metric.values())) for metric
                         in metrics]
        return metric_names, metric_params
    else:
        raise ValueError("metrics must be a list of string or dictionary.")


def cast_embedding_model(key, value):
    if key == 'embedding_model':
        return key, embedding_models[value]
    else:
        return key, value

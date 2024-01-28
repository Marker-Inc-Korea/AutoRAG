import logging
import os
import pathlib
from typing import List, Callable, Dict, Tuple

import pandas as pd

from autorag import embedding_models
from autorag.nodes.retrieval import bm25, vectordb
from autorag.nodes.retrieval.base import load_bm25_corpus, load_chroma_collection
from autorag.nodes.retrieval.run import evaluate_retrieval_node
from autorag.schema import Module
from autorag.strategy import measure_speed, filter_by_threshold, select_best_average
from autorag.utils.util import make_module_file_name, fetch_contents, make_combinations

logger = logging.getLogger("AutoRAG")


def run_query_expansion_node(modules: List[Callable],
                             module_params: List[Dict],
                             previous_result: pd.DataFrame,
                             node_line_dir: str,
                             strategies: Dict,
                             ) -> pd.DataFrame:
    """
    Run evaluation and select the best module among query expansion node results.

    :param modules: Query expansion modules to run.
    :param module_params: Query expansion module parameters.
    :param previous_result: Previous result dataframe.
        In this case, it would be qa data.
    :param node_line_dir: This node line's directory.
    :param strategies: Strategies for query expansion node.
    :return: The best result dataframe.
    """
    if not os.path.exists(node_line_dir):
        os.makedirs(node_line_dir)
    project_dir = pathlib.PurePath(node_line_dir).parent.parent
    resources_dir = os.path.join(project_dir, "resources")
    data_dir = os.path.join(project_dir, "data")
    retrieval_gt = pd.read_parquet(os.path.join(project_dir, "data", "qa.parquet"))['retrieval_gt'].tolist()

    # run query expansion
    result_queries, execution_times = zip(*map(lambda task: measure_speed(
        task[0], project_dir=project_dir, previous_result=previous_result, **task[1]), zip(modules, module_params)))
    average_times = list(map(lambda x: x / len(result_queries[0]), execution_times))

    # pop top_k from strategies
    if "top_k" in strategies.keys():
        top_k = strategies.pop("top_k")
    else:
        top_k = 10  # default value

    # get retrieval module values
    if "retrieval_module" in strategies.keys():
        retrieval_modules = strategies.pop("retrieval_module")
    else:
        retrieval_modules = [{"module_type": "bm25"}]  # default value

    # get the best retrieval result for each module
    results = list(map(lambda x: module_best_retrieval(x, top_k, retrieval_modules, resources_dir,
                                                       retrieval_gt, data_dir, strategies), result_queries))

    # save results to folder
    save_dir = os.path.join(node_line_dir, "query_expansion")  # node name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filepaths = list(map(lambda x: os.path.join(save_dir, make_module_file_name(x[0].__name__, x[1])),
                         zip(modules, module_params)))
    list(map(lambda x: x[0].to_parquet(x[1], index=False), zip(results, filepaths)))  # execute save to parquet
    filenames = list(map(lambda x: os.path.basename(x), filepaths))

    summary_df = pd.DataFrame({
        'filename': filenames,
        'module_name': list(map(lambda module: module.__name__, modules)),
        'module_params': module_params,
        'execution_time': average_times,
        **{metric: list(map(lambda result: result[metric].mean(), results)) for metric in strategies.get('metrics')},
    })

    # filter by strategies
    if strategies.get('speed_threshold') is not None:
        results, filenames = filter_by_threshold(results, average_times, strategies['speed_threshold'], filenames)
    selected_result, selected_filename = select_best_average(results, strategies.get('metrics'), filenames)
    best_result = pd.concat([previous_result, selected_result], axis=1)

    # add summary.csv 'is_best' column
    summary_df['is_best'] = summary_df['filename'] == selected_filename

    # save the result files
    best_result.to_parquet(os.path.join(save_dir, f'best_{os.path.splitext(selected_filename)[0]}.parquet'),
                           index=False)
    summary_df.to_parquet(os.path.join(save_dir, 'summary.parquet'), index=False)
    return best_result


def retrieval_by_retrieval_module(retrieval_module: Dict, resources_dir: str, decomposed_queries: List[List[str]],
                                  top_k: int) -> Tuple[List[List[str]], List[List[float]]]:
    module = Module.from_dict(retrieval_module)
    retrieval_module_type = module.module_type

    # set parameters for retrieval module
    if retrieval_module_type == "bm25":
        # check if bm25_path and file exists
        bm25_path = os.path.join(resources_dir, 'bm25.pkl')
        assert bm25_path is not None, "bm25_path must be specified for using bm25 retrieval."
        assert os.path.exists(bm25_path), f"bm25_path {bm25_path} does not exist. Please ingest first."

        # run retrieval function
        bm25_corpus = load_bm25_corpus(bm25_path)
        original_bm25 = bm25.__wrapped__
        ids, scores = original_bm25(queries=decomposed_queries, bm25_corpus=bm25_corpus, top_k=top_k)

    elif retrieval_module_type == "vectordb":
        # check if chroma_path and file exists
        chroma_path = os.path.join(resources_dir, 'chroma')
        assert chroma_path is not None, "chroma_path must be specified for using vectordb retrieval."
        assert os.path.exists(chroma_path), f"chroma_path {chroma_path} does not exist. Please ingest first."

        # set embedding model
        embedding_model_str = module.module_param["embedding_model"]
        if embedding_model_str in embedding_models:
            embedding_model = embedding_models[embedding_model_str]
        else:
            logger.error(f"embedding_model_str {embedding_model_str} does not exist.")
            raise KeyError(f"embedding_model_str {embedding_model_str} does not exist.")

        # run retrieval function
        chroma_collection = load_chroma_collection(db_path=chroma_path, collection_name=embedding_model_str)
        original_vectordb = vectordb.__wrapped__
        ids, scores = original_vectordb(queries=decomposed_queries, collection=chroma_collection,
                                        embedding_model=embedding_model, top_k=top_k)
    else:
        raise ValueError(f"invalid f{retrieval_module} for using query_expansion_io decorator.")

    return ids, scores


def process_retrieval_module(retrieval_module, resources_dir, expanded_queries, top_k, data_dir):
    ids, scores = retrieval_by_retrieval_module(retrieval_module=retrieval_module, resources_dir=resources_dir,
                                                decomposed_queries=expanded_queries, top_k=top_k)
    # fetch data from corpus_data
    corpus_data = pd.read_parquet(os.path.join(data_dir, "corpus.parquet"))
    contents = fetch_contents(corpus_data, ids)

    # create retrieval df
    retrieval_df = pd.DataFrame({
        'retrieved_contents': contents,
        'retrieved_ids': ids,
        'retrieve_scores': scores,
    })
    return retrieval_df


def module_best_retrieval(result_df: pd.DataFrame, top_k: int, retrieval_modules: List[Dict], resources_dir,
                          retrieval_gt, data_dir, strategies):
    # get expanded_queries to list
    expanded_queries = result_df["expanded_queries"].tolist()

    # get all combinations of retrieval modules
    final_retrieval_modules = [item for retrieval_module in retrieval_modules
                               for item in make_combinations(retrieval_module)]

    # get retrieval results
    retrieval_results = list(map(lambda x: process_retrieval_module(x, resources_dir, expanded_queries,
                                                                    top_k, data_dir), final_retrieval_modules))
    # get best retrieval result for each retrieval module
    if strategies.get('metrics') is None:
        raise ValueError("You must at least one metrics for retrieval evaluation.")
    results = list(map(lambda x: evaluate_retrieval_node(x, retrieval_gt, strategies.get('metrics')), retrieval_results))

    # filter best result in retrieval_modules
    best_result = select_best_average(results, strategies.get('metrics'))
    return best_result[0]

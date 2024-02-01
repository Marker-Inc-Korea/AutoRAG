import os.path
import pathlib
import shutil

import chromadb
import pandas as pd
import pytest
from llama_index import OpenAIEmbedding

from autorag.nodes.retrieval import bm25, vectordb, hybrid_rrf
from autorag.nodes.retrieval.run import run_retrieval_node
from autorag.nodes.retrieval.vectordb import vectordb_ingest

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
resources_dir = os.path.join(root_dir, "resources")


@pytest.fixture
def node_line_dir():
    test_project_dir = os.path.join(resources_dir, "test_project")
    sample_project_dir = os.path.join(resources_dir, "sample_project")
    # copy & paste all folders and files in sample_project folder
    shutil.copytree(sample_project_dir, test_project_dir)

    chroma_path = os.path.join(test_project_dir, "resources", "chroma")
    os.makedirs(chroma_path)
    db = chromadb.PersistentClient(path=chroma_path)
    collection = db.create_collection(name="openai", metadata={"hnsw:space": "cosine"})
    corpus_path = os.path.join(test_project_dir, "data", "corpus.parquet")
    corpus_df = pd.read_parquet(corpus_path)
    vectordb_ingest(collection, corpus_df, OpenAIEmbedding())

    test_trail_dir = os.path.join(test_project_dir, "test_trial")
    os.makedirs(test_trail_dir)
    node_line_dir = os.path.join(test_trail_dir, "test_node_line")
    os.makedirs(node_line_dir)
    yield node_line_dir
    # teardown
    shutil.rmtree(test_project_dir)


def test_run_retrieval_node(node_line_dir):
    modules = [bm25, vectordb, hybrid_rrf]
    module_params = [
        {'top_k': 4},
        {'top_k': 4, 'embedding_model': 'openai'},
        {'top_k': 4, 'rrf_k': 2, 'target_modules': ('bm25', 'vectordb')}
    ]
    project_dir = pathlib.PurePath(node_line_dir).parent.parent
    qa_path = os.path.join(project_dir, "data", "qa.parquet")
    strategies = {
        'metrics': ['retrieval_f1', 'retrieval_recall'],
        'speed_threshold': 5,
    }
    previous_result = pd.read_parquet(qa_path)
    best_result = run_retrieval_node(modules, module_params, previous_result, node_line_dir, strategies)
    assert os.path.exists(os.path.join(node_line_dir, "retrieval"))
    expect_columns = ['qid', 'query', 'retrieval_gt', 'generation_gt',
                      'retrieved_contents', 'retrieved_ids', 'retrieve_scores', 'retrieval_f1', 'retrieval_recall']
    assert all([expect_column in best_result.columns for expect_column in expect_columns])
    # test summary feature
    summary_path = os.path.join(node_line_dir, "retrieval", "summary.parquet")
    bm25_top_k_path = os.path.join(node_line_dir, "retrieval", "bm25=>top_k_4.parquet")
    assert os.path.exists(os.path.join(node_line_dir, "retrieval", "bm25=>top_k_4.parquet"))
    bm25_top_k_df = pd.read_parquet(bm25_top_k_path)
    assert os.path.exists(summary_path)
    summary_df = pd.read_parquet(summary_path)
    assert set(summary_df.columns) == {'filename', 'retrieval_f1', 'retrieval_recall',
                                       'module_name', 'module_params', 'execution_time', 'is_best'}
    assert len(summary_df) == 3
    assert summary_df['filename'][0] == "bm25=>top_k_4.parquet"
    assert summary_df['retrieval_f1'][0] == bm25_top_k_df['retrieval_f1'].mean()
    assert summary_df['retrieval_recall'][0] == bm25_top_k_df['retrieval_recall'].mean()
    assert summary_df['module_name'][0] == "bm25"
    assert summary_df['module_params'][0] == {'top_k': 4}
    assert summary_df['execution_time'][0] > 0
    # test the best file is saved properly
    best_path = summary_df[summary_df['is_best'] == True]['filename'].values[0]
    assert os.path.exists(best_path)
    best_df = pd.read_parquet(best_path)
    assert all([expect_column in best_df.columns for expect_column in expect_columns])

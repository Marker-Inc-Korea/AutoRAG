import os.path
import pathlib
import shutil
import tempfile

import chromadb
import pandas as pd
import pytest
from llama_index import OpenAIEmbedding

from autorag.nodes.retrieval import bm25, vectordb, hybrid_rrf, hybrid_cc
from autorag.nodes.retrieval.run import run_retrieval_node, select_result_for_hybrid, get_ids_and_scores
from autorag.nodes.retrieval.vectordb import vectordb_ingest
from autorag.utils.util import load_summary_file

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
    modules = [bm25, vectordb, hybrid_rrf, hybrid_cc, hybrid_cc]
    module_params = [
        {'top_k': 4},
        {'top_k': 4, 'embedding_model': 'openai'},
        {'top_k': 4, 'rrf_k': 2, 'target_modules': ('bm25', 'vectordb')},
        {'top_k': 4, 'target_modules': ('bm25', 'vectordb'), 'weights': (0.3, 0.7)},
        {'top_k': 4, 'target_modules': ('bm25', 'vectordb'), 'weights': (0.5, 0.5)},
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
    summary_path = os.path.join(node_line_dir, "retrieval", "summary.csv")
    bm25_top_k_path = os.path.join(node_line_dir, "retrieval", "0.parquet")
    assert os.path.exists(bm25_top_k_path)
    bm25_top_k_df = pd.read_parquet(bm25_top_k_path)
    assert os.path.exists(summary_path)
    summary_df = load_summary_file(summary_path)
    assert set(summary_df.columns) == {'filename', 'retrieval_f1', 'retrieval_recall',
                                       'module_name', 'module_params', 'execution_time', 'is_best'}
    assert len(summary_df) == 5
    assert summary_df['filename'][0] == "0.parquet"
    assert summary_df['retrieval_f1'][0] == bm25_top_k_df['retrieval_f1'].mean()
    assert summary_df['retrieval_recall'][0] == bm25_top_k_df['retrieval_recall'].mean()
    assert summary_df['module_name'][0] == "bm25"
    assert summary_df['module_params'][0] == {'top_k': 4}
    assert summary_df['execution_time'][0] > 0
    # assert average times
    assert summary_df['execution_time'][0] + summary_df['execution_time'][1] == pytest.approx(
        summary_df['execution_time'][2])
    assert summary_df['execution_time'][0] + summary_df['execution_time'][1] == pytest.approx(
        summary_df['execution_time'][3])

    assert summary_df['filename'].nunique() == len(summary_df)
    assert len(summary_df[summary_df['is_best'] == True]) == 1

    # test summary_df hybrid retrieval convert well
    assert all(summary_df['module_params'].apply(lambda x: 'ids' not in x))
    assert all(summary_df['module_params'].apply(lambda x: 'scores' not in x))
    hybrid_summary_df = summary_df[summary_df['module_name'].str.contains('hybrid')]
    assert all(hybrid_summary_df['module_params'].apply(lambda x: 'target_modules' in x))
    assert all(hybrid_summary_df['module_params'].apply(lambda x: 'target_module_params' in x))
    assert all(hybrid_summary_df['module_params'].apply(lambda x: x['target_modules'] == ('bm25', 'vectordb')))
    assert all(hybrid_summary_df['module_params'].apply(
        lambda x: x['target_module_params'] == ({'top_k': 4}, {'top_k': 4, 'embedding_model': 'openai'})))

    # test the best file is saved properly
    best_filename = summary_df[summary_df['is_best'] == True]['filename'].values[0]
    best_path = os.path.join(node_line_dir, "retrieval", f'best_{best_filename}')
    assert os.path.exists(best_path)
    best_df = pd.read_parquet(best_path)
    assert all([expect_column in best_df.columns for expect_column in expect_columns])


@pytest.fixture
def pseudo_node_dir():
    summary_df = pd.DataFrame({
        'filename': ['0.parquet', '1.parquet', '2.parquet'],
        'module_name': ['bm25', 'vectordb', 'vectordb'],
        'module_params': [
            {'top_k': 3},
            {'top_k': 3, 'embedding_model': 'openai'},
            {'top_k': 3, 'embedding_model': 'huggingface'},
        ],
        'execution_time': [1, 1, 1],
        'retrieval_f1': [0.1, 0.2, 0.3],
        'retrieval_recall': [0.2, 0.55, 0.5],
    })
    bm25_df = pd.DataFrame({
        'query': ['query-1', 'query-2', 'query-3'],
        'retrieved_ids': [['id-1', 'id-2', 'id-3'],
                          ['id-1', 'id-2', 'id-3'],
                          ['id-1', 'id-2', 'id-3']],
        'retrieve_scores': [[0.1, 0.2, 0.3],
                            [0.1, 0.2, 0.3],
                            [0.1, 0.2, 0.3]],
        'retrieval_f1': [0.05, 0.1, 0.15],
        'retrieval_recall': [0.1, 0.275, 0.25],
    })
    vector_openai_df = pd.DataFrame({
        'query': ['query-1', 'query-2', 'query-3'],
        'retrieved_ids': [['id-4', 'id-5', 'id-6'],
                          ['id-4', 'id-5', 'id-6'],
                          ['id-4', 'id-5', 'id-6']],
        'retrieve_scores': [[0.3, 0.4, 0.5],
                            [0.3, 0.4, 0.5],
                            [0.3, 0.4, 0.5]],
        'retrieval_f1': [0.15, 0.2, 0.25],
        'retrieval_recall': [0.3, 0.55, 0.5],
    })
    vector_huggingface_df = pd.DataFrame({
        'query': ['query-1', 'query-2', 'query-3'],
        'retrieved_ids': [['id-7', 'id-8', 'id-9'],
                          ['id-7', 'id-8', 'id-9'],
                          ['id-7', 'id-8', 'id-9']],
        'retrieve_scores': [[0.5, 0.6, 0.7],
                            [0.5, 0.6, 0.7],
                            [0.5, 0.6, 0.7]],
        'retrieval_f1': [0.25, 0.3, 0.35],
        'retrieval_recall': [0.5, 0.825, 0.75],
    })

    with tempfile.TemporaryDirectory() as node_dir:
        summary_df.to_csv(os.path.join(node_dir, "summary.csv"))
        bm25_df.to_parquet(os.path.join(node_dir, "0.parquet"))
        vector_openai_df.to_parquet(os.path.join(node_dir, "1.parquet"))
        vector_huggingface_df.to_parquet(os.path.join(node_dir, "2.parquet"))
        yield node_dir


def test_select_result_for_hybrid(pseudo_node_dir):
    filenames = select_result_for_hybrid(pseudo_node_dir, ("bm25", "vectordb"))
    dict_id_scores = get_ids_and_scores(pseudo_node_dir, filenames)
    ids = dict_id_scores['ids']
    scores = dict_id_scores['scores']
    assert len(ids) == len(scores) == 2
    assert len(ids[0]) == len(scores[0]) == 3
    assert len(ids[1]) == len(scores[1]) == 3
    assert ids[0] == [['id-1', 'id-2', 'id-3'],
                      ['id-1', 'id-2', 'id-3'],
                      ['id-1', 'id-2', 'id-3']]
    assert scores[0] == [[0.1, 0.2, 0.3],
                         [0.1, 0.2, 0.3],
                         [0.1, 0.2, 0.3]]
    assert ids[1] == [['id-7', 'id-8', 'id-9'],
                      ['id-7', 'id-8', 'id-9'],
                      ['id-7', 'id-8', 'id-9']]
    assert scores[1] == [[0.5, 0.6, 0.7],
                         [0.5, 0.6, 0.7],
                         [0.5, 0.6, 0.7]]


def test_run_retrieval_node_only_hybrid(node_line_dir):
    modules = [hybrid_cc]
    module_params = [
        {'top_k': 4, 'target_modules': ('bm25', 'vectordb'), 'weights': (0.3, 0.7),
         'target_module_params': ({'top_k': 3}, {'top_k': 3, 'embedding_model': 'openai'})},
    ]
    project_dir = pathlib.PurePath(node_line_dir).parent.parent
    qa_path = os.path.join(project_dir, "data", "qa.parquet")
    strategies = {
        'metrics': ['retrieval_f1', 'retrieval_recall'],
    }
    previous_result = pd.read_parquet(qa_path)
    best_result = run_retrieval_node(modules, module_params, previous_result, node_line_dir, strategies)
    assert os.path.exists(os.path.join(node_line_dir, "retrieval"))
    expect_columns = ['qid', 'query', 'retrieval_gt', 'generation_gt',
                      'retrieved_contents', 'retrieved_ids', 'retrieve_scores', 'retrieval_f1', 'retrieval_recall']
    assert all([expect_column in best_result.columns for expect_column in expect_columns])
    # test summary feature
    summary_path = os.path.join(node_line_dir, "retrieval", "summary.csv")
    single_result_path = os.path.join(node_line_dir, "retrieval", "0.parquet")
    assert os.path.exists(single_result_path)
    single_result_df = pd.read_parquet(single_result_path)
    assert os.path.exists(summary_path)
    summary_df = load_summary_file(summary_path)
    assert set(summary_df.columns) == {'filename', 'retrieval_f1', 'retrieval_recall',
                                       'module_name', 'module_params', 'execution_time', 'is_best'}
    assert len(summary_df) == 1
    assert summary_df['filename'][0] == "0.parquet"
    assert summary_df['retrieval_f1'][0] == single_result_df['retrieval_f1'].mean()
    assert summary_df['retrieval_recall'][0] == single_result_df['retrieval_recall'].mean()
    assert summary_df['module_name'][0] == "hybrid_cc"
    assert summary_df['module_params'][0] == {'top_k': 4, 'target_modules': ('bm25', 'vectordb'), 'weights': (0.3, 0.7),
         'target_module_params': ({'top_k': 3}, {'top_k': 3, 'embedding_model': 'openai'})}
    assert summary_df['execution_time'][0] > 0
    assert summary_df['is_best'][0] == True
    assert summary_df['filename'].nunique() == len(summary_df)

    # test the best file is saved properly
    best_filename = summary_df[summary_df['is_best'] == True]['filename'].values[0]
    best_path = os.path.join(node_line_dir, "retrieval", f'best_{best_filename}')
    assert os.path.exists(best_path)
    best_df = pd.read_parquet(best_path)
    assert all([expect_column in best_df.columns for expect_column in expect_columns])

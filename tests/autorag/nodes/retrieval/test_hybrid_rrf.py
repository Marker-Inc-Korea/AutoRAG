import os
import tempfile
from datetime import datetime

import chromadb
import pandas as pd
import pytest
from llama_index import OpenAIEmbedding

from autorag.nodes.retrieval.bm25 import bm25_ingest
from autorag.nodes.retrieval.hybrid_rrf import rrf_pure
from autorag.nodes.retrieval import hybrid_rrf
from autorag.nodes.retrieval.vectordb import vectordb_ingest
from tests.autorag.nodes.retrieval.test_run_retrieval_node import pseudo_node_dir


def test_hybrid_rrf():
    sample_ids = ([
                      ['id-1', 'id-2', 'id-3'],
                      ['id-2', 'id-3', 'id-4']
                  ], [
                      ['id-1', 'id-4', 'id-3'],
                      ['id-2', 'id-5', 'id-4']
                  ])
    sample_scores = ([
                         [1, 3, 5],
                         [2, 4, 6]
                     ], [
                         [4, 2, 6],
                         [5, 3, 7]
                     ])
    result_id, result_scores = hybrid_rrf.__wrapped__(sample_ids, sample_scores, top_k=3, rrf_k=1)
    assert result_id == [
        ['id-3', 'id-1', 'id-2'],
        ['id-4', 'id-2', 'id-3']
    ]
    assert result_scores[0] == pytest.approx([1.0, (1 / 4) + (1 / 3), (1 / 3)])
    assert result_scores[1] == pytest.approx([1.0, (1 / 4) + (1 / 3), (1 / 3)])


def test_rrf_pure():
    sample_ids = (['id-1', 'id-2', 'id-3'],
                  ['id-1', 'id-4', 'id-3'])
    sample_scores = ([1, 3, 5], [4, 2, 6])
    result_id, result_scores = rrf_pure(sample_ids, sample_scores,
                                        rrf_k=1, top_k=3)
    assert result_scores == pytest.approx([1.0, (1 / 4) + (1 / 3), (1 / 3)])
    assert result_id == ['id-3', 'id-1', 'id-2']


@pytest.fixture
def pseudo_project_dir():
    with tempfile.TemporaryDirectory() as project_dir:
        corpus_df = pd.DataFrame({
            'doc_id': ['id-1', 'id-2', 'id-3', 'id-4', 'id-5', 'id-6', 'id-7', 'id-8', 'id-9'],
            'contents': ['doc-1', 'doc-2', 'doc-3', 'doc-4', 'doc-5', 'doc-6', 'doc-7', 'doc-8', 'doc-9'],
            'metadata': [{'last_modified_date': datetime.now()} for _ in range(9)]
        })
        os.makedirs(os.path.join(project_dir, "data"))
        corpus_df.to_parquet(os.path.join(project_dir, "data", 'corpus.parquet'))
        resource_dir = os.path.join(project_dir, "resources")
        os.makedirs(resource_dir)
        bm25_ingest(os.path.join(resource_dir, 'bm25.pkl'), corpus_df)
        chroma_path = os.path.join(resource_dir, 'chroma')
        db = chromadb.PersistentClient(path=chroma_path)
        collection = db.create_collection(name="openai", metadata={"hnsw:space": "cosine"})
        vectordb_ingest(collection, corpus_df, OpenAIEmbedding())
        yield project_dir

previous_result = pd.DataFrame({
        'qid': ['query-1', 'query-2', 'query-3'],
        'query': ['query-1', 'query-2', 'query-3'],
        'retrieval_gt': [
            [['id-1'], ['id-2'], ['id-3']],
            [['id-1'], ['id-2'], ['id-3']],
            [['id-1'], ['id-2'], ['id-3']]
        ],
        'generation_gt': [
            ['gen-1', 'gen-2'],
            ['gen-1', 'gen-2'],
            ['gen-1', 'gen-2']
        ]
    })


def test_hybrid_rrf_node(pseudo_project_dir):
    modules = {
        'ids': ([['id-1', 'id-2', 'id-3'],
                 ['id-1', 'id-2', 'id-3'],
                 ['id-1', 'id-2', 'id-3']],
                [['id-7', 'id-8', 'id-9'],
                ['id-7', 'id-8', 'id-9'],
                ['id-7', 'id-8', 'id-9']]),
        'scores': ([
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3]
        ],[
            [0.5, 0.6, 0.7],
            [0.5, 0.6, 0.7],
            [0.5, 0.6, 0.7]
        ]),
        'top_k': 3,
        'rrf_k': 1,
    }
    result_df = hybrid_rrf(project_dir=pseudo_project_dir, previous_result=previous_result, **modules)
    assert len(result_df) == 3
    assert isinstance(result_df, pd.DataFrame)
    assert set(result_df.columns) == {'retrieved_contents', 'retrieved_ids', 'retrieve_scores'}
    assert set(result_df['retrieved_ids'].tolist()[0]) == {'id-9', 'id-3', 'id-2'}
    assert result_df['retrieve_scores'].tolist()[0] == pytest.approx([0.5, 0.5, 1 / 3])
    assert set(result_df['retrieved_contents'].tolist()[0]) == {'doc-9', 'doc-3', 'doc-2'}


def test_hybrid_rrf_node_deploy(pseudo_project_dir):
    modules = {
        'target_modules': ('bm25', 'vectordb'),
        'target_module_params': [
            {'top_k': 3},
            {'embedding_model': 'openai', 'top_k': 3}
        ],
        'top_k': 3,
        'rrf_k': 1,
    }
    result_df = hybrid_rrf(project_dir=pseudo_project_dir, previous_result=previous_result, **modules)
    assert len(result_df) == 3
    assert isinstance(result_df, pd.DataFrame)
    assert set(result_df.columns) == {'retrieved_contents', 'retrieved_ids', 'retrieve_scores'}
    assert len(result_df['retrieved_ids'].tolist()[0]) == 3
    assert len(result_df['retrieve_scores'].tolist()[0]) == 3
    assert len(result_df['retrieved_contents'].tolist()[0]) == 3

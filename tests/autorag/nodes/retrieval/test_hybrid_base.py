import os
import tempfile
from datetime import datetime

import pandas as pd
import pytest
import yaml

from autorag.nodes.lexicalretrieval.bm25 import bm25_ingest
from autorag.nodes.semanticretrieval.vectordb import vectordb_ingest_api
from autorag.schema.metricinput import MetricInput
from autorag.utils.util import get_event_loop
from autorag.vectordb.chroma import Chroma

sample_ids = (
    [["id-1", "id-2", "id-3"], ["id-2", "id-3", "id-4"]],
    [["id-1", "id-4", "id-3"], ["id-2", "id-5", "id-4"]],
)
sample_scores = ([[1, 3, 5], [2, 4, 6]], [[4, 2, 6], [5, 3, 7]])
sample_ids_2 = (
    ["id-1", "id-2", "id-3", "id-4", "id-5"],
    ["id-1", "id-4", "id-3", "id-5", "id-2"],
)
sample_scores_2 = ([5, 3, 1, 0.4, 0.2], [6, 2, 1, 0.5, 0.1])

sample_ids_3 = (
    [
        ["id-1", "id-2", "id-3", "id-4", "id-5"],
        ["id-2", "id-3", "id-4", "id-5", "id-6"],
    ],
    [
        ["id-1", "id-4", "id-3", "id-5", "id-2"],
        ["id-2", "id-5", "id-4", "id-6", "id-3"],
    ],
)
sample_scores_3 = (
    [[5, 3, 1, 0.4, 0.2], [6, 4, 2, 1.4, 1.2]],
    [
        [6, 2, 1, 0.5, 0.1],
        [7, 3, 2, 1.5, 1.1],
    ],
)
sample_retrieval_gt_3 = [
    [["id-3"]],
    [["id-2"], ["id-6"]],
]

sample_ids_4 = (
    [
        ["id-1", "id-2", "id-3", "id-4", "id-5"],
        ["id-2", "id-10", "id-4", "id-5", "id-6"],
    ],
    [
        ["id-1", "id-4", "id-7", "id-5", "id-2"],
        ["id-2", "id-5", "id-4", "id-6", "id-3"],
    ],
)
sample_scores_4 = (
    [[5, 3, 1, 0.4, 0.2], [6, 10, 0, 1.4, 1.2]],
    [
        [6, 10, 0, 0.5, 0.1],
        [7, 4, 2, 1.5, 1.1],
    ],
)

sample_ids_non_overlap = (
    ["id-1", "id-2", "id-3", "id-4", "id-5"],
    ["id-6", "id-4", "id-3", "id-7", "id-2"],
)

previous_result = pd.DataFrame(
    {
        "qid": ["query-1", "query-2", "query-3"],
        "query": ["query-1", "query-2", "query-3"],
        "retrieval_gt": [
            [["id-1"], ["id-2"], ["id-3"]],
            [["id-1"], ["id-2"], ["id-3"]],
            [["id-1"], ["id-2"], ["id-3"]],
        ],
        "generation_gt": [["gen-1", "gen-2"], ["gen-1", "gen-2"], ["gen-1", "gen-2"]],
        "retrieved_contents_semantic": [
            ["doc-1", "doc-2", "doc-3"],
            ["doc-4", "doc-5", "doc-6"],
            ["doc-7", "doc-8", "doc-9"],
        ],
        "retrieved_contents_lexical": [
            ["doc-1", "doc-2", "doc-3"],
            ["doc-4", "doc-5", "doc-6"],
            ["doc-7", "doc-8", "doc-9"],
        ],
        "retrieved_ids_semantic": [
            ["id-1", "id-2", "id-3"],
            ["id-4", "id-5", "id-6"],
            ["id-7", "id-8", "id-9"],
        ],
        "retrieved_ids_lexical": [
            ["id-1", "id-2", "id-6"],
            ["id-4", "id-5", "id-3"],
            ["id-7", "id-8", "id-2"],
        ],
        "retrieve_scores_semantic": [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ],
        "retrieve_scores_lexical": [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ],
    }
)

modules_with_weights = {
    "top_k": 3,
    "strategy": {
        "metrics": ["retrieval_f1", "retrieval_recall", "retrieval_precision"],
    },
    "input_metrics": [
        MetricInput(retrieval_gt=[["id-1"]]),
        MetricInput(retrieval_gt=[["id-2"]]),
    ],
}


@pytest.fixture
def pseudo_project_dir():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as project_dir:
        corpus_df = pd.DataFrame(
            {
                "doc_id": [
                    "id-1",
                    "id-2",
                    "id-3",
                    "id-4",
                    "id-5",
                    "id-6",
                    "id-7",
                    "id-8",
                    "id-9",
                ],
                "contents": [
                    "doc-1",
                    "doc-2",
                    "doc-3",
                    "doc-4",
                    "doc-5",
                    "doc-6",
                    "doc-7",
                    "doc-8",
                    "doc-9",
                ],
                "metadata": [{"last_modified_date": datetime.now()} for _ in range(9)],
            }
        )
        qa_df = pd.DataFrame(
            {
                "qid": ["havertz", "hanjunsu"],
                "query": ["What is JAX?", "Donggeon twerking jax havertz?"],
                "retrieval_gt": [
                    [["id-1", "id-4"]],
                    [["id-3", "id-8"], ["id-2"]],
                ],
                "generation_gt": [
                    ["JAX is minsingjin."],
                    ["Donggeon is the god."],
                ],
            }
        )
        os.makedirs(os.path.join(project_dir, "data"))
        corpus_df.to_parquet(os.path.join(project_dir, "data", "corpus.parquet"))
        qa_df.to_parquet(os.path.join(project_dir, "data", "qa.parquet"))
        resource_dir = os.path.join(project_dir, "resources")
        os.makedirs(resource_dir)
        bm25_ingest(os.path.join(resource_dir, "bm25_porter_stemmer.pkl"), corpus_df)
        chroma_path = os.path.join(resource_dir, "chroma")

        vectordb_config_path = os.path.join(resource_dir, "vectordb.yaml")
        with open(vectordb_config_path, "w") as f:
            vectordb_dict = {
                "vectordb": [
                    {
                        "name": "test_default",
                        "db_type": "chroma",
                        "embedding_model": "mock",
                        "collection_name": "openai",
                        "path": chroma_path,
                        "similarity_metric": "cosine",
                    }
                ]
            }
            yaml.safe_dump(vectordb_dict, f)

        chroma = Chroma(
            embedding_model="mock",
            collection_name="openai",
            similarity_metric="cosine",
            client_type="persistent",
            path=chroma_path,
        )
        loop = get_event_loop()
        loop.run_until_complete(vectordb_ingest_api(chroma, corpus_df))
        yield project_dir


def base_hybrid_weights_node_test(hybrid_func, pseudo_project_dir, retrieve_scores):
    result = hybrid_func(
        project_dir=pseudo_project_dir,
        previous_result=previous_result,
        **modules_with_weights,
    )
    assert len(result["best_result"]) == 3
    assert isinstance(result["best_result"], pd.DataFrame)
    assert set(result["best_result"].columns) == {
        "retrieval_f1",
        "retrieval_recall",
        "retrieval_precision",
        "retrieved_contents",
        "retrieved_ids",
        "retrieve_scores",
    }
    assert result["best_result"]["retrieved_ids"].tolist()[0] == [
        "id-6",
        "id-2",
        "id-1",
    ]
    assert result["best_result"]["retrieve_scores"].tolist()[0] == pytest.approx(
        retrieve_scores
    )
    assert result["best_result"]["retrieved_contents"].tolist()[0] == [
        "doc-6",
        "doc-2",
        "doc-1",
    ]

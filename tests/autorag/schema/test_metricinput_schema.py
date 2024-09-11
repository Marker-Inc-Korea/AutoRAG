import pandas as pd
import pytest

from autorag.schema.metricinput import MetricInput


@pytest.fixture
def sample_metric_input():
    return MetricInput(
        query="test query",
        queries=["q1", "q2"],
        retrieval_gt_contents=[["gc1"], ["gc2"]]
    )


def test_default_constructor():
    metric_input = MetricInput()
    assert metric_input.query is None
    assert metric_input.queries is None
    assert metric_input.retrieval_gt_contents is None


def test_is_fields_notnone(sample_metric_input):
    assert sample_metric_input.is_fields_notnone(["query", "queries", "retrieval_gt_contents"])

    sample_metric_input.query = None
    assert not sample_metric_input.is_fields_notnone(["query", "queries", "retrieval_gt_contents"])


def test_from_dataframe_multiple_rows():
    df = pd.DataFrame({
        'query': ['test query 1', 'test query 2'],
        'queries': [['q1', 'q2'], ['q3', 'q4']],
        'retrieval_gt_contents': [['gc1', 'gc2'], ['gc3', 'gc4']]
    })

    metricinputs = MetricInput.from_dataframe(df)
    assert len(metricinputs) == 2
    assert metricinputs[0].query == 'test query 1'
    assert metricinputs[1].query == 'test query 2'


def test_from_dataframe_empty_values():
    df = pd.DataFrame({
        'query': ['', 'valid query'],
        'queries': [[], ['q1', 'q2']],
        'retrieval_gt_contents': [None, ['gc1', 'gc2']]
    })

    metricinputs = MetricInput.from_dataframe(df)
    assert len(metricinputs) == 2
    assert metricinputs[0].query is None
    assert metricinputs[0].queries is None
    assert metricinputs[0].retrieval_gt_contents is None
    assert metricinputs[1].query == 'valid query'
    assert metricinputs[1].queries == ['q1', 'q2']
    assert metricinputs[1].retrieval_gt_contents == ['gc1', 'gc2']

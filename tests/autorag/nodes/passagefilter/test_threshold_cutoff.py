from autorag.nodes.passagefilter import threshold_cutoff
from tests.autorag.nodes.passagefilter.test_passage_filter_base import queries_example, contents_example, \
    scores_example, ids_example, base_passage_filter_test, project_dir, previous_result, base_passage_filter_node_test


def test_threshold_cutoff():
    original_cutoff = threshold_cutoff.__wrapped__
    contents, ids, scores = original_cutoff(
        queries_example, contents_example, scores_example, ids_example, threshold=0.6)
    base_passage_filter_test(contents, ids, scores)


def test_threshold_cutoff_node():
    result_df = threshold_cutoff(
        project_dir=project_dir, previous_result=previous_result, threshold=0.9)
    base_passage_filter_node_test(result_df)

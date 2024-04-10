from autorag.nodes.passagefilter import similarity_threshold_cutoff
from tests.autorag.nodes.passagefilter.test_passage_filter_base import queries_example, contents_example, \
    scores_example, ids_example, base_passage_filter_test, project_dir, previous_result, base_passage_filter_node_test


def test_similarity_threshold_cutoff():
    original_cutoff = similarity_threshold_cutoff.__wrapped__
    contents, ids, scores = original_cutoff(
        queries_example, contents_example, scores_example, ids_example, threshold=0.85,
        embedding_model='openai_embed_3_large', batch=64)
    base_passage_filter_test(contents, ids, scores)


def test_similarity_threshold_cutoff_node():
    result_df = similarity_threshold_cutoff(
        project_dir=project_dir, previous_result=previous_result, threshold=0.9)
    base_passage_filter_node_test(result_df)

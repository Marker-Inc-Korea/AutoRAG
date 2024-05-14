from autorag.nodes.passagefilter import similarity_percentile_cutoff
from tests.autorag.nodes.passagefilter.test_passage_filter_base import queries_example, contents_example, \
    scores_example, ids_example, base_passage_filter_test, project_dir, previous_result, base_passage_filter_node_test


def test_similarity_percentile_cutoff():
    original_cutoff = similarity_percentile_cutoff.__wrapped__
    contents, ids, scores = original_cutoff(
        queries_example, contents_example, scores_example, ids_example, percentile=0.85,
        embedding_model='openai_embed_3_large', batch=64)
    num_top_k = int(len(contents_example[0]) * 0.85)
    assert len(contents[0]) == len(contents[1]) == num_top_k
    base_passage_filter_test(contents, ids, scores)


def test_similarity_percentile_cutoff_node():
    result_df = similarity_percentile_cutoff(
        project_dir=project_dir, previous_result=previous_result, percentile=0.9)
    base_passage_filter_node_test(result_df)

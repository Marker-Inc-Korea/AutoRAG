from autorag.nodes.passagefilter import similarity_threshold_cutoff
from tests.autorag.nodes.passagefilter.test_passage_filter_base import queries_example, contents_example, \
    scores_example, ids_example, base_passage_filter_test


def test_similarity_threshold_cutoff():
    original_cutoff = similarity_threshold_cutoff.__wrapped__
    contents, ids, scores = original_cutoff(
        queries_example, contents_example, scores_example, ids_example, threshold=0.5,
        embedding_model='openai_embed_3_large', batch=64)
    base_passage_filter_test(contents, ids, scores)

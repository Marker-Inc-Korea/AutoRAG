from autorag.nodes.passagereranker import sentence_transformer_reranker

from tests.autorag.nodes.passagereranker.test_passage_reranker_base import queries_example, contents_example, \
    scores_example, ids_example, base_reranker_test, project_dir, previous_result, base_reranker_node_test


def test_sentence_transformer():
    top_k = 3
    original_sentence_transformer = sentence_transformer_reranker.__wrapped__
    contents_result, id_result, score_result \
        = original_sentence_transformer(queries_example, contents_example, scores_example, ids_example, top_k)
    base_reranker_test(contents_result, id_result, score_result, top_k)


def test_sentence_transformer_node():
    top_k = 1
    result_df = sentence_transformer_reranker(project_dir=project_dir, previous_result=previous_result, top_k=top_k)
    base_reranker_node_test(result_df, top_k)

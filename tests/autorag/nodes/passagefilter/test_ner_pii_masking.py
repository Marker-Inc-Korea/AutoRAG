from autorag.nodes.passagefilter import ner_pii_masking
from tests.autorag.nodes.passagefilter.test_passage_filter_base import base_passage_filter_test, contents_example, \
    ids_example, scores_example, project_dir, previous_result, base_passage_filter_node_test


def test_ner_pii_masking():
    original_ner = ner_pii_masking.__wrapped__
    contents, ids, scores = original_ner(contents_example, scores_example, ids_example)
    assert contents[1][3] == "[PER_0] is one of the members of [ORG_34]."
    base_passage_filter_test(contents, ids, scores)


def test_ner_pii_masking_node():
    result_df = ner_pii_masking(project_dir=project_dir, previous_result=previous_result)
    base_passage_filter_node_test(result_df)

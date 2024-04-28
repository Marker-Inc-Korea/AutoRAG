from autorag.nodes.passageaugmenter import recursive_chunk

from tests.autorag.nodes.passageaugmenter.test_base_passage_augmenter import ids_list, project_dir, \
    previous_result, contents_list, corpus_data


def test_recursive_chunk():
    ids, contents = recursive_chunk.__wrapped__(ids_list, contents_list)
    assert len(ids) == len(contents) == 3
    assert len(ids[0]) == len(contents[0]) == 23


def test_recursive_chunk_node():
    result_df = recursive_chunk(project_dir=project_dir, previous_result=previous_result, top_k=2)
    contents = result_df["retrieved_contents"].tolist()
    ids = result_df["retrieved_ids"].tolist()
    scores = result_df["retrieve_scores"].tolist()
    assert len(contents) == len(ids) == len(scores) == 2
    assert len(contents[0]) == len(ids[0]) == len(scores[0]) == 2
    for content_list, id_list, score_list in zip(contents, ids, scores):
        for i, (content, _id, score) in enumerate(zip(content_list, id_list, score_list)):
            assert isinstance(content, str)
            assert isinstance(_id, str)
            assert isinstance(score, float)
            assert _id in corpus_data["doc_id"].tolist()
            if i >= 1:
                assert score_list[i - 1] >= score_list[i]

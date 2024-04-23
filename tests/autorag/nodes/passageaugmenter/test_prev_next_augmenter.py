from autorag.nodes.passageaugmenter import prev_next_augmenter

from tests.autorag.nodes.passageaugmenter.test_base_passage_augmenter import ids_list, project_dir, \
    previous_result, corpus_data, doc_id_list


def test_prev_next_augmenter_next():
    results = prev_next_augmenter.__wrapped__(ids_list, corpus_data, num_passages=1, mode='next')
    assert results == [[doc_id_list[1], doc_id_list[2]],
                       [doc_id_list[3], doc_id_list[4]],
                       [doc_id_list[0], doc_id_list[1], doc_id_list[29]]]


def test_prev_next_augmenter_prev():
    results = prev_next_augmenter.__wrapped__(ids_list, corpus_data, num_passages=1, mode='prev')
    assert results == [[doc_id_list[0], doc_id_list[1]],
                       [doc_id_list[2], doc_id_list[3]],
                       [doc_id_list[0], doc_id_list[28], doc_id_list[29]]]


def test_prev_next_augmenter_both():
    results = prev_next_augmenter.__wrapped__(ids_list, corpus_data, num_passages=1, mode='both')
    assert results == [[doc_id_list[0], doc_id_list[1], doc_id_list[2]],
                       [doc_id_list[2], doc_id_list[3], doc_id_list[4]],
                       [doc_id_list[0], doc_id_list[1], doc_id_list[28], doc_id_list[29]]]


def test_prev_next_augmenter_multi_passages():
    results = prev_next_augmenter.__wrapped__(ids_list, corpus_data, num_passages=3, mode='prev')
    assert results == [[doc_id_list[0], doc_id_list[1]],
                       [doc_id_list[0], doc_id_list[1], doc_id_list[2], doc_id_list[3]],
                       [doc_id_list[0], doc_id_list[26], doc_id_list[27], doc_id_list[28], doc_id_list[29]]]


def test_prev_next_augmenter_node():
    result_df = prev_next_augmenter(project_dir=project_dir, previous_result=previous_result, mode='next', top_k=2)
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
            assert content == corpus_data[corpus_data["doc_id"] == _id]["contents"].values[0]
            if i >= 1:
                assert score_list[i - 1] >= score_list[i]

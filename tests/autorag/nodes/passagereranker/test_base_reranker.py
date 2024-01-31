from uuid import uuid4

queries_example = ["What is the capital of France?",
                   "How many members are in Newjeans?"]
contents_example = [["NomaDamas is Great Team", "Paris is the capital of France.", "havertz is suck at soccer"],
                    ["i am hungry", "LA is a country in the United States.", "Newjeans has 5 members."]]
ids_example = [[str(uuid4()) for _ in range(len(contents_example[0]))],
               [str(uuid4()) for _ in range(len(contents_example[1]))]]
scores_example = [[0.1, 0.8, 0.1], [0.1, 0.2, 0.7]]


def base_reranker_test(contents, ids, scores):
    assert len(contents) == len(ids) == len(scores) == 2
    for content_list, id_list, score_list in zip(contents, ids, scores):
        assert isinstance(content_list, list)
        assert isinstance(id_list, list)
        assert isinstance(score_list, list)
        for content, _id, score in zip(content_list, id_list, score_list):
            assert isinstance(content, str)
            assert isinstance(_id, str)
            assert isinstance(score, float)
        for i in range(1, len(score_list)):
            assert score_list[i - 1] >= score_list[i]

    assert contents[0][0] == "Paris is the capital of France."
    assert contents[1][0] == "Newjeans has 5 members."

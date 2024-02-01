import os
import pathlib
import pandas as pd

from uuid import uuid4

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
project_dir = os.path.join(root_dir, "resources", "sample_project")
qa_data = pd.read_parquet(os.path.join(project_dir, "data", "qa.parquet"))
corpus_data = pd.read_parquet(os.path.join(project_dir, "data", "corpus.parquet"))
previous_result = qa_data.sample(2)

queries_example = ["What is the capital of France?",
                   "How many members are in Newjeans?"]
contents_example = [["NomaDamas is Great Team", "Paris is the capital of France.", "havertz is suck at soccer"],
                    ["i am hungry", "LA is a country in the United States.", "Newjeans has 5 members."]]
ids_example = [[str(uuid4()) for _ in range(len(contents_example[0]))],
               [str(uuid4()) for _ in range(len(contents_example[1]))]]
scores_example = [[0.1, 0.8, 0.1], [0.1, 0.2, 0.7]]
f1_example = [0.4, 0.4]
recall_example = [1.0, 1.0]

previous_result['query'] = queries_example
previous_result['retrieved_contents'] = contents_example
previous_result['retrieved_ids'] = ids_example
previous_result['retrieve_scores'] = scores_example
previous_result['retrieval_f1'] = f1_example
previous_result['retrieval_recall'] = recall_example


def base_reranker_test(contents, ids, scores, top_k):
    assert len(contents) == len(ids) == len(scores) == 2
    assert len(contents[0]) == len(ids[0]) == len(scores[0]) == top_k
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
    assert ids[0][0] in ids_example[0][1]
    assert contents[1][0] == "Newjeans has 5 members."
    assert ids[1][0] in ids_example[1][2]


def base_reranker_node_test(result_df, top_k):
    contents = result_df["retrieved_contents"].tolist()
    ids = result_df["retrieved_ids"].tolist()
    scores = result_df["retrieve_scores"].tolist()
    base_reranker_test(contents, ids, scores, top_k)

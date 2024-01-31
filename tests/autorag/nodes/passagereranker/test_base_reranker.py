from uuid import uuid4

queries_example = ["What is the capital of France?",
                   "How many members are in Newjeans?"]
contents_example = [["NomaDamas is Great Team", "Paris is the capital of France.", "havertz is suck at soccer"],
                    ["i am hungry", "LA is a country in the United States.", "Newjeans has 5 members."]]
ids_example = [[uuid4() for _ in range(len(contents_example[0]))], [uuid4() for _ in range(len(contents_example[1]))]]
scores_example = [[0.1, 0.8, 0.1], [0.1, 0.2, 0.7]]


def rerank_test(reranker):
    result = reranker(queries_example, contents_example, scores_example, ids_example)

    first_contents = result[0][0]
    second_contents = result[1][0]

    assert first_contents[0] == "Paris is the capital of France."
    assert second_contents[0] == "Newjeans has 5 members."

from typing import List

import pandas as pd

queries = [
	"What is the capital of France?",
	"What is the meaning of life?",
]
retrieved_contents = [
	[
		"Paris is the capital of France.",
		"France is a country in Europe.",
		"France is a member of the EU.",
	],
	[
		"The meaning of life is 42.",
		"The meaning of life is to be happy.",
		"The meaning of life is to be kind.",
	],
]

df = pd.DataFrame(
	{
		"query": queries,
		"retrieved_contents": retrieved_contents,
		"retrieved_ids": [["id-1", "id-2", "id-3"], ["id-4", "id-5", "id-6"]],
		"retrieve_scores": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
	}
)


def check_result(result: List[str]):
	assert len(result) == len(queries)
	for r in result:
		assert isinstance(r, str)
		assert len(r) > 0
		assert bool(r) is True

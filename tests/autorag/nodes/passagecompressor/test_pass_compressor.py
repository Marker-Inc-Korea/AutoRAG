import pandas as pd

from autorag.nodes.passagecompressor import PassCompressor

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


def test_pass_compressor():
	df = pd.DataFrame(
		{
			"query": queries,
			"retrieved_contents": retrieved_contents,
			"retrieved_ids": [["id-1", "id-2", "id-3"], ["id-4", "id-5", "id-6"]],
			"retrieve_scores": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
		}
	)
	result_df = PassCompressor.run_evaluator("project_dir", df)
	assert result_df["retrieved_contents"].tolist() == retrieved_contents

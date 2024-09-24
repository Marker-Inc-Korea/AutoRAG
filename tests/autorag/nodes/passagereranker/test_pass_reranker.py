from autorag.nodes.passagereranker import PassReranker
from tests.autorag.nodes.passagereranker.test_passage_reranker_base import (
	project_dir,
	previous_result,
	ids_example,
)


def test_pass_reranker():
	top_k = 1
	result_df = PassReranker.run_evaluator(
		project_dir=project_dir, previous_result=previous_result, top_k=top_k
	)
	assert "retrieved_contents" in result_df.columns
	assert "retrieved_ids" in result_df.columns
	assert "retrieve_scores" in result_df.columns

	contents = result_df["retrieved_contents"].tolist()
	assert contents == [["NomaDamas is Great Team"], ["i am hungry"]]
	ids = result_df["retrieved_ids"].tolist()
	assert ids == [[ids_example[0][0]], [ids_example[1][0]]]
	scores = result_df["retrieve_scores"].tolist()
	assert scores == [[0.1], [0.1]]

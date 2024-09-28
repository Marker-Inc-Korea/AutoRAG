import pandas as pd

from autorag.data.qa.schema import QA

passage = """NewJeans (뉴진스) is a 5-member girl group under ADOR and HYBE Labels.
The members consist of Minji, Hanni, Danielle, Haerin, and Hyein.
They released their debut single “Attention” on July 22, 2022,
followed by their debut extended play, New Jeans, which was released on August 1, 2022."""
question = "How many members are in the New Jeans?"
qa_df = pd.DataFrame(
	{
		"qid": ["jax"],
		"retrieval_gt": [[["havertz"]]],
		"retrieval_gt_contents": [[[passage]]],
		"query": [question],
	}
)


def check_generation_gt(result_qa: QA):
	assert isinstance(result_qa, QA)
	assert isinstance(result_qa.data, pd.DataFrame)
	assert set(result_qa.data.columns) == {
		"qid",
		"retrieval_gt",
		"retrieval_gt_contents",
		"query",
		"generation_gt",
	}
	assert len(result_qa.data) == len(qa_df)
	assert result_qa.data["generation_gt"].iloc[0]
	assert all(isinstance(x, list) for x in result_qa.data["generation_gt"].tolist())

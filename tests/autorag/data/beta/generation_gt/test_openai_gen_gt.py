from openai import AsyncClient


from autorag.data.beta.generation_gt.openai_gen_gt import (
	make_concise_gen_gt,
	make_basic_gen_gt,
)
from autorag.schema.data import QA
from tests.autorag.data.beta.generation_gt.base_test_generation_gt import (
	qa_df,
	check_generation_gt,
)

client = AsyncClient()


def test_make_concise_gen_gt():
	qa = QA(qa_df)
	result_qa = qa.batch_apply(
		lambda row: make_concise_gen_gt(
			row, client, model_name="gpt-4o-mini-2024-07-18"
		)
	)
	check_generation_gt(result_qa)


def test_make_basic_gen_gt():
	qa = QA(qa_df)
	result_qa = qa.batch_apply(lambda row: make_basic_gen_gt(row, client))
	check_generation_gt(result_qa)

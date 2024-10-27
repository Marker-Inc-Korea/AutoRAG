from llama_index.core.llms import MockLLM

from autorag.data.qa.generation_gt.llama_index_gen_gt import (
	make_concise_gen_gt,
	make_basic_gen_gt,
	make_custom_gen_gt,
)
from autorag.data.qa.schema import QA
from tests.autorag.data.qa.generation_gt.base_test_generation_gt import (
	qa_df,
	check_generation_gt,
)

llm = MockLLM()


def test_make_concise_gen_gt():
	qa = QA(qa_df)
	result_qa = qa.batch_apply(make_concise_gen_gt, llm=llm)
	check_generation_gt(result_qa)


def test_make_basic_gen_gt():
	qa = QA(qa_df)
	result_qa = qa.batch_apply(make_basic_gen_gt, llm=llm)
	check_generation_gt(result_qa)


def test_make_basic_gen_gt_ko():
	qa = QA(qa_df)
	result_qa = qa.batch_apply(make_basic_gen_gt, llm=llm, lang="ko")
	check_generation_gt(result_qa)


def test_make_basic_gen_gt_ja():
	qa = QA(qa_df)
	result_qa = qa.batch_apply(make_basic_gen_gt, llm=llm, lang="ja")
	check_generation_gt(result_qa)


def test_make_multiple_gen_gt():
	qa = QA(qa_df)
	result_qa = qa.batch_apply(make_basic_gen_gt, llm=llm).batch_apply(
		make_concise_gen_gt, llm=llm
	)
	check_generation_gt(result_qa)
	assert all(len(x) == 2 for x in result_qa.data["generation_gt"].tolist())


def test_make_custom_gen_gt():
	system_prompt = """As an expert AI assistant focused on providing accurate and concise responses, generate an answer for the question based strictly on the given **Text**.
Your answer should:
- Be derived only from the provided **Text** without using pre-trained knowledge.
- Contain only the answer itself, whether it is a full sentence or a single word, without any introductory phrases or extra commentary.
- If the information is unavailable within the **Text**, respond with "I don't know."
- be same as query's language"""
	qa = QA(qa_df)
	result_qa = qa.batch_apply(make_custom_gen_gt, llm=llm, system_prompt=system_prompt)
	check_generation_gt(result_qa)

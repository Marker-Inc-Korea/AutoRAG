import pandas as pd
import pytest

from autorag.nodes.generator import Vllm
from tests.autorag.nodes.generator.test_generator_base import (
	prompts,
	check_generated_texts,
	check_generated_tokens,
	check_generated_log_probs,
)


@pytest.mark.skip(reason="This test needs CUDA enabled machine with vllm installed.")
def test_vllm():
	previous_result = pd.DataFrame(
		{"prompts": prompts, "qid": ["id-1", "id-2", "id-3"]}
	)
	result_df = Vllm.run_evaluator(
		project_dir=".",
		previous_result=previous_result,
		tensor_parallel_size=1,
		llm="facebook/opt-125m",
		max_tokens=5,
		temperature=0.5,
	)
	tokens = result_df["generated_tokens"].tolist()
	log_probs = result_df["generated_log_probs"].tolist()
	check_generated_texts(result_df["generated_texts"].tolist())
	check_generated_tokens(tokens)
	check_generated_log_probs(log_probs)
	assert all(len(tokens[i]) == len(log_probs[i]) for i in range(len(tokens)))

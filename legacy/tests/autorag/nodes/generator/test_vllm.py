import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

import pandas as pd

from autorag.nodes.generator import Vllm
from tests.autorag.nodes.generator.test_generator_base import (  # noqa: E402
	prompts,
	chat_prompts,
	check_generated_texts,
	check_generated_tokens,
	check_generated_log_probs,
	check_generated_texts_chat,
	check_generated_tokens_chat,
	check_generated_log_probs_chat,
)


@dataclass
class FakeLogProb:
	logprob: float


class FakeSamplingParams:
	@classmethod
	def from_optional(cls, **kwargs):
		return kwargs

	def __init__(self, **kwargs):
		self.kwargs = kwargs


class FakeLLM:
	def __init__(self, model, **kwargs):
		self.model = model
		self.kwargs = kwargs

	def _outputs(self, prompts):
		return [
			SimpleNamespace(
				outputs=[
					SimpleNamespace(
						text=f"generated: {prompt}",
						token_ids=[1, 2, 3],
						logprobs=[
							{1: FakeLogProb(-0.1)},
							{2: FakeLogProb(-0.2)},
							{3: FakeLogProb(-0.3)},
						],
					)
				]
			)
			for prompt in prompts
		]

	def generate(self, prompts, sampling_params, **kwargs):
		return self._outputs(prompts)

	def chat(self, prompts, sampling_params, **kwargs):
		return self._outputs(
			[message["content"] for prompt in prompts for message in prompt[-1:]]
		)


def install_fake_vllm(monkeypatch):
	fake_vllm = types.ModuleType("vllm")
	fake_outputs = types.ModuleType("vllm.outputs")
	fake_logprobs = types.ModuleType("vllm.logprobs")
	setattr(fake_vllm, "LLM", FakeLLM)
	setattr(fake_vllm, "SamplingParams", FakeSamplingParams)
	setattr(fake_outputs, "RequestOutput", SimpleNamespace)
	setattr(fake_logprobs, "SampleLogprobs", dict)
	monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
	monkeypatch.setitem(sys.modules, "vllm.outputs", fake_outputs)
	monkeypatch.setitem(sys.modules, "vllm.logprobs", fake_logprobs)


def test_vllm(monkeypatch):
	install_fake_vllm(monkeypatch)

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


def test_vllm_chat_prompt(monkeypatch):
	install_fake_vllm(monkeypatch)

	previous_result = pd.DataFrame(
		{"prompts": chat_prompts, "qid": ["id-1", "id-2", "id-3"]}
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
	check_generated_texts_chat(result_df["generated_texts"].tolist())
	check_generated_tokens_chat(tokens)
	check_generated_log_probs_chat(log_probs)
	assert all(len(tokens[i]) == len(log_probs[i]) for i in range(len(tokens)))


def test_vllm_chat_prompt_think(monkeypatch):
	install_fake_vllm(monkeypatch)

	previous_result = pd.DataFrame(
		{"prompts": chat_prompts, "qid": ["id-1", "id-2", "id-3"]}
	)
	result_df = Vllm.run_evaluator(
		project_dir=".",
		previous_result=previous_result,
		tensor_parallel_size=1,
		llm="facebook/opt-125m",
		max_tokens=5,
		temperature=0.5,
		thinking=True,
	)
	tokens = result_df["generated_tokens"].tolist()
	log_probs = result_df["generated_log_probs"].tolist()
	check_generated_texts_chat(result_df["generated_texts"].tolist())
	check_generated_tokens_chat(tokens)
	check_generated_log_probs_chat(log_probs)
	assert all(len(tokens[i]) == len(log_probs[i]) for i in range(len(tokens)))

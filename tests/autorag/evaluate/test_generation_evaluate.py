import os
from unittest.mock import patch

import openai
import pandas as pd
import pytest
from llama_index.embeddings.openai import OpenAIEmbedding
from openai.types.chat import (
	ChatCompletion,
	ChatCompletionMessage,
	ChatCompletionTokenLogprob,
)
from openai.types.chat.chat_completion import Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_token_logprob import TopLogprob
from transformers import AutoTokenizer

from autorag.evaluation.generation import evaluate_generation
from autorag.schema.metricinput import MetricInput
from tests.mock import mock_get_text_embedding_batch

generation_gts = [
	["The dog had bit the man.", "The man had bitten the dog."],
	["I want to be a artist, but I end up to be a programmer."],
	[
		"To be a artist these days, you can overcome by AI.",
		"To be a programmer these days, you can overcome by AI.",
		"To be a lawyer these days, you can overcome by AI.",
	],
]
pseudo_generations = [
	"The dog bit the man.",
	"It really like to be a programmer, but I think artist is my passion.",
	"To be a artist these days, you can overcome by AI.",
]

tokenizer = AutoTokenizer.from_pretrained("gpt2")
pseudo_tokens = list(map(lambda x: tokenizer.tokenize(x), pseudo_generations))
pseudo_log_probs = list(map(lambda x: [0.1] * len(x), pseudo_tokens))


@evaluate_generation(
	metric_inputs=[MetricInput(generation_gt=gen_gt) for gen_gt in generation_gts],
	metrics=[
		{"metric_name": "bleu"},
		{"metric_name": "meteor"},
		{"metric_name": "rouge"},
		{"metric_name": "sem_score", "embedding_model": "openai"},
		{"metric_name": "g_eval"},
	],
)
def pseudo_generation():
	return pseudo_generations


@evaluate_generation(
	metric_inputs=[MetricInput(generation_gt=gen_gt) for gen_gt in generation_gts],
	metrics=["bleu", "meteor", "donggeon_metric"],
)
def pseudo_generation_with_log_probs():
	return pseudo_generations, pseudo_tokens, pseudo_log_probs


@evaluate_generation(
	metric_inputs=[MetricInput(generation_gt=gen_gt) for gen_gt in generation_gts],
	metrics=[
		{"metric_name": "bleu"},
		{"metric_name": "sem_score", "embedding_model": "openai"},
	],
)
def pseudo_generation_dict_metrics():
	return pseudo_generations


async def mock_g_eval_openai_create(*args, **kwargs):
	sample_choice = Choice(
		finish_reason="stop",
		index=0,
		message=ChatCompletionMessage(
			content="2",
			role="assistant",
		),
		logprobs=ChoiceLogprobs(
			content=[
				ChatCompletionTokenLogprob(
					token="2",
					logprob=2.8,
					top_logprobs=[
						TopLogprob(token="2", logprob=2.8),
					],
				)
			],
		),
	)
	if "n" not in kwargs.keys():
		n = 1
	else:
		n = kwargs["n"]
	return ChatCompletion(
		id="_id",
		choices=[sample_choice] * n,
		created=1713363661,
		model=kwargs["model"],
		object="chat.completion",
	)


@patch.object(
	openai.resources.chat.completions.AsyncCompletions,
	"create",
	mock_g_eval_openai_create,
)
@patch.object(
	OpenAIEmbedding,
	"get_text_embedding_batch",
	mock_get_text_embedding_batch,
)
def test_evaluate_generation():
	os.environ["OPENAI_API_KEY"] = "mock_openai_api_key"
	result_df = pseudo_generation()
	assert isinstance(result_df, pd.DataFrame)
	assert len(result_df) == 3
	assert len(result_df.columns) == 6
	assert set(result_df.columns) == {
		"generated_texts",
		"bleu",
		"meteor",
		"rouge",
		"sem_score",
		"g_eval",
	}

	with pytest.warns():
		result_df_log_probs = pseudo_generation_with_log_probs()
	assert isinstance(result_df_log_probs, pd.DataFrame)
	assert len(result_df_log_probs) == 3
	assert len(result_df_log_probs.columns) == 5
	assert set(result_df_log_probs.columns) == {
		"generated_texts",
		"bleu",
		"meteor",
		"generated_tokens",
		"generated_log_probs",
	}

	assert result_df_log_probs["generated_texts"].tolist() == pseudo_generations
	assert result_df_log_probs["generated_tokens"].tolist() == pseudo_tokens
	assert result_df_log_probs["generated_log_probs"].tolist() == pseudo_log_probs

	assert all(
		list(
			map(
				lambda x: x[0] == pytest.approx(x[1], 0.001),
				zip(result_df["bleu"].tolist(), [51.1507, 23.5783, 100.0]),
			)
		)
	)
	assert all(
		list(
			map(
				lambda x: x[0] == pytest.approx(x[1], 0.001),
				zip(result_df["meteor"].tolist(), [0.853462, 0.5859375, 1.0]),
			)
		)
	)
	assert all(
		list(
			map(
				lambda x: x[0] == pytest.approx(x[1], 0.001),
				zip(result_df["rouge"].tolist(), [0.909, 0.35714, 1.0]),
			)
		)
	)

	result_df = pseudo_generation_dict_metrics()
	assert isinstance(result_df, pd.DataFrame)
	assert len(result_df) == 3
	assert len(result_df.columns) == 3
	assert set(result_df.columns) == {"generated_texts", "bleu", "sem_score"}

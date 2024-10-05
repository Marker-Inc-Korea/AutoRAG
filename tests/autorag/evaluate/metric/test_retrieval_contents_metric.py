import pytest

from autorag.evaluation.metric.retrieval_contents import (
	single_token_f1,
	retrieval_token_f1,
	retrieval_token_precision,
	retrieval_token_recall,
)
from autorag.schema.metricinput import MetricInput

gt = [
	[["Enough for drinking water", "Just looking for a water bottle"]],
	[["Do you want to buy some?"]],
	[[""]],
	[[]],
]
pred = [
	[
		"Enough for mixing water",
		"I want to do a nothing",
		"Just looking is a very healthy",
	],
	["Do you want to buy some?", "I want to buy some", "I want to buy some water"],
	["Who is son? He is great player in the world"],
	["i love havertz", "i love kai havertz"],
]
metric_inputs = [
	MetricInput(retrieval_gt_contents=g, retrieved_contents=p) for g, p in zip(gt, pred)
]


def test_single_token_f1():
	precision, recall, f1 = single_token_f1(gt[0][0][0], pred[0][0])
	assert precision == 0.75
	assert recall == 0.75
	assert f1 == 0.75

	precision, recall, f1 = single_token_f1(gt[0][0][1], pred[0][2])
	assert precision == 0.4
	assert recall == 0.4
	assert f1 == pytest.approx(0.4)


def test_retrieval_token_f1():
	f1 = retrieval_token_f1.__wrapped__(
		MetricInput(retrieval_gt_contents=gt[0], retrieved_contents=pred[0])
	)
	assert f1 == pytest.approx(0.38333, rel=0.001)

	f1 = retrieval_token_f1.__wrapped__(
		MetricInput(retrieval_gt_contents=gt[1], retrieved_contents=pred[1])
	)
	assert f1 == pytest.approx(0.797979, rel=0.001)

	result_f1 = retrieval_token_f1(metric_inputs=metric_inputs)
	assert result_f1 == pytest.approx([0.38333, 0.797979, None, None], rel=0.001)


def test_retrieval_token_precision():
	result_precision = retrieval_token_precision(metric_inputs=metric_inputs)
	assert result_precision == pytest.approx(
		[0.383333, 0.8222222, None, None], rel=0.001
	)


def test_retrieval_token_recall():
	result_recall = retrieval_token_recall(metric_inputs=metric_inputs)
	assert result_recall == pytest.approx([0.383333, 0.777777, None, None], rel=0.001)

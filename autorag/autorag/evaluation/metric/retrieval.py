import itertools
import math

from autorag.evaluation.metric.util import autorag_metric
from autorag.schema.metricinput import MetricInput


@autorag_metric(fields_to_check=["retrieval_gt", "retrieved_ids"])
def retrieval_f1(metric_input: MetricInput):
	"""
	Compute f1 score for retrieval.

	:param metric_input: The MetricInput schema for AutoRAG metric.
	:return: The f1 score.
	"""
	recall_score = retrieval_recall.__wrapped__(metric_input)
	precision_score = retrieval_precision.__wrapped__(metric_input)
	if recall_score + precision_score == 0:
		return 0
	else:
		return 2 * (recall_score * precision_score) / (recall_score + precision_score)


@autorag_metric(fields_to_check=["retrieval_gt", "retrieved_ids"])
def retrieval_recall(metric_input: MetricInput) -> float:
	gt, pred = metric_input.retrieval_gt, metric_input.retrieved_ids

	gt_sets = [frozenset(g) for g in gt]
	pred_set = set(pred)
	hits = sum(any(pred_id in gt_set for pred_id in pred_set) for gt_set in gt_sets)
	recall = hits / len(gt) if len(gt) > 0 else 0.0
	return recall


@autorag_metric(fields_to_check=["retrieval_gt", "retrieved_ids"])
def retrieval_precision(metric_input: MetricInput) -> float:
	gt, pred = metric_input.retrieval_gt, metric_input.retrieved_ids

	gt_sets = [frozenset(g) for g in gt]
	pred_set = set(pred)
	hits = sum(any(pred_id in gt_set for gt_set in gt_sets) for pred_id in pred_set)
	precision = hits / len(pred) if len(pred) > 0 else 0.0
	return precision


@autorag_metric(fields_to_check=["retrieval_gt", "retrieved_ids"])
def retrieval_ndcg(metric_input: MetricInput) -> float:
	gt, pred = metric_input.retrieval_gt, metric_input.retrieved_ids

	gt_sets = [frozenset(g) for g in gt]
	pred_set = set(pred)
	relevance_scores = {
		pred_id: 1 if any(pred_id in gt_set for gt_set in gt_sets) else 0
		for pred_id in pred_set
	}

	dcg = sum(
		(2 ** relevance_scores[doc_id] - 1) / math.log2(i + 2)
		for i, doc_id in enumerate(pred)
	)

	len_flatten_gt = len(list(itertools.chain.from_iterable(gt)))
	len_pred = len(pred)
	ideal_pred = [1] * min(len_flatten_gt, len_pred) + [0] * max(
		0, len_pred - len_flatten_gt
	)
	idcg = sum(relevance / math.log2(i + 2) for i, relevance in enumerate(ideal_pred))

	ndcg = dcg / idcg if idcg > 0 else 0
	return ndcg


@autorag_metric(fields_to_check=["retrieval_gt", "retrieved_ids"])
def retrieval_mrr(metric_input: MetricInput) -> float:
	"""
	Reciprocal Rank (RR) is the reciprocal of the rank of the first relevant item.
	Mean of RR in whole queries is MRR.
	"""
	gt, pred = metric_input.retrieval_gt, metric_input.retrieved_ids

	# Flatten the ground truth list of lists into a single set of relevant documents
	gt_sets = [frozenset(g) for g in gt]

	rr_list = []
	for gt_set in gt_sets:
		for i, pred_id in enumerate(pred):
			if pred_id in gt_set:
				rr_list.append(1.0 / (i + 1))
				break
	return sum(rr_list) / len(gt_sets) if rr_list else 0.0


@autorag_metric(fields_to_check=["retrieval_gt", "retrieved_ids"])
def retrieval_map(metric_input: MetricInput) -> float:
	"""
	Mean Average Precision (MAP) is the mean of Average Precision (AP) for all queries.
	"""
	gt, pred = metric_input.retrieval_gt, metric_input.retrieved_ids

	gt_sets = [frozenset(g) for g in gt]

	ap_list = []

	for gt_set in gt_sets:
		pred_hits = [1 if pred_id in gt_set else 0 for pred_id in pred]
		precision_list = [
			sum(pred_hits[: i + 1]) / (i + 1)
			for i, hit in enumerate(pred_hits)
			if hit == 1
		]
		ap_list.append(
			sum(precision_list) / len(precision_list) if precision_list else 0.0
		)

	return sum(ap_list) / len(gt_sets) if ap_list else 0.0

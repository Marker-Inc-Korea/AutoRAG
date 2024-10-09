from autorag import embedding_models
from autorag.evaluation.util import cast_metrics


def test_cast_metrics():
	metric1 = ["bleu", "meteor", "rouge"]
	metric_names, metric_params = cast_metrics(metric1)
	assert metric_names == ["bleu", "meteor", "rouge"]
	assert metric_params == [{}, {}, {}]

	metric2 = [
		{"metric_name": "bleu"},
		{"metric_name": "meteor"},
		{"metric_name": "rouge"},
	]
	metric_names, metric_params = cast_metrics(metric2)
	assert metric_names == ["bleu", "meteor", "rouge"]
	assert metric_params == [{}, {}, {}]

	metric3 = [
		{"metric_name": "bleu"},
		{"metric_name": "sem_score", "embedding_model": "openai"},
	]
	metric_names, metric_params = cast_metrics(metric3)
	assert metric_names == ["bleu", "sem_score"]
	assert metric_params == [{}, {"embedding_model": embedding_models["openai"]()}]

	metric4 = [
		{"metric_name": "bleu", "extra_param": "extra"},
		{
			"metric_name": "sem_score",
			"embedding_model": "openai",
			"extra_param": "extra",
		},
	]
	metric_names, metric_params = cast_metrics(metric4)
	assert metric_names == ["bleu", "sem_score"]
	assert metric_params == [
		{"extra_param": "extra"},
		{"embedding_model": embedding_models["openai"](), "extra_param": "extra"},
	]

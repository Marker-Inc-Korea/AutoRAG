import asyncio
import itertools
import os
from typing import List, Optional

import evaluate
import nltk
import pandas as pd
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import AsyncOpenAI
from pydantic import BaseModel
from rouge_score import tokenizers
from rouge_score.rouge_scorer import RougeScorer
from sacrebleu.metrics.bleu import BLEU

from autorag.embedding.base import embedding_models
from autorag.evaluation.metric.deepeval_prompt import FaithfulnessTemplate
from autorag.evaluation.metric.util import (
	autorag_metric_loop,
	calculate_cosine_similarity,
)
from autorag.nodes.generator import OpenAILLM
from autorag.nodes.generator.base import BaseGenerator
from autorag.schema.metricinput import MetricInput
from autorag.support import get_support_modules
from autorag.utils.util import (
	get_event_loop,
	process_batch,
	openai_truncate_by_token,
	convert_inputs_to_list,
	pop_params,
	empty_cuda_cache,
)


@convert_inputs_to_list
def huggingface_evaluate(
	instance, key: str, metric_inputs: List[MetricInput], **kwargs
) -> List[float]:
	"""
	Compute huggingface evaluate metric.

	:param instance: The instance of huggingface evaluates metric.
	:param key: The key to retrieve result score from huggingface evaluate result.
	:param metric_inputs: A list of MetricInput schema
	:param kwargs: The additional arguments for metric function.
	:return: The list of scores.
	"""

	def compute_score(gt: List[str], pred: str) -> float:
		return max(
			list(
				map(
					lambda x: instance.compute(
						predictions=[pred], references=[x], **kwargs
					)[key],
					gt,
				)
			)
		)

	result = list(
		map(lambda x: compute_score(x.generation_gt, x.generated_texts), metric_inputs)
	)
	return result


def make_generator_instance(generator_module_type: str, llm: str, **kwargs):
	llm_class = get_support_modules(generator_module_type)
	init_params = pop_params(llm_class.__init__, kwargs)
	return llm_class(project_dir="", llm=llm, **init_params)


@autorag_metric_loop(fields_to_check=["retrieval_gt_contents", "generated_texts"])
def deepeval_faithfulness(
	metric_inputs: List[MetricInput],
	generator_module_type: str = "openai_llm",
	lang: str = "en",
	llm: str = "gpt-4o-2024-08-06",
	batch: int = 16,
	**kwargs,
) -> List[float]:
	"""
	Compute deepeval faithfulness metric.
	Its default model is gpt-4o-2024-08-06.
	Since it uses OpenAI model, please be aware of the expensive cost.

	:param metric_inputs: The list of MetricInput schema (Required Field -> "generation_gt", "generated_texts")
	:param generator_module_type: Generator module type.
	The default is "openai_llm".
		You can use like "llama_index_llm" or "vllm".
	:param lang: The prompt language that you want to use.
	"en", "ko" and "ja" are supported.
	Korean prompt is not officially supported by DeepEval, but it can be translated by AutoRAG developers.
		Default is "en".
	:param llm: The model name to use for generation.
		Or llm if using llama_index_llm.
		The default is "gpt-4o-2024-08-06".
	:param batch: The batch size for processing.
		Default is 16.
	:param kwargs: The extra parameters for initializing the llm instance.
	:return: The metric scores.
	"""

	class Truth(BaseModel):
		truths: List[str]

	class Claim(BaseModel):
		claims: List[str]

	class Verdict(BaseModel):
		verdict: str
		reason: Optional[str]

	class FaithfulnessVerdicts(BaseModel):
		verdicts: List[Verdict]

	def calculate_score(verdicts: List[Verdict]) -> float:
		number_of_verdicts = len(verdicts)
		if number_of_verdicts == 0:
			return 1

		faithfulness_count = 0
		for verdict in verdicts:
			if verdict.verdict.strip().lower() != "no":
				faithfulness_count += 1

		score = faithfulness_count / number_of_verdicts
		return score

	retrieval_contexts = list(map(lambda x: x.retrieval_gt_contents, metric_inputs))
	truth_prompts = list(
		map(lambda x: FaithfulnessTemplate.generate_truths(x, lang), retrieval_contexts)
	)

	generated_texts = list(map(lambda x: x.generated_texts, metric_inputs))
	claim_prompts = list(
		map(lambda x: FaithfulnessTemplate.generate_claims(x, lang), generated_texts)
	)

	generator: BaseGenerator = make_generator_instance(
		generator_module_type, llm=llm, batch=batch, **kwargs
	)
	if isinstance(generator, OpenAILLM):  # Because of the event loop error at the httpx
		# TODO: Fix the httpx APIConnectionError at the many repetitive request to the OpenAILLM on the same instance
		truth_responses: List[Truth] = generator.structured_output(truth_prompts, Truth)
		claim_responses: List[Claim] = make_generator_instance(
			generator_module_type, llm=llm, batch=batch, **kwargs
		).structured_output(claim_prompts, Claim)
		verdict_prompts = list(
			map(
				lambda claim, truth: FaithfulnessTemplate.generate_verdicts(
					"\n\n".join(claim.claims), "\n\n".join(truth.truths), lang
				),
				claim_responses,
				truth_responses,
			)
		)
		verdict_responses: List[FaithfulnessVerdicts] = make_generator_instance(
			generator_module_type, llm=llm, batch=batch, **kwargs
		).structured_output(verdict_prompts, FaithfulnessVerdicts)
	else:
		truth_responses: List[Truth] = generator.structured_output(truth_prompts, Truth)
		claim_responses: List[Claim] = generator.structured_output(claim_prompts, Claim)
		verdict_prompts = list(
			map(
				lambda claim, truth: FaithfulnessTemplate.generate_verdicts(
					"\n\n".join(claim.claims), "\n\n".join(truth.truths), lang
				),
				claim_responses,
				truth_responses,
			)
		)
		verdict_responses: List[FaithfulnessVerdicts] = generator.structured_output(
			verdict_prompts, FaithfulnessVerdicts
		)

	result = list(map(lambda x: calculate_score(x.verdicts), verdict_responses))
	return result


@autorag_metric_loop(fields_to_check=["generation_gt", "generated_texts"])
def bleu(
	metric_inputs: List[MetricInput],
	tokenize: Optional[str] = None,
	smooth_method: str = "exp",
	smooth_value: Optional[float] = None,
	max_ngram_order: int = 4,
	trg_lang: str = "",
	effective_order: bool = True,
	**kwargs,
) -> List[float]:
	"""
	Computes the BLEU metric given pred and ground-truth.

	:param metric_inputs: A list of MetricInput schema (Required Field -> "generation_gt", "generated_texts")
	:param tokenize: The tokenizer to use. If None, defaults to language-specific tokenizers with '13a' as the fallback default. check #https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/bleu.py
	:param smooth_method: The smoothing method to use ('floor', 'add-k', 'exp' or 'none').
	:param smooth_value: The smoothing value for `floor` and `add-k` methods. `None` falls back to default value.
	:param max_ngram_order: If given, it overrides the maximum n-gram order (default: 4) when computing precisions.
	:param trg_lang: An optional language code to raise potential tokenizer warnings.
	:param effective_order: If `True`, stop including n-gram orders for which precision is 0. This should be
	`True`, if sentence-level BLEU will be computed.
	"""
	bleu_instance = BLEU(
		tokenize=tokenize,
		smooth_method=smooth_method,
		smooth_value=smooth_value,
		max_ngram_order=max_ngram_order,
		trg_lang=trg_lang,
		effective_order=effective_order,
		**kwargs,
	)

	result = list(
		map(
			lambda x: bleu_instance.sentence_score(
				x.generated_texts, x.generation_gt
			).score,
			metric_inputs,
		)
	)
	return result


@autorag_metric_loop(fields_to_check=["generation_gt", "generated_texts"])
def meteor(
	metric_inputs: List[MetricInput],
	alpha: float = 0.9,
	beta: float = 3.0,
	gamma: float = 0.5,
) -> List[float]:
	"""
	Compute meteor score for generation.

	:param metric_inputs: A list of MetricInput schema (Required Field -> "generation_gt", "generated_texts")
	:param alpha: Parameter for controlling relative weights of precision and recall.
	    Default is 0.9.
	:param beta: Parameter for controlling shape of penalty as a
	    function of as a function of fragmentation.
	    Default is 3.0.
	:param gamma: Relative weight assigned to fragmentation penalty.
	    Default is 0.5.
	:return: A list of computed metric scores.
	"""
	nltk.download("punkt", quiet=True)
	meteor_instance = evaluate.load("meteor")
	result = huggingface_evaluate(
		meteor_instance,
		"meteor",
		metric_inputs,
		alpha=alpha,
		beta=beta,
		gamma=gamma,
	)
	del meteor_instance
	return result


@autorag_metric_loop(fields_to_check=["generation_gt", "generated_texts"])
def rouge(
	metric_inputs: List[MetricInput],
	rouge_type: Optional[str] = "rougeL",
	use_stemmer: bool = False,
	split_summaries: bool = False,
	batch: int = os.cpu_count(),
) -> List[float]:
	"""
	Compute rouge score for generation.

	:param metric_inputs: A list of MetricInput schema (Required Field -> "generation_gt", "generated_texts")
	:param rouge_type: A rouge type to use for evaluation.
	    Default is 'RougeL'.
	    Choose between rouge1, rouge2, rougeL, and rougeLSum.
	    - rouge1: unigram (1-gram) based scoring.
	    - rouge2: bigram (2-gram) based scoring.
	    - rougeL: Longest Common Subsequence based scoring.
	    - rougeLSum: splits text using "\n"
	:param use_stemmer: Bool indicating whether Porter stemmer should be used to
	    strip word suffixes to improve matching. This arg is used in the
	    DefaultTokenizer, but other tokenizers might or might not choose to
	    use this. Default is False.
	:param split_summaries: Whether to add newlines between sentences for rougeLsum.
	    Default is False.
	:param batch: The batch size for processing.
	    Default is your cpu count.
	:return: A list of computed metric scores.
	"""
	rouge_instance = RougeScorer(
		rouge_types=[rouge_type],
		use_stemmer=use_stemmer,
		split_summaries=split_summaries,
		tokenizer=tokenizers.DefaultTokenizer(use_stemmer),
	)

	async def compute(gt: List[str], pred: str) -> float:
		return rouge_instance.score_multi(targets=gt, prediction=pred)[
			rouge_type
		].fmeasure

	tasks = [
		compute(metric_input.generation_gt, metric_input.generated_texts)
		for metric_input in metric_inputs
	]
	loop = get_event_loop()
	result = loop.run_until_complete(process_batch(tasks, batch_size=batch))

	del rouge_instance
	return result


@autorag_metric_loop(fields_to_check=["generation_gt", "generated_texts"])
def sem_score(
	metric_inputs: List[MetricInput],
	embedding_model: Optional[BaseEmbedding] = None,
	batch: int = 128,
) -> List[float]:
	"""
	Compute sem score between generation gt and pred with cosine similarity.

	:param metric_inputs: A list of MetricInput schema (Required Field -> "generation_gt", "generated_texts")
	:param embedding_model: Embedding model to use for compute cosine similarity.
	    Default is all-mpnet-base-v2 embedding model.
	    The paper used this embedding model.
	:param batch: The batch size for processing.
	    Default is 128
	:return: A list of computed metric scores.
	"""
	generations = [metric_input.generated_texts for metric_input in metric_inputs]
	generation_gt = [metric_input.generation_gt for metric_input in metric_inputs]
	if embedding_model is None:
		embedding_model = embedding_models.get("huggingface_all_mpnet_base_v2")()

	embedding_model.embed_batch_size = batch

	openai_embedding_max_length = 8000
	if isinstance(embedding_model, OpenAIEmbedding):
		generations = openai_truncate_by_token(
			generations, openai_embedding_max_length, embedding_model.model_name
		)

	embedded_pred: List[List[float]] = embedding_model.get_text_embedding_batch(
		generations, show_progress=True
	)
	gt_lengths = list(map(len, generation_gt))
	flatten_gt = list(itertools.chain.from_iterable(generation_gt))
	if isinstance(embedding_model, OpenAIEmbedding):
		flatten_gt = openai_truncate_by_token(
			flatten_gt, openai_embedding_max_length, embedding_model.model_name
		)
	embedded_gt_flatten = embedding_model.get_text_embedding_batch(
		flatten_gt, show_progress=True
	)
	# re-group embedded_gt_flatten with gt_lengths
	iterator = iter(embedded_gt_flatten)
	embedded_gt: List[List[List[float]]] = [
		list(itertools.islice(iterator, length)) for length in gt_lengths
	]

	result = []
	for gt, pred in zip(embedded_gt, embedded_pred):
		similarity_scores: List[float] = list(
			map(lambda x: calculate_cosine_similarity(x, pred), gt)
		)
		result.append(max(similarity_scores))

	del embedding_model
	empty_cuda_cache()

	return result


@autorag_metric_loop(fields_to_check=["generation_gt", "generated_texts"])
def g_eval(
	metric_inputs: List[MetricInput],
	metrics: Optional[List[str]] = None,
	model: str = "gpt-4-0125-preview",
	batch_size: int = 8,
) -> List[float]:
	"""
	Calculate G-Eval score.
	G-eval is a metric that uses high-performance LLM model to evaluate generation performance.
	It evaluates the generation result by coherence, consistency, fluency, and relevance.
	It uses only 'openai' model, and we recommend to use gpt-4 for evaluation accuracy.

	:param metric_inputs: A list of MetricInput schema (Required Field -> "generation_gt", "generated_texts")
	:param metrics: A list of metrics to use for evaluation.
	    Default is all metrics, which is ['coherence', 'consistency', 'fluency', 'relevance'].
	:param model: OpenAI model name.
	    Default is 'gpt-4-0125-preview'.
	:param batch_size: The batch size for processing.
	    Default is 8.
	:return: G-Eval score.
	"""
	generations = [metric_input.generated_texts for metric_input in metric_inputs]
	generation_gt = [metric_input.generation_gt for metric_input in metric_inputs]
	loop = get_event_loop()
	tasks = [
		async_g_eval(gt, pred, metrics, model)
		for gt, pred in zip(generation_gt, generations)
	]
	result = loop.run_until_complete(process_batch(tasks, batch_size=batch_size))
	return result


async def async_g_eval(
	generation_gt: List[str],
	pred: str,
	metrics: Optional[List[str]] = None,
	model: str = "gpt-4-0125-preview",
) -> float:
	available_metrics = ["coherence", "consistency", "fluency", "relevance"]
	if metrics is None:
		metrics = available_metrics
	else:
		assert len(metrics) > 0, "metrics must be a list of string"
		metrics = [metric for metric in metrics if metric in available_metrics]

	current_path = os.path.dirname(os.path.realpath(__file__))
	prompt_path = os.path.join(current_path, "g_eval_prompts")
	g_eval_prompts = {
		"coherence": open(os.path.join(prompt_path, "coh_detailed.txt")).read(),
		"consistency": open(os.path.join(prompt_path, "con_detailed.txt")).read(),
		"fluency": open(os.path.join(prompt_path, "flu_detailed.txt")).read(),
		"relevance": open(os.path.join(prompt_path, "rel_detailed.txt")).read(),
	}

	client = AsyncOpenAI()

	async def g_eval_score(prompt: str, gen_gt: List[str], pred: str):
		scores = []
		for gt in gen_gt:
			input_prompt = prompt.replace("{{Document}}", gt).replace(
				"{{Summary}}", pred
			)
			response = await client.chat.completions.create(
				model=model,
				messages=[{"role": "system", "content": input_prompt}],
				logprobs=True,
				top_logprobs=5,
				temperature=0,
				max_tokens=2,
				frequency_penalty=0,
				presence_penalty=0,
				stop=None,
				n=20,
			)
			if "(1-3):" in prompt:
				scores.append(get_g_eval_score(response, max_score=3))
			else:
				scores.append(get_g_eval_score(response))
		return max(scores)

	def get_g_eval_score(responses, max_score: int = 5) -> int:
		target_tokens = {str(i): 0 for i in range(1, max_score + 1)}
		for choice in responses.choices:
			first_top_log_probs = choice.logprobs.content[0].top_logprobs
			for i, top_log_prob in enumerate(
				list(map(lambda x: x.token, first_top_log_probs))
			):
				if top_log_prob in target_tokens:
					target_tokens[top_log_prob] += 5 - i

		return int(max(target_tokens, key=target_tokens.get))

	g_eval_scores = await asyncio.gather(
		*(g_eval_score(g_eval_prompts[x], generation_gt, pred) for x in metrics)
	)
	return sum(g_eval_scores) / len(g_eval_scores)


@autorag_metric_loop(fields_to_check=["generation_gt", "generated_texts"])
def bert_score(
	metric_inputs: List[MetricInput],
	lang: str = "en",
	batch: int = 128,
	n_threads: int = os.cpu_count(),
) -> List[float]:
	generations = [metric_input.generated_texts for metric_input in metric_inputs]
	generation_gt = [metric_input.generation_gt for metric_input in metric_inputs]
	evaluator = evaluate.load("bertscore")

	df = pd.DataFrame(
		{
			"reference": generation_gt,
			"prediction": generations,
			"lang": lang,
		}
	)

	df = df.explode("reference", ignore_index=False)
	df["bert_score"] = evaluator.compute(
		predictions=df["prediction"].tolist(),
		references=df["reference"].tolist(),
		lang=lang,
		nthreads=n_threads,
		batch_size=batch,
	)["f1"]

	del evaluator
	empty_cuda_cache()

	return df.groupby(level=0)["bert_score"].max().tolist()

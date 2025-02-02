import pandas as pd
from autorag.data.qa.filter.passage_dependency import passage_dependency_filter_llama_index
from autorag.data.qa.query.llama_gen_query import factoid_query_gen
from autorag.data.qa.sample import random_single_hop
from autorag.data.qa.schema import Corpus, QA
from autorag.data.qa.generation_gt.llama_index_gen_gt import (
    make_basic_gen_gt,
    make_concise_gen_gt,
)
from autorag.data.qa.filter.dontknow import dontknow_filter_rule_based
from llama_index.core.base.llms.base import BaseLLM
from autorag.data.qa.evolve.llama_index_query_evolve import reasoning_evolve_ragas
from autorag.data.qa.evolve.llama_index_query_evolve import compress_ragas


def default_create(corpus_df, llm: BaseLLM, n: int = 100, lang: str = "en",
				   batch_size: int = 32) -> QA:
	corpus_instance = Corpus(corpus_df)
	if len(corpus_instance.data) < n:
		n = len(corpus_instance.data)
	sampled_corpus = corpus_instance.sample(random_single_hop, n=n)
	mapped_corpus = sampled_corpus.map(lambda df: df.reset_index(drop=True))
	retrieval_gt_contents = mapped_corpus.make_retrieval_gt_contents()
	query_generated = retrieval_gt_contents.batch_apply(factoid_query_gen, llm=llm, lang=lang, batch_size=batch_size)
	basic_answers = query_generated.batch_apply(make_basic_gen_gt, llm=llm, lang=lang, batch_size=batch_size)
	concise_answers = basic_answers.batch_apply(make_concise_gen_gt, llm=llm, lang=lang, batch_size=batch_size)
	filtered_answers = concise_answers.filter(dontknow_filter_rule_based, lang=lang)
	initial_qa = filtered_answers.batch_filter(passage_dependency_filter_llama_index, llm=llm, lang=lang, batch_size=batch_size)
	return initial_qa


def fast_create(corpus_df, llm: BaseLLM, n: int = 100, lang: str = "en",
				batch_size: int = 32) -> QA:
	corpus_instance = Corpus(corpus_df)
	if len(corpus_instance.data) < n:
		n = len(corpus_instance.data)

	sampled_corpus = corpus_instance.sample(random_single_hop, n=n)
	mapped_corpus = sampled_corpus.map(lambda df: df.reset_index(drop=True))

	retrieval_gt_contents = mapped_corpus.make_retrieval_gt_contents()

	query_generated = retrieval_gt_contents.batch_apply(factoid_query_gen, llm=llm, lang=lang, batch_size=batch_size)

	basic_answers = query_generated.batch_apply(make_basic_gen_gt, llm=llm, lang=lang, batch_size=batch_size)

	concise_answers = basic_answers.batch_apply(make_concise_gen_gt, llm=llm, lang=lang, batch_size=batch_size)

	initial_qa = concise_answers

	return initial_qa


def advanced_create(corpus_df, llm: BaseLLM, n: int = 100, lang: str = "en",
					batch_size: int = 32) -> QA:
	"""
	Mix hard and easy question.
	"""
	corpus_instance = Corpus(corpus_df)
	if len(corpus_instance.data) < n:
		n = len(corpus_instance.data)
	sampled_corpus = corpus_instance.sample(random_single_hop, n=n)
	mapped_corpus = sampled_corpus.map(lambda df: df.reset_index(drop=True))
	retrieval_gt_contents = mapped_corpus.make_retrieval_gt_contents()
	query_generated = retrieval_gt_contents.batch_apply(factoid_query_gen, llm=llm, lang=lang, batch_size=batch_size)
	basic_answers = query_generated.batch_apply(make_basic_gen_gt, llm=llm, lang=lang, batch_size=batch_size)
	concise_answers = basic_answers.batch_apply(make_concise_gen_gt, llm=llm, lang=lang, batch_size=batch_size)
	filtered_answers = concise_answers.filter(dontknow_filter_rule_based, lang=lang)
	initial_qa = filtered_answers.batch_filter(passage_dependency_filter_llama_index, llm=llm, lang=lang, batch_size=batch_size)
	cut_idx = n // 2
	reasoning_qa = initial_qa.map(lambda df: df.iloc[:cut_idx]).batch_apply(
		reasoning_evolve_ragas,
		llm=llm,
		lang=lang,
		batch_size=batch_size,
	)
	compressed_qa = initial_qa.map(lambda df: df.iloc[cut_idx:]).map(lambda df: df.reset_index(drop=True)).batch_apply(
		compress_ragas,
		llm=llm,
		lang=lang,
		batch_size=batch_size,
	)
	final_qa = QA(pd.concat([reasoning_qa.data, compressed_qa.data], ignore_index=True),
				  linked_corpus=corpus_instance)

	return final_qa

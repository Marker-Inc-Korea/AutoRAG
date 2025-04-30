import asyncio
import os
import pickle
import re
from typing import List, Dict, Tuple, Callable, Union, Iterable, Optional

import numpy as np
import pandas as pd
from llama_index.core.indices.keyword_table.utils import simple_extract_keywords
from nltk import PorterStemmer
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from autorag.nodes.retrieval.base import (
	evenly_distribute_passages,
	BaseRetrieval,
	get_bm25_pkl_name,
)
from autorag.utils import validate_corpus_dataset, fetch_contents
from autorag.utils.util import (
	get_event_loop,
	normalize_string,
	result_to_dataframe,
	pop_params,
)


def tokenize_ko_kiwi(texts: List[str]) -> List[List[str]]:
	try:
		from kiwipiepy import Kiwi, Token
	except ImportError:
		raise ImportError(
			"You need to install kiwipiepy to use 'ko_kiwi' tokenizer. "
			"Please install kiwipiepy by running 'pip install kiwipiepy'. "
			"Or install Korean version of AutoRAG by running 'pip install AutoRAG[ko]'."
		)
	texts = list(map(lambda x: x.strip().lower(), texts))
	kiwi = Kiwi()
	tokenized_list: Iterable[List[Token]] = kiwi.tokenize(texts)

	def extract_form_safe(x):
		try:
			return x.form
		except UnicodeDecodeError:
			return " "

	return [
		list(map(lambda x: extract_form_safe(x), token_list))
		for token_list in tokenized_list
	]


def tokenize_ko_kkma(texts: List[str]) -> List[List[str]]:
	try:
		from konlpy.tag import Kkma
	except ImportError:
		raise ImportError(
			"You need to install konlpy to use 'ko_kkma' tokenizer. "
			"Please install konlpy by running 'pip install konlpy'. "
			"Or install Korean version of AutoRAG by running 'pip install AutoRAG[ko]'."
		)
	tokenizer = Kkma()
	tokenized_list: List[List[str]] = list(map(lambda x: tokenizer.morphs(x), texts))
	return tokenized_list


def tokenize_ko_okt(texts: List[str]) -> List[List[str]]:
	try:
		from konlpy.tag import Okt
	except ImportError:
		raise ImportError(
			"You need to install konlpy to use 'ko_kkma' tokenizer. "
			"Please install konlpy by running 'pip install konlpy'. "
			"Or install Korean version of AutoRAG by running 'pip install AutoRAG[ko]'."
		)
	tokenizer = Okt()
	tokenized_list: List[List[str]] = list(map(lambda x: tokenizer.morphs(x), texts))
	return tokenized_list


def tokenize_porter_stemmer(texts: List[str]) -> List[List[str]]:
	def tokenize_remove_stopword(text: str, stemmer) -> List[str]:
		text = text.lower()
		words = list(simple_extract_keywords(text))
		return [stemmer.stem(word) for word in words]

	stemmer = PorterStemmer()
	tokenized_list: List[List[str]] = list(
		map(lambda x: tokenize_remove_stopword(x, stemmer), texts)
	)
	return tokenized_list


def tokenize_space(texts: List[str]) -> List[List[str]]:
	def tokenize_space_text(text: str) -> List[str]:
		text = normalize_string(text)
		return re.split(r"\s+", text.strip())

	return list(map(tokenize_space_text, texts))


def load_bm25_corpus(bm25_path: str) -> Dict:
	if bm25_path is None:
		return {}
	with open(bm25_path, "rb") as f:
		bm25_corpus = pickle.load(f)
	return bm25_corpus


def tokenize_ja_sudachipy(texts: List[str]) -> List[List[str]]:
	try:
		from sudachipy import dictionary, tokenizer
	except ImportError:
		raise ImportError(
			"You need to install SudachiPy to use 'sudachipy' tokenizer. "
			"Please install SudachiPy by running 'pip install sudachipy'."
		)

	# Initialize SudachiPy with the default tokenizer
	tokenizer_obj = dictionary.Dictionary(dict="core").create()

	# Choose the tokenizer mode: NORMAL, SEARCH, A
	mode = tokenizer.Tokenizer.SplitMode.A

	# Tokenize the input texts
	tokenized_list = []
	for text in texts:
		tokens = tokenizer_obj.tokenize(text, mode)
		tokenized_list.append([token.surface() for token in tokens])

	return tokenized_list


BM25_TOKENIZER = {
	"porter_stemmer": tokenize_porter_stemmer,
	"ko_kiwi": tokenize_ko_kiwi,
	"space": tokenize_space,
	"ko_kkma": tokenize_ko_kkma,
	"ko_okt": tokenize_ko_okt,
	"sudachipy": tokenize_ja_sudachipy,
}


class BM25(BaseRetrieval):
	def __init__(self, project_dir: str, *args, **kwargs):
		"""
		Initialize BM25 module.
		(Retrieval)

		:param project_dir: The project directory path.
		:param bm25_tokenizer: The tokenizer name that is used to the BM25.
		    It supports 'porter_stemmer', 'ko_kiwi', and huggingface `AutoTokenizer`.
		    You can pass huggingface tokenizer name.
		    Default is porter_stemmer.
		:param kwargs: The optional arguments.
		"""

		super().__init__(project_dir)
		# check if bm25_path and file exist
		bm25_tokenizer = kwargs.get("bm25_tokenizer", None)
		if bm25_tokenizer is None:
			bm25_tokenizer = "porter_stemmer"
		bm25_path = os.path.join(self.resources_dir, get_bm25_pkl_name(bm25_tokenizer))

		assert (
			bm25_path is not None
		), "bm25_path must be specified for using bm25 retrieval."
		assert os.path.exists(
			bm25_path
		), f"bm25_path {bm25_path} does not exist. Please ingest first."

		self.bm25_corpus = load_bm25_corpus(bm25_path)
		assert (
			"tokens" and "passage_id" in list(self.bm25_corpus.keys())
		), "bm25_corpus must contain tokens and passage_id. Please check you ingested bm25 corpus correctly."
		self.tokenizer = select_bm25_tokenizer(bm25_tokenizer)
		assert self.bm25_corpus["tokenizer_name"] == bm25_tokenizer, (
			f"The bm25 corpus tokenizer is {self.bm25_corpus['tokenizer_name']}, but your input is {bm25_tokenizer}. "
			f"You need to ingest again. Delete bm25 pkl file and re-ingest it."
		)
		self.bm25_instance = BM25Okapi(self.bm25_corpus["tokens"])

	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		queries = self.cast_to_run(previous_result)
		pure_params = pop_params(self._pure, kwargs)
		ids, scores = self._pure(queries, *args, **pure_params)
		contents = fetch_contents(self.corpus_df, ids)
		return contents, ids, scores

	def _pure(
		self,
		queries: List[List[str]],
		top_k: int,
		ids: Optional[List[List[str]]] = None,
	) -> Tuple[List[List[str]], List[List[float]]]:
		"""
		BM25 retrieval function.
		You have to load a pickle file that is already ingested.

		:param queries: 2-d list of query strings.
		    Each element of the list is a query strings of each row.
		:param top_k: The number of passages to be retrieved.
		:param ids: The optional list of ids that you want to retrieve.
		    You don't need to specify this in the general use cases.
		    Default is None.
		:return: The 2-d list contains a list of passage ids that retrieved from bm25 and 2-d list of its scores.
		    It will be a length of queries. And each element has a length of top_k.
		"""
		if ids is not None:
			score_result = list(
				map(
					lambda query_list, id_list: get_bm25_scores(
						query_list,
						id_list,
						self.tokenizer,
						self.bm25_instance,
						self.bm25_corpus,
					),
					queries,
					ids,
				)
			)
			return ids, score_result

		# run async bm25_pure function
		tasks = [
			bm25_pure(
				input_queries,
				top_k,
				self.tokenizer,
				self.bm25_instance,
				self.bm25_corpus,
			)
			for input_queries in queries
		]
		loop = get_event_loop()
		results = loop.run_until_complete(asyncio.gather(*tasks))
		id_result = list(map(lambda x: x[0], results))
		score_result = list(map(lambda x: x[1], results))
		return id_result, score_result


async def bm25_pure(
	queries: List[str], top_k: int, tokenizer, bm25_api: BM25Okapi, bm25_corpus: Dict
) -> Tuple[List[str], List[float]]:
	"""
	Async BM25 retrieval function.
	Its usage is for async retrieval of bm25 row by row.

	:param queries: A list of query strings.
	:param top_k: The number of passages to be retrieved.
	:param tokenizer: A tokenizer that will be used to tokenize queries.
	:param bm25_api: A bm25 api instance that will be used to retrieve passages.
	:param bm25_corpus: A dictionary containing the bm25 corpus, which is doc_id from corpus and tokenized corpus.
	    Its data structure looks like this:

	    .. Code:: python

	        {
	            "tokens": [], # 2d list of tokens
	            "passage_id": [], # 2d list of passage_id. Type must be str.
	        }
	:return: The tuple contains a list of passage ids that retrieved from bm25 and its scores.
	"""
	# I don't make queries operation to async, because queries length might be small, so it will occur overhead.
	tokenized_queries = tokenize(queries, tokenizer)
	id_result = []
	score_result = []
	for query in tokenized_queries:
		scores = bm25_api.get_scores(query)
		sorted_scores = sorted(scores, reverse=True)
		top_n_index = np.argsort(scores)[::-1][:top_k]
		ids = [bm25_corpus["passage_id"][i] for i in top_n_index]
		id_result.append(ids)
		score_result.append(sorted_scores[:top_k])

	# make a total result to top_k
	id_result, score_result = evenly_distribute_passages(id_result, score_result, top_k)
	# sort id_result and score_result by score
	result = [
		(_id, score)
		for score, _id in sorted(
			zip(score_result, id_result), key=lambda pair: pair[0], reverse=True
		)
	]
	id_result, score_result = zip(*result)
	return list(id_result), list(score_result)


def get_bm25_scores(
	queries: List[str],
	ids: List[str],
	tokenizer,
	bm25_api: BM25Okapi,
	bm25_corpus: Dict,
) -> List[float]:
	if len(ids) == 0 or not bool(ids):
		return []
	tokenized_queries = tokenize(queries, tokenizer)
	result_dict = {id_: [] for id_ in ids}
	for query in tokenized_queries:
		scores = bm25_api.get_scores(query)
		for i, id_ in enumerate(ids):
			result_dict[id_].append(scores[bm25_corpus["passage_id"].index(id_)])
	result_df = pd.DataFrame(result_dict)
	return result_df.max(axis=0).tolist()


def tokenize(queries: List[str], tokenizer) -> List[List[int]]:
	if isinstance(tokenizer, PreTrainedTokenizerBase):
		tokenized_queries = tokenizer(queries).input_ids
	else:
		tokenized_queries = tokenizer(queries)
	return tokenized_queries


def bm25_ingest(
	corpus_path: str, corpus_data: pd.DataFrame, bm25_tokenizer: str = "porter_stemmer"
):
	if not corpus_path.endswith(".pkl"):
		raise ValueError(f"Corpus path {corpus_path} is not a pickle file.")
	validate_corpus_dataset(corpus_data)
	ids = corpus_data["doc_id"].tolist()

	# Initialize bm25_corpus
	bm25_corpus = pd.DataFrame()

	# Load the BM25 corpus if it exists and get the passage ids
	if os.path.exists(corpus_path) and os.path.getsize(corpus_path) > 0:
		with open(corpus_path, "rb") as r:
			corpus = pickle.load(r)
			bm25_corpus = pd.DataFrame.from_dict(corpus)
		duplicated_passage_rows = bm25_corpus[bm25_corpus["passage_id"].isin(ids)]
		new_passage = corpus_data[
			~corpus_data["doc_id"].isin(duplicated_passage_rows["passage_id"])
		]
	else:
		new_passage = corpus_data

	if not new_passage.empty:
		tokenizer = select_bm25_tokenizer(bm25_tokenizer)
		if isinstance(tokenizer, PreTrainedTokenizerBase):
			tokenized_corpus = tokenizer(new_passage["contents"].tolist()).input_ids
		else:
			tokenized_corpus = tokenizer(new_passage["contents"].tolist())
		new_bm25_corpus = pd.DataFrame(
			{
				"tokens": tokenized_corpus,
				"passage_id": new_passage["doc_id"].tolist(),
			}
		)

		if not bm25_corpus.empty:
			bm25_corpus_updated = pd.concat(
				[bm25_corpus, new_bm25_corpus], ignore_index=True
			)
			bm25_dict = bm25_corpus_updated.to_dict("list")
		else:
			bm25_dict = new_bm25_corpus.to_dict("list")

		# add tokenizer name to bm25_dict
		bm25_dict["tokenizer_name"] = bm25_tokenizer

		with open(corpus_path, "wb") as w:
			pickle.dump(bm25_dict, w)


def select_bm25_tokenizer(
	bm25_tokenizer: str,
) -> Callable[[str], List[Union[int, str]]]:
	if bm25_tokenizer in list(BM25_TOKENIZER.keys()):
		return BM25_TOKENIZER[bm25_tokenizer]

	return AutoTokenizer.from_pretrained(bm25_tokenizer, use_fast=False)

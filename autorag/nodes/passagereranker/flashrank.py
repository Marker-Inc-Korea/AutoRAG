import json
from pathlib import Path

import pandas as pd
import numpy as np
import os
import zipfile
import requests
from tqdm import tqdm
import collections
from typing import List, Dict, Tuple

from autorag.nodes.passagereranker.base import BasePassageReranker
from autorag.utils import result_to_dataframe
from autorag.utils.util import (
	flatten_apply,
	sort_by_scores,
	select_top_k,
	make_batch,
	empty_cuda_cache,
)

model_url = "https://huggingface.co/prithivida/flashrank/resolve/main/{}.zip"

model_file_map = {
	"ms-marco-TinyBERT-L-2-v2": "flashrank-TinyBERT-L-2-v2.onnx",
	"ms-marco-MiniLM-L-12-v2": "flashrank-MiniLM-L-12-v2_Q.onnx",
	"ms-marco-MultiBERT-L-12": "flashrank-MultiBERT-L12_Q.onnx",
	"rank-T5-flan": "flashrank-rankt5_Q.onnx",
	"ce-esci-MiniLM-L12-v2": "flashrank-ce-esci-MiniLM-L12-v2_Q.onnx",
	"miniReranker_arabic_v1": "miniReranker_arabic_v1.onnx",
}


class FlashRankReranker(BasePassageReranker):
	def __init__(
		self, project_dir: str, model: str = "ms-marco-TinyBERT-L-2-v2", *args, **kwargs
	):
		"""
		Initialize FlashRank rerank node.

		:param project_dir: The project directory path.
		:param model: The model name for FlashRank rerank.
		    You can get the list of available models from https://github.com/PrithivirajDamodaran/FlashRank.
		    Default is "ms-marco-TinyBERT-L-2-v2".
		    Not support “rank_zephyr_7b_v1_full” due to parallel inference issue.
		:param kwargs: Extra arguments that are not affected
		"""
		super().__init__(project_dir)
		try:
			from tokenizers import Tokenizer
		except ImportError:
			raise ImportError(
				"Tokenizer is not installed. Please install tokenizers to use FlashRank reranker."
			)

		cache_dir = kwargs.pop("cache_dir", "/tmp")
		max_length = kwargs.pop("max_length", 512)

		self.cache_dir: Path = Path(cache_dir)
		self.model_dir: Path = self.cache_dir / model
		self._prepare_model_dir(model)
		model_file = model_file_map[model]

		try:
			import onnxruntime as ort
		except ImportError:
			raise ImportError(
				"onnxruntime is not installed. Please install onnxruntime to use FlashRank reranker."
			)

		self.session = ort.InferenceSession(str(self.model_dir / model_file))
		self.tokenizer: Tokenizer = self._get_tokenizer(max_length)

	def __del__(self):
		del self.session
		del self.tokenizer
		empty_cuda_cache()
		super().__del__()

	def _prepare_model_dir(self, model_name: str):
		if not self.cache_dir.exists():
			self.cache_dir.mkdir(parents=True, exist_ok=True)

		if not self.model_dir.exists():
			self._download_model_files(model_name)

	def _download_model_files(self, model_name: str):
		local_zip_file = self.cache_dir / f"{model_name}.zip"
		formatted_model_url = model_url.format(model_name)

		with requests.get(formatted_model_url, stream=True) as r:
			r.raise_for_status()
			total_size = int(r.headers.get("content-length", 0))
			with (
				open(local_zip_file, "wb") as f,
				tqdm(
					desc=local_zip_file.name,
					total=total_size,
					unit="iB",
					unit_scale=True,
					unit_divisor=1024,
				) as bar,
			):
				for chunk in r.iter_content(chunk_size=8192):
					size = f.write(chunk)
					bar.update(size)

		with zipfile.ZipFile(local_zip_file, "r") as zip_ref:
			zip_ref.extractall(self.cache_dir)
		os.remove(local_zip_file)

	def _get_tokenizer(self, max_length: int = 512):
		try:
			from tokenizers import AddedToken, Tokenizer
		except ImportError:
			raise ImportError(
				"Pytorch is not installed. Please install pytorch to use FlashRank reranker."
			)
		config = json.load(open(str(self.model_dir / "config.json")))
		tokenizer_config = json.load(
			open(str(self.model_dir / "tokenizer_config.json"))
		)
		tokens_map = json.load(open(str(self.model_dir / "special_tokens_map.json")))
		tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))

		tokenizer.enable_truncation(
			max_length=min(tokenizer_config["model_max_length"], max_length)
		)
		tokenizer.enable_padding(
			pad_id=config["pad_token_id"], pad_token=tokenizer_config["pad_token"]
		)

		for token in tokens_map.values():
			if isinstance(token, str):
				tokenizer.add_special_tokens([token])
			elif isinstance(token, dict):
				tokenizer.add_special_tokens([AddedToken(**token)])

		vocab_file = self.model_dir / "vocab.txt"
		if vocab_file.exists():
			tokenizer.vocab = self._load_vocab(vocab_file)
			tokenizer.ids_to_tokens = collections.OrderedDict(
				[(ids, tok) for tok, ids in tokenizer.vocab.items()]
			)
		return tokenizer

	def _load_vocab(self, vocab_file: Path) -> Dict[str, int]:
		vocab = collections.OrderedDict()
		with open(vocab_file, "r", encoding="utf-8") as reader:
			tokens = reader.readlines()
		for index, token in enumerate(tokens):
			token = token.rstrip("\n")
			vocab[token] = index
		return vocab

	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		queries, contents, _, ids = self.cast_to_run(previous_result)
		top_k = kwargs.pop("top_k")
		batch = kwargs.pop("batch", 64)
		return self._pure(queries, contents, ids, top_k, batch)

	def _pure(
		self,
		queries: List[str],
		contents_list: List[List[str]],
		ids_list: List[List[str]],
		top_k: int,
		batch: int = 64,
	) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
		"""
		Rerank a list of contents with FlashRank rerank models.

		:param queries: The list of queries to use for reranking
		:param contents_list: The list of lists of contents to rerank
		:param ids_list: The list of lists of ids retrieved from the initial ranking
		:param top_k: The number of passages to be retrieved
		:param batch: The number of queries to be processed in a batch
		:return: Tuple of lists containing the reranked contents, ids, and scores
		"""
		nested_list = [
			list(map(lambda x: [query, x], content_list))
			for query, content_list in zip(queries, contents_list)
		]

		rerank_scores = flatten_apply(
			flashrank_run_model,
			nested_list,
			session=self.session,
			batch_size=batch,
			tokenizer=self.tokenizer,
		)

		df = pd.DataFrame(
			{
				"contents": contents_list,
				"ids": ids_list,
				"scores": rerank_scores,
			}
		)
		df[["contents", "ids", "scores"]] = df.apply(
			sort_by_scores, axis=1, result_type="expand"
		)
		results = select_top_k(df, ["contents", "ids", "scores"], top_k)

		return (
			results["contents"].tolist(),
			results["ids"].tolist(),
			results["scores"].tolist(),
		)


def flashrank_run_model(input_texts, tokenizer, session, batch_size: int):
	batch_input_texts = make_batch(input_texts, batch_size)
	results = []

	for batch_texts in tqdm(batch_input_texts):
		input_text = tokenizer.encode_batch(batch_texts)
		input_ids = np.array([e.ids for e in input_text])
		token_type_ids = np.array([e.type_ids for e in input_text])
		attention_mask = np.array([e.attention_mask for e in input_text])

		use_token_type_ids = token_type_ids is not None and not np.all(
			token_type_ids == 0
		)

		onnx_input = {
			"input_ids": input_ids.astype(np.int64),
			"attention_mask": attention_mask.astype(np.int64),
		}
		if use_token_type_ids:
			onnx_input["token_type_ids"] = token_type_ids.astype(np.int64)

		outputs = session.run(None, onnx_input)

		logits = outputs[0]

		if logits.shape[1] == 1:
			scores = 1 / (1 + np.exp(-logits.flatten()))
		else:
			exp_logits = np.exp(logits)
			scores = exp_logits[:, 1] / np.sum(exp_logits, axis=1)
		results.extend(scores)
	return results

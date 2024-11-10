from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd

from autorag.nodes.passagereranker.base import BasePassageReranker


from autorag.utils.util import (
	make_batch,
	sort_by_scores,
	flatten_apply,
	select_top_k,
	result_to_dataframe,
	pop_params,
	empty_cuda_cache,
)


class OpenVINOReranker(BasePassageReranker):
	def __init__(
		self,
		project_dir: str,
		model: str = "BAAI/bge-reranker-large",
		*args,
		**kwargs,
	):
		super().__init__(project_dir)

		try:
			from huggingface_hub import HfApi
			from transformers import AutoTokenizer
		except ImportError as e:
			raise ValueError(
				"Could not import huggingface_hub python package. "
				"Please install it with: "
				"`pip install -U huggingface_hub`."
			) from e

		def require_model_export(
			model_id: str, revision: Any = None, subfolder: Any = None
		) -> bool:
			model_dir = Path(model_id)
			if subfolder is not None:
				model_dir = model_dir / subfolder
			if model_dir.is_dir():
				return (
					not (model_dir / "openvino_model.xml").exists()
					or not (model_dir / "openvino_model.bin").exists()
				)
			hf_api = HfApi()
			try:
				model_info = hf_api.model_info(model_id, revision=revision or "main")
				normalized_subfolder = (
					None if subfolder is None else Path(subfolder).as_posix()
				)
				model_files = [
					file.rfilename
					for file in model_info.siblings
					if normalized_subfolder is None
					or file.rfilename.startswith(normalized_subfolder)
				]
				ov_model_path = (
					"openvino_model.xml"
					if subfolder is None
					else f"{normalized_subfolder}/openvino_model.xml"
				)
				return (
					ov_model_path not in model_files
					or ov_model_path.replace(".xml", ".bin") not in model_files
				)
			except Exception:
				return True

		try:
			from optimum.intel.openvino import OVModelForSequenceClassification
		except ImportError:
			raise ImportError(
				"Please install optimum package to use OpenVINOReranker"
				"pip install 'optimum[openvino,nncf]'"
			)

		model_kwargs = pop_params(
			OVModelForSequenceClassification.from_pretrained, kwargs
		)

		if require_model_export(model):
			# use remote model
			self.model = OVModelForSequenceClassification.from_pretrained(
				model, export=True, **model_kwargs
			)
		else:
			# use local model
			self.model = OVModelForSequenceClassification.from_pretrained(
				model, **model_kwargs
			)

		self.tokenizer = AutoTokenizer.from_pretrained(model)

	def __del__(self):
		del self.model
		del self.tokenizer
		empty_cuda_cache()
		super().__del__()

	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		queries, contents, _, ids = self.cast_to_run(previous_result)
		top_k = kwargs.get("top_k", 3)
		batch = kwargs.get("batch", 64)
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
		Rerank a list of contents based on their relevance to a query using MonoT5.

		:param queries: The list of queries to use for reranking
		:param contents_list: The list of lists of contents to rerank
		:param ids_list: The list of lists of ids retrieved from the initial ranking
		:param top_k: The number of passages to be retrieved

		:param batch: The number of queries to be processed in a batch
		:return: tuple of lists containing the reranked contents, ids, and scores
		"""
		# Retrieve the tokens used by the model to represent false and true predictions

		nested_list = [
			list(map(lambda x: [query, x], content_list))
			for query, content_list in zip(queries, contents_list)
		]

		rerank_scores = flatten_apply(
			openvino_run_model,
			nested_list,
			model=self.model,
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


def openvino_run_model(
	input_texts,
	model,
	batch_size: int,
	tokenizer,
):
	batch_input_texts = make_batch(input_texts, batch_size)
	results = []
	for batch_texts in batch_input_texts:
		input_tensors = tokenizer(
			batch_texts,
			padding=True,
			truncation=True,
			return_tensors="pt",
		)

		outputs = model(**input_tensors, return_dict=True)
		if outputs[0].shape[1] > 1:
			scores = outputs[0][:, 1]
		else:
			scores = outputs[0].flatten()

		scores = list(map(float, (1 / (1 + np.exp(-np.array(scores))))))
		results.extend(scores)
	return results

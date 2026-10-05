import os
from typing import List, Tuple

import aiohttp
import pandas as pd

from autorag.nodes.passagereranker.base import BasePassageReranker
from autorag.utils.util import get_event_loop, process_batch, result_to_dataframe

JINA_API_URL = "https://api.jina.ai/v1/rerank"


class JinaReranker(BasePassageReranker):
	def __init__(self, project_dir: str, api_key: str = None, *args, **kwargs):
		"""
		Initialize Jina rerank node.

		:param project_dir: The project directory path.
		:param api_key: The API key for Jina rerank.
		You can set it in the environment variable JINAAI_API_KEY.
		Or, you can directly set it on the config YAML file using this parameter.
		Default is env variable "JINAAI_API_KEY".
		:param kwargs: Extra arguments that are not affected
		"""
		super().__init__(project_dir)
		if api_key is None:
			api_key = os.getenv("JINAAI_API_KEY", None)
			if api_key is None:
				raise ValueError(
					"API key is not provided."
					"You can set it as an argument or as an environment variable 'JINAAI_API_KEY'"
				)
		# Open the session lazily on the loop that runs the requests. aiohttp's loop=
		# kwarg is a deprecated no-op, so a session created here would bind to whatever
		# loop happens to exist at construction time instead of the request loop.
		self._headers = {
			"Authorization": f"Bearer {api_key}",
			"Accept-Encoding": "identity",
		}
		self.session = None

	def _ensure_session(self) -> aiohttp.ClientSession:
		if self.session is None or self.session.closed:
			self.session = aiohttp.ClientSession(headers=self._headers)
		return self.session

	def __del__(self):
		# Best-effort safety net. _pure closes the session on the request loop; calling
		# session.close() without awaiting it leaks the session and warns at shutdown.
		try:
			session = getattr(self, "session", None)
			if session is not None and not session.closed:
				loop = get_event_loop()
				if not loop.is_running() and not loop.is_closed():
					loop.run_until_complete(session.close())
		except Exception:
			pass
		try:
			super().__del__()
		except Exception:
			pass

	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		queries, contents, _, ids = self.cast_to_run(previous_result)
		top_k = kwargs.pop("top_k")
		batch = kwargs.pop("batch", 8)
		model = kwargs.pop("model", "jina-reranker-v1-base-en")
		return self._pure(queries, contents, ids, top_k, model, batch)

	def _pure(
		self,
		queries: List[str],
		contents_list: List[List[str]],
		ids_list: List[List[str]],
		top_k: int,
		model: str = "jina-reranker-v1-base-en",
		batch: int = 8,
	) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
		"""
		Rerank a list of contents with Jina rerank models.
		You can get the API key from https://jina.ai/reranker and set it in the environment variable JINAAI_API_KEY.

		:param queries: The list of queries to use for reranking
		:param contents_list: The list of lists of contents to rerank
		:param ids_list: The list of lists of ids retrieved from the initial ranking
		:param top_k: The number of passages to be retrieved
		:param model: The model name for Cohere rerank.
		    You can choose between "jina-reranker-v1-base-en" and "jina-colbert-v1-en".
		    Default is "jina-reranker-v1-base-en".
		:param batch: The number of queries to be processed in a batch
		:return: Tuple of lists containing the reranked contents, ids, and scores
		"""
		loop = get_event_loop()

		async def _run_batch():
			# Create the session here, on the loop that runs the requests. aiohttp 3.13
			# resolves the session loop from the running loop and rejects loop=.
			session = self._ensure_session()
			tasks = [
				jina_reranker_pure(
					session, query, contents, ids, top_k=top_k, model=model
				)
				for query, contents, ids in zip(queries, contents_list, ids_list)
			]
			try:
				return await process_batch(tasks, batch)
			finally:
				if not session.closed:
					await session.close()

		results = loop.run_until_complete(_run_batch())

		content_result, id_result, score_result = zip(*results)

		return list(content_result), list(id_result), list(score_result)


async def jina_reranker_pure(
	session,
	query: str,
	contents: List[str],
	ids: List[str],
	top_k: int,
	model: str = "jina-reranker-v1-base-en",
) -> Tuple[List[str], List[str], List[float]]:
	async with session.post(
		JINA_API_URL,
		json={
			"query": query,
			"documents": contents,
			"model": model,
			"top_n": top_k,
		},
	) as resp:
		resp_json = await resp.json()
		if "results" not in resp_json:
			raise RuntimeError(f"Invalid response from Jina API: {resp_json['detail']}")

		results = resp_json["results"]
		indices = list(map(lambda x: x["index"], results))
		score_result = list(map(lambda x: x["relevance_score"], results))
		id_result = list(map(lambda x: ids[x], indices))
		content_result = list(map(lambda x: contents[x], indices))

		return content_result, id_result, score_result

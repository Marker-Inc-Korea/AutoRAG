import uuid
from typing import Optional

import pandas as pd
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from autorag.data.utils.util import corpus_df_to_langchain_documents
from autorag.utils import cast_qa_dataset


def generate_qa_ragas(
	corpus_df: pd.DataFrame,
	test_size: int,
	distributions: Optional[dict] = None,
	generator_llm: Optional[BaseChatModel] = None,
	critic_llm: Optional[BaseChatModel] = None,
	embedding_model: Optional[Embeddings] = None,
	**kwargs,
) -> pd.DataFrame:
	"""
	QA dataset generation using RAGAS.
	Returns qa dataset dataframe.

	:param corpus_df: Corpus dataframe.
	:param test_size: Number of queries to generate.
	:param distributions: Distributions of different types of questions.
	    Default is "simple is 0.5, multi_context is 0.4, and reasoning is 0.1."
	    Each type of questions refers to Ragas evolution types.
	:param generator_llm: Generator language model from Langchain.
	:param critic_llm: Critic language model from Langchain.
	:param embedding_model: Embedding model from Langchain.
	:param kwargs: The additional option to pass to the 'generate_with_langchain_docs' method.
	    You can input 'with_debugging_logs', 'is_async', 'raise_exceptions', and 'run_config'.
	:return: QA dataset dataframe.
	"""
	from ragas.testset import TestsetGenerator
	from ragas.testset.evolutions import simple, reasoning, multi_context

	if generator_llm is None:
		generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
	if critic_llm is None:
		critic_llm = ChatOpenAI(model="gpt-4-turbo")
	if embedding_model is None:
		embedding_model = OpenAIEmbeddings()
	if distributions is None:
		distributions = {simple: 0.5, multi_context: 0.4, reasoning: 0.1}

	assert sum(list(distributions.values())) == 1.0, "Sum of distributions must be 1.0"

	generator = TestsetGenerator.from_langchain(
		generator_llm, critic_llm, embedding_model
	)

	langchain_docs = corpus_df_to_langchain_documents(corpus_df)

	test_df = generator.generate_with_langchain_docs(
		langchain_docs, test_size, distributions=distributions, **kwargs
	).to_pandas()

	result_df = pd.DataFrame(
		{
			"qid": [str(uuid.uuid4()) for _ in range(len(test_df))],
			"query": test_df["question"].tolist(),
			"generation_gt": list(map(lambda x: x, test_df["ground_truth"].tolist())),
		}
	)

	result_df["retrieval_gt"] = test_df["metadata"].apply(
		lambda x: list(map(lambda y: y["filename"], x))
	)
	result_df = cast_qa_dataset(result_df)

	return result_df

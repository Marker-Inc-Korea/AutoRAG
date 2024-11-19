import os
import pathlib
import uuid
from typing import Callable

import pandas as pd


def generate_qa_row(llm, corpus_data_row):
	"""
	this sample code to generate rag dataset using OpenAI chat model

	:param llm: guidance model
	:param corpus_data_row: need "contents" column
	:return: should to be dict which has "query", "generation_gt" columns at least.
	"""
	from guidance import gen
	import guidance

	temp_llm = llm
	with guidance.user():
		temp_llm += f"""
    You have to found a passge to solve "the problem".
    You need to build a clean and clear set of (problem, passage, answer) in json format
    so that you don't have to ask about "the problem" again.
    problem need to end with question mark("?").
    The process of approaching the answer based on the information of the given passage
    must be clearly and neatly displayed in the answer.\n
    \n
    Here is set of (problem, passage, answer) in JSON format:\n
    {{\n
        "passage": {corpus_data_row["contents"]}\n
        "problem":
    """

	with guidance.assistant():
		temp_llm += gen("query", stop="?")
	with guidance.user():
		temp_llm += """
        "answer":
        """
	with guidance.assistant():
		temp_llm += gen("generation_gt")

	corpus_data_row["metadata"]["qa_generation"] = "simple"

	response = {"query": temp_llm["query"], "generation_gt": temp_llm["generation_gt"]}
	return response


def generate_simple_qa_dataset(
	llm,
	corpus_data: pd.DataFrame,
	output_filepath: str,
	generate_row_function: Callable,
	**kwargs,
):
	"""
	corpus_data to qa_dataset
	qa_dataset will be saved to filepath(file_dir/filename)

	:param llm: guidance.models.Model
	:param corpus_data: pd.DataFrame. refer to the basic structure
	:param output_filepath: file_dir must exist, filepath must not exist. file extension must be .parquet
	:param generate_row_function: input(llm, corpus_data_row, kwargs) output(dict[columns contain "query" and "generation_gt"])
	:param kwargs: if generate_row_function requires more args, use kwargs
	:return: qa_dataset as pd.DataFrame
	"""
	output_file_dir = pathlib.PurePath(output_filepath).parent
	if not os.path.isdir(output_file_dir):
		raise NotADirectoryError(f"directory {output_file_dir}  not found.")
	if not output_filepath.endswith("parquet"):
		raise NameError(
			f'file path: {output_filepath}  filename extension need to be ".parquet"'
		)
	if os.path.exists(output_filepath):
		raise FileExistsError(
			f"{output_filepath.split('/')[-1]} already exists in {output_file_dir}."
		)

	qa_data_lst = []
	for _, corpus_data_row in corpus_data.iterrows():
		response = generate_row_function(
			llm=llm, corpus_data_row=corpus_data_row, **kwargs
		)
		qa_data_lst.append(
			{
				"qid": str(uuid.uuid4()),
				"query": response["query"],
				"retrieval_gt": [[corpus_data_row["doc_id"]]],
				"generation_gt": [response["generation_gt"]],
				"metadata": corpus_data_row["metadata"],
			}
		)

	qa_dataset = pd.DataFrame(qa_data_lst)
	qa_dataset.to_parquet(output_filepath, index=False)

	return qa_dataset

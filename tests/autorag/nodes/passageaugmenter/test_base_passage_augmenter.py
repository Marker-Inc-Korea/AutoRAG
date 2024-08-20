import os
import pathlib

import pandas as pd

root_dir = pathlib.PurePath(
	os.path.dirname(os.path.realpath(__file__))
).parent.parent.parent
project_dir = os.path.join(root_dir, "resources", "sample_project")
qa_data = pd.read_parquet(os.path.join(project_dir, "data", "qa.parquet"))
corpus_data = pd.read_parquet(os.path.join(project_dir, "data", "corpus.parquet"))
doc_id_list = corpus_data["doc_id"].tolist()
previous_result = pd.DataFrame(
	{
		"qid": [1, 2],
		"query": [
			"What is the capital of France?",
			"How many members are in Newjeans?",
		],
		"retrieved_ids": [
			[doc_id_list[1], doc_id_list[3]],
			[doc_id_list[5], doc_id_list[7]],
		],
		"retrieved_contents": [
			[
				"Paris is the capital of France.",
				"Paris is one of the capital from France. Isn't it?",
			],
			["Newjeans has 5 members.", "Danielle is one of the members of Newjeans."],
		],
		"retrieve_scores": [[0.8, 0.1], [0.2, 0.1]],
		"retrieval_gt": [1, 1],
		"generation_gt": ["answer", "answer"],
		"retrieval_f1": [0.4, 1.0],
		"retrieval_recall": [1.0, 0.3],
	}
)
ids_list = [[doc_id_list[1]], [doc_id_list[3]], [doc_id_list[0], doc_id_list[29]]]

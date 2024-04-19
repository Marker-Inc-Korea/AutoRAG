import os
import pathlib

import pandas as pd

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
project_dir = os.path.join(root_dir, "resources", "sample_project")
qa_data = pd.read_parquet(os.path.join(project_dir, "data", "qa.parquet"))
corpus_data = pd.read_parquet(os.path.join(project_dir, "data", "corpus.parquet"))
previous_result = pd.DataFrame({
    'qid': [1, 2],
    'query': ['What is the capital of France?', 'How many members are in Newjeans?'],
    'retrieved_ids': [['2ec51121-3640-43d7-85db-1259cddaa4c9', 'dc5dc6d5-b53f-4a08-888d-2d6e5c85cf4b'],
                      ['69a07e70-1767-468e-9eb1-010e8b0f61d0', 'ba7032bf-fe9e-40aa-a207-7a211d817309']],
    'retrieved_contents': [['Paris is the capital of France.', 'Paris is one of the capital from France. Isn\'t it?'],
                           ['Newjeans has 5 members.', 'Danielle is one of the members of Newjeans.']],
    'retrieve_scores': [[0.1, 0.8], [0.1, 0.2]],
    'retrieval_gt': [1, 1],
    'generation_gt': ['answer', 'answer'],
    'retrieval_f1': [0.4, 1.0],
    'retrieval_recall': [1.0, 0.3]
})
doc_id_list = corpus_data["doc_id"].tolist()
ids_list = [[doc_id_list[1]], [doc_id_list[3]], [doc_id_list[0], doc_id_list[29]]]

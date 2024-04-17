import os
import pathlib

import pandas as pd

ids_list = [['2'], ['3'], ['0', '6']]

all_ids_list = [str(i) for i in range(7)]

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
project_dir = os.path.join(root_dir, "resources", "sample_project")
corpus_data = pd.read_parquet(os.path.join(project_dir, "data", "corpus.parquet"))
previous_result = pd.DataFrame({
    'qid': [1, 2],
    'query': ['What is the capital of France?', 'How many members are in Newjeans?'],
    'retrieved_ids': [['2ec51121-3640-43d7-85db-1259cddaa4c9', 'dc5dc6d5-b53f-4a08-888d-2d6e5c85cf4b'],
                      ['69a07e70-1767-468e-9eb1-010e8b0f61d0', 'ba7032bf-fe9e-40aa-a207-7a211d817309']],
    'retrieval_gt': [1, 1],
    'generation_gt': ['answer', 'answer']
})

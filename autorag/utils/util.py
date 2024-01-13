from typing import List

import pandas as pd
import swifter


def fetch_contents(corpus_data: pd.DataFrame, ids: List[List[str]]) -> List[List[str]]:
    assert isinstance(ids[0], list), "ids must be a list of list of ids."
    id_df = pd.DataFrame(ids, columns=[f'id_{i}' for i in range(len(ids[0]))])
    contents_df = id_df.swifter.applymap(
        lambda x: corpus_data.loc[lambda row: row['doc_id'] == x]['contents'].values[0])
    return contents_df.values.tolist()

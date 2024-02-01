from typing import Tuple, List

import pandas as pd
import pytest

from autorag.evaluate import evaluate_retrieval_contents

gt = [
    ['Enough for drinking water', 'Just looking for a water bottle'],
    ['Do you want to buy some?']
]
pred = [
    ['Enough for mixing water', 'I want to do a nothing', 'Just looking is a very healthy'],
    ['Do you want to buy some?', 'I want to buy some', 'I want to buy some water']
]


@evaluate_retrieval_contents(retrieval_gt=gt,
                             metrics=['retrieval_token_recall', 'retrieval_token_precision', 'retrieval_token_f1'])
def pseudo_retrieval() -> Tuple[List[List[str]], List[List[float]], List[List[str]]]:
    return pred, [[0.3, 0.2, 0.1], [0.3, 0.2, 0.1]], [['pred-0', 'pred-1', 'pred-2'], ['pred-3', 'pred-4', 'pred-5']]


def test_evaluate_retrieval_contents():
    result_df = pseudo_retrieval()
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 2
    assert set(result_df.columns) == {
        'retrieved_contents', 'retrieved_ids', 'retrieve_scores',
        'retrieval_token_recall', 'retrieval_token_precision', 'retrieval_token_f1'
    }
    assert result_df['retrieval_token_recall'].tolist() == pytest.approx([0.383333, 0.777777], rel=0.001)
    assert result_df['retrieval_token_precision'].tolist() == pytest.approx([0.383333, 0.8222222], rel=0.001)
    assert result_df['retrieval_token_f1'].tolist() == pytest.approx([0.38333, 0.797979], rel=0.001)


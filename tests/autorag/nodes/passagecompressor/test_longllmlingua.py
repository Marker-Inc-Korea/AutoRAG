import pandas as pd

from autorag.nodes.passagecompressor import longllmlingua
from tests.autorag.nodes.passagecompressor.test_base_passage_compressor import (queries, retrieved_contents,
                                                                                check_result, df)


def test_longllmlingua_default():
    result = longllmlingua.__wrapped__(queries, retrieved_contents, [], [])
    check_result(result)


def test_refine_node():
    result = longllmlingua(
        "project_dir",
        df,
        max_tokens=75,
    )
    assert isinstance(result, pd.DataFrame)
    contents = result['retrieved_contents'].tolist()
    assert isinstance(contents, list)
    assert len(contents) == len(queries)
    assert isinstance(contents[0], list)
    assert len(contents[0]) == 1
    assert isinstance(contents[0][0], str)
    assert bool(contents[0][0]) is True

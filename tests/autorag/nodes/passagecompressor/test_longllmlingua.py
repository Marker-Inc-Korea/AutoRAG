import pandas as pd
import pytest

from autorag.nodes.passagecompressor import longllmlingua
from tests.autorag.nodes.passagecompressor.test_base_passage_compressor import (
	queries,
	retrieved_contents,
	check_result,
	df,
)


@pytest.mark.skip(reason="This test needs CUDA enabled machine.")
def test_longllmlingua():
	result = longllmlingua.__wrapped__(queries, retrieved_contents, [], [])
	check_result(result)


@pytest.mark.skip(reason="This test needs CUDA enabled machine.")
def test_longllmlingua_node():
	result = longllmlingua(
		"project_dir",
		df,
		target_token=75,
	)
	assert isinstance(result, pd.DataFrame)
	contents = result["retrieved_contents"].tolist()
	assert isinstance(contents, list)
	assert len(contents) == len(queries)
	assert isinstance(contents[0], list)
	assert len(contents[0]) == 1
	assert isinstance(contents[0][0], str)
	assert bool(contents[0][0]) is True

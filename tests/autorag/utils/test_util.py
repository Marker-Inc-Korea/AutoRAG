import itertools
import os
import pathlib
import tempfile

import pandas as pd
import pytest

from autorag.utils import fetch_contents
from autorag.utils.util import find_best_result_path, make_module_file_name

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent


@pytest.fixture
def module_name():
    return "test_module"


@pytest.fixture
def module_params():
    return {
        "param1": "value1",
        "param2": "value2",
        "param3": "value3",
    }


def test_make_module_file_name(module_name, module_params):
    module_file_name = make_module_file_name(module_name, module_params)
    assert module_file_name == "test_module=>param1_value1-param2_value2-param3_value3.parquet"

    module_file_name = make_module_file_name(module_name, {})
    assert module_file_name == "test_module.parquet"

    module_file_name = make_module_file_name(module_name, {"param1": "value1"})
    assert module_file_name == "test_module=>param1_value1.parquet"


def test_fetch_contents():
    corpus_data_path = os.path.join(root_dir, "resources", "corpus_data_sample.parquet")
    corpus_data = pd.read_parquet(corpus_data_path)
    search_rows = corpus_data.sample(n=10)
    find_contents = fetch_contents(corpus_data, list(map(lambda x: [x], search_rows['doc_id'].tolist())))
    assert len(find_contents) == len(search_rows)
    assert list(itertools.chain.from_iterable(find_contents)) == search_rows['contents'].tolist()


def test_find_best_result_path():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Set up the test files
        paths = [
            "best_result.parquet",
            "average_result.parquet",
            "worst_result.parquet",
            "best_other.txt"
        ]
        for file_name in paths:
            with open(os.path.join(tmpdirname, file_name), 'w') as f:
                f.write("test data")

        # Run the function under test
        best_path = find_best_result_path(tmpdirname)

        # Check that the function returns the correct path
        assert best_path == "best_result.parquet"

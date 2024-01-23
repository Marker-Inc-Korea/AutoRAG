import itertools
import os
import pathlib
import tempfile

import pandas as pd

from autorag.utils import fetch_contents
from autorag.utils.util import find_best_result_path

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent


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

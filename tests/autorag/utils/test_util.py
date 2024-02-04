import asyncio
import itertools
import os
import pathlib
import tempfile

import pandas as pd
import pytest
from llama_index.core.llms.types import CompletionResponse
from llama_index.llms import MockLLM

from autorag.utils import fetch_contents
from autorag.utils.util import find_best_result_path, load_summary_file, result_to_dataframe, \
    make_combinations, explode, replace_value_in_dict, normalize_string, convert_string_to_tuple_in_dict, process_batch

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent

summary_df = pd.DataFrame({
    'best_module_name': ['bm25', 'upr', 'gpt-4'],
    'best_module_params': [{'top_k': 50}, {'model': 'llama-2', 'havertz': 'chelsea'}, {'top_p': 0.9}],
})


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


@pytest.fixture
def summary_path():
    with tempfile.TemporaryDirectory() as tmp_dir:
        summary_path = os.path.join(tmp_dir, "summary.csv")
        summary_df.to_csv(summary_path, index=False)
        yield summary_path


def test_fetch_contents():
    corpus_data_path = os.path.join(root_dir, "resources", "corpus_data_sample.parquet")
    corpus_data = pd.read_parquet(corpus_data_path)
    search_rows = corpus_data.sample(n=10)
    find_contents = fetch_contents(corpus_data, list(map(lambda x: [x], search_rows['doc_id'].tolist())))
    assert len(find_contents) == len(search_rows)
    assert list(itertools.chain.from_iterable(find_contents)) == search_rows['contents'].tolist()

    corpus_data = pd.DataFrame({
        'doc_id': ['doc1', 'doc2', 'doc3'],
        'contents': ['apple', 'banana', 'cherry'],
    })
    find_contents = fetch_contents(corpus_data, [['doc3', 'doc1'], ['doc2']])
    assert find_contents[0] == ['cherry', 'apple']
    assert find_contents[1] == ['banana']


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


def test_load_summary_file(summary_path):
    with pytest.raises(ValueError):
        load_summary_file(summary_path)
    df = load_summary_file(summary_path, ['best_module_params'])
    assert df.equals(summary_df)


def test_result_to_dataframe():
    @result_to_dataframe(['col_1', 'col_2'])
    def func1():
        return [1, 2], [3, 4]

    result1 = func1()
    assert isinstance(result1, pd.DataFrame)
    assert result1.columns.tolist() == ['col_1', 'col_2']
    assert result1['col_1'].tolist() == [1, 2]
    assert result1['col_2'].tolist() == [3, 4]

    @result_to_dataframe(['col_1'])
    def func2():
        return [1, 2, 3]

    result2 = func2()
    assert isinstance(result2, pd.DataFrame)
    assert result2.columns.tolist() == ['col_1']
    assert result2['col_1'].tolist() == [1, 2, 3]


def test_make_combinations():
    target_dict = {'key1': 'value1', 'key2': ['value1', 'value2'], 'key3': 'value3', 'key4': ['value4', 'value5']}
    solution = [
        {'key1': 'value1', 'key2': 'value1', 'key3': 'value3', 'key4': 'value4'},
        {'key1': 'value1', 'key2': 'value1', 'key3': 'value3', 'key4': 'value5'},
        {'key1': 'value1', 'key2': 'value2', 'key3': 'value3', 'key4': 'value4'},
        {'key1': 'value1', 'key2': 'value2', 'key3': 'value3', 'key4': 'value5'}
    ]
    combinations = make_combinations(target_dict)
    assert len(combinations) == len(solution)
    assert all([combination in solution for combination in combinations])

    elem1 = {
        'key5': 'value5',
        'key6': ['value6', 'value7']
    }
    elem2 = {'key7': 'value8'}
    value_of_key_4 = [elem1, elem2]
    target_dict = {'key1': 'value1', 'key2': ['value1', 'value2'], 'key3': 'value3', 'key4': value_of_key_4}
    combinations = make_combinations(target_dict)
    solution = [
        {'key1': 'value1', 'key2': 'value1', 'key3': 'value3', 'key4': elem1},
        {'key1': 'value1', 'key2': 'value2', 'key3': 'value3', 'key4': elem1},
        {'key1': 'value1', 'key2': 'value1', 'key3': 'value3', 'key4': elem2},
        {'key1': 'value1', 'key2': 'value2', 'key3': 'value3', 'key4': elem2},
    ]
    assert len(combinations) == len(solution)
    assert all([combination in solution for combination in combinations])

    target_dict = {'key1': 'value1', 'key2': ['value1', 'value2'], 'key3': 'value3', 'key4': ('value4', 'value5')}
    solution = [
        {'key1': 'value1', 'key2': 'value1', 'key3': 'value3', 'key4': ('value4', 'value5')},
        {'key1': 'value1', 'key2': 'value2', 'key3': 'value3', 'key4': ('value4', 'value5')},
    ]
    combinations = make_combinations(target_dict)
    assert len(combinations) == len(solution)
    assert all([combination in solution for combination in combinations])


def test_explode():
    index_values = ['a', 'b', 'c']
    explode_values = [
        ['apple', 'banana', 'cherry'],
        ['april', 'may'],
        ['alpha'],
    ]
    result_index, result_values = explode(index_values, explode_values)
    assert result_index == ['a', 'a', 'a', 'b', 'b', 'c']
    assert result_values == ['apple', 'banana', 'cherry', 'april', 'may', 'alpha']


def test_replace_value_in_dict():
    target_dict = {
        'key1': 'value1',
        'key2': 'value2',
        'key3': 'value3',
    }
    result_dict = replace_value_in_dict(target_dict, 'key1', 'value4')
    assert result_dict == {
        'key1': 'value4',
        'key2': 'value2',
        'key3': 'value3',
    }
    result_dict = replace_value_in_dict(target_dict, 'key4', 'value4')
    assert result_dict == target_dict


def test_normalize_string():
    text = "This IS a TEST Text."
    expected = "this is test text"
    assert normalize_string(text) == expected

    text = "Hello, world! This is a test."
    expected = "hello world this is test"
    assert normalize_string(text) == expected

    text = "The quick brown fox jumps over the lazy dog."
    expected = "quick brown fox jumps over lazy dog"
    assert normalize_string(text) == expected

    text = "This    is      a test    text."
    expected = "this is test text"
    assert normalize_string(text) == expected

    text = "The, QUICK Brown-Fox; jumps over... the LAZY dog!"
    expected = "quick brownfox jumps over lazy dog"
    assert normalize_string(text) == expected


def test_convert_string_to_tuple_in_dict():
    # Example usage
    data = {
        'key1': '(1, \'two\', 3)',
        'key2': ['(4, 5, \'six\')', {'nested_key': '(7, 8, \'nine\')'},
                 {'key4': 'value2'}],
        'key3': {'nested_key2': '(10, \'eleven\', 12)',
                 'nested_key3': 'value1',
                 'nested_key4': {'nested_key5': '(\'thirteen\', 14, 15)'}},
    }
    result = convert_string_to_tuple_in_dict(data)
    assert result == {
        'key1': (1, 'two', 3),
        'key2': [
            (4, 5, 'six'),
            {
                'nested_key': (7, 8, 'nine')
            },
            {
                'key4': 'value2'
            }
        ],
        'key3': {
            'nested_key2': (10, 'eleven', 12),
            'nested_key3': 'value1',
            'nested_key4': {
                'nested_key5': ('thirteen', 14, 15)
            }
        }
    }


def test_process_batch():
    prompts = [str(i) for i in range(1000)]
    results = [CompletionResponse(text=prompt) for prompt in prompts]
    mock_llm = MockLLM()

    tasks = [mock_llm.acomplete(prompt) for prompt in prompts]
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(process_batch(tasks, batch_size=64))

    assert result == results

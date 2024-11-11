import asyncio
import itertools
import os
import pathlib
import tempfile
from datetime import datetime, date

import numpy as np
import pandas as pd
import pytest
import tiktoken
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import CompletionResponse
from llama_index.embeddings.openai import OpenAIEmbedding

from autorag.utils import fetch_contents
from autorag.utils.util import (
	load_summary_file,
	result_to_dataframe,
	make_combinations,
	explode,
	replace_value_in_dict,
	normalize_string,
	convert_string_to_tuple_in_dict,
	process_batch,
	convert_env_in_dict,
	openai_truncate_by_token,
	convert_datetime_string,
	split_dataframe,
	find_trial_dir,
	find_node_summary_files,
	normalize_unicode,
	demojize,
	preprocess_text,
	dict_to_markdown,
	dict_to_markdown_table,
	convert_inputs_to_list,
	to_list,
	get_event_loop,
	find_key_values,
	pop_params,
	apply_recursive,
)
from tests.mock import MockLLM

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent

summary_df = pd.DataFrame(
	{
		"best_module_name": ["bm25", "upr", "gpt-4"],
		"best_module_params": [
			{"top_k": 50},
			{"model": "llama-2", "havertz": "chelsea"},
			{"top_p": 0.9},
		],
	}
)


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
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
		summary_path = os.path.join(tmp_dir, "summary.csv")
		summary_df.to_csv(summary_path, index=False)
		yield summary_path


def test_fetch_contents():
	corpus_data_path = os.path.join(root_dir, "resources", "corpus_data_sample.parquet")
	corpus_data = pd.read_parquet(corpus_data_path)
	search_rows = corpus_data.sample(n=10)
	find_contents = fetch_contents(
		corpus_data, list(map(lambda x: [x], search_rows["doc_id"].tolist()))
	)
	assert len(find_contents) == len(search_rows)
	assert (
		list(itertools.chain.from_iterable(find_contents))
		== search_rows["contents"].tolist()
	)

	corpus_data = pd.DataFrame(
		{
			"doc_id": ["doc1", "doc2", "doc3"],
			"contents": ["apple", "banana", "cherry"],
			"metadata": [
				{"last_modified_datetime": datetime(2022, 1, 1, 0, 0, 0)},
				{"last_modified_datetime": datetime(2022, 1, 2, 0, 0, 0)},
				{"last_modified_datetime": datetime(2022, 1, 3, 0, 0, 0)},
			],
		}
	)
	find_contents = fetch_contents(corpus_data, [["doc3", "doc1"], ["doc2"]])
	assert find_contents[0] == ["cherry", "apple"]
	assert find_contents[1] == ["banana"]

	find_metadatas = fetch_contents(
		corpus_data, [["doc3", "doc1"], ["doc2"]], "metadata"
	)
	assert find_metadatas[0] == [
		{"last_modified_datetime": datetime(2022, 1, 3, 0, 0, 0)},
		{"last_modified_datetime": datetime(2022, 1, 1, 0, 0, 0)},
	]
	assert find_metadatas[1] == [
		{"last_modified_datetime": datetime(2022, 1, 2, 0, 0, 0)}
	]

	find_empty = fetch_contents(corpus_data, [[], ["doc2"]])
	assert find_empty[0] == [None]
	assert find_empty[1] == ["banana"]

	find_blank = fetch_contents(corpus_data, [[""], ["doc2"]])
	assert find_blank[0] == [None]
	assert find_blank[1] == ["banana"]


def test_load_summary_file(summary_path):
	with pytest.raises(ValueError):
		load_summary_file(summary_path)
	df = load_summary_file(summary_path, ["best_module_params"])
	assert df.equals(summary_df)


def test_load_summary_file_recency_filter():
	df = pd.DataFrame(
		{
			"module_name": ["havertz", "recency_filter"],
			"module_params": [
				{"jazz": "eastsidegunn"},
				{"threshold": datetime(2022, 1, 3, 0, 1, 3)},
			],
		}
	)
	with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as csv_file:
		df.to_csv(csv_file.name, index=False)
		load_df = load_summary_file(csv_file.name)
		assert load_df.equals(df)
		csv_file.close()
		os.unlink(csv_file.name)


def test_convert_datetime_string():
	datetime_dict = {"threshold": datetime(2022, 1, 3, 0, 0, 3)}
	date_dict = {"threshold": date(2001, 7, 11)}
	result1 = convert_datetime_string(str(datetime_dict))
	result2 = convert_datetime_string(str(date_dict))

	assert result1 == datetime_dict["threshold"]
	assert result2 == date_dict["threshold"]


def test_result_to_dataframe():
	@result_to_dataframe(["col_1", "col_2"])
	def func1():
		return [1, 2], [3, 4]

	result1 = func1()
	assert isinstance(result1, pd.DataFrame)
	assert result1.columns.tolist() == ["col_1", "col_2"]
	assert result1["col_1"].tolist() == [1, 2]
	assert result1["col_2"].tolist() == [3, 4]

	@result_to_dataframe(["col_1"])
	def func2():
		return [1, 2, 3]

	result2 = func2()
	assert isinstance(result2, pd.DataFrame)
	assert result2.columns.tolist() == ["col_1"]
	assert result2["col_1"].tolist() == [1, 2, 3]


def test_make_combinations():
	target_dict = {
		"key1": "value1",
		"key2": ["value1", "value2"],
		"key3": "value3",
		"key4": ["value4", "value5"],
	}
	solution = [
		{"key1": "value1", "key2": "value1", "key3": "value3", "key4": "value4"},
		{"key1": "value1", "key2": "value1", "key3": "value3", "key4": "value5"},
		{"key1": "value1", "key2": "value2", "key3": "value3", "key4": "value4"},
		{"key1": "value1", "key2": "value2", "key3": "value3", "key4": "value5"},
	]
	combinations = make_combinations(target_dict)
	assert len(combinations) == len(solution)
	assert all([combination in solution for combination in combinations])

	elem1 = {"key5": "value5", "key6": ["value6", "value7"]}
	elem2 = {"key7": "value8"}
	value_of_key_4 = [elem1, elem2]
	target_dict = {
		"key1": "value1",
		"key2": ["value1", "value2"],
		"key3": "value3",
		"key4": value_of_key_4,
	}
	combinations = make_combinations(target_dict)
	solution = [
		{"key1": "value1", "key2": "value1", "key3": "value3", "key4": elem1},
		{"key1": "value1", "key2": "value2", "key3": "value3", "key4": elem1},
		{"key1": "value1", "key2": "value1", "key3": "value3", "key4": elem2},
		{"key1": "value1", "key2": "value2", "key3": "value3", "key4": elem2},
	]
	assert len(combinations) == len(solution)
	assert all([combination in solution for combination in combinations])

	target_dict = {
		"key1": "value1",
		"key2": ["value1", "value2"],
		"key3": "value3",
		"key4": ("value4", "value5"),
	}
	solution = [
		{
			"key1": "value1",
			"key2": "value1",
			"key3": "value3",
			"key4": ("value4", "value5"),
		},
		{
			"key1": "value1",
			"key2": "value2",
			"key3": "value3",
			"key4": ("value4", "value5"),
		},
	]
	combinations = make_combinations(target_dict)
	assert len(combinations) == len(solution)
	assert all([combination in solution for combination in combinations])


def test_explode():
	index_values = ["a", "b", "c"]
	explode_values = [
		["apple", "banana", "cherry"],
		["april", "may"],
		["alpha"],
	]
	result_index, result_values = explode(index_values, explode_values)
	assert result_index == ["a", "a", "a", "b", "b", "c"]
	assert result_values == ["apple", "banana", "cherry", "april", "may", "alpha"]


def test_replace_value_in_dict():
	target_dict = {
		"key1": "value1",
		"key2": "value2",
		"key3": "value3",
	}
	result_dict = replace_value_in_dict(target_dict, "key1", "value4")
	assert result_dict == {
		"key1": "value4",
		"key2": "value2",
		"key3": "value3",
	}
	result_dict = replace_value_in_dict(target_dict, "key4", "value4")
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
		"key1": "(1, 'two', 3)",
		"key2": ["(4, 5, 'six')", {"nested_key": "(7, 8, 'nine')"}, {"key4": "value2"}],
		"key3": {
			"nested_key2": "(10, 'eleven', 12)",
			"nested_key3": "value1",
			"nested_key4": {"nested_key5": "('thirteen', 14, 15)"},
		},
	}
	result = convert_string_to_tuple_in_dict(data)
	assert result == {
		"key1": (1, "two", 3),
		"key2": [(4, 5, "six"), {"nested_key": (7, 8, "nine")}, {"key4": "value2"}],
		"key3": {
			"nested_key2": (10, "eleven", 12),
			"nested_key3": "value1",
			"nested_key4": {"nested_key5": ("thirteen", 14, 15)},
		},
	}


def test_convert_env_in_dict():
	os.environ["ENV_VAR1"] = "value1"
	os.environ["ENV_VAR2"] = "value2"
	os.environ["ENV_VAR3"] = "value3"
	data = {
		"key1": "value1",
		"key2": [
			"value1",
			"${ENV_VAR1}",
		],
		"key3": "${ENV_VAR2}",
		"key4": {
			"key5": "value1",
			"key6": "${ENV_VAR3}",
			"key7": [
				"value1",
				"${ENV_VAR4}",
			],
		},
		"prompt": "This is a prompt with ${ENV_VAR1} and ${ENV_VAR2}.",
	}
	result = convert_env_in_dict(data)
	assert result == {
		"key1": "value1",
		"key2": [
			"value1",
			"value1",
		],
		"key3": "value2",
		"key4": {
			"key5": "value1",
			"key6": "value3",
			"key7": [
				"value1",
				"",
			],
		},
		"prompt": "This is a prompt with value1 and value2.",
	}


def test_process_batch():
	prompts = [str(i) for i in range(1000)]
	results = [CompletionResponse(text=prompt) for prompt in prompts]
	mock_llm = MockLLM()

	tasks = [mock_llm.acomplete(prompt) for prompt in prompts]
	loop = get_event_loop()
	result = loop.run_until_complete(process_batch(tasks, batch_size=64))

	assert result == results


def test_openai_truncate_by_token():
	base_text = "This is a test text."
	t1 = base_text * 5
	t2 = base_text * 200000
	t3 = base_text * 20

	truncated = openai_truncate_by_token([t1, t2, t3], 8000, "text-embedding-ada-002")
	assert len(truncated) == 3
	assert truncated[0] == base_text * 5
	assert len(truncated[1]) < len(t2)
	assert (
		len(tiktoken.encoding_for_model("text-embedding-ada-002").encode(truncated[1]))
		== 8000
	)
	assert len(truncated[2]) == len(t3)

	truncated = openai_truncate_by_token([t1, t3], 8000, "solar-1-mini-embedding")
	assert len(truncated) == 2
	assert truncated[0] == base_text * 5
	assert truncated[1] == base_text * 20


def test_split_dataframe():
	df = pd.DataFrame({"a": list(range(10)), "b": list(range(10, 20))})

	df_list_1 = split_dataframe(df, chunk_size=5)
	assert len(df_list_1) == 2
	assert len(df_list_1[0]) == 5
	assert pd.DataFrame({"a": list(range(5)), "b": list(range(10, 15))}).equals(
		df_list_1[0]
	)

	df_list_2 = split_dataframe(df, chunk_size=3)
	assert len(df_list_2) == 4
	assert len(df_list_2[0]) == 3
	assert len(df_list_2[-1]) == 1
	assert pd.DataFrame({"a": list(range(3)), "b": list(range(10, 13))}).equals(
		df_list_2[0]
	)


def test_find_trial_dir():
	project_dir = os.path.join(root_dir, "resources", "result_project")
	trial_dirs = find_trial_dir(project_dir)

	assert len(trial_dirs) == 4
	assert all(isinstance(int(os.path.basename(path)), int) for path in trial_dirs)


def test_find_node_summary_files():
	trial_dir = os.path.join(root_dir, "resources", "result_project", "2")
	node_summary_paths = find_node_summary_files(trial_dir)

	assert len(node_summary_paths) == 4
	assert all(os.path.basename(path) == "summary.csv" for path in node_summary_paths)


def test_demojize():
	str = "👍엄지엄지척"
	new_str = demojize(str)
	assert new_str == ":thumbs_up:엄지엄지척"


def test_normalize_unicode():
	str1 = "전국보행자전용도로표준데이터"
	str2 = "전국보행자전용도로표준데이터"
	assert len(str1) == 14
	assert len(str2) == 34
	assert str1 != str2

	new_str1 = normalize_unicode(str1)
	new_str2 = normalize_unicode(str2)

	assert len(new_str1) == 14
	assert len(new_str2) == 14
	assert new_str1 == new_str2


def test_preprocess():
	str1 = (
		"👍전국보행자전용도로표준데이터👍"  # ":thumbs_up:" is added on both sides + 22
	)
	str2 = "👍전국보행자전용도로표준데이터👍"
	assert len(str1) == 16
	assert len(str2) == 36
	assert str1 != str2

	new_str1 = preprocess_text(str1)
	new_str2 = preprocess_text(str2)

	assert len(new_str1) == 36
	assert len(new_str2) == 36
	assert new_str1 == new_str2


def test_dict_to_markdown():
	data = {
		"Title": "Sample Document",
		"Author": "John Doe",
		"Content": {
			"Introduction": "This is the introduction.",
			"Body": [
				"First point",
				"Second point",
				{"Subsection": "Details about the second point"},
			],
			"Conclusion": "This is the conclusion.",
		},
	}
	markdown_text = dict_to_markdown(data)
	result_text = """# Title
Sample Document
# Author
John Doe
# Content
## Introduction
This is the introduction.
## Body
- First point
- Second point
### Subsection
Details about the second point
## Conclusion
This is the conclusion.
"""
	assert result_text == markdown_text


def test_dict_to_markdown_table():
	data = {"key1": "value1", "key2": "value2"}
	result = dict_to_markdown_table(data, "havertz", "william")
	result_text = """| havertz | william |
| :---: | :-----: |
| key1 | value1 |
| key2 | value2 |
"""
	assert result == result_text


@convert_inputs_to_list
def convert_inputs_to_list_function(int_type, str_type, iterable_type, iterable_type2):
	assert isinstance(int_type, int)
	assert isinstance(str_type, str)
	assert isinstance(iterable_type, list)
	assert isinstance(iterable_type2, list)


def test_convert_inputs_to_list():
	convert_inputs_to_list_function(1, "jax", (2, 3), (5, 6, [4, 66]))
	convert_inputs_to_list_function(
		1, "jax", np.array([3, 4]), [pd.Series([12, 13]), 14]
	)
	convert_inputs_to_list_function(
		4, "jax", pd.Series([7, 8, 9]), np.array([[3, 4], [4, 5]])
	)


def test_to_list():
	embedding_model = OpenAIEmbedding()
	new_model = to_list(embedding_model)
	assert isinstance(new_model, BaseEmbedding)


def test_find_key_values():
	# Test case 1: Simple dictionary
	data = {"a": 1, "b": 2, "c": {"a": 3, "d": 4}}
	target_key = "a"
	expected = [1, 3]
	assert find_key_values(data, target_key) == expected

	# Test case 2: Nested dictionary with lists
	data = {"a": 1, "b": [{"a": 2}, {"c": {"a": 3}}]}
	target_key = "a"
	expected = [1, 2, 3]
	assert find_key_values(data, target_key) == expected

	# Test case 3: List of dictionaries
	data = [{"a": 1}, {"b": 2, "c": {"a": 3}}]
	target_key = "a"
	expected = [1, 3]
	assert find_key_values(data, target_key) == expected

	# Test case 4: No matching key
	data = {"b": 2, "c": {"d": 4}}
	target_key = "a"
	expected = []
	assert find_key_values(data, target_key) == expected

	# Test case 5: Empty dictionary
	data = {}
	target_key = "a"
	expected = []
	assert find_key_values(data, target_key) == expected

	# Test case 6: Empty list
	data = []
	target_key = "a"
	expected = []
	assert find_key_values(data, target_key) == expected

	# Test case 7: Complex nested structure
	data = {"a": {"b": [{"a": 1}, {"c": {"a": 2}}]}, "d": [{"e": {"a": 3}}]}
	target_key = "a"
	expected = [{"b": [{"a": 1}, {"c": {"a": 2}}]}, 1, 2, 3]
	assert find_key_values(data, target_key) == expected


def test_pop_params():
	# Test function with no self or cls
	def func_no_self_cls(param1, param2):
		pass

	kwargs = {"param1": "value1", "param2": "value2", "extra_param": "extra_value"}
	expected = {"param1": "value1", "param2": "value2"}
	result = pop_params(func_no_self_cls, kwargs)
	assert result == expected
	assert kwargs == {"extra_param": "extra_value"}

	# Test function with self
	class TestClass:
		def func_with_self(self, param1, param2):
			pass

	kwargs = {
		"param1": "value1",
		"param2": "value2",
		"extra_param": "extra_value",
		"self": "self",
	}
	expected = {"param1": "value1", "param2": "value2"}
	result = pop_params(TestClass.func_with_self, kwargs)
	assert result == expected
	assert kwargs == {"extra_param": "extra_value", "self": "self"}

	# Test function with cls
	class TestClassWithCls:
		@classmethod
		def func_with_cls(cls, param1, param2):
			pass

	kwargs = {
		"param1": "value1",
		"param2": "value2",
		"extra_param": "extra_value",
		"cls": "cls",
	}
	expected = {"param1": "value1", "param2": "value2"}
	result = pop_params(TestClassWithCls.func_with_cls, kwargs)
	assert result == expected
	assert kwargs == {"extra_param": "extra_value", "cls": "cls"}

	# Test function with no parameters
	def func_no_params():
		pass

	kwargs = {"extra_param": "extra_value"}
	expected = {}
	result = pop_params(func_no_params, kwargs)
	assert result == expected
	assert kwargs == {"extra_param": "extra_value"}

	# Test function with mixed parameters
	def func_mixed(param1, param2, *args, **kwargs):
		pass

	kwargs = {"param1": "value1", "param2": "value2", "extra_param": "extra_value"}
	expected = {"param1": "value1", "param2": "value2"}
	result = pop_params(func_mixed, kwargs)
	assert result == expected
	assert kwargs == {"extra_param": "extra_value"}


def test_apply_recursive():
	data = [1, 2, 3, 4]
	result = apply_recursive(lambda x: x * 2, data)
	assert result == [2, 4, 6, 8]

	data = [[1, 2], [3, 4]]
	result = apply_recursive(lambda x: x * 2, data)
	assert result == [[2, 4], [6, 8]]

	data = [[1, [2, 3]], [4, [5, 6]]]
	result = apply_recursive(lambda x: x * 2, data)
	assert result == [[2, [4, 6]], [8, [10, 12]]]

	data = []
	result = apply_recursive(lambda x: x * 2, data)
	assert result == []

	data = 5
	result = apply_recursive(lambda x: x * 2, data)
	assert result == 10

	data = [(4, 5), (6, 7), [5, [6, 7]], np.array([4, 5]), pd.Series([4, 5])]
	result = apply_recursive(lambda x: x * 2, data)
	assert result == [[8, 10], [12, 14], [10, [12, 14]], [8, 10], [8, 10]]

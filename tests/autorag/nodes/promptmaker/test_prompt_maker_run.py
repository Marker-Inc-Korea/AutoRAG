import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from autorag.nodes.generator import llama_index_llm
from autorag.nodes.promptmaker import fstring
from autorag.nodes.promptmaker.run import evaluate_generator_result, evaluate_one_prompt_maker_node, \
    run_prompt_maker_node
from autorag.utils.util import load_summary_file

prompts = ['Hello, Do you know the world without war?',
           'Hi, I am dreaming about the world without any war.']
sample_generated_texts = ['hello. This is the world speaking',
                          'Hello, Do you know the world without war?']
sample_generation_gt = [
    ['Hello from space. Hi! This is the world speaking.'],
    ['Hello, Do you know the world without war?', 'Hi, I am dreaming about the world without any war.']
]
metrics = ['bleu', 'rouge']
previous_result = pd.DataFrame({
    'query': ['What is the war?', 'Who is John Lennon?'],
    'retrieved_contents': [['War is horrible.', 'War is bad.'],
                           ['John Lennon is a musician.', 'John Lennon is a singer.']],
    'test_column': ['test_value_1', 'test_value_2'],
})


def test_evaluate_generator_result():
    sample_df = pd.DataFrame({'generated_texts': sample_generated_texts})
    result_df = evaluate_generator_result(sample_df, sample_generation_gt, metrics)
    assert all(metric_name in result_df.columns for metric_name in metrics)
    assert len(result_df) == len(sample_generated_texts)


def test_evaluate_one_prompt_maker_node():
    generator_funcs = [llama_index_llm, llama_index_llm]
    generator_params = [{'llm': 'openai', 'model': 'gpt-3.5-turbo'},
                        {'llm': 'openai', 'model': 'gpt-3.5-turbo-1106'}]
    project_dir = '_'
    best_result = evaluate_one_prompt_maker_node(generator_funcs, generator_params, prompts, sample_generation_gt,
                                                 metrics, project_dir)
    assert isinstance(best_result, pd.DataFrame)
    assert all(metric_name in best_result.columns for metric_name in metrics)
    assert len(best_result) == len(prompts)


@pytest.fixture
def node_line_dir():
    with tempfile.TemporaryDirectory() as project_dir:
        data_dir = os.path.join(project_dir, "data")
        os.makedirs(data_dir)
        qa_data = pd.DataFrame({
            'qid': ['id-1', 'id-2'],
            'query': ['What is the war?', 'Who is John Lennon?'],
            'retrieval_gt': [[['doc-1']], [['doc-2']]],
            'generation_gt': sample_generation_gt,
        })
        qa_data.to_parquet(os.path.join(data_dir, "qa.parquet"), index=False)
        trial_dir = os.path.join(project_dir, "trial")
        os.makedirs(trial_dir)
        node_line_path = os.path.join(trial_dir, "node_line_1")
        os.makedirs(node_line_path)
        yield node_line_path


def check_best_result(best_df: pd.DataFrame):
    assert isinstance(best_df, pd.DataFrame)
    assert len(best_df) == len(previous_result)
    assert set(best_df.columns) == {
        'query', 'retrieved_contents', 'test_column', 'prompts', 'prompt_maker_bleu', 'prompt_maker_rouge'
    }


def check_summary_df(node_line_dir):
    # check the files saved properly
    summary_path = os.path.join(node_line_dir, "prompt_maker", "summary.csv")
    assert os.path.exists(summary_path)
    summary_df = load_summary_file(summary_path)
    assert len(summary_df) == len(previous_result)
    assert set(summary_df.columns) == {'filename', 'module_name', 'module_params', 'execution_time',
                                       'prompt_maker_bleu', 'prompt_maker_rouge', 'is_best'}
    best_filename = summary_df[summary_df['is_best']]['filename'].values[0]
    return best_filename


def test_run_prompt_maker_node(node_line_dir):
    modules = [fstring, fstring]
    params = [{'prompt': 'Tell me something about the question: {query} \n\n {retrieved_contents}'},
              {'prompt': 'Question: {query} \n Something to read: {retrieved_contents} \n What\'s your answer?'}]
    strategies = {
        'metrics': metrics,
        'speed_threshold': 5,
        'generator_modules': [{
            'module_type': 'llama_index_llm',
            'llm': 'openai',
            'model': ['gpt-3.5-turbo', 'gpt-3.5-turbo-1106'],
        }]
    }
    best_result = run_prompt_maker_node(modules, params, previous_result, node_line_dir, strategies)
    check_best_result(best_result)
    best_filename = check_summary_df(node_line_dir)
    best_result_path = os.path.join(node_line_dir, "prompt_maker", f"best_{best_filename}")
    assert os.path.exists(best_result_path)

    assert os.path.exists(os.path.join(node_line_dir, "prompt_maker", "0.parquet"))
    assert os.path.exists(os.path.join(node_line_dir, "prompt_maker", "1.parquet"))


def test_run_prompt_maker_node_default(node_line_dir):
    modules = [fstring, fstring]
    params = [{'prompt': 'Tell me something about the question: {query} \n\n {retrieved_contents}'},
              {'prompt': 'Question: {query} \n Something to read: {retrieved_contents} \n What\'s your answer?'}]
    strategies = {
        'metrics': metrics
    }
    best_result = run_prompt_maker_node(modules, params, previous_result, node_line_dir, strategies)
    check_best_result(best_result)
    best_filename = check_summary_df(node_line_dir)
    best_result_path = os.path.join(node_line_dir, "prompt_maker", f"best_{best_filename}")
    assert os.path.exists(best_result_path)


def test_run_prompt_maker_one_module(node_line_dir):
    modules = [fstring]
    params = [{'prompt': 'Tell me something about the question: {query} \n\n {retrieved_contents}'}]
    strategies = {
        'metrics': metrics
    }
    best_result = run_prompt_maker_node(modules, params, previous_result, node_line_dir, strategies)
    assert set(best_result.columns) == {
        'query', 'retrieved_contents', 'test_column', 'prompts'  # automatically skip evaluation
    }
    summary_filepath = os.path.join(node_line_dir, "prompt_maker", "summary.csv")
    assert os.path.exists(summary_filepath)
    summary_df = load_summary_file(summary_filepath)
    assert set(summary_df) == {
        'filename', 'module_name', 'module_params', 'execution_time', 'is_best'
    }
    best_filepath = os.path.join(node_line_dir, "prompt_maker", f"best_{summary_df['filename'].values[0]}")
    assert os.path.exists(best_filepath)


def test_summary_metric_name_at_threshold_cutoff():
    evaluation_results = [
        pd.DataFrame({
            'test_column': ['test_value_1', 'test_value_2', 'test_value_3'],
            'bleu': [0.1, 0.2, 0.3],
            'rouge': [0.4, 0.5, 0.6],
        })
    ]
    summary_df = pd.DataFrame({
        'filename': ['filename_1', 'filename_2'],
        'execution_time': [1, 2],
    })
    filenames = ['filename_2']

    evaluation_df = pd.DataFrame({
        'filename': filenames,
        **{f'prompt_maker_{metric_name}': list(map(lambda x: x[metric_name].mean(), evaluation_results))
           for metric_name in metrics}
    })
    summary_df = pd.merge(on='filename', left=summary_df, right=evaluation_df, how='left')

    assert set(summary_df.columns) == {'filename', 'execution_time', 'prompt_maker_bleu', 'prompt_maker_rouge'}
    assert np.isnan(summary_df['prompt_maker_bleu'].values[0])
    assert np.isnan(summary_df['prompt_maker_rouge'].values[0])
    assert summary_df['prompt_maker_bleu'].values[1] == pytest.approx(0.2)
    assert summary_df['prompt_maker_rouge'].values[1] == pytest.approx(0.5)

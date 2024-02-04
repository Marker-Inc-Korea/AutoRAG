import os
import tempfile

import pandas as pd
import pytest

from autorag.nodes.generator import llama_index_llm
from autorag.nodes.generator.run import run_generator_node
from autorag.utils.util import load_summary_file

qa_df = pd.DataFrame({
    'qid': ['id-1', 'id-2', 'id-3'],
    'query': ['query-1', 'query-2', 'query-3'],
    'generation_gt': [
        ['The dog had bit the man.', 'The man had bitten the dog.'],
        ['I want to be a artist, but I end up to be a programmer.'],
        ['To be a artist these days, you can overcome by AI.',
         'To be a programmer these days, you can overcome by AI.',
         'To be a lawyer these days, you can overcome by AI.'],
    ],
})

previous_df = pd.DataFrame({
    'qid': ['id-1', 'id-2', 'id-3'],
    'query': ['query-1', 'query-2', 'query-3'],
    'prompts': [
        'What was the dog doing with the man?',
        'What is your dream job? And what is your current job?',
        'Is AI can overcome and replace all jobs in the future?',
    ],
})


@pytest.fixture
def node_line_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        os.makedirs(os.path.join(temp_dir, "data"))
        qa_df.to_parquet(os.path.join(temp_dir, "data", "qa.parquet"), index=False)
        trial_dir = os.path.join(temp_dir, "test_trial")
        os.makedirs(trial_dir)
        node_line_dir = os.path.join(trial_dir, "test_node_line")
        yield node_line_dir


def test_run_generator_node(node_line_dir):
    modules = [llama_index_llm, llama_index_llm]
    module_params = [{'llm': 'openai', 'temperature': 0.5, 'top_p': 0.9, 'max_tokens': 128, 'batch': 8},
                     {'llm': 'openai', 'temperature': 1.5, 'top_p': 0.9, 'max_tokens': 128, 'batch': 8}]
    strategies = {
        'metrics': ['bleu', 'meteor', 'rouge'],
        'speed_threshold': 5,
    }
    best_result = run_generator_node(modules, module_params, previous_df, node_line_dir, strategies)
    assert os.path.exists(os.path.join(node_line_dir, "generator"))
    expect_columns = {'qid', 'query', 'prompts', 'generated_texts', 'generated_tokens', 'generated_log_probs',
                      'bleu', 'meteor', 'rouge'}
    assert set(best_result.columns) == expect_columns

    summary_path = os.path.join(node_line_dir, "generator", "summary.csv")
    assert os.path.exists(summary_path)
    summary_df = load_summary_file(summary_path)
    expect_columns = {'filename', 'bleu', 'meteor', 'rouge', 'module_name', 'module_params', 'execution_time',
                      'is_best'}
    assert set(summary_df.columns) == expect_columns
    assert len(summary_df) == 2
    assert summary_df['module_params'][0] == {'llm': 'openai', 'temperature': 0.5, 'top_p': 0.9, 'max_tokens': 128,
                                              'batch': 8}

    first_path = os.path.join(node_line_dir, "generator", "0.parquet")
    assert os.path.exists(first_path)

    best_path = summary_df[summary_df['is_best']]['filename'].values[0]
    assert os.path.exists(os.path.join(node_line_dir, "generator", f'best_{best_path}'))

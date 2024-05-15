from unittest.mock import patch

import openai.resources.chat
import pandas as pd

from autorag.nodes.generator import openai_llm
from tests.autorag.nodes.generator.test_generator_base import prompts, check_generated_texts, check_generated_tokens, \
    check_generated_log_probs
from tests.mock import mock_openai_chat_create


@patch.object(openai.resources.chat.completions.AsyncCompletions, 'create', mock_openai_chat_create)
def test_openai_llm():
    openai_original = openai_llm.__wrapped__
    model = "gpt-3.5-turbo"
    answers, tokens, log_probs = openai_original(prompts, model, batch=1, temperature=0.5, logprobs=False, n=3)
    check_generated_texts(answers)
    check_generated_tokens(tokens)
    check_generated_log_probs(log_probs)


@patch.object(openai.resources.chat.completions.AsyncCompletions, 'create', mock_openai_chat_create)
def test_openai_llm_node():
    previous_result = pd.DataFrame(
        {
            'prompts': prompts,
            'qid': ['id-1', 'id-2', 'id-3']
        })
    result_df = openai_llm(project_dir='.', previous_result=previous_result, llm='gpt-4o')
    check_generated_texts(result_df['generated_texts'].tolist())
    check_generated_tokens(result_df['generated_tokens'].tolist())
    check_generated_log_probs(result_df['generated_log_probs'].tolist())


@patch.object(openai.resources.chat.completions.AsyncCompletions, 'create', mock_openai_chat_create)
def test_openai_llm_truncate():
    openai_original = openai_llm.__wrapped__
    prompt = [f'havertz on the block and I am {i}th player on the Arsenal.' for i in range(50_000)]
    prompt = ' '.join(prompt)
    answers, tokens, log_probs = openai_original([prompt] * 3)
    check_generated_texts(answers)
    check_generated_tokens(tokens)
    check_generated_log_probs(log_probs)

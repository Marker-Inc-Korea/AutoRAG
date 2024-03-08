import pandas as pd

from autorag import generator_models
from autorag.nodes.generator import llama_index_llm
from tests.autorag.nodes.generator.test_generator_base import (prompts, check_generated_texts, check_generated_tokens,
                                                               check_generated_log_probs)
from tests.mock import MockLLM


def test_llama_index_llm():
    llama_index_llm_original = llama_index_llm.__wrapped__
    answers, tokens, log_probs = llama_index_llm_original(prompts, MockLLM())
    check_generated_texts(answers)
    check_generated_tokens(tokens)
    check_generated_log_probs(log_probs)
    assert all(all(log_prob == 0.5 for log_prob in log_prob_list) for log_prob_list in log_probs)
    assert all(len(tokens[i]) == len(log_probs[i]) for i in range(len(tokens)))


def test_llama_index_llm_node():
    generator_models['mock'] = MockLLM
    previous_result = pd.DataFrame(
        {
            'prompts': prompts,
            'qid': ['id-1', 'id-2', 'id-3']
        })
    result_df = llama_index_llm(project_dir='.', previous_result=previous_result, llm='mock',
                                temperature=0.5, top_p=0.9)
    check_generated_texts(result_df['generated_texts'].tolist())
    check_generated_tokens(result_df['generated_tokens'].tolist())
    check_generated_log_probs(result_df['generated_log_probs'].tolist())

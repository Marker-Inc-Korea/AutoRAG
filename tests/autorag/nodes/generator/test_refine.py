import pandas as pd

from autorag import generator_models
from autorag.nodes.generator.refine import refine
from tests.autorag.nodes.generator.test_generator_base import (prompts, retrieved_contents, check_generated_texts,
                                                               check_generated_tokens, check_generated_log_probs)
from tests.mock import MockLLM


def test_refine_default():
    llm = MockLLM()
    answers, tokens, log_probs = refine.__wrapped__(prompts, retrieved_contents, llm)
    check_generated_texts(answers)
    check_generated_tokens(tokens)
    check_generated_log_probs(log_probs)
    assert all(all(log_prob == 0.5 for log_prob in log_prob_list) for log_prob_list in log_probs)
    assert all(len(tokens[i]) == len(log_probs[i]) for i in range(len(tokens)))


def test_refine_structured_answer_filtering():
    llm = MockLLM()
    answers, tokens, log_probs = refine.__wrapped__(prompts, retrieved_contents, llm, structured_answer_filtering=True)
    check_generated_texts(answers)
    check_generated_tokens(tokens)
    check_generated_log_probs(log_probs)
    assert all(all(log_prob == 0.5 for log_prob in log_prob_list) for log_prob_list in log_probs)
    assert all(len(tokens[i]) == len(log_probs[i]) for i in range(len(tokens)))


def test_refine_custom_prompt():
    llm = MockLLM()
    prompt = "This is a custom prompt. {context_msg} {query_str}"
    answers, tokens, log_probs = refine.__wrapped__(prompts, retrieved_contents, llm, prompt=prompt)
    check_generated_texts(answers)
    check_generated_tokens(tokens)
    check_generated_log_probs(log_probs)
    assert all(all(log_prob == 0.5 for log_prob in log_prob_list) for log_prob_list in log_probs)
    assert all(len(tokens[i]) == len(log_probs[i]) for i in range(len(tokens)))


def test_refine_node():
    generator_models['mock'] = MockLLM
    previous_result = pd.DataFrame(
        {
            'query': prompts,
            'retrieved_contents': retrieved_contents,
            'prompts': prompts,
            'qid': ['id-1', 'id-2', 'id-3']
        })
    result_df = refine(project_dir='.', previous_result=previous_result, llm='mock')
    check_generated_texts(result_df['generated_texts'].tolist())
    check_generated_tokens(result_df['generated_tokens'].tolist())
    check_generated_log_probs(result_df['generated_log_probs'].tolist())

import pandas as pd
from llama_index.llms import OpenAI

from autorag.nodes.generator import llama_index_llm

prompts = [
    "Who is the strongest Avenger?",
    "Who is the best soccer player in the world?",
    "Who is the president of the United States?",
]


def check_generated_texts(generated_texts):
    assert len(generated_texts) == len(prompts)
    assert isinstance(generated_texts[0], str)
    assert all(bool(text) is True for text in generated_texts)


def check_generated_tokens(tokens):
    assert len(tokens) == len(prompts)
    assert isinstance(tokens[0], list)
    assert isinstance(tokens[0][0], int)


def check_generated_log_probs(log_probs):
    assert len(log_probs) == len(prompts)
    assert isinstance(log_probs[0], list)
    assert isinstance(log_probs[0][0], float)
    assert all(all(log_prob == 0.5 for log_prob in log_prob_list) for log_prob_list in log_probs)


def test_llama_index_llm():
    llama_index_llm_original = llama_index_llm.__wrapped__
    answers, tokens, log_probs = llama_index_llm_original(prompts, OpenAI())
    check_generated_texts(answers)
    check_generated_tokens(tokens)
    check_generated_log_probs(log_probs)
    assert all(len(tokens[i]) == len(log_probs[i]) for i in range(len(tokens)))


def test_llama_index_llm_node():
    previous_result = pd.DataFrame(
        {
            'prompts': prompts,
            'qid': ['id-1', 'id-2', 'id-3']
        })
    result_df = llama_index_llm(project_dir='.', previous_result=previous_result, llm='openai',
                                temperature=0.5, top_p=0.9)
    check_generated_texts(result_df['generated_texts'].tolist())
    check_generated_tokens(result_df['generated_tokens'].tolist())
    check_generated_log_probs(result_df['generated_log_probs'].tolist())

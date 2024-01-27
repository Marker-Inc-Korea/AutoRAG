from llama_index.llms import OpenAI

from autorag.nodes.generator import llama_index_llm

prompts = [
    "Who is the strongest Avenger?",
    "Who is the best soccer player in the world?",
    "Who is the president of the United States?",
]


def test_llama_index_llm():
    answers, tokens, log_probs = llama_index_llm(prompts, OpenAI())
    assert len(answers) == len(prompts)
    assert len(tokens) == len(prompts)
    assert len(log_probs) == len(prompts)
    assert all(len(tokens[i]) == len(log_probs[i]) for i in range(len(tokens)))
    assert isinstance(answers[0], str)
    assert all(bool(answer) is True for answer in answers)
    assert isinstance(tokens[0], list)
    assert isinstance(log_probs[0], list)
    assert isinstance(tokens[0][0], int)
    assert isinstance(log_probs[0][0], float)
    assert all(all(log_prob == 0.5 for log_prob in log_prob_list) for log_prob_list in log_probs)

from autorag.nodes.promptmaker import long_context_reorder
from tests.autorag.nodes.promptmaker.test_prompt_maker_base import (prompt, queries, retrieved_contents,
                                                                    retrieve_scores, previous_result)


def test_long_context_reorder():
    long_context_reorder_original = long_context_reorder.__wrapped__
    result_prompts = long_context_reorder_original(prompt, queries, retrieved_contents, retrieve_scores)
    assert len(result_prompts) == 2
    assert isinstance(result_prompts, list)
    assert result_prompts[
               0] == 'Answer this question: What is the capital of Japan? \n\n Tokyo is the capital of Japan.\n\nTokyo, the capital of Japan, is a huge metropolitan city.\n\nTokyo is the capital of Japan.'
    assert result_prompts[
               1] == 'Answer this question: What is the capital of China? \n\n Beijing is the capital of China.\n\nBeijing, the capital of China, is a huge metropolitan city.\n\nBeijing is the capital of China.'


def test_long_context_reorder_node():
    result = long_context_reorder(project_dir="pseudo_project_dir",
                                  previous_result=previous_result,
                                  prompt=prompt)
    assert len(result) == 2
    assert result.columns == ["prompts"]
    assert result['prompts'][
               0] == "Answer this question: What is the capital of Japan? \n\n Tokyo is the capital of Japan.\n\nTokyo, the capital of Japan, is a huge metropolitan city.\n\nTokyo is the capital of Japan."
    assert result['prompts'][
               1] == "Answer this question: What is the capital of China? \n\n Beijing is the capital of China.\n\nBeijing, the capital of China, is a huge metropolitan city.\n\nBeijing is the capital of China."

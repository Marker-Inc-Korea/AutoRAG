import pandas as pd

from autorag.nodes.promptmaker import fstring

prompt = "Answer this question: {query} \n\n {retrieved_contents}"
queries = ["What is the capital of Japan?", "What is the capital of China?"]
retrieved_contents = [
    ["Tokyo is the capital of Japan.", "Tokyo, the capital of Japan, is a huge metropolitan city."],
    ["Beijing is the capital of China.", "Beijing, the capital of China, is a huge metropolitan city."]]


def test_fstring():
    fstring_original = fstring.__wrapped__
    result_prompts = fstring_original(prompt, queries, retrieved_contents)
    assert len(result_prompts) == 2
    assert isinstance(result_prompts, list)
    assert result_prompts[0] == "Answer this question: What is the capital of Japan? \n\n Tokyo is the capital of Japan.\n\nTokyo, the capital of Japan, is a huge metropolitan city."
    assert result_prompts[1] == "Answer this question: What is the capital of China? \n\n Beijing is the capital of China.\n\nBeijing, the capital of China, is a huge metropolitan city."


def test_fstring_node():
    previous_result = pd.DataFrame({
        "query": queries,
        "retrieved_contents": retrieved_contents
    })
    result = fstring(project_dir="pseudo_project_dir",
                     previous_result=previous_result,
                     prompt=prompt)
    assert len(result) == 2
    assert result.columns == ["prompts"]
    assert result['prompts'][0] == "Answer this question: What is the capital of Japan? \n\n Tokyo is the capital of Japan.\n\nTokyo, the capital of Japan, is a huge metropolitan city."
    assert result['prompts'][1] == "Answer this question: What is the capital of China? \n\n Beijing is the capital of China.\n\nBeijing, the capital of China, is a huge metropolitan city."

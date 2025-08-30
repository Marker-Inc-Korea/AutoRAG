prompts = [
    "Who is the strongest Avenger?",
    "Who is the best soccer player in the world?",
    "Who is the president of the United States?",
]

chat_prompts = [
    [
        {
            "role": "system",
            "content": "You are a helpful assistant that helps people find information.",
        },
        {"role": "user", "content": "Who is the strongest Avenger?"},
    ],
    [
        {
            "role": "system",
            "content": "You are a helpful assistant that helps people find information.",
        },
        {"role": "user", "content": "Who is the best soccer player in the world?"},
    ],
    [
        {
            "role": "system",
            "content": "You are a helpful assistant that helps people find information.",
        },
        {"role": "user", "content": "Who is the president of the United States?"},
    ],
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


def check_generated_texts_chat(generated_texts):
    assert len(generated_texts) == len(chat_prompts)
    assert isinstance(generated_texts[0], str)
    assert all(bool(text) is True for text in generated_texts)


def check_generated_tokens_chat(tokens):
    assert len(tokens) == len(chat_prompts)
    assert isinstance(tokens[0], list)
    assert isinstance(tokens[0][0], int)


def check_generated_log_probs_chat(log_probs):
    assert len(log_probs) == len(chat_prompts)
    assert isinstance(log_probs[0], list)
    assert isinstance(log_probs[0][0], float)

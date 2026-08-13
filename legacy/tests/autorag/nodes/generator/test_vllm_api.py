from autorag.nodes.generator.vllm_api import VllmAPI


class FakeVllmAPI(VllmAPI):
	def __init__(self):
		self.max_model_len = 10_000

	def encoding_for_model(self, text):
		return {"tokens": list(text)}

	def decoding_for_model(self, tokens):
		return {"prompt": "".join(tokens)}


def test_vllm_api_truncate_preserves_chat_messages():
	messages = [
		{"role": "system", "content": "Follow the conversation."},
		{"role": "user", "content": "What is the capital of France?"},
		{"role": "assistant", "content": "Paris."},
		{"role": "user", "content": "And Germany?"},
	]

	result = FakeVllmAPI().truncate_by_token(messages)

	assert result == messages
	assert result is not messages
	assert [message["role"] for message in result] == [
		"system",
		"user",
		"assistant",
		"user",
	]

from unittest.mock import Mock

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


def test_vllm_api_sends_chat_roles_unchanged(monkeypatch):
	messages = [
		{"role": "system", "content": "Follow the conversation."},
		{"role": "user", "content": "Question"},
		{"role": "assistant", "content": "Earlier answer"},
		{"role": "user", "content": "Follow-up"},
	]
	response = Mock()
	response.raise_for_status.return_value = None
	response.json.return_value = {"choices": []}
	post = Mock(return_value=response)
	monkeypatch.setattr("autorag.nodes.generator.vllm_api.requests.post", post)

	api = FakeVllmAPI()
	api.llm = "mock-model"
	api.uri = "http://localhost:8000"
	api.max_token_size = 100
	api.call_vllm_api(messages)

	assert post.call_args.kwargs["json"]["messages"] == messages

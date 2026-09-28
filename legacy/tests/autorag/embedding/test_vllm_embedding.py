import sys
import types
from types import SimpleNamespace

from autorag.embedding.vllm import VllmEmbedding


class FakeLLM:
	def __init__(self, model, **kwargs):
		if "task" in kwargs:
			raise TypeError("task is not a supported vLLM constructor argument")
		self.model = model
		self.kwargs = kwargs
		self.llm_engine = SimpleNamespace()

	def embed(self, inputs):
		return [SimpleNamespace(outputs=SimpleNamespace(embedding=[0.1, 0.2])) for _ in inputs]


def install_fake_vllm(monkeypatch):
	fake_vllm = types.ModuleType("vllm")
	setattr(fake_vllm, "LLM", FakeLLM)
	monkeypatch.setitem(sys.modules, "vllm", fake_vllm)


def test_vllm_embedding_uses_supported_constructor_and_preserves_kwargs(monkeypatch):
	install_fake_vllm(monkeypatch)

	embedding = VllmEmbedding(
		model_name="fake-model",
		embed_batch_size=4,
		vllm_kwargs={"gpu_memory_utilization": 0.5},
	)

	assert embedding._client.kwargs["gpu_memory_utilization"] == 0.5
	assert "task" not in embedding._client.kwargs
	assert embedding._get_text_embedding("hello") == [0.1, 0.2]

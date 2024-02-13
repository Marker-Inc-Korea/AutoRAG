import logging
import logging.config
import os
import sys

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbeddingModelType
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.gradient import GradientBaseModelLLM
from llama_index.llms.gradient import GradientModelAdapterLLM
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.langchain import LangChainLLM
from llama_index.llms.litellm import LiteLLM
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.openllm import OpenLLM
from llama_index.llms.palm import PaLM
from llama_index.llms.predibase import PredibaseLLM
from llama_index.llms.replicate import Replicate
from llama_index.llms.xinference import Xinference
from rich.logging import RichHandler

version_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'VERSION')

with open(version_path, 'r') as f:
    __version__ = f.read().strip()

embedding_models = {
    'openai': OpenAIEmbedding(),  # default model is OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002
    'openai_babbage': OpenAIEmbedding(model=OpenAIEmbeddingModelType.BABBAGE),
    'openai_ada': OpenAIEmbedding(model=OpenAIEmbeddingModelType.ADA),
    'openai_davinci': OpenAIEmbedding(model=OpenAIEmbeddingModelType.DAVINCI),
    'openai_curie': OpenAIEmbedding(model=OpenAIEmbeddingModelType.CURIE),
    # you can use your own model in this way.
    'huggingface_baai_bge_small': HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    'huggingface_cointegrated_rubert_tiny2': HuggingFaceEmbedding(model_name="cointegrated/rubert-tiny2"),
    'huggingface_all_mpnet_base_v2': HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
}

generator_models = {
    'openai': OpenAI,
    'anthropic': Anthropic,
    'azureopenai': AzureOpenAI,
    'huggingfacellm': HuggingFaceLLM,
    'langchainllm': LangChainLLM,
    'gradientbasemodelllm': GradientBaseModelLLM,
    'gradientmodeladapterllm': GradientModelAdapterLLM,
    'litellm': LiteLLM,
    'llamacpp': LlamaCPP,
    'openailike': OpenAILike,
    'openllm': OpenLLM,
    'palm': PaLM,
    'predibasellm': PredibaseLLM,
    'replicate': Replicate,
    'xinference': Xinference,
}

rich_format = "[%(filename)s:%(lineno)s] >> %(message)s"
logging.basicConfig(
    level="INFO",
    format=rich_format,
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("AutoRAG")


def handle_exception(exc_type, exc_value, exc_traceback):
    logger = logging.getLogger("AutoRAG")
    logger.error("Unexpected exception",
                 exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

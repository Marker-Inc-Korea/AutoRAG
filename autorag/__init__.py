import os
import logging
import logging.config
import sys

from rich.logging import RichHandler

from llama_index import OpenAIEmbedding
from llama_index.llms import OpenAI

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
version_path = os.path.join(root_path, 'VERSION')

with open(version_path, 'r') as f:
    __version__ = f.read().strip()

embedding_models = {
    'openai': OpenAIEmbedding(),
}

generator_models = {
    'openai': OpenAI,
}

rich_format = "[%(filename)s:%(lineno)s] >> %(message)s"
logging.basicConfig(
    level="NOTSET",
    format=rich_format,
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("AutoRAG")


def handle_exception(exc_type, exc_value, exc_traceback):
    logger = logging.getLogger("AutoRAG")
    logger.error("Unexpected exception",
                 exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

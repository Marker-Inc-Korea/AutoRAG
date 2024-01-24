__version__ = '0.0.1'

import logging
import logging.config
import sys

from rich.logging import RichHandler

from llama_index import OpenAIEmbedding
from llama_index.llms import OpenAI

embedding_models = {
    'openai': OpenAIEmbedding(),
}

generator_models = {
    'openai': OpenAI(),
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

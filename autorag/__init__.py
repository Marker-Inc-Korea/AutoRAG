__version__ = '0.0.1'

from llama_index import OpenAIEmbedding
from llama_index.llms import OpenAI

embedding_models = {
    'openai': OpenAIEmbedding(),
}

generator_models = {
    'openai': OpenAI(),
}

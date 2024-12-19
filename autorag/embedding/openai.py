from autorag.embedding.base import EmbeddingModel

class EmbeddingOpenAI(EmbeddingModel):
    
    def __init__(self):
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding
            from llama_index.embeddings.openai import OpenAIEmbeddingModelType
        except Exception as e:
            print(e)
        super().__init__()
        
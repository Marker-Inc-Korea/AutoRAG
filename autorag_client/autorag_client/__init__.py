import logging
import os

from .client import AutoRAGClient
from .models import Project, RAGPipeline, RetrievalResults, Retrieval
from .exceptions import APIError, AuthenticationError

__all__ = [
	"AutoRAGClient",
	"Project",
	"RAGPipeline",
	"RetrievalResults",
	"Retrieval",
	"APIError",
	"AuthenticationError",
]

# Set log level based on environment variable
log_level = logging.DEBUG if os.getenv("AUTORAG_ENV") == "dev" else logging.WARNING
logger = logging.getLogger("AutoRAG-Client")
logger.setLevel(log_level)

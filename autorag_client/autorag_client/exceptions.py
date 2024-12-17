class AutoRAGClientError(Exception):
	"""Base exception for AutoRAG client errors"""

	pass


class AuthenticationError(AutoRAGClientError):
	"""Raised when authentication fails"""

	pass


class APIError(AutoRAGClientError):
	"""Raised when the API returns an error"""

	pass

# Authentication decorator
from functools import wraps

import jwt
from quart import request, jsonify


def require_auth():
	def decorator(f):
		@wraps(f)
		async def decorated_function(*args, **kwargs):
			auth_header = request.headers.get('Authorization')

			if not auth_header:
				return jsonify({'error': 'No authorization header'}), 401

			try:
				token_type, token = auth_header.split()
				if token_type.lower() != 'bearer':
					return jsonify({'error': 'Invalid token type'}), 401

				# Verify token (implement your token verification logic here)
				# This is a simple example - adjust according to your needs
				# TODO: Make JWT auth server at remote location (AutoRAG private)

				# Check permissions (implement your permission logic here)
				if token != 'good':
					return jsonify({'error': 'Insufficient permissions'}), 403

			except jwt.InvalidTokenError:
				return jsonify({'error': 'Invalid token'}), 401
			except Exception as e:
				return jsonify({'error': str(e)}), 401

			return await f(*args, **kwargs)

		return decorated_function

	return decorator

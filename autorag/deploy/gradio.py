from autorag.deploy.base import BaseRunner


class GradioRunner(BaseRunner):
	def run_web(
		self,
		server_name: str = "0.0.0.0",
		server_port: int = 7680,
		share: bool = False,
		**kwargs,
	):
		"""
		Run web interface to interact pipeline.
		You can access the web interface at `http://server_name:server_port` in your browser

		:param server_name: The host of the web. Default is 0.0.0.0.
		:param server_port: The port of the web. Default is 7680.
		:param share: Whether to create a publicly shareable link. Default is False.
		:param kwargs: Other arguments for gr.ChatInterface.launch.
		"""

		logger.info(f"Run web interface at http://{server_name}:{server_port}")

		def get_response(message, _):
			return self.run(message)

		gr.ChatInterface(
			get_response, title="📚 AutoRAG", retry_btn=None, undo_btn=None
		).launch(
			server_name=server_name, server_port=server_port, share=share, **kwargs
		)

"""Expose Ollama's local EmbeddingGemma API using MinSync's TEI shape."""

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import Request, urlopen

OLLAMA_URL = os.environ.get("OLLAMA_EMBEDDINGS_URL", "http://127.0.0.1:11434/api/embeddings")
MODEL = "embeddinggemma:latest"


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path != "/embed":
            self.send_error(404)
            return

        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length))
        embeddings = []
        for text in payload.get("inputs", []):
            request = Request(
                OLLAMA_URL,
                data=json.dumps({"model": MODEL, "prompt": text}).encode(),
                headers={"content-type": "application/json"},
            )
            with urlopen(request, timeout=120) as response:
                embeddings.append(json.load(response)["embedding"])

        body = json.dumps(embeddings).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 18080), Handler).serve_forever()

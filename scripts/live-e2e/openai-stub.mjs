/**
 * A loopback OpenAI-compatible embeddings server for the live lanes. It runs in its OWN process: the live
 * stack drives `autorag` through blocking spawns, and a stub on the same event loop could never answer
 * the embedding requests those spawns make.
 *
 * Prints `{"port": N}` once listening, answers `POST /v1/embeddings` with deterministic unit vectors, and on
 * SIGTERM prints `{"requests": N}` and exits.
 */
import { createServer } from "node:http";

const dimension = Number(process.argv[2] ?? 1024);
let requests = 0;
const vector = (text) => {
	const values = Array.from({ length: dimension }, (_, index) => ((text.length * 31 + index * 17) % 97) / 97);
	const norm = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));
	return values.map((value) => value / norm);
};
const server = createServer(async (request, response) => {
	let body = "";
	for await (const chunk of request) body += chunk;
	response.setHeader("content-type", "application/json");
	if (request.method !== "POST" || request.url !== "/v1/embeddings") {
		response.statusCode = 404;
		response.end("{}");
		return;
	}
	requests += 1;
	const input = JSON.parse(body).input;
	response.end(JSON.stringify({ data: input.map((text, index) => ({ index, embedding: vector(text) })) }));
});
process.on("SIGTERM", () => {
	console.log(JSON.stringify({ requests }));
	process.exit(0);
});
server.listen(0, "127.0.0.1", () => console.log(JSON.stringify({ port: server.address().port })));

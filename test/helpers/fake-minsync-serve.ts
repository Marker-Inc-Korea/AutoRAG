import { chmodSync, writeFileSync } from "node:fs";
import { join } from "node:path";

export interface FakeServeOptions {
	readonly model?: string;
	readonly dimension?: number;
	/** Exit non-zero before serving, with this stderr text. */
	readonly failWith?: string;
}

/**
 * A `minsync` stand-in that implements only `serve`: it listens on loopback, publishes the same
 * discovery files as the real server (`<HOME>/.cache/minsync/serve/<instance>/{host,port,pid}`),
 * answers `GET /health`, TEI `POST /embed` and OpenAI `POST /v1/embeddings`, and removes its files on SIGTERM.
 * Vectors are deterministic: component i of a text is `(text.length + i) / 100`.
 */
export function writeFakeMinSyncServe(directory: string, options: FakeServeOptions = {}): string {
	const binaryPath = join(directory, "minsync");
	const script = `#!/usr/bin/env node
const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");
const model = ${JSON.stringify(options.model ?? "Qwen/Qwen3-Embedding-0.6B")};
const dimension = ${options.dimension ?? 4};
const failWith = ${JSON.stringify(options.failWith ?? "")};
const args = process.argv.slice(2);
if (args[0] !== "serve") { console.error("unexpected command: " + args.join(" ")); process.exit(2); }
if (failWith) { console.error(failWith); process.exit(1); }
const dir = path.join(process.env.HOME, ".cache", "minsync", "serve", "fake-instance");
fs.mkdirSync(dir, { recursive: true });
const health = () => ({ status: "ok", model_id: model, dimension, dtype: "f32", device: "cpu", attention: "eager", max_length: 2048, max_inputs: 512, instance: "fake-instance", pid: process.pid, host: "127.0.0.1", port: server.address().port, url: "http://127.0.0.1:" + server.address().port, version: "0.4.11" });
const vector = (text) => Array.from({ length: dimension }, (_, i) => (text.length + i) / 100);
const server = http.createServer((req, res) => {
  if (req.method === "GET" && req.url === "/health") { res.setHeader("content-type", "application/json"); res.end(JSON.stringify(health())); return; }
  if (req.method === "POST" && req.url === "/embed") {
    let body = "";
    req.on("data", (c) => { body += c; });
    req.on("end", () => {
      const parsed = JSON.parse(body);
      const inputs = Array.isArray(parsed.inputs) ? parsed.inputs : [parsed.inputs];
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify(inputs.map(vector)));
    });
    return;
  }
  if (req.method === "POST" && req.url === "/v1/embeddings") {
    let body = "";
    req.on("data", (c) => { body += c; });
    req.on("end", () => {
      const parsed = JSON.parse(body);
      const input = Array.isArray(parsed.input) ? parsed.input : [parsed.input];
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ object: "list", model, data: input.map((t, index) => ({ object: "embedding", index, embedding: vector(t) })), usage: { prompt_tokens: 0, total_tokens: 0 } }));
    });
    return;
  }
  res.statusCode = 404; res.end("{}");
});
const cleanup = () => { for (const f of ["host", "port", "pid"]) { try { fs.unlinkSync(path.join(dir, f)); } catch {} } };
process.on("SIGTERM", () => { cleanup(); server.close(); process.exit(0); });
server.listen(0, "127.0.0.1", () => {
  fs.writeFileSync(path.join(dir, "host"), "127.0.0.1");
  fs.writeFileSync(path.join(dir, "port"), String(server.address().port));
  fs.writeFileSync(path.join(dir, "pid"), String(process.pid));
  console.log(JSON.stringify({ event: "serving", ...health() }));
});
`;
	writeFileSync(binaryPath, script);
	chmodSync(binaryPath, 0o755);
	return binaryPath;
}

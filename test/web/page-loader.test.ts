import { createServer, type Server } from "node:http";
import iconv from "iconv-lite";
import { afterEach, describe, expect, it } from "vitest";
import { loadPage } from "../../src/web/fetch/page-loader.ts";

const servers: Server[] = [];

async function serve(handler: Parameters<typeof createServer>[0]): Promise<string> {
	const server = createServer(handler);
	servers.push(server);
	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
	const address = server.address();
	if (!address || typeof address === "string") throw new Error("server did not bind");
	return `http://127.0.0.1:${address.port}`;
}

afterEach(async () => {
	await Promise.all(servers.splice(0).map((server) => new Promise<void>((resolve) => server.close(() => resolve()))));
});

describe("loadPage", () => {
	it("rotates the user agent after a bot-block page", async () => {
		const userAgents: string[] = [];
		const base = await serve((request, response) => {
			userAgents.push(request.headers["user-agent"] ?? "");
			if (userAgents.length === 1) {
				response.writeHead(403, { "Content-Type": "text/html" });
				response.end("<html>Cloudflare bot challenge</html>");
				return;
			}
			response.writeHead(200, { "Content-Type": "text/plain" });
			response.end("allowed");
		});

		const result = await loadPage(base);
		expect(result.ok).toBe(true);
		expect(result.content).toBe("allowed");
		expect(userAgents).toHaveLength(2);
		expect(userAgents[1]).not.toBe(userAgents[0]);
	});

	it("retries one 429 response using Retry-After without changing user agent", async () => {
		const userAgents: string[] = [];
		const base = await serve((request, response) => {
			userAgents.push(request.headers["user-agent"] ?? "");
			if (userAgents.length === 1) {
				response.writeHead(429, { "Content-Type": "text/plain", "Retry-After": "0" });
				response.end("slow down");
				return;
			}
			response.writeHead(200, { "Content-Type": "text/plain" });
			response.end("retried");
		});

		const result = await loadPage(base);
		expect(result.content).toBe("retried");
		expect(userAgents).toHaveLength(2);
		expect(userAgents[1]).toBe(userAgents[0]);
	});

	it("caps streamed response bytes and reports truncation", async () => {
		const base = await serve((_request, response) => {
			response.writeHead(200, { "Content-Type": "text/plain" });
			response.end("abcdefghijklmnopqrstuvwxyz");
		});

		const result = await loadPage(base, { maxBytes: 10 });
		expect(result.ok).toBe(true);
		expect(result.truncated).toBe(true);
		expect(result.content).toContain("abcdefghij");
	});

	it("decodes a declared non-UTF-8 charset", async () => {
		const expected = "안녕하세요";
		const encoded = iconv.encode(expected, "euc-kr");
		const base = await serve((_request, response) => {
			response.writeHead(200, { "Content-Type": "text/plain; charset=euc-kr" });
			response.end(encoded);
		});

		const result = await loadPage(base);
		expect(result.content).toBe(expected);
	});

	it.each([
		["application/json", '{"ok":true}'],
		["text/plain", "plain text"],
	])("passes through %s bodies", async (contentType, body) => {
		const base = await serve((_request, response) => {
			response.writeHead(200, { "Content-Type": contentType });
			response.end(body);
		});
		const result = await loadPage(base);
		expect(result.content).toBe(body);
		expect(result.contentType).toBe(contentType);
	});
});

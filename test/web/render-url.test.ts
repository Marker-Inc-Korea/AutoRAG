import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { afterEach, describe, expect, it } from "vitest";
import { renderUrl } from "../../src/web/fetch/render.ts";

const servers: Server[] = [];
async function serve(handler: (request: IncomingMessage, response: ServerResponse) => void): Promise<string> {
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

const longText = "Useful article text. ".repeat(12);

describe("renderUrl", () => {
	it("renders HTML with native Turndown", async () => {
		const base = await serve((_request, response) => {
			response.writeHead(200, { "Content-Type": "text/html" });
			response.end(`<html><body><h1>Title</h1><p>${longText}</p></body></html>`);
		});
		const result = await renderUrl(base, { timeoutSeconds: 2 });
		expect(result.method).toBe("native");
		expect(result.content).toContain("# Title");
	});

	it("routes the primary page load through an injected transport", async () => {
		const base = await serve((_request, response) => {
			response.writeHead(200, { "Content-Type": "text/html" });
			response.end(`<html><body><p>server body that must not be read</p></body></html>`);
		});
		const requested: string[] = [];
		const result = await renderUrl(base, {
			timeoutSeconds: 2,
			fetch: (async (input: string | URL | Request) => {
				requested.push(String(input));
				return new Response(`<html><body><h1>Injected</h1><p>${longText}</p></body></html>`, {
					status: 200,
					headers: { "content-type": "text/html" },
				});
			}) as typeof fetch,
		});
		expect(requested[0]).toContain("127.0.0.1");
		expect(result.content).toContain("# Injected");
		expect(result.content).not.toContain("must not be read");
	});

	it("uses a markdown alternate link", async () => {
		const base = await serve((request, response) => {
			if (request.url === "/page.md") {
				response.writeHead(200, { "Content-Type": "text/markdown" });
				response.end(`# Alternate\n\n${longText}`);
				return;
			}
			response.writeHead(200, { "Content-Type": "text/html" });
			response.end(
				`<html><head><link rel="alternate" type="text/markdown" href="/page.md"></head><body>fallback</body></html>`,
			);
		});
		const result = await renderUrl(`${base}/page`, { timeoutSeconds: 2 });
		expect(result.method).toBe("alternate-markdown");
		expect(result.content).toContain("# Alternate");
	});

	it("uses the URL.md suffix", async () => {
		const base = await serve((request, response) => {
			if (request.url === "/doc.md") {
				response.writeHead(200, { "Content-Type": "text/markdown" });
				response.end(`# Suffix\n\n${longText}`);
				return;
			}
			response.writeHead(200, { "Content-Type": "text/html" });
			response.end("<html><body>fallback</body></html>");
		});
		const result = await renderUrl(`${base}/doc`, { timeoutSeconds: 2 });
		expect(result.method).toBe("md-suffix");
	});

	it("parses RSS feeds to markdown", async () => {
		const base = await serve((_request, response) => {
			response.writeHead(200, { "Content-Type": "application/rss+xml" });
			response.end(
				`<rss><channel><title>News</title><item><title>Story</title><link>https://example.test/story</link><description>${longText}</description></item></channel></rss>`,
			);
		});
		const result = await renderUrl(base, {});
		expect(result.method).toBe("feed");
		expect(result.content).toContain("# News");
	});

	it("pretty-prints JSON and returns raw mode verbatim", async () => {
		const base = await serve((_request, response) => {
			response.writeHead(200, { "Content-Type": "application/json" });
			response.end('{"a":1,"nested":{"b":true}}');
		});
		const pretty = await renderUrl(base, {});
		expect(pretty.method).toBe("json");
		expect(pretty.content).toContain('\n  "nested"');
		const raw = await renderUrl(base, { raw: true });
		expect(raw.method).toBe("raw");
		expect(raw.content).toBe('{"a":1,"nested":{"b":true}}');
	});

	it("returns a textual binary notice for unsupported binary payloads", async () => {
		const base = await serve((_request, response) => {
			response.writeHead(200, { "Content-Type": "application/zip" });
			response.end(Buffer.from([0x50, 0x4b, 0x03, 0x04, 0x00, 0x00, 0x00, 0x00]));
		});
		const result = await renderUrl(`${base}/archive.zip`, {});
		expect(result.method).toBe("binary");
		expect(result.content).toContain("Binary content");
	});

	it("reports non-200 status in notes", async () => {
		const base = await serve((_request, response) => {
			response.writeHead(404, { "Content-Type": "text/plain" });
			response.end("not found");
		});
		const result = await renderUrl(base, {});
		expect(result.method).toBe("failed");
		expect(result.notes.join(" ")).toContain("HTTP 404");
	});
});

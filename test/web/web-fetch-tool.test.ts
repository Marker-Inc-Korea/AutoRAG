import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createWebFetchTool, WEB_FETCH_TOOL_NAME } from "../../src/agent/web-fetch-tool.ts";

let server: Server;
let baseUrl: string;

beforeEach(async () => {
	server = createServer((req, res) => {
		if (req.url === "/page") {
			res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
			res.end(
				"<!doctype html><html><head><title>Refund Policy</title></head><body><main><h1>Refund Policy</h1><p>Refunds require director approval before payout. Finance acknowledged the policy in the July review, and the operations team reconfirmed it during the annual audit cycle.</p></main></body></html>",
			);
			return;
		}
		if (req.url === "/data.json") {
			res.writeHead(200, { "content-type": "application/json" });
			res.end(JSON.stringify({ refund: { approver: "director" } }));
			return;
		}
		res.writeHead(404, { "content-type": "text/plain" });
		res.end("not found");
	});
	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
	const { port } = server.address() as AddressInfo;
	baseUrl = `http://127.0.0.1:${port}`;
});

afterEach(async () => {
	await new Promise<void>((resolve) => server.close(() => resolve()));
});

describe("web_fetch tool", () => {
	it("exposes the tool contract", () => {
		const tool = createWebFetchTool();
		expect(tool.name).toBe(WEB_FETCH_TOOL_NAME);
		expect(tool.label).toBe("Web Fetch");
		const props = (tool.parameters as { properties: Record<string, unknown> }).properties;
		expect(Object.keys(props)).toEqual(expect.arrayContaining(["url", "raw"]));
	});

	it("renders an HTML page as text with response metadata", async () => {
		const tool = createWebFetchTool();
		const result = await tool.execute("f-1", { url: `${baseUrl}/page` });
		const details = result.details as {
			method: string;
			finalUrl: string;
			contentType: string;
			available: boolean;
		};
		expect(details.method).toBe(WEB_FETCH_TOOL_NAME);
		expect(details.available).toBe(true);
		expect(details.contentType).toContain("html");
		const text = (result.content[0] as { type: "text"; text: string } | undefined)?.text ?? "";
		expect(text).toContain(`URL: ${baseUrl}/page`);
		expect(text).toContain("Refund Policy");
		expect(text).toContain("director approval");
		expect(text).not.toContain("<h1>");
	});

	it("pretty-prints JSON responses", async () => {
		const tool = createWebFetchTool();
		const result = await tool.execute("f-2", { url: `${baseUrl}/data.json` });
		expect((result.content[0] as { text: string }).text).toContain('"approver": "director"');
	});

	it("reports HTTP failures as notes instead of throwing", async () => {
		const tool = createWebFetchTool();
		const result = await tool.execute("f-3", { url: `${baseUrl}/missing` });
		const details = result.details as { available: boolean };
		expect(details.available).toBe(false);
		expect((result.content[0] as { text: string }).text).toContain("404");
	});

	it("rejects non-http URLs without fetching", async () => {
		const tool = createWebFetchTool();
		for (const bad of ["/tmp/file.txt", "file:///etc/passwd", "/kakao/instance/chunks/1", "ftp://x"]) {
			const result = await tool.execute("f-4", { url: bad });
			const details = result.details as { available: boolean; error?: string };
			expect(details.available).toBe(false);
			expect(details.error).toBe("non-http URL");
		}
	});
});

import { randomBytes } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import type { AutoRAGLite } from "../../src/core.ts";
import { type McpHttpServer, startMcpHttpServer } from "../../src/mcp/http.ts";
import { type PreflightStep, runPreflight } from "../../src/mcp/preflight.ts";

const cleanups: (() => Promise<void> | void)[] = [];

afterEach(async () => {
	while (cleanups.length > 0) await cleanups.pop()?.();
});

const TOKEN = `owner-${randomBytes(24).toString("base64url")}`;

function fakeLite(): AutoRAGLite {
	const root = mkdtempSync(join(tmpdir(), "autorag-preflight-"));
	cleanups.push(() => rmSync(root, { recursive: true, force: true }));
	mkdirSync(join(root, ".autorag"), { recursive: true });
	writeFileSync(
		join(root, ".autorag", "refresh-complete.json"),
		JSON.stringify({ version: 1, completed: true, parsed: true }),
	);
	return {
		config: { searchPaths: [], excludePaths: [], workspacePath: root, memoryPath: join(root, "memory.json") },
		getRefreshStatus: async () => ({
			state: "success",
			inFlight: false,
			stale: false,
			diagnostics: [],
			components: {},
		}),
		getRetrievalEngine: () => ({ getMethodRegistry: () => ({ list: () => [] }) }),
		listDatasources: () => [],
		searchSelected: async () => ({ results: [], diagnostics: [], unsearched: [] }),
		refresh: async () => ({ ok: true }),
	} as unknown as AutoRAGLite;
}

async function freePort(): Promise<number> {
	const probe = createServer();
	await new Promise<void>((resolve) => probe.listen(0, "127.0.0.1", resolve));
	const address = probe.address();
	await new Promise<void>((resolve) => probe.close(() => resolve()));
	if (address === null || typeof address === "string") throw new Error("no port");
	return address.port;
}

/** A server whose public URL is its own loopback origin, so the preflight can reach every endpoint it advertises. */
async function oauthServer(overrides: Partial<Parameters<typeof startMcpHttpServer>[0]> = {}): Promise<{
	server: McpHttpServer;
	publicMcpUrl: string;
}> {
	const port = await freePort();
	const server = await startMcpHttpServer({
		lite: fakeLite(),
		host: "127.0.0.1",
		port,
		publicUrl: `http://127.0.0.1:${port}`,
		ownerToken: TOKEN,
		...overrides,
	});
	cleanups.push(() => server.close());
	return { server, publicMcpUrl: `http://127.0.0.1:${port}/mcp` };
}

function byName(steps: readonly PreflightStep[], name: string): PreflightStep {
	const step = steps.find((entry) => entry.name === name);
	if (step === undefined) throw new Error(`no step ${name}; got ${steps.map((entry) => entry.name).join(", ")}`);
	return step;
}

describe("runPreflight", () => {
	it("passes every step against a correctly configured server and reports the tools it saw", async () => {
		const { publicMcpUrl } = await oauthServer();
		const result = await runPreflight(publicMcpUrl, { token: TOKEN });
		expect(result.steps.filter((step) => step.status === "fail")).toEqual([]);
		expect(result.ok).toBe(true);
		expect(byName(result.steps, "challenge").status).toBe("pass");
		expect(byName(result.steps, "resource-metadata").status).toBe("pass");
		expect(byName(result.steps, "authorization-server-metadata").status).toBe("pass");
		expect(byName(result.steps, "dynamic-registration").status).toBe("pass");
		expect(byName(result.steps, "consent-page").status).toBe("pass");
		const tools = byName(result.steps, "tools");
		expect(tools.status).toBe("pass");
		expect(tools.detail).toContain("autorag.search");
	});

	it("flags write tools as a warning, not a failure, so an operator who opted in still gets exit 0", async () => {
		const { publicMcpUrl } = await oauthServer({ readOnly: false });
		const result = await runPreflight(publicMcpUrl, { token: TOKEN });
		expect(result.ok).toBe(true);
		const tools = byName(result.steps, "tools");
		expect(tools.status).toBe("warn");
		expect(tools.detail).toContain("autorag.refresh");
	});

	it("fails the token step for a wrong token and never prints the token", async () => {
		const { publicMcpUrl } = await oauthServer();
		const wrong = `wrong-${randomBytes(24).toString("base64url")}`;
		const result = await runPreflight(publicMcpUrl, { token: wrong });
		expect(result.ok).toBe(false);
		expect(byName(result.steps, "tools").status).toBe("fail");
		expect(JSON.stringify(result)).not.toContain(wrong);
		expect(JSON.stringify(result)).not.toContain(TOKEN);
	});

	it("skips the authenticated steps, instead of failing, when no token is given", async () => {
		const { publicMcpUrl } = await oauthServer();
		const result = await runPreflight(publicMcpUrl, {});
		expect(byName(result.steps, "tools").status).toBe("skip");
		expect(result.ok).toBe(true);
	});

	it("fails when the server has no OAuth surface (a bearer-only server cannot serve ChatGPT or Muse)", async () => {
		const port = await freePort();
		const server = await startMcpHttpServer({ lite: fakeLite(), host: "127.0.0.1", port, ownerToken: TOKEN });
		cleanups.push(() => server.close());
		const result = await runPreflight(server.url, { token: TOKEN });
		expect(result.ok).toBe(false);
		expect(byName(result.steps, "resource-metadata").status).toBe("fail");
		// The bearer path still works, and the preflight says so rather than hiding it.
		expect(byName(result.steps, "tools").status).toBe("pass");
	});

	it("fails when the metadata advertises a different resource than the URL it was fetched for", async () => {
		const { publicMcpUrl } = await oauthServer();
		const real = globalThis.fetch;
		const lying: typeof fetch = async (input, init) => {
			const response = await real(input, init);
			if (!String(input).includes("/.well-known/oauth-protected-resource")) return response;
			const body: unknown = await response.json();
			return new Response(JSON.stringify({ ...Object(body), resource: "https://elsewhere.example/mcp" }), {
				status: 200,
				headers: { "content-type": "application/json" },
			});
		};
		const result = await runPreflight(publicMcpUrl, { token: TOKEN, fetch: lying });
		expect(byName(result.steps, "resource-metadata").status).toBe("fail");
		expect(result.ok).toBe(false);
	});

	it("reports an unreachable server as one failed step, not an exception", async () => {
		const port = await freePort();
		const result = await runPreflight(`http://127.0.0.1:${port}/mcp`, { token: TOKEN });
		expect(result.ok).toBe(false);
		expect(byName(result.steps, "challenge").status).toBe("fail");
	});

	it("rejects a URL that is not an MCP endpoint address", async () => {
		await expect(runPreflight("not a url", {})).rejects.toThrow(/url/i);
		await expect(runPreflight("ftp://example.com/mcp", {})).rejects.toThrow(/http/i);
	});
});

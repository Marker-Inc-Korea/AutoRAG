import { createHash, randomBytes } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { request } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Client, StreamableHTTPClientTransport } from "@modelcontextprotocol/client";
import { afterEach, describe, expect, it } from "vitest";
import type { AutoRAGLite } from "../../src/core.ts";
import { type McpHttpServer, startMcpHttpServer } from "../../src/mcp/http.ts";

const cleanups: (() => Promise<void> | void)[] = [];

afterEach(async () => {
	while (cleanups.length > 0) await cleanups.pop()?.();
});

const OWNER_TOKEN = `owner-${randomBytes(24).toString("base64url")}`;

function workspace(): string {
	const root = mkdtempSync(join(tmpdir(), "autorag-mcp-http-"));
	cleanups.push(() => rmSync(root, { recursive: true, force: true }));
	mkdirSync(join(root, ".autorag"), { recursive: true });
	writeFileSync(
		join(root, ".autorag", "refresh-complete.json"),
		JSON.stringify({ version: 1, completed: true, parsed: true }),
	);
	return root;
}

function fakeLite(root: string): AutoRAGLite {
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
		searchSelected: async () => ({
			results: [
				{
					id: "minsync:1",
					source: "docs/refund-policy.md",
					score: 0.9,
					metadata: { method: "minsync" },
					content: "Refund exceptions require director approval before payout.",
				},
			],
			diagnostics: [],
			unsearched: [],
		}),
		refresh: async () => ({ ok: true }),
	} as unknown as AutoRAGLite;
}

async function start(overrides: Partial<Parameters<typeof startMcpHttpServer>[0]> = {}): Promise<McpHttpServer> {
	const server = await startMcpHttpServer({
		lite: fakeLite(workspace()),
		host: "127.0.0.1",
		port: 0,
		ownerToken: OWNER_TOKEN,
		...overrides,
	});
	cleanups.push(() => server.close());
	return server;
}

async function connect(server: McpHttpServer, token: string = OWNER_TOKEN): Promise<Client> {
	const client = new Client({ name: "autorag-http-test", version: "1.0.0" });
	const transport = new StreamableHTTPClientTransport(new URL(server.url), {
		requestInit: { headers: { authorization: `Bearer ${token}` } },
	});
	cleanups.push(() => client.close());
	await client.connect(transport);
	return client;
}

const INITIALIZE = {
	jsonrpc: "2.0",
	id: 1,
	method: "initialize",
	params: { protocolVersion: "2025-03-26", capabilities: {}, clientInfo: { name: "probe", version: "0" } },
};

/** A JSON response body as a plain record; anything else is a test failure, not a silent `undefined`. */
async function jsonBody(response: Response): Promise<Record<string, unknown>> {
	const body: unknown = await response.json();
	if (typeof body !== "object" || body === null || Array.isArray(body))
		throw new Error("response body is not an object");
	return Object.fromEntries(Object.entries(body));
}

function text(body: Record<string, unknown>, key: string): string {
	const value = body[key];
	if (typeof value !== "string") throw new Error(`${key} is not a string`);
	return value;
}

function seconds(body: Record<string, unknown>, key: string): number {
	const value = body[key];
	if (typeof value !== "number") throw new Error(`${key} is not a number`);
	return value;
}

/** POST with a hand-set Host header, which `fetch` forbids overriding. */
function postWithHost(server: McpHttpServer, headers: Record<string, string>): Promise<number> {
	const url = new URL(server.url);
	// tsconfig lib is ES2022, which predates Promise.withResolvers.
	return new Promise<number>((resolve, reject) => {
		const req = request(
			{
				host: url.hostname,
				port: url.port,
				path: url.pathname,
				method: "POST",
				headers: {
					authorization: `Bearer ${OWNER_TOKEN}`,
					"content-type": "application/json",
					accept: "application/json, text/event-stream",
					...headers,
				},
			},
			(res) => {
				res.resume();
				resolve(res.statusCode ?? 0);
			},
		);
		req.on("error", reject);
		req.end(JSON.stringify(INITIALIZE));
	});
}

function post(server: McpHttpServer, headers: Record<string, string> = {}, body: unknown = INITIALIZE) {
	return fetch(server.url, {
		method: "POST",
		headers: { "content-type": "application/json", accept: "application/json, text/event-stream", ...headers },
		body: JSON.stringify(body),
	});
}

describe("autorag-mcp over HTTP: transport and bearer auth", () => {
	it("refuses a request with no credentials and advertises how to authenticate", async () => {
		const server = await start();
		const response = await post(server);
		expect(response.status).toBe(401);
		expect(response.headers.get("www-authenticate")).toMatch(/^Bearer/);
	});

	it("refuses a wrong token, a wrong scheme, and a token that only shares a prefix with the owner token", async () => {
		const server = await start();
		for (const authorization of [
			"Bearer wrong",
			`Basic ${OWNER_TOKEN}`,
			OWNER_TOKEN,
			`Bearer ${OWNER_TOKEN.slice(0, -1)}`,
			`Bearer ${OWNER_TOKEN}x`,
		]) {
			const response = await post(server, { authorization });
			expect(response.status, authorization).toBe(401);
		}
	});

	it("serves the MCP tools to a real client holding the owner token", async () => {
		const server = await start();
		const client = await connect(server);
		const { tools } = await client.listTools();
		const names = tools.map((tool) => tool.name);
		expect(names).toContain("autorag.search");
		expect(names).toContain("autorag.status");
		const result = await client.callTool({ name: "autorag.search", arguments: { query: "refund approval" } });
		const text = JSON.stringify(result.content);
		expect(text).toContain("Refund exceptions require director approval before payout.");
	});

	it("is read-only by default: nothing that writes is offered to a remote agent", async () => {
		const server = await start();
		const client = await connect(server);
		const names = (await client.listTools()).tools.map((tool) => tool.name);
		for (const writer of ["autorag.refresh", "autorag.report", "autorag.convert"])
			expect(names).not.toContain(writer);
	});

	it("offers the write tools only when the operator opts in", async () => {
		const server = await start({ readOnly: false });
		const client = await connect(server);
		const names = (await client.listTools()).tools.map((tool) => tool.name);
		expect(names).toContain("autorag.refresh");
		expect(names).toContain("autorag.report");
	});

	it("honors the tool allowlist", async () => {
		const server = await start({ tools: ["autorag.search"] });
		const client = await connect(server);
		expect((await client.listTools()).tools.map((tool) => tool.name)).toEqual(["autorag.search"]);
	});

	it("serves only its configured path", async () => {
		const server = await start();
		const response = await fetch(new URL("/other", server.url), {
			method: "POST",
			headers: { authorization: `Bearer ${OWNER_TOKEN}`, "content-type": "application/json" },
			body: "{}",
		});
		expect(response.status).toBe(404);
	});

	it("answers /healthz without credentials and without leaking anything about the corpus", async () => {
		const server = await start();
		const response = await fetch(new URL("/healthz", server.url));
		expect(response.status).toBe(200);
		expect(await response.json()).toEqual({ ok: true });
	});

	it("rejects a Host header that is not loopback or the public URL (DNS rebinding)", async () => {
		const server = await start();
		expect(await postWithHost(server, { host: "evil.example" })).toBe(403);
	});

	it("rejects a browser Origin that is not the public URL, even with a valid token", async () => {
		const server = await start();
		const response = await post(server, { authorization: `Bearer ${OWNER_TOKEN}`, origin: "https://evil.example" });
		expect(response.status).toBe(403);
	});

	it("accepts the public URL's host and origin when one is configured", async () => {
		const server = await start({ publicUrl: "https://autorag.example.test" });
		expect(await postWithHost(server, { host: "autorag.example.test", origin: "https://autorag.example.test" })).toBe(
			200,
		);
	});

	it("rejects an oversized body before parsing it", async () => {
		const server = await start({ maxBodyBytes: 1024 });
		const response = await post(server, { authorization: `Bearer ${OWNER_TOKEN}` }, { pad: "x".repeat(4096) });
		expect(response.status).toBe(413);
	});

	it("refuses to start without an owner token", async () => {
		await expect(startMcpHttpServer({ lite: fakeLite(workspace()), ownerToken: "" })).rejects.toThrow(/owner token/i);
		await expect(startMcpHttpServer({ lite: fakeLite(workspace()), ownerToken: "short" })).rejects.toThrow(
			/at least 32/i,
		);
	});

	it("refuses a non-loopback bind without a public URL", async () => {
		await expect(
			startMcpHttpServer({ lite: fakeLite(workspace()), host: "0.0.0.0", port: 0, ownerToken: OWNER_TOKEN }),
		).rejects.toThrow(/public url/i);
	});
});

const PUBLIC = "https://autorag.example.test";

function pkce() {
	const verifier = randomBytes(32).toString("base64url");
	const challenge = createHash("sha256").update(verifier).digest("base64url");
	return { verifier, challenge };
}

async function oauthServer(): Promise<McpHttpServer> {
	return start({ publicUrl: PUBLIC });
}

/** Everything is addressed to the real listener; the public URL is only the identity the metadata advertises. */
function local(server: McpHttpServer, path: string): URL {
	return new URL(path, server.url);
}

async function register(server: McpHttpServer, redirectUris: string[] = ["https://chatgpt.example/callback"]) {
	const response = await fetch(local(server, "/register"), {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({
			client_name: "ChatGPT",
			redirect_uris: redirectUris,
			token_endpoint_auth_method: "none",
			grant_types: ["authorization_code", "refresh_token"],
			response_types: ["code"],
		}),
	});
	return { response, body: await jsonBody(response) };
}

async function authorize(
	server: McpHttpServer,
	params: Record<string, string>,
	token: string = OWNER_TOKEN,
): Promise<Response> {
	return fetch(local(server, "/authorize"), {
		method: "POST",
		redirect: "manual",
		headers: { "content-type": "application/x-www-form-urlencoded" },
		body: new URLSearchParams({ ...params, owner_token: token }).toString(),
	});
}

function tokenRequest(server: McpHttpServer, params: Record<string, string>): Promise<Response> {
	return fetch(local(server, "/token"), {
		method: "POST",
		headers: { "content-type": "application/x-www-form-urlencoded" },
		body: new URLSearchParams(params).toString(),
	});
}

async function fullFlow(server: McpHttpServer) {
	const { body } = await register(server);
	const clientId = String(body.client_id);
	const { verifier, challenge } = pkce();
	const authParams = {
		response_type: "code",
		client_id: clientId,
		redirect_uri: "https://chatgpt.example/callback",
		code_challenge: challenge,
		code_challenge_method: "S256",
		state: "st-1",
		resource: `${PUBLIC}/mcp`,
	};
	const approved = await authorize(server, authParams);
	const location = new URL(approved.headers.get("location") ?? "about:blank");
	return { clientId, verifier, authParams, approved, location, code: location.searchParams.get("code") ?? "" };
}

describe("autorag-mcp over HTTP: OAuth for connector-only clients", () => {
	it("publishes protected-resource and authorization-server metadata for the public URL", async () => {
		const server = await oauthServer();
		const resource = await jsonBody(await fetch(local(server, "/.well-known/oauth-protected-resource/mcp")));
		expect(resource).toMatchObject({ resource: `${PUBLIC}/mcp`, authorization_servers: [PUBLIC] });
		const meta = await jsonBody(await fetch(local(server, "/.well-known/oauth-authorization-server")));
		expect(meta).toMatchObject({
			issuer: PUBLIC,
			authorization_endpoint: `${PUBLIC}/authorize`,
			token_endpoint: `${PUBLIC}/token`,
			registration_endpoint: `${PUBLIC}/register`,
			code_challenge_methods_supported: ["S256"],
			token_endpoint_auth_methods_supported: ["none"],
		});
		expect(meta.client_id_metadata_document_supported).not.toBe(true);
	});

	it("points the 401 challenge at the resource metadata", async () => {
		const server = await oauthServer();
		const response = await post(server);
		expect(response.status).toBe(401);
		expect(response.headers.get("www-authenticate")).toContain(
			`resource_metadata="${PUBLIC}/.well-known/oauth-protected-resource/mcp"`,
		);
	});

	it("has no OAuth surface when no public URL is configured", async () => {
		const server = await start();
		for (const path of ["/.well-known/oauth-authorization-server", "/register", "/authorize", "/token"]) {
			const response = await fetch(local(server, path));
			expect(response.status, path).toBe(404);
		}
	});

	it("registers a public client and refuses a non-https, non-loopback redirect", async () => {
		const server = await oauthServer();
		const ok = await register(server);
		expect(ok.response.status).toBe(201);
		expect(ok.body.client_id).toEqual(expect.any(String));
		expect(ok.body.client_secret).toBeUndefined();
		const bad = await register(server, ["http://evil.example/cb"]);
		expect(bad.response.status).toBe(400);
		expect(bad.body.error).toBe("invalid_redirect_uri");
		const none = await register(server, []);
		expect(none.response.status).toBe(400);
	});

	it("shows the consent page only for a registered redirect and a valid S256 request", async () => {
		const server = await oauthServer();
		const { body } = await register(server);
		const { challenge } = pkce();
		const base = {
			response_type: "code",
			client_id: String(body.client_id),
			redirect_uri: "https://chatgpt.example/callback",
			code_challenge: challenge,
			code_challenge_method: "S256",
			state: "s",
		};
		const good = await fetch(`${local(server, "/authorize")}?${new URLSearchParams(base)}`);
		expect(good.status).toBe(200);
		expect(good.headers.get("content-type")).toContain("text/html");
		expect(good.headers.get("x-frame-options")).toBe("DENY");
		expect(await good.text()).toContain("owner_token");
		const patches: Record<string, string>[] = [
			{ redirect_uri: "https://evil.example/cb" },
			{ code_challenge_method: "plain" },
			{ code_challenge: "" },
			{ response_type: "token" },
			{ client_id: "forged" },
			{ resource: "https://other.example/mcp" },
		];
		for (const patch of patches) {
			const response = await fetch(`${local(server, "/authorize")}?${new URLSearchParams({ ...base, ...patch })}`, {
				redirect: "manual",
			});
			expect(response.status, JSON.stringify(patch)).toBe(400);
		}
	});

	it("never redirects to a redirect_uri that was not registered, even for a valid owner token", async () => {
		const server = await oauthServer();
		const { body } = await register(server);
		const response = await authorize(server, {
			response_type: "code",
			client_id: String(body.client_id),
			redirect_uri: "https://evil.example/cb",
			code_challenge: pkce().challenge,
			code_challenge_method: "S256",
			state: "s",
		});
		expect(response.status).toBe(400);
		expect(response.headers.get("location")).toBeNull();
	});

	it("issues no code for a wrong owner token and throttles guessing", async () => {
		const server = await oauthServer();
		const { body } = await register(server);
		const params = {
			response_type: "code",
			client_id: String(body.client_id),
			redirect_uri: "https://chatgpt.example/callback",
			code_challenge: pkce().challenge,
			code_challenge_method: "S256",
			state: "s",
		};
		const first = await authorize(server, params, "guess-0");
		expect(first.status).toBe(401);
		expect(first.headers.get("location")).toBeNull();
		let throttled = false;
		for (let attempt = 1; attempt < 12 && !throttled; attempt++) {
			throttled = (await authorize(server, params, `guess-${attempt}`)).status === 429;
		}
		expect(throttled).toBe(true);
		// Once throttled, even the right token is refused until the window passes.
		expect((await authorize(server, params)).status).toBe(429);
	});

	it("runs the full code + PKCE flow and the issued access token opens the MCP endpoint", async () => {
		const server = await oauthServer();
		const { clientId, verifier, authParams, approved, location, code } = await fullFlow(server);
		expect(approved.status).toBe(302);
		expect(`${location.origin}${location.pathname}`).toBe("https://chatgpt.example/callback");
		expect(location.searchParams.get("state")).toBe("st-1");
		expect(location.searchParams.get("iss")).toBe(PUBLIC);
		expect(code).not.toBe("");

		const issued = await tokenRequest(server, {
			grant_type: "authorization_code",
			code,
			client_id: clientId,
			redirect_uri: authParams.redirect_uri,
			code_verifier: verifier,
			resource: authParams.resource,
		});
		expect(issued.status).toBe(200);
		expect(issued.headers.get("cache-control")).toBe("no-store");
		const tokens = await jsonBody(issued);
		expect(tokens.token_type).toBe("Bearer");
		expect(text(tokens, "access_token")).toEqual(expect.any(String));
		expect(text(tokens, "refresh_token")).toEqual(expect.any(String));
		expect(text(tokens, "access_token")).not.toContain(OWNER_TOKEN);

		const client = await connect(server, text(tokens, "access_token"));
		expect((await client.listTools()).tools.length).toBeGreaterThan(0);
	});

	it("rejects a replayed code, a wrong verifier, a different client, and a different redirect", async () => {
		const server = await oauthServer();
		const { clientId, verifier, authParams, code } = await fullFlow(server);
		const base = {
			grant_type: "authorization_code",
			code,
			client_id: clientId,
			redirect_uri: authParams.redirect_uri,
			code_verifier: verifier,
		};
		for (const patch of [
			{ code_verifier: pkce().verifier },
			{ client_id: "another" },
			{ redirect_uri: "https://chatgpt.example/other" },
			{ code: "garbage" },
		]) {
			const response = await tokenRequest(server, { ...base, ...patch });
			expect(response.status, JSON.stringify(patch)).toBe(400);
			expect((await jsonBody(response)).error).toBe("invalid_grant");
		}
		expect((await tokenRequest(server, base)).status).toBe(200);
		const replay = await tokenRequest(server, base);
		expect(replay.status).toBe(400);
	});

	it("refreshes an access token and rejects a refresh token presented as an access token", async () => {
		const server = await oauthServer();
		const { clientId, verifier, authParams, code } = await fullFlow(server);
		const tokens = await jsonBody(
			await tokenRequest(server, {
				grant_type: "authorization_code",
				code,
				client_id: clientId,
				redirect_uri: authParams.redirect_uri,
				code_verifier: verifier,
			}),
		);

		const refreshed = await tokenRequest(server, {
			grant_type: "refresh_token",
			refresh_token: text(tokens, "refresh_token"),
			client_id: clientId,
		});
		expect(refreshed.status).toBe(200);
		const next = await jsonBody(refreshed);
		expect((await post(server, { authorization: `Bearer ${text(next, "access_token")}` })).status).toBe(200);

		// A refresh token must never open the MCP endpoint, nor an access token be refreshed.
		expect((await post(server, { authorization: `Bearer ${text(tokens, "refresh_token")}` })).status).toBe(401);
		const wrongKind = await tokenRequest(server, {
			grant_type: "refresh_token",
			refresh_token: text(tokens, "access_token"),
			client_id: clientId,
		});
		expect(wrongKind.status).toBe(400);
		const otherClient = await tokenRequest(server, {
			grant_type: "refresh_token",
			refresh_token: text(tokens, "refresh_token"),
			client_id: "another",
		});
		expect(otherClient.status).toBe(400);
	});

	it("rejects an expired access token and one signed under a different owner token", async () => {
		let now = 1_000_000;
		const first = await start({ publicUrl: PUBLIC, now: () => now });
		const { clientId, verifier, authParams, code } = await fullFlow(first);
		const tokens = await jsonBody(
			await tokenRequest(first, {
				grant_type: "authorization_code",
				code,
				client_id: clientId,
				redirect_uri: authParams.redirect_uri,
				code_verifier: verifier,
			}),
		);
		expect((await post(first, { authorization: `Bearer ${text(tokens, "access_token")}` })).status).toBe(200);

		const rotated = await start({
			publicUrl: PUBLIC,
			ownerToken: `rotated-${randomBytes(24).toString("base64url")}`,
		});
		expect((await post(rotated, { authorization: `Bearer ${text(tokens, "access_token")}` })).status).toBe(401);

		now += (seconds(tokens, "expires_in") + 1) * 1000;
		expect((await post(first, { authorization: `Bearer ${text(tokens, "access_token")}` })).status).toBe(401);
	});

	it("survives a restart: tokens issued before still verify against the same owner token", async () => {
		const first = await start({ publicUrl: PUBLIC });
		const { clientId, verifier, authParams, code } = await fullFlow(first);
		const tokens = await jsonBody(
			await tokenRequest(first, {
				grant_type: "authorization_code",
				code,
				client_id: clientId,
				redirect_uri: authParams.redirect_uri,
				code_verifier: verifier,
			}),
		);
		await first.close();
		const second = await start({ publicUrl: PUBLIC });
		expect((await post(second, { authorization: `Bearer ${text(tokens, "access_token")}` })).status).toBe(200);
	});

	it("binds tokens to this resource: an OAuth token issued for another resource is refused", async () => {
		const server = await oauthServer();
		const { body } = await register(server);
		const { challenge } = pkce();
		const response = await authorize(server, {
			response_type: "code",
			client_id: String(body.client_id),
			redirect_uri: "https://chatgpt.example/callback",
			code_challenge: challenge,
			code_challenge_method: "S256",
			state: "s",
			resource: "https://other.example/mcp",
		});
		expect(response.status).toBe(400);
	});
});

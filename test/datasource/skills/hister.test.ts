import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it } from "vitest";
import { filterDatasourceScope } from "../../../src/datasource/scope.ts";
import { buildDatasourceSkills } from "../../../src/datasource/skills/factory.ts";
import { HisterClient, HisterSkill } from "../../../src/datasource/skills/hister/index.ts";

interface Recorded {
	readonly method: string;
	readonly path: string;
	readonly headers: IncomingMessage["headers"];
	readonly query: Record<string, unknown> | undefined;
}

type Handler = (request: Recorded, response: ServerResponse) => void;

const servers: Server[] = [];

afterEach(async () => {
	await Promise.all(servers.splice(0).map((server) => new Promise<void>((done) => server.close(() => done()))));
});

/** A real loopback HTTP server standing in for `hister listen`: the HTTP boundary is genuine. */
async function fakeHister(handler: Handler): Promise<{ baseUrl: string; calls: Recorded[] }> {
	const calls: Recorded[] = [];
	const server = createServer((request, response) => {
		const url = new URL(request.url ?? "/", "http://127.0.0.1");
		const raw = url.searchParams.get("query");
		const recorded: Recorded = {
			method: request.method ?? "GET",
			path: url.pathname,
			headers: request.headers,
			query: raw === null ? undefined : (JSON.parse(raw) as Record<string, unknown>),
		};
		calls.push(recorded);
		handler(recorded, response);
	});
	servers.push(server);
	await new Promise<void>((done) => server.listen(0, "127.0.0.1", done));
	return { baseUrl: `http://127.0.0.1:${(server.address() as AddressInfo).port}`, calls };
}

function json(response: ServerResponse, body: unknown, status = 200): void {
	response.writeHead(status, { "Content-Type": "application/json" });
	response.end(JSON.stringify(body));
}

function doc(url: string, overrides: Record<string, unknown> = {}): Record<string, unknown> {
	const parsed = new URL(url);
	return {
		id: url,
		url,
		domain: parsed.hostname,
		title: `Title of ${parsed.pathname}`,
		text: "A <mark>refund</mark> policy &amp; &#34;exceptions&#34; apply.",
		score: 2.5,
		added: 1_792_552_000,
		updated: 1_792_552_122,
		type: 0,
		language: "en",
		metadata: {},
		...overrides,
	};
}

const WEB_PAGE = "https://example.com/policy/refund";

describe("HisterSkill lexical retrieval", () => {
	it("asks Hister for web documents only, as JSON, with highlighted snippets", async () => {
		const { baseUrl, calls } = await fakeHister((_request, response) => json(response, { total: 0, documents: [] }));
		const skill = new HisterSkill({ baseUrl });
		const [lexical] = skill.retrievalMethods();

		await lexical?.retrieve("refund policy", { topK: 7 });

		expect(calls).toHaveLength(1);
		const call = calls[0];
		expect(call?.method).toBe("GET");
		expect(call?.path).toBe("/search");
		// Hister answers 500 to a request without this header.
		expect(call?.headers.accept).toContain("application/json");
		expect(call?.query).toMatchObject({ highlight: "HTML", limit: 7, semantic_enabled: false });
		expect(call?.query?.text).toBe("(refund|policy) type:web");
	});

	it("turns a natural-language question into an alternation of its terms so long questions still match", async () => {
		const { baseUrl, calls } = await fakeHister((_request, response) => json(response, { total: 0, documents: [] }));
		const [lexical] = new HisterSkill({ baseUrl }).retrievalMethods();

		await lexical?.retrieve("When did I read about the inverted-index AND OR NOT 인증서를?", {});

		const text = String(calls[0]?.query?.text);
		expect(text).toBe("(When|did|I|read|about|the|inverted|index|인증서를) type:web");
	});

	it("never sends the word OR: Hister searches it as a literal word and matches most English pages", async () => {
		const { baseUrl, calls } = await fakeHister((_request, response) => json(response, { total: 0, documents: [] }));
		const [lexical] = new HisterSkill({ baseUrl }).retrievalMethods();

		await lexical?.retrieve("refund or policy OR approval", {});
		await lexical?.retrieve("맥 파인더를 대체하는 앱", {});

		for (const call of calls) expect(String(call.query?.text)).not.toMatch(/\bOR\b/i);
		// A user's own `or`/`OR` is dropped, not forwarded as a search term.
		expect(calls[0]?.query?.text).toBe("(refund|policy|approval) type:web");
		expect(calls[1]?.query?.text).toBe("(맥|파인더를|대체하는|앱) type:web");
	});

	it("passes Hister query syntax through, still confined to web documents", async () => {
		const { baseUrl, calls } = await fakeHister((_request, response) => json(response, { total: 0, documents: [] }));
		const [lexical] = new HisterSkill({ baseUrl }).retrievalMethods();

		await lexical?.retrieve('"inverted index" domain:wikipedia.org added:<7d', {});
		await lexical?.retrieve("sort:date title:refund", {});
		await lexical?.retrieve("sort:-date", {});

		expect(calls[0]?.query?.text).toBe('("inverted index" domain:wikipedia.org added:<7d) type:web');
		// Hister only honours a sort directive outside the parenthesised group.
		expect(calls[1]?.query?.text).toBe("(title:refund) type:web sort:date");
		expect(calls[2]?.query?.text).toBe("type:web sort:-date");
	});

	it("maps hits to opaque domain-scoped sources with readable content and traceable metadata", async () => {
		const { baseUrl } = await fakeHister((_request, response) =>
			json(response, { total: 1, documents: [doc(WEB_PAGE)] }),
		);
		const skill = new HisterSkill({ baseUrl, datasourceId: "browser", instanceId: "chrome" });
		const [lexical] = skill.retrievalMethods();

		const [hit] = (await lexical?.retrieve("refund", {})) ?? [];

		expect(hit?.source).toMatch(/^\/browser\/chrome\/pages\/example\.com\/[0-9a-f]{16}$/);
		expect(hit?.source).not.toContain("#");
		expect(hit?.content).toContain("Title of /policy/refund");
		expect(hit?.content).toContain(WEB_PAGE);
		expect(hit?.content).toContain('A refund policy & "exceptions" apply.');
		expect(hit?.content).not.toContain("<mark>");
		expect(hit?.score).toBe(2.5);
		expect(hit?.metadata).toMatchObject({
			datasourceId: "browser",
			instanceId: "chrome",
			url: WEB_PAGE,
			domain: "example.com",
			title: "Title of /policy/refund",
			visitedAt: new Date(1_792_552_122 * 1000).toISOString(),
		});
		// Same page, same identity across calls: evidence ids must not drift.
		const again = (await lexical?.retrieve("refund", {})) ?? [];
		expect(again[0]?.source).toBe(hit?.source);
	});

	it("never returns a local-file document, even if the server hands one back", async () => {
		const { baseUrl } = await fakeHister((_request, response) =>
			json(response, {
				total: 3,
				documents: [
					doc("file:///Users/me/secret.md", { type: 1, domain: "local" }),
					doc("file:///Users/me/snapshot.md", { type: 2, domain: "local" }),
					doc(WEB_PAGE),
				],
			}),
		);
		const [lexical] = new HisterSkill({ baseUrl }).retrievalMethods();

		const hits = (await lexical?.retrieve("secret", {})) ?? [];

		expect(hits.map((hit) => hit.metadata.url)).toEqual([WEB_PAGE]);
	});

	it("sends the access token from the named environment variable, never from argv or config", async () => {
		const { baseUrl, calls } = await fakeHister((_request, response) => json(response, { total: 0, documents: [] }));
		process.env.HISTER_TEST_TOKEN = "s3cret";
		try {
			const skill = new HisterSkill({ baseUrl, tokenEnv: "HISTER_TEST_TOKEN" });
			await skill.retrievalMethods()[0]?.retrieve("x", {});
		} finally {
			delete process.env.HISTER_TEST_TOKEN;
		}
		expect(calls[0]?.headers["x-access-token"]).toBe("s3cret");
	});

	it("answers an empty question without calling Hister", async () => {
		const { baseUrl, calls } = await fakeHister((_request, response) => json(response, { total: 0, documents: [] }));
		const [lexical] = new HisterSkill({ baseUrl }).retrievalMethods();
		expect(await lexical?.retrieve("   ", {})).toEqual([]);
		expect(await lexical?.retrieve("?!", {})).toEqual([]);
		expect(calls).toHaveLength(0);
	});
});

describe("HisterSkill scope", () => {
	it("does not call Hister for a scope outside this datasource", async () => {
		const { baseUrl, calls } = await fakeHister((_request, response) => json(response, { total: 0, documents: [] }));
		const [lexical] = new HisterSkill({ baseUrl }).retrievalMethods();
		expect(await lexical?.retrieve("refund", { scope: "/kakao/**" })).toEqual([]);
		expect(calls).toHaveLength(0);
	});

	it("pushes a literal domain scope down to Hister as domain: and keeps only matching hits", async () => {
		const { baseUrl, calls } = await fakeHister((_request, response) =>
			json(response, {
				total: 2,
				documents: [doc(WEB_PAGE), doc("https://other.org/refund")],
			}),
		);
		const skill = new HisterSkill({ baseUrl });
		const [lexical] = skill.retrievalMethods();

		const hits = (await lexical?.retrieve("refund", { scope: "/hister/default/pages/example.com/**" })) ?? [];

		expect(calls[0]?.query?.text).toBe("(refund) domain:example.com type:web");
		expect(hits.map((hit) => hit.metadata.domain)).toEqual(["example.com"]);
	});

	it("filters through the shared scope helper like every datasource", async () => {
		const { baseUrl } = await fakeHister((_request, response) =>
			json(response, { total: 1, documents: [doc(WEB_PAGE)] }),
		);
		const skill = new HisterSkill({ baseUrl });
		const methods = skill.retrievalMethods();
		const byMethod = new Map([[methods[0]?.describe().name ?? "", (await methods[0]?.retrieve("refund", {})) ?? []]]);
		const kept = filterDatasourceScope(byMethod, methods, "/hister/default/pages/**");
		expect([...kept.values()].flat()).toHaveLength(1);
		expect(
			[...filterDatasourceScope(byMethod, methods, "/hister/default/pages/other.org/**").values()].flat(),
		).toHaveLength(0);
	});
});

describe("HisterSkill failures are reported verbatim", () => {
	it("throws the server's own status and body", async () => {
		const { baseUrl } = await fakeHister((_request, response) => {
			response.writeHead(401, { "Content-Type": "text/plain" });
			response.end("invalid token\n");
		});
		const [lexical] = new HisterSkill({ baseUrl }).retrievalMethods();
		await expect(lexical?.retrieve("x", {})).rejects.toThrow(/401.*invalid token/s);
	});

	it("throws with the connection error when Hister is not running", async () => {
		const { baseUrl } = await fakeHister((_request, response) => json(response, {}));
		const dead = new HisterSkill({ baseUrl: baseUrl.replace(/:\d+$/, ":1") });
		await expect(dead.retrievalMethods()[0]?.retrieve("x", {})).rejects.toThrow(/hister/i);
	});

	it("throws when the response is not a Hister search result", async () => {
		const { baseUrl } = await fakeHister((_request, response) => {
			response.writeHead(200, { "Content-Type": "text/html" });
			response.end("<html>not json</html>");
		});
		const [lexical] = new HisterSkill({ baseUrl }).retrievalMethods();
		await expect(lexical?.retrieve("x", {})).rejects.toThrow(/not.*JSON|unexpected/i);
	});
});

describe("HisterSkill semantic retrieval", () => {
	it("registers no semantic method unless the operator enabled it", () => {
		const types = new HisterSkill({ baseUrl: "http://127.0.0.1:1" }).retrievalMethods().map((m) => m.describe().type);
		expect(types).toEqual(["remote"]);
		const both = new HisterSkill({ baseUrl: "http://127.0.0.1:1", semantic: true })
			.retrievalMethods()
			.map((m) => m.describe().type);
		expect(both).toEqual(["remote", "vector"]);
	});

	it("reads semantic_hits, keeps web pages only, and scores by similarity", async () => {
		const { baseUrl, calls } = await fakeHister((_request, response) =>
			json(response, {
				total: 0,
				semantic_enabled: true,
				documents: [],
				semantic_hits: [
					{ doc_id: WEB_PAGE, similarity: 0.82, matched_chunk: "refunds need approval", document: doc(WEB_PAGE) },
					{
						doc_id: "file:///Users/me/secret.md",
						similarity: 0.9,
						matched_chunk: "private",
						document: doc("file:///Users/me/secret.md", { type: 1, domain: "local" }),
					},
				],
			}),
		);
		const skill = new HisterSkill({ baseUrl, semantic: true });
		const semantic = skill.retrievalMethods()[1];

		const hits = (await semantic?.retrieve("how do refunds get approved", { topK: 5 })) ?? [];

		expect(calls[0]?.query).toMatchObject({ semantic_enabled: true, limit: 5 });
		expect(hits).toHaveLength(1);
		expect(hits[0]?.score).toBe(0.82);
		expect(hits[0]?.content).toContain("refunds need approval");
		expect(hits[0]?.metadata.url).toBe(WEB_PAGE);
	});

	it("resolves a semantic hit that overlaps a keyword hit from the keyword document, and drops unverifiable ones", async () => {
		const { baseUrl } = await fakeHister((_request, response) =>
			json(response, {
				total: 1,
				semantic_enabled: true,
				documents: [doc(WEB_PAGE)],
				semantic_hits: [
					{ doc_id: WEB_PAGE, similarity: 0.7, matched_chunk: "chunk" },
					{ doc_id: "https://ghost.example/unknown", similarity: 0.99, matched_chunk: "no type information" },
				],
			}),
		);
		const semantic = new HisterSkill({ baseUrl, semantic: true }).retrievalMethods()[1];
		const hits = (await semantic?.retrieve("refund", {})) ?? [];
		expect(hits.map((hit) => hit.metadata.url)).toEqual([WEB_PAGE]);
	});

	it("fails loudly when semantic is configured but the Hister server has it disabled", async () => {
		const { baseUrl } = await fakeHister((_request, response) =>
			json(response, { total: 0, semantic_enabled: false, documents: [] }),
		);
		const semantic = new HisterSkill({ baseUrl, semantic: true }).retrievalMethods()[1];
		await expect(semantic?.retrieve("refund", {})).rejects.toThrow(/semantic_search/);
	});
});

describe("HisterSkill index probe", () => {
	function probeServer(
		overrides: { nonWeb?: number; configStatus?: number } = {},
	): Promise<{ baseUrl: string; calls: Recorded[] }> {
		return fakeHister((request, response) => {
			if (request.path === "/api/config") {
				if (overrides.configStatus !== undefined) {
					response.writeHead(overrides.configStatus, { "Content-Type": "text/plain" });
					response.end("invalid token");
					return;
				}
				json(response, { authMode: "none", authenticated: true, semanticEnabled: false });
				return;
			}
			const text = String(request.query?.text ?? "");
			const total = text.includes("-type:web") ? (overrides.nonWeb ?? 0) : 42;
			json(response, { total, documents: [] });
		});
	}

	it("reports the web-page count and says nothing about local files when there are none", async () => {
		const { baseUrl } = await probeServer();
		const result = await new HisterSkill({ baseUrl }).index();
		expect(result).toMatchObject({ ok: true, chunkCount: 42 });
		expect(result.diagnostics.map((d) => d.message).join("\n")).not.toMatch(/local/i);
	});

	it("tells the operator, verbatim, that non-web documents exist and are excluded", async () => {
		const { baseUrl } = await probeServer({ nonWeb: 5 });
		const result = await new HisterSkill({ baseUrl }).index();
		expect(result.ok).toBe(true);
		expect(result.diagnostics).toContainEqual(
			expect.objectContaining({ severity: "warning", message: expect.stringMatching(/5 non-web.*excluded/s) }),
		);
	});

	it("fails with an auth diagnostic and the server's own words on 401", async () => {
		const { baseUrl } = await probeServer({ configStatus: 401 });
		const result = await new HisterSkill({ baseUrl }).index();
		expect(result).toMatchObject({ ok: false, code: "datasource-auth-error" });
		expect(result.ok === false && result.message).toMatch(/invalid token/);
	});

	it("fails as unavailable, naming how to start Hister, when nothing listens", async () => {
		const result = await new HisterSkill({ baseUrl: "http://127.0.0.1:1" }).index();
		expect(result).toMatchObject({ ok: false, code: "datasource-unavailable" });
		expect(result.ok === false && result.message).toMatch(/hister listen/);
	});
});

describe("HisterClient construction guards", () => {
	it("refuses to send an access token over plain http to a non-loopback host", () => {
		process.env.HISTER_TEST_TOKEN = "s3cret";
		try {
			expect(
				() => new HisterClient({ baseUrl: "http://hister.example.com:4433", tokenEnv: "HISTER_TEST_TOKEN" }),
			).toThrow(/https/);
			expect(
				() => new HisterClient({ baseUrl: "https://hister.example.com", tokenEnv: "HISTER_TEST_TOKEN" }),
			).not.toThrow();
			expect(
				() => new HisterClient({ baseUrl: "http://127.0.0.1:4433", tokenEnv: "HISTER_TEST_TOKEN" }),
			).not.toThrow();
		} finally {
			delete process.env.HISTER_TEST_TOKEN;
		}
	});

	it("rejects a base URL that is not http(s)", () => {
		expect(() => new HisterClient({ baseUrl: "file:///etc/passwd" })).toThrow(/http/);
	});
});

describe("hister through the datasource factory", () => {
	it("builds from config and exposes a path-opaque, license-honest manifest", () => {
		const { skills, unknown } = buildDatasourceSkills({
			browser: { type: "hister", connector: { baseUrl: "http://127.0.0.1:4433", semantic: true } },
		});
		expect(unknown).toEqual([]);
		const skill = skills[0];
		expect(skill?.describe()).toMatchObject({
			name: "browser",
			type: "hister-web-history",
			requiresExternalCli: true,
		});
		expect(skill?.retrievalMethods().map((m) => m.describe().name)).toEqual(["browser-lexical", "browser-semantic"]);
		const manifest = skill?.skillManifest();
		expect(manifest?.name).toBe("datasource-browser");
		const content = manifest?.content ?? "";
		// The tool the agent must call, and the licence notice that keeps Hister's use visible.
		expect(content).toContain("search_datasource_browser");
		expect(content).toContain("Hister");
		expect(content).toContain("AGPL-3.0");
		expect(content).toContain("https://github.com/asciimoo/hister");
	});

	it("defaults to the plain `hister` identity and a single lexical method", () => {
		const { skills } = buildDatasourceSkills({ hister: true });
		expect(skills[0]?.describe().name).toBe("hister");
		expect(skills[0]?.retrievalMethods().map((m) => m.describe().name)).toEqual(["hister-lexical"]);
	});
});

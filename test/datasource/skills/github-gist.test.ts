import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { GitHubGistConnector } from "../../../src/datasource/skills/github-gist/connector.ts";
import { GitHubGistSkill } from "../../../src/datasource/skills/github-gist/skill.ts";
import type { GistEmbedder } from "../../../src/datasource/skills/github-gist/semantic.ts";
import { createMockFetch } from "../../fixtures/mock-fetch.ts";

const GIST_LIST = [
	{
		id: "g1",
		description: "BM25 tokenizer benchmark",
		public: false,
		updated_at: "2024-03-01T00:00:00.000Z",
		html_url: "https://gist.github.com/g1",
		files: { "bench.md": { filename: "bench.md", truncated: false } },
	},
	{
		id: "g2",
		description: "VFS design notes",
		public: true,
		updated_at: "2024-03-02T00:00:00.000Z",
		html_url: "https://gist.github.com/g2",
		files: { "vfs.md": { filename: "vfs.md", truncated: false } },
	},
];

const GIST_FULL = {
	g1: {
		id: "g1",
		description: "BM25 tokenizer benchmark",
		public: false,
		updated_at: "2024-03-01T00:00:00.000Z",
		html_url: "https://gist.github.com/g1",
		files: { "bench.md": { filename: "bench.md", content: "Making benchmark of different tokenizer in BM25", truncated: false } },
	},
	g2: {
		id: "g2",
		description: "VFS design notes",
		public: true,
		updated_at: "2024-03-02T00:00:00.000Z",
		html_url: "https://gist.github.com/g2",
		files: { "vfs.md": { filename: "vfs.md", content: "A trap-free virtual file system design.", truncated: false } },
	},
};

function listMock(list: unknown) {
	return createMockFetch([
		{ match: "/gists/g1", json: GIST_FULL.g1 },
		{ match: "/gists/g2", json: GIST_FULL.g2 },
		{ match: "/gists?", json: list },
	]);
}

const NO_TOKEN = { tokenEnv: "GITHUB_GIST_TEST_UNSET", ghCliFallback: false } as const;

const tempDirs: string[] = [];
function tempWorkspace(): string {
	const dir = mkdtempSync(join(tmpdir(), "autorag-gist-test-"));
	tempDirs.push(dir);
	return dir;
}

afterEach(() => {
	for (const dir of tempDirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

describe("GitHubGistConnector", () => {
	it("fails as not-configured when no token resolves", async () => {
		const result = await new GitHubGistConnector({ ...NO_TOKEN }).fetch();
		expect(result).toMatchObject({ ok: false, reason: "not-configured" });
	});

	it("falls back to the gh CLI token runner when no env token exists", async () => {
		const seenAuth: string[] = [];
		const fetchImpl = (async (input: string | URL | Request, init?: RequestInit) => {
			const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
			const headers = (init?.headers ?? {}) as Record<string, string>;
			seenAuth.push(headers.Authorization ?? "");
			if (url.includes("/gists?")) return new Response(JSON.stringify([GIST_LIST[0]]), { status: 200 });
			return new Response(JSON.stringify(GIST_FULL.g1), { status: 200 });
		}) as typeof fetch;
		const result = await new GitHubGistConnector({
			tokenEnv: "GITHUB_GIST_TEST_UNSET",
			ghTokenRunner: async () => "gh-fallback-token",
			fetchImpl,
		}).fetch();
		expect(result.ok).toBe(true);
		expect(seenAuth[0]).toBe("Bearer gh-fallback-token");
	});

	it("prefers an explicit token over the env and gh fallback", async () => {
		const seenAuth: string[] = [];
		const fetchImpl = (async (input: string | URL | Request, init?: RequestInit) => {
			const headers = (init?.headers ?? {}) as Record<string, string>;
			seenAuth.push(headers.Authorization ?? "");
			const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
			return new Response(JSON.stringify(url.includes("/gists?") ? [] : {}), { status: 200 });
		}) as typeof fetch;
		await new GitHubGistConnector({
			token: "explicit-token",
			ghTokenRunner: async () => "gh-fallback-token",
			fetchImpl,
		}).fetch();
		expect(seenAuth[0]).toBe("Bearer explicit-token");
	});

	it("fetches every gist fully on the first sync and maps documents", async () => {
		const mock = listMock(GIST_LIST);
		const result = await new GitHubGistConnector({ ...NO_TOKEN, token: "t", fetchImpl: mock.fetchImpl }).fetch();
		expect(result.ok).toBe(true);
		if (!result.ok) return;
		expect(result.documents).toHaveLength(2);
		const gist = result.documents.find((d) => d.docId === "g1");
		expect(gist).toMatchObject({
			hierarchy: ["gists"],
			title: "BM25 tokenizer benchmark",
			metadata: { gistId: "g1", public: false, htmlUrl: "https://gist.github.com/g1" },
		});
		expect(gist?.content).toContain("bench.md");
		expect(gist?.content).toContain("Making benchmark of different tokenizer in BM25");
		expect(result.changed).not.toBe(false);
		expect(mock.requests.some((r) => r.includes("/gists/g1"))).toBe(true);
		expect(mock.requests.some((r) => r.includes("/gists/g2"))).toBe(true);
	});

	it("reports changed:false and skips content fetches when nothing changed", async () => {
		const workspace = tempWorkspace();
		const statePath = join(workspace, "state.json");
		const first = listMock(GIST_LIST);
		await new GitHubGistConnector({ ...NO_TOKEN, token: "t", fetchImpl: first.fetchImpl, statePath }).fetch();

		const second = listMock(GIST_LIST);
		const result = await new GitHubGistConnector({
			...NO_TOKEN,
			token: "t",
			fetchImpl: second.fetchImpl,
			statePath,
		}).fetch();
		expect(result).toMatchObject({ ok: true, changed: false, documents: [] });
		expect(second.requests.some((r) => /\/gists\/g[12]/.test(r))).toBe(false);
	});

	it("re-fetches only gists whose updated_at changed", async () => {
		const workspace = tempWorkspace();
		const statePath = join(workspace, "state.json");
		const first = listMock(GIST_LIST);
		await new GitHubGistConnector({ ...NO_TOKEN, token: "t", fetchImpl: first.fetchImpl, statePath }).fetch();

		const updatedList = [
			{ ...GIST_LIST[0], updated_at: "2024-03-05T00:00:00.000Z" },
			GIST_LIST[1],
		];
		const updatedFull = {
			...GIST_FULL,
			g1: { ...(GIST_FULL.g1 as object), updated_at: "2024-03-05T00:00:00.000Z" },
		};
		const second = createMockFetch([
			{ match: "/gists/g1", json: updatedFull.g1 },
			{ match: "/gists/g2", json: updatedFull.g2 },
			{ match: "/gists?", json: updatedList },
		]);
		const result = await new GitHubGistConnector({
			...NO_TOKEN,
			token: "t",
			fetchImpl: second.fetchImpl,
			statePath,
		}).fetch();
		expect(result.ok).toBe(true);
		if (!result.ok) return;
		expect(result.documents.map((d) => d.docId)).toEqual(["g1"]);
		expect(result.deletedDocIds).toEqual([]);
		expect(second.requests.some((r) => r.includes("/gists/g1"))).toBe(true);
		expect(second.requests.some((r) => r.includes("/gists/g2"))).toBe(false);
	});

	it("reports deleted gists via deletedDocIds", async () => {
		const workspace = tempWorkspace();
		const statePath = join(workspace, "state.json");
		const first = listMock(GIST_LIST);
		await new GitHubGistConnector({ ...NO_TOKEN, token: "t", fetchImpl: first.fetchImpl, statePath }).fetch();

		const second = listMock([GIST_LIST[1]]);
		const result = await new GitHubGistConnector({
			...NO_TOKEN,
			token: "t",
			fetchImpl: second.fetchImpl,
			statePath,
		}).fetch();
		expect(result.ok).toBe(true);
		if (!result.ok) return;
		expect(result.documents).toEqual([]);
		expect(result.deletedDocIds).toEqual(["g1"]);
	});

	it("maps 401 to auth and 403 to rate-limited", async () => {
		const authMock = createMockFetch([{ match: "/gists", status: 401 }]);
		expect(
			await new GitHubGistConnector({ ...NO_TOKEN, token: "t", fetchImpl: authMock.fetchImpl }).fetch(),
		).toMatchObject({ ok: false, reason: "auth" });
		const rateMock = createMockFetch([{ match: "/gists", status: 403 }]);
		expect(
			await new GitHubGistConnector({ ...NO_TOKEN, token: "t", fetchImpl: rateMock.fetchImpl }).fetch(),
		).toMatchObject({ ok: false, reason: "rate-limited" });
	});
});

describe("GitHubGistSkill", () => {
	function skillWithMock(workspace: string, list: unknown, semantic?: { embedder: GistEmbedder } | { enabled: false }) {
		const mock = listMock(list);
		return new GitHubGistSkill({
			workspaceRoot: workspace,
			...(semantic !== undefined ? { semantic } : { semantic: { enabled: false } }),
			connector: new GitHubGistConnector({
				...NO_TOKEN,
				token: "t",
				fetchImpl: mock.fetchImpl,
				statePath: join(workspace, "state.json"),
			}),
		});
	}

	it("indexes gists and retrieves them lexically with opaque sources", async () => {
		const skill = skillWithMock(tempWorkspace(), GIST_LIST);
		const result = await skill.index();
		expect(result.ok).toBe(true);
		if (!result.ok) return;
		expect(result.chunkCount).toBeGreaterThan(0);
		const lexical = skill.retrievalMethods().find((m) => m.describe().name === "github-gist-lexical");
		expect(lexical).toBeDefined();
		const hits = await lexical!.retrieve("tokenizer BM25 benchmark", { topK: 5 });
		expect(hits.length).toBeGreaterThan(0);
		expect(hits[0]?.source).toMatch(/^\/github-gist\/default\/chunks\/g1/);
		expect(hits[0]?.content).toContain("tokenizer in BM25");
	});

	it("keeps chunks on a no-op reindex and drops deleted gists", async () => {
		const workspace = tempWorkspace();
		const mock = listMock(GIST_LIST);
		const connector = new GitHubGistConnector({
			...NO_TOKEN,
			token: "t",
			fetchImpl: mock.fetchImpl,
			statePath: join(workspace, "state.json"),
		});
		const skill = new GitHubGistSkill({ workspaceRoot: workspace, semantic: { enabled: false }, connector });
		const first = await skill.index();
		const second = await skill.index();
		expect(second.ok).toBe(true);
		if (!second.ok || !first.ok) return;
		expect(second.chunkCount).toBe(first.chunkCount);

		const reduced = listMock([GIST_LIST[1]]);
		const skill2 = new GitHubGistSkill({
			workspaceRoot: workspace,
			semantic: { enabled: false },
			connector: new GitHubGistConnector({
				...NO_TOKEN,
				token: "t",
				fetchImpl: reduced.fetchImpl,
				statePath: join(workspace, "state.json"),
			}),
		});
		await skill2.index();
		const lexical = skill2.retrievalMethods().find((m) => m.describe().name === "github-gist-lexical");
		const hits = await lexical!.retrieve("tokenizer BM25 benchmark", { topK: 5 });
		expect(hits).toEqual([]);
	});

	it("narrows retrieval by scope", async () => {
		const skill = skillWithMock(tempWorkspace(), GIST_LIST);
		await skill.index();
		const lexical = skill.retrievalMethods().find((m) => m.describe().name === "github-gist-lexical");
		const outside = await lexical!.retrieve("tokenizer BM25 benchmark", { topK: 5, scope: "/other-datasource" });
		expect(outside).toEqual([]);
	});
});

/** Deterministic embedder stub: vector = [count of 'alpha', count of 'beta', 1]. */
function stubEmbedder(dimension = 3): GistEmbedder & { calls: string[][] } {
	const calls: string[][] = [];
	return {
		calls,
		identity: async () => ({ provider: "stub", model: "stub-model", dimension }),
		async embed(texts: readonly string[]): Promise<readonly (readonly number[])[]> {
			calls.push([...texts]);
			return texts.map((text) => [text.includes("alpha") ? 1 : 0, text.includes("beta") ? 1 : 0, 1]);
		},
	};
}

describe("GitHubGistSkill semantic retrieval", () => {
	it("adds a semantic method that ranks by cosine similarity", async () => {
		const workspace = tempWorkspace();
		const embedder = stubEmbedder();
		const alphaList = [
			{ ...GIST_LIST[0], id: "g1", description: "alpha notes" },
			{ ...GIST_LIST[1], id: "g2", description: "beta notes" },
		];
		const mock = createMockFetch([
			{
				match: "/gists/g1",
				json: { ...(GIST_FULL.g1 as object), description: "alpha notes", files: { "a.md": { filename: "a.md", content: "alpha alpha topic", truncated: false } } },
			},
			{
				match: "/gists/g2",
				json: { ...(GIST_FULL.g2 as object), description: "beta notes", files: { "b.md": { filename: "b.md", content: "beta beta topic", truncated: false } } },
			},
			{ match: "/gists?", json: alphaList },
		]);
		const skill = new GitHubGistSkill({
			workspaceRoot: workspace,
			semantic: { embedder },
			connector: new GitHubGistConnector({
				...NO_TOKEN,
				token: "t",
				fetchImpl: mock.fetchImpl,
				statePath: join(workspace, "state.json"),
			}),
		});
		const result = await skill.index();
		expect(result.ok).toBe(true);

		const semantic = skill.retrievalMethods().find((m) => m.describe().name === "github-gist-semantic");
		expect(semantic).toBeDefined();
		const hits = await semantic!.retrieve("alpha query", { topK: 1 });
		expect(hits).toHaveLength(1);
		expect(hits[0]?.source).toContain("/github-gist/default/chunks/g1");
	});

	it("re-embeds everything when the embedding identity changes", async () => {
		const workspace = tempWorkspace();
		const embedderA = stubEmbedder(3);
		const make = (embedder: GistEmbedder) =>
			new GitHubGistSkill({
				workspaceRoot: workspace,
				semantic: { embedder },
				connector: new GitHubGistConnector({
					...NO_TOKEN,
					token: "t",
					fetchImpl: listMock(GIST_LIST).fetchImpl,
					statePath: join(workspace, "state.json"),
				}),
			});
		await make(embedderA).index();
		const firstCalls = embedderA.calls.flat().length;
		expect(firstCalls).toBeGreaterThan(0);

		// Same identity, no content change → no re-embed.
		const embedderA2 = stubEmbedder(3);
		await make(embedderA2).index();
		expect(embedderA2.calls.flat()).toEqual([]);

		// Changed identity → full re-embed.
		const embedderB = stubEmbedder(4);
		await make(embedderB).index();
		expect(embedderB.calls.flat().length).toBe(firstCalls);
	});

	it("degrades with a semantic-unavailable diagnostic when embedding fails", async () => {
		const workspace = tempWorkspace();
		const failing: GistEmbedder = {
			identity: async () => ({ provider: "stub", model: "stub-model", dimension: 3 }),
			async embed() {
				throw new Error("gateway down");
			},
		};
		const skill = new GitHubGistSkill({
			workspaceRoot: workspace,
			semantic: { embedder: failing },
			connector: new GitHubGistConnector({
				...NO_TOKEN,
				token: "t",
				fetchImpl: listMock(GIST_LIST).fetchImpl,
				statePath: join(workspace, "state.json"),
			}),
		});
		const result = await skill.index();
		expect(result.ok).toBe(true);
		expect(result.diagnostics.some((d) => d.message.includes("semantic-unavailable"))).toBe(true);

		const semantic = skill.retrievalMethods().find((m) => m.describe().name === "github-gist-semantic");
		const hits = await semantic!.retrieve("anything", { topK: 5 });
		expect(hits).toEqual([]);
	});
});

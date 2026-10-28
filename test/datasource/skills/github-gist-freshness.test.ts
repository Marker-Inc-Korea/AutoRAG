import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { DatasourceChunkStore } from "../../../src/datasource/chunk-store.ts";
import type { ConnectorDocument } from "../../../src/datasource/connector.ts";
import { GitHubGistConnector } from "../../../src/datasource/skills/github-gist/connector.ts";
import { GistSemanticIndex, GitHubGistSemanticMethod } from "../../../src/datasource/skills/github-gist/semantic.ts";
import { GitHubGistSkill } from "../../../src/datasource/skills/github-gist/skill.ts";
import type { Embedder, EmbedOptions } from "../../../src/embedding/embedder.ts";
import { createMockFetch } from "../../fixtures/mock-fetch.ts";

const tempDirs: string[] = [];
function workspace(): string {
	const dir = mkdtempSync(join(tmpdir(), "autorag-gist-freshness-"));
	tempDirs.push(dir);
	return dir;
}

afterEach(() => {
	for (const dir of tempDirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

function document(docId: string, content: string, title = "notes"): ConnectorDocument {
	return { docId, content, title };
}

// Deterministic program checks: these vectors do not evaluate model quality.
function controlledEmbedder() {
	return {
		failDocuments: false,
		documentCalls: [] as string[][],
		identity: async () => ({ provider: "stub", model: "freshness", dimension: 2 }),
		async embed(texts: readonly string[], options?: EmbedOptions): Promise<readonly (readonly number[])[]> {
			if (options?.role === "query") return texts.map(() => [1, 0]);
			this.documentCalls.push([...texts]);
			if (this.failDocuments) throw new Error("embedding gateway unavailable");
			return texts.map((text) => (text.includes("steady") ? [1, 1] : [1, 0]));
		},
	};
}

function store(workspaceRoot?: string) {
	return new DatasourceChunkStore({ skillName: "github-gist", instanceId: "default", workspaceRoot });
}

function method(chunks: DatasourceChunkStore, index: GistSemanticIndex, embedder: Embedder) {
	return new GitHubGistSemanticMethod({
		skillName: "github-gist",
		skillType: "github-gist",
		instanceId: "default",
		tags: [],
		store: chunks,
		index,
		embedder,
	});
}

function signal() {
	let resolve!: () => void;
	const promise = new Promise<void>((done) => {
		resolve = done;
	});
	return { promise, resolve };
}

describe("GitHub Gist vector freshness", () => {
	it("rejects changed content after failed embedding and keeps metadata-only updates searchable", async () => {
		const chunks = store();
		const index = new GistSemanticIndex();
		const embedder = controlledEmbedder();
		chunks.replaceDocuments([document("changed", "old text"), document("keeper", "steady text")]);
		expect(await index.sync(chunks.chunks(), embedder)).toEqual({ ok: true, embedded: 2 });

		chunks.updateDocuments([{ ...document("keeper", "steady text"), metadata: { revision: 2 } }]);
		expect(await index.sync(chunks.chunks(), embedder)).toEqual({ ok: true, embedded: 0 });
		expect(embedder.documentCalls).toHaveLength(1);
		chunks.updateDocuments([document("changed", "replacement text")]);
		embedder.failDocuments = true;
		expect(await index.sync(chunks.chunks(), embedder)).toEqual({
			ok: false,
			message: "embedding gateway unavailable",
		});
		const hits = await method(chunks, index, embedder).retrieve("query", { topK: 5 });
		expect(hits.map((hit) => hit.content)).toEqual(["steady text"]);
		expect(hits[0]?.metadata).toMatchObject({ revision: 2, chunkId: "keeper" });
	});

	it("invalidates a vector when only the title changes", async () => {
		const chunks = store();
		const index = new GistSemanticIndex();
		const embedder = controlledEmbedder();
		chunks.replaceDocuments([document("gist", "same body", "old title")]);
		await index.sync(chunks.chunks(), embedder);
		chunks.updateDocuments([document("gist", "same body", "new title")]);
		embedder.failDocuments = true;
		expect(await index.sync(chunks.chunks(), embedder)).toMatchObject({ ok: false });
		expect(await method(chunks, index, embedder).retrieve("query", { topK: 1 })).toEqual([]);
	});

	it("rejects persisted stale vectors before refresh after restarting", async () => {
		const root = workspace();
		const statePath = join(root, "vectors.json");
		const chunks = store(root);
		const index = new GistSemanticIndex({ statePath });
		const embedder = controlledEmbedder();
		chunks.replaceDocuments([document("changed", "old text"), document("keeper", "steady text")]);
		await index.sync(chunks.chunks(), embedder);
		chunks.updateDocuments([document("changed", "new persisted text")]);

		const restoredIndex = new GistSemanticIndex({ statePath });
		// Existing low-level callers can still rank without supplying a chunk snapshot.
		expect(restoredIndex.searchByVector([1, 0], 1)[0]?.chunkId).toBe("changed");
		const hits = await method(store(root), restoredIndex, embedder).retrieve("query", { topK: 5 });
		expect(hits.map((hit) => hit.content)).toEqual(["steady text"]);
	});

	it("does not let stale and deleted high-scoring entries consume the candidate limit", async () => {
		const chunks = store();
		const index = new GistSemanticIndex();
		const embedder = controlledEmbedder();
		const outdated = Array.from({ length: 8 }, (_, i) => document(`old-${i}`, "high-scoring old text"));
		chunks.replaceDocuments([...outdated, document("keeper", "steady text")]);
		await index.sync(chunks.chunks(), embedder);
		chunks.updateDocuments(
			outdated.slice(0, 4).map((doc) => ({ ...doc, content: "changed text" })),
			outdated.slice(4).map((doc) => doc.docId),
		);
		const hits = await method(chunks, index, embedder).retrieve("query", { topK: 1 });
		expect(hits.map((hit) => hit.content)).toEqual(["steady text"]);
	});

	it("validates and maps the current chunks after query embedding resumes", async () => {
		const chunks = store();
		const index = new GistSemanticIndex();
		const base = controlledEmbedder();
		chunks.replaceDocuments([document("changed", "old text"), document("keeper", "steady text")]);
		await index.sync(chunks.chunks(), base);
		const started = signal();
		const release = signal();
		const embedder: Embedder = {
			identity: base.identity,
			async embed(texts, options) {
				started.resolve();
				await release.promise;
				return base.embed(texts, options);
			},
		};
		const pending = method(chunks, index, embedder).retrieve("query", { topK: 5 });
		await started.promise;
		chunks.updateDocuments([
			document("changed", "text updated during query"),
			{ ...document("keeper", "steady text"), metadata: { revision: 3 } },
		]);
		release.resolve();
		const hits = await pending;
		expect(hits.map((hit) => hit.content)).toEqual(["steady text"]);
		expect(hits[0]?.metadata.revision).toBe(3);
	});

	it("rejects vectors from an in-flight sync whose chunk snapshot became outdated", async () => {
		const chunks = store();
		const index = new GistSemanticIndex();
		const base = controlledEmbedder();
		chunks.replaceDocuments([document("gist", "original text")]);
		const started = signal();
		const release = signal();
		const embedder: Embedder = {
			identity: base.identity,
			async embed(texts, options) {
				if (options?.role === "document") {
					started.resolve();
					await release.promise;
				}
				return base.embed(texts, options);
			},
		};
		const pending = index.sync(chunks.chunks(), embedder);
		await started.promise;
		chunks.updateDocuments([document("gist", "newer text")]);
		release.resolve();
		expect(await pending).toEqual({ ok: true, embedded: 1 });
		expect(await method(chunks, index, embedder).retrieve("query", { topK: 1 })).toEqual([]);
	});

	it("recovers on an unchanged connector snapshot without fetching gist content again", async () => {
		const root = workspace();
		const embedder = controlledEmbedder();
		const snapshot = (revision: number, content: string) => {
			const gist = {
				id: "g1",
				description: "notes",
				updated_at: `2024-03-0${revision}T00:00:00.000Z`,
				html_url: "https://gist.github.com/g1",
				files: { "notes.md": { filename: "notes.md", content, truncated: false } },
			};
			return createMockFetch([
				{ match: "/gists/g1", json: gist },
				{ match: "/gists?", json: [gist] },
			]);
		};
		const make = (mock: ReturnType<typeof snapshot>) =>
			new GitHubGistSkill({
				workspaceRoot: root,
				semantic: { embedder },
				connector: new GitHubGistConnector({
					token: "fixture-token",
					ghCliFallback: false,
					fetchImpl: mock.fetchImpl,
					statePath: join(root, "state.json"),
				}),
			});
		expect((await make(snapshot(1, "original text")).index()).ok).toBe(true);
		const updated = snapshot(2, "replacement text");
		const skill = make(updated);
		embedder.failDocuments = true;
		const failed = await skill.index();
		expect(failed.ok).toBe(true);
		expect(failed.diagnostics).toContainEqual(
			expect.objectContaining({
				code: "datasource-index-failed",
				message: "semantic-unavailable: embedding gateway unavailable",
			}),
		);
		const lexical = skill.retrievalMethods().find((entry) => entry.describe().type === "bm25");
		const semantic = skill.retrievalMethods().find((entry) => entry.describe().type === "vector");
		expect((await lexical!.retrieve("replacement", { topK: 1 }))[0]?.content).toContain("replacement text");
		expect(await semantic!.retrieve("query", { topK: 1 })).toEqual([]);

		embedder.failDocuments = false;
		const recovered = await skill.index();
		expect(recovered.ok).toBe(true);
		expect(recovered.diagnostics.some((diagnostic) => diagnostic.message.includes("semantic-unavailable"))).toBe(
			false,
		);
		expect((await semantic!.retrieve("query", { topK: 1 }))[0]?.content).toContain("replacement text");
		expect(updated.requests.filter((request) => request.includes("/gists/g1"))).toHaveLength(1);
		expect(updated.requests.filter((request) => request.includes("/gists?"))).toHaveLength(2);
	});
});

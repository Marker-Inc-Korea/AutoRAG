import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { EvidenceLedger } from "../../src/agent/evidence-ledger.ts";
import { createJikjiFindTool } from "../../src/agent/jikji-find-tool.ts";
import { createSearchAllDocumentsTool } from "../../src/agent/search-all-tool.ts";
import { createSearchMinSyncDocumentsTool } from "../../src/agent/search-minsync-tool.ts";
import { createSingleDatasourceSearchTools } from "../../src/agent/search-single-datasource-tool.ts";
import { createWebFetchTool } from "../../src/agent/web-fetch-tool.ts";
import { createWebSearchTool } from "../../src/agent/web-search-tool.ts";
import type { MinSyncVectorMethod } from "../../src/minsync/method.ts";
import type { RetrievalResult } from "../../src/retrieval/types.ts";
import { clearRegisteredSearchProviders, registerSearchProvider } from "../../src/web/search/provider.ts";
import { SEARCH_PROVIDER_ORDER, type SearchProviderId } from "../../src/web/search/types.ts";

function row(id: string, source: string, content: string, method = "bm25"): RetrievalResult {
	return { id, source, content, score: 0.5, metadata: { method } };
}

function textOf(result: { content: ReadonlyArray<{ type: string; text?: string }> }): string {
	return result.content.map((part) => (part.type === "text" ? (part.text ?? "") : "")).join("");
}

function resolve(ledger: EvidenceLedger, ref: string) {
	return ledger.resolve([ref], { label: "t", number: 1, fallbackContent: "", allowLocalFiles: false })[0];
}

let ledger: EvidenceLedger;
beforeEach(() => {
	ledger = new EvidenceLedger();
});
afterEach(() => {
	clearRegisteredSearchProviders();
});

describe("retrieval tools issue evidence ids", () => {
	it("search_all_documents prints an id per result and records exactly what it printed", async () => {
		const tool = createSearchAllDocumentsTool(
			{
				searchAllDocuments: async () => ({
					results: [row("a", "/docs/a.txt", "alpha text"), row("b", "/docs/b.txt", "beta text")],
					diagnostics: [],
				}),
			},
			ledger,
		);

		const out = await tool.execute("c1", { query: "q" });
		const text = textOf(out);

		expect(text).toContain("[e1] /docs/a.txt");
		expect(text).toContain("[e2] /docs/b.txt");
		expect(text).not.toMatch(/^\[1\] /mu);
		expect(resolve(ledger, "e2")).toMatchObject({ source: "/docs/b.txt", content: "beta text", method: "bm25" });
	});

	it("gives the same chunk the same id when two tools surface it", async () => {
		const results = [row("a", "/docs/a.txt", "alpha text")];
		const all = createSearchAllDocumentsTool(
			{ searchAllDocuments: async () => ({ results, diagnostics: [] }) },
			ledger,
		);
		const method = { isBinaryMissing: () => false, retrieve: async () => results } as unknown as MinSyncVectorMethod;
		const minsync = createSearchMinSyncDocumentsTool(() => method, undefined, ledger);

		await all.execute("c1", { query: "q" });
		const out = await minsync.execute("c2", { query: "q" });

		expect(textOf(out)).toContain("[e1] /docs/a.txt");
	});

	it("semantic_search_local_docs prints ids", async () => {
		const method = {
			isBinaryMissing: () => false,
			retrieve: async () => [row("m", "/docs/m.txt", "semantic hit", "minsync")],
		} as unknown as MinSyncVectorMethod;
		const tool = createSearchMinSyncDocumentsTool(() => method, undefined, ledger);

		const out = await tool.execute("c1", { query: "q" });

		expect(textOf(out)).toContain("[e1] /docs/m.txt");
		expect(resolve(ledger, "e1")).toMatchObject({ method: "minsync", content: "semantic hit" });
	});

	it("search_datasource_<id> prints ids for virtual sources", async () => {
		const [tool] = createSingleDatasourceSearchTools(
			{
				searchSingleDatasourceDocuments: async () => ({
					results: [row("k", "/kakao/acct/chunks/1", "lunch at noon", "kakao-lexical")],
					diagnostics: [],
				}),
			},
			[{ datasourceId: "kakao", description: "chats", instanceScopes: [] }],
			ledger,
		);

		const out = await tool?.execute("c1", { query: "lunch" });

		expect(out ? textOf(out) : "").toContain("[e1] /kakao/acct/chunks/1");
		expect(resolve(ledger, "/kakao/acct/chunks/1")).toMatchObject({ method: "kakao-lexical" });
	});

	it("an empty result set issues no ids", async () => {
		const tool = createSearchAllDocumentsTool(
			{ searchAllDocuments: async () => ({ results: [], diagnostics: [] }) },
			ledger,
		);
		await tool.execute("c1", { query: "q" });
		expect(() => resolve(ledger, "e1")).toThrow();
	});

	it("jikji_find registers each answer path so it can be cited", async () => {
		const tool = createJikjiFindTool(
			{
				findJikji: async () => ({
					answerPack: {
						answerPaths: ["/docs/refund.txt"],
						paths: ["/docs/refund.txt"],
						candidates: [{ path: "/docs/refund.txt", nextRead: "original", label: "Refund policy" }],
						evidencePack: [{ path: "/docs/refund.txt", nextRead: "original" }],
						handoffAction: "raw_fallback_after_retry",
					},
					policy: undefined,
					diagnostics: [],
					perRoot: [],
				}),
			} as never,
			ledger,
		);

		const out = await tool.execute("c1", { query: "refund" });

		expect(textOf(out)).toContain("[e1] /docs/refund.txt");
		expect(resolve(ledger, "e1")).toMatchObject({ source: "/docs/refund.txt", method: "jikji_find" });
	});

	it("web_search prints an id per source and records the url as the source", async () => {
		registerSearchProvider({
			id: "duckduckgo",
			label: "DuckDuckGo",
			isAvailable: () => true,
			search: async () => ({
				provider: "duckduckgo",
				sources: [{ title: "Refund Policy", url: "https://example.com/refunds", snippet: "Director approval" }],
			}),
		} as never);
		const keep: SearchProviderId = "duckduckgo";
		const tool = createWebSearchTool(
			{ order: [keep], exclude: SEARCH_PROVIDER_ORDER.filter((id) => id !== keep) },
			ledger,
		);

		const out = await tool.execute("c1", { query: "refund" });

		expect(textOf(out)).toContain("[e1] Refund Policy");
		expect(resolve(ledger, "e1")).toMatchObject({
			source: "https://example.com/refunds",
			method: "web_search",
			content: expect.stringContaining("Director approval"),
		});
	});

	it("web_fetch rejects non-http urls without registering evidence", async () => {
		const tool = createWebFetchTool({}, ledger);
		await tool.execute("c1", { url: "/etc/passwd" });
		expect(() => resolve(ledger, "e1")).toThrow();
	});
});

describe("ledger scoping", () => {
	let root: string;
	beforeEach(() => {
		root = mkdtempSync(join(tmpdir(), "autorag-ids-"));
	});
	afterEach(() => {
		rmSync(root, { recursive: true, force: true });
	});

	it("never lets one run's ids resolve in the next run", async () => {
		const tool = createSearchAllDocumentsTool(
			{ searchAllDocuments: async () => ({ results: [row("a", join(root, "a.txt"), "x")], diagnostics: [] }) },
			ledger,
		);
		await tool.execute("c1", { query: "q" });
		ledger.clear();
		expect(() => resolve(ledger, "e1")).toThrow();
	});
});

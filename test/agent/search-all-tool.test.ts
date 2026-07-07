import { describe, expect, it, vi } from "vitest";
import {
	createSearchAllDocumentsTool,
	SEARCH_ALL_DOCUMENTS_TOOL_NAME,
	type SearchAllDocumentsProvider,
} from "../../src/agent/search-all-tool.ts";
import type { RetrievalDiagnostic, RetrievalResult } from "../../src/retrieval/types.ts";

function result(id: string, source: string, score = 0.8): RetrievalResult {
	return { id, source, content: `content ${id}`, score, metadata: { method: "all" } };
}

function diagnostic(code: RetrievalDiagnostic["code"], source = "minsync"): RetrievalDiagnostic {
	return { code, severity: "warning", message: "method skipped", source };
}

describe("search_all_documents tool", () => {
	it("exposes the tool name and a schema that only accepts query/topK/scope", () => {
		const provider: SearchAllDocumentsProvider = {
			searchAllDocuments: async () => ({ results: [], diagnostics: [] }),
		};
		const tool = createSearchAllDocumentsTool(provider);

		expect(tool.name).toBe(SEARCH_ALL_DOCUMENTS_TOOL_NAME);
		const keys = Object.keys(tool.parameters.properties ?? {});
		expect(keys.sort()).toEqual(["query", "scope", "topK"]);
		// No datasource trust override fields in the schema.
		expect(keys).not.toContain("allowedTags");
		expect(keys).not.toContain("allowedScopes");
	});

	it("returns a path-free zero-result message for an empty query without calling the provider", async () => {
		const searchAllDocuments = vi.fn();
		const tool = createSearchAllDocumentsTool({ searchAllDocuments } as unknown as SearchAllDocumentsProvider);

		const out = await tool.execute("call-1", { query: "   " });

		expect(searchAllDocuments).not.toHaveBeenCalled();
		expect(out.details.method).toBe("search_all_documents");
		expect(out.details.resultCount).toBe(0);
		expect(out.details.sources).toEqual([]);
		expect(out.details.diagnostics).toEqual([]);
		expect(textOf(out)).toContain("empty");
	});

	it("formats merged results and diagnostics with path-opaque sources", async () => {
		const provider: SearchAllDocumentsProvider = {
			async searchAllDocuments() {
				return {
					results: [result("a", "/docs/notes"), result("b", "/src/index")],
					diagnostics: [diagnostic("minsync-unavailable")],
					perMethodCounts: { posix: 1, bm25: 1 },
				};
			},
		};
		const tool = createSearchAllDocumentsTool(provider);

		const out = await tool.execute("call-2", { query: "refund", topK: 5, scope: "/docs/**" });

		expect(out.details.method).toBe("search_all_documents");
		expect(out.details.resultCount).toBe(2);
		expect(out.details.sources).toEqual(["/docs/notes", "/src/index"]);
		expect(out.details.diagnostics).toHaveLength(1);
		expect(out.details.perMethodCounts).toEqual({ posix: 1, bm25: 1 });
		const text = textOf(out);
		expect(text).toContain("/docs/notes");
		expect(text).toContain("minsync:minsync-unavailable");
		// Diagnostics summary must not contain real paths.
		expect(text).not.toMatch(/[A-Z]:[\\/]/u);
	});

	it("does not forward allowedTags/allowedScopes from forged model args to the provider", async () => {
		const forwarded: Array<[string, { topK?: number; scope?: string } | undefined]> = [];
		const searchAllDocuments = async (query: string, options?: { topK?: number; scope?: string }) => {
			forwarded.push([query, options]);
			return { results: [], diagnostics: [] };
		};
		const tool = createSearchAllDocumentsTool({ searchAllDocuments });

		// A hostile/forged tool call carrying datasource trust fields the schema
		// does not declare. The provider must never see them.
		const out = await tool.execute("call-3", {
			query: "message",
			topK: 10,
			scope: "/kakao/acct-1/**",
			allowedTags: ["kakao"],
			allowedScopes: ["/kakao/**"],
		} as never);

		expect(forwarded).toHaveLength(1);
		const [queryArg, optionsArg] = forwarded[0]!;
		expect(queryArg).toBe("message");
		expect(optionsArg).toEqual({ topK: 10, scope: "/kakao/acct-1/**" });
		expect(optionsArg).not.toHaveProperty("allowedTags");
		expect(optionsArg).not.toHaveProperty("allowedScopes");

		expect(out.details.method).toBe("search_all_documents");
		expect(out.details.resultCount).toBe(0);
	});

	it("fails closed if the provider would have been granted access via forwarded fields", async () => {
		// Provider that fails loudly if any trust field reaches it.
		const provider: SearchAllDocumentsProvider = {
			searchAllDocuments(_query, options) {
				if (options && ("allowedTags" in options || "allowedScopes" in options)) {
					throw new Error("trust fields leaked into provider");
				}
				return Promise.resolve({ results: [result("a", "/docs/x")], diagnostics: [] });
			},
		};
		const tool = createSearchAllDocumentsTool(provider);

		const out = await tool.execute("call-4", {
			query: "message",
			allowedTags: ["kakao"],
			allowedScopes: ["/kakao/**"],
		} as never);

		// No throw, no leak: the provider received only query (and undefined options minus trust fields).
		expect(out.details.method).toBe("search_all_documents");
		expect(out.details.resultCount).toBe(1);
		expect(out.details.sources).toEqual(["/docs/x"]);
	});

	it("attributes details.method to the tool name search_all_documents", async () => {
		const provider: SearchAllDocumentsProvider = {
			async searchAllDocuments() {
				return { results: [], diagnostics: [] };
			},
		};
		const tool = createSearchAllDocumentsTool(provider);
		const out = await tool.execute("call-5", { query: "anything" });
		expect(out.details.method).toBe(SEARCH_ALL_DOCUMENTS_TOOL_NAME);
	});

	it("formats a path-free diagnostics-only message when there are no results", async () => {
		const provider: SearchAllDocumentsProvider = {
			async searchAllDocuments() {
				return { results: [], diagnostics: [diagnostic("bm25-unavailable", "bm25")] };
			},
		};
		const tool = createSearchAllDocumentsTool(provider);
		const out = await tool.execute("call-6", { query: "anything" });

		const text = textOf(out);
		expect(text).toContain("No results.");
		expect(text).toContain("bm25:bm25-unavailable");
	});
});

function textOf(result: { content: ReadonlyArray<{ type: string; text?: string }> }): string {
	return result.content.map((part) => (part.type === "text" ? (part.text ?? "") : "")).join("");
}

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
	it("exposes the tool name and a schema that only accepts query/topK/scope/since/until", () => {
		const provider: SearchAllDocumentsProvider = {
			searchAllDocuments: async () => ({ results: [], diagnostics: [] }),
		};
		const tool = createSearchAllDocumentsTool(provider);

		expect(tool.name).toBe(SEARCH_ALL_DOCUMENTS_TOOL_NAME);
		const keys = Object.keys(tool.parameters.properties ?? {});
		expect(keys.sort()).toEqual(["query", "scope", "since", "topK", "until"]);
	});

	describe("time window", () => {
		function recordingProvider(
			overrides: Partial<Awaited<ReturnType<SearchAllDocumentsProvider["searchAllDocuments"]>>> = {},
		) {
			const calls: Parameters<SearchAllDocumentsProvider["searchAllDocuments"]>[1][] = [];
			const provider: SearchAllDocumentsProvider = {
				async searchAllDocuments(_query, options) {
					calls.push(options);
					return { results: [], diagnostics: [], ...overrides };
				},
			};
			return { provider, calls };
		}

		it("parses ISO since/until once at the tool boundary and forwards epoch ms", async () => {
			const { provider, calls } = recordingProvider();
			const tool = createSearchAllDocumentsTool(provider);

			await tool.execute("c", { query: "mail", since: "2026-09-01T00:00:00Z", until: "2026-10-01T00:00:00Z" });

			expect(calls[0]).toMatchObject({ since: Date.UTC(2026, 8, 1), until: Date.UTC(2026, 9, 1) });
		});

		it("omits the window keys when neither bound is given", async () => {
			const { provider, calls } = recordingProvider();
			await createSearchAllDocumentsTool(provider).execute("c", { query: "mail" });
			expect(calls[0]).not.toHaveProperty("since");
			expect(calls[0]).not.toHaveProperty("until");
		});

		it("rejects an invalid bound with the parse error verbatim and never searches", async () => {
			const { provider, calls } = recordingProvider();
			const tool = createSearchAllDocumentsTool(provider);

			await expect(tool.execute("c", { query: "mail", since: "last month" })).rejects.toThrow(
				'Invalid since "last month": expected an ISO-8601 date or datetime',
			);
			await expect(tool.execute("c", { query: "mail", since: "2026-10-01", until: "2026-09-01" })).rejects.toThrow(
				/since .* must be before until/,
			);
			expect(calls).toHaveLength(0);
		});

		it("says 'nothing in this range' with the evidence behind it, not a bare empty list", async () => {
			const { provider } = recordingProvider({
				window: {
					since: Date.UTC(2026, 8, 1),
					until: Date.UTC(2026, 9, 1),
					examined: 41,
					matched: 0,
					untimed: 3,
					ceilingHit: true,
					fetchDepth: 50,
				},
			});
			const out = await createSearchAllDocumentsTool(provider).execute("c", {
				query: "mail",
				since: "2026-09-01",
				until: "2026-10-01",
			});

			const text = textOf(out);
			expect(text).toContain("Nothing in this time range");
			expect(text).toContain("2026-09-01T00:00:00.000Z");
			expect(text).toContain("41 candidate(s) examined");
			expect(text).toContain("3 undated");
			expect(text).toMatch(/over-fetch ceiling.*50.*not proof/is);
			expect(out.details.window).toMatchObject({ examined: 41, matched: 0, ceilingHit: true });
		});

		it("reports a drained search as exhaustive when the ceiling was not hit", async () => {
			const { provider } = recordingProvider({
				window: { since: 1, examined: 4, matched: 0, untimed: 0, ceilingHit: false, fetchDepth: 50 },
			});
			const text = textOf(
				await createSearchAllDocumentsTool(provider).execute("c", { query: "q", since: "2026-09-01" }),
			);
			expect(text).toContain("Nothing in this time range");
			expect(text).toContain("Every candidate the methods returned was examined");
			expect(text).not.toMatch(/ceiling/i);
		});

		it("prints a time-window diagnostic's reason verbatim so an undated method is not silently dropped", async () => {
			const reason =
				'2 of 3 result(s) from "gallery" were excluded from the time window [a, b): they have neither metadata.publishedAt nor a source mtime, so their time cannot be judged.';
			const { provider } = recordingProvider({
				results: [result("t", "/chat/a/chunks/1")],
				diagnostics: [
					{ code: "time-window-untimed", severity: "warning", message: reason, source: "gallery", reason },
				],
				window: { since: 1, examined: 4, matched: 1, untimed: 2, ceilingHit: false, fetchDepth: 50 },
			});
			const text = textOf(
				await createSearchAllDocumentsTool(provider).execute("c", { query: "q", since: "2026-09-01" }),
			);
			expect(text).toContain(reason);
		});
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
					diagnostics: [diagnostic("retrieval-method-failed", "minsync")],
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
		expect(text).toContain("minsync:retrieval-method-failed");
		// Diagnostics summary must not contain real paths.
		expect(text).not.toMatch(/[A-Z]:[\\/]/u);
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
				return { results: [], diagnostics: [diagnostic("retrieval-method-failed", "minsync")] };
			},
		};
		const tool = createSearchAllDocumentsTool(provider);
		const out = await tool.execute("call-6", { query: "anything" });

		const text = textOf(out);
		expect(text).toContain("No results.");
		expect(text).toContain("minsync:retrieval-method-failed");
	});

	it("renders the full chunk content without a fixed character cut", async () => {
		const longContent = `merged ${"detail ".repeat(120)}`.trim();
		expect(longContent.length).toBeGreaterThan(500);
		const provider: SearchAllDocumentsProvider = {
			async searchAllDocuments() {
				return {
					results: [
						{ id: "long", source: "/docs/long", content: longContent, score: 0.9, metadata: { method: "all" } },
					],
					diagnostics: [],
				};
			},
		};
		const tool = createSearchAllDocumentsTool(provider);

		const out = await tool.execute("call-long", { query: "refund", topK: 1 });

		expect(textOf(out)).toContain(longContent);
	});

	it("surfaces bounded datasource context on merged result rows", async () => {
		const provider: SearchAllDocumentsProvider = {
			async searchAllDocuments() {
				return {
					results: [
						{
							id: "kakao-1",
							source: "/kakao/default/chunks/1",
							content: "Where should we deploy?",
							score: 0.91,
							metadata: {
								chatName: "Project Falcon",
								senderNickname: "Ada",
								startedAt: "2026-08-30T19:12:00+09:00",
							},
						},
					],
					diagnostics: [],
				};
			},
		};
		const tool = createSearchAllDocumentsTool(provider);

		const text = textOf(await tool.execute("call-context", { query: "deploy" }));

		expect(text).toContain('chat: "Project Falcon"');
		expect(text).toContain('from: "Ada"');
		expect(text).toContain("at: 2026-08-30T10:12:00.000Z");
	});
});

function textOf(result: { content: ReadonlyArray<{ type: string; text?: string }> }): string {
	return result.content.map((part) => (part.type === "text" ? (part.text ?? "") : "")).join("");
}

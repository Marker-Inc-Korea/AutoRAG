import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	createSearchMinSyncDocumentsTool,
	SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME,
} from "../../src/agent/search-minsync-tool.ts";
import { MinSyncRequiredError } from "../../src/minsync/errors.ts";
import { MinSyncVectorMethod } from "../../src/minsync/method.ts";
import { RetrievalScopeError } from "../../src/retrieval/scope.ts";
import type { RetrievalResult } from "../../src/retrieval/types.ts";

let tmpDir: string;

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-minsync-tool-test-"));
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
});

/** Minimal stub satisfying the surface the tool touches (`retrieve`). */
interface StubMethod {
	retrieve(query: string, options: { topK?: number; scope?: string }): Promise<RetrievalResult[]>;
}

function stubMethod(rows: readonly RetrievalResult[]): MinSyncVectorMethod {
	return {
		retrieve: async (_query: string, options: { topK?: number; scope?: string }) =>
			rows.slice(0, options.topK ?? rows.length),
	} as unknown as MinSyncVectorMethod;
}

function result(id: string, source: string): RetrievalResult {
	return { id, source, content: `semantic content ${id}`, score: 0.9, metadata: { method: "minsync" } };
}

describe("semantic_search_local_docs tool", () => {
	it("exposes the tool name and path-opaque-only schema fields", () => {
		const tool = createSearchMinSyncDocumentsTool(() => stubMethod([]));
		expect(tool.name).toBe(SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME);
		const keys = Object.keys(tool.parameters.properties ?? {});
		expect(keys.sort()).toEqual(["query", "scope", "topK"]);
	});

	it("fails with MinSyncRequiredError when the binary is missing, without leaking the machine's paths", async () => {
		// Real MinSyncVectorMethod with a binaryPath that does not exist.
		const missing = join(tmpDir, "does-not-exist", "minsync");
		const method = new MinSyncVectorMethod({ root: tmpDir, binaryPath: missing });
		expect(method.isBinaryMissing()).toBe(true);
		expect(existsSync(missing)).toBe(false);

		const tool = createSearchMinSyncDocumentsTool(() => method);
		const failure = await tool.execute("call-2", { query: "meaning" }).catch((error: unknown) => error);

		expect(failure).toBeInstanceOf(MinSyncRequiredError);
		expect((failure as Error).message).toContain("cargo install minsync");
		expect((failure as Error).message).not.toContain(tmpDir);
	});

	it("returns a zero-result message for an empty query without calling MinSync", async () => {
		const tool = createSearchMinSyncDocumentsTool(() => {
			throw new Error("MinSync must not be resolved for an empty query");
		});
		const out = await tool.execute("call-3", { query: "  " });

		expect(out.details.method).toBe("semantic_search_local_docs");
		expect(out.details.resultCount).toBe(0);
		expect(textOf(out)).toContain("empty");
	});

	it("formats successful rows using opaque sources only", async () => {
		const tool = createSearchMinSyncDocumentsTool(() =>
			stubMethod([result("a", "/docs/notes"), result("b", "/docs/guide")]),
		);
		const out = await tool.execute("call-4", { query: "concept", topK: 2 });

		expect(out.details.method).toBe("semantic_search_local_docs");
		expect(out.details.resultCount).toBe(2);
		expect(out.details.sources).toEqual(["/docs/notes", "/docs/guide"]);
		const text = textOf(out);
		expect(text).toContain("/docs/notes");
		expect(text).toContain("/docs/guide");
		expect(text).not.toContain(tmpDir);
	});

	it("normalizes model-supplied physical scopes before MinSync retrieval", async () => {
		const seenScopes: Array<string | undefined> = [];
		const method: StubMethod = {
			retrieve(_query, options) {
				seenScopes.push(options.scope);
				return Promise.resolve([result("a", "/docs/notes")]);
			},
		};
		const tool = createSearchMinSyncDocumentsTool(
			() => method as never,
			() => "/docs",
		);

		await tool.execute("call-scope", { query: "concept", scope: join(tmpDir, "docs") });

		expect(seenScopes).toEqual(["/docs"]);
	});

	it("preserves coded scope errors at the MinSync tool boundary", async () => {
		const method: StubMethod = {
			retrieve: () => Promise.resolve([]),
		};
		const tool = createSearchMinSyncDocumentsTool(
			() => method as never,
			() => {
				throw new RetrievalScopeError(["/docs"]);
			},
		);

		await expect(tool.execute("call-invalid-scope", { query: "concept", scope: "/outside" })).rejects.toMatchObject({
			code: "invalid-retrieval-scope",
		});
	});

	it("lets a retrieval failure surface with its real message", async () => {
		const throwing: StubMethod = {
			retrieve(): Promise<never> {
				return Promise.reject(new Error("minsync query exited with code 3: index is corrupt"));
			},
		};
		const tool = createSearchMinSyncDocumentsTool(() => throwing as never);

		await expect(tool.execute("call-5", { query: "concept" })).rejects.toThrow("index is corrupt");
	});
});

function textOf(result: { content: ReadonlyArray<{ type: string; text?: string }> }): string {
	return result.content.map((part) => (part.type === "text" ? (part.text ?? "") : "")).join("");
}

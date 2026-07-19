import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	createSearchMinSyncDocumentsTool,
	SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME,
} from "../../src/agent/search-minsync-tool.ts";
import { MinSyncVectorMethod } from "../../src/minsync/method.ts";
import type { RetrievalResult } from "../../src/retrieval/types.ts";

let tmpDir: string;

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-minsync-tool-test-"));
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
});

/** Minimal stub satisfying the surface the tool touches (`isBinaryMissing`, `retrieve`). */
interface StubMethod {
	isBinaryMissing(): boolean;
	retrieve(query: string, options: { topK?: number; scope?: string }): Promise<RetrievalResult[]>;
}

function stubMethod(rows: readonly RetrievalResult[], opts: { binaryMissing?: boolean } = {}): MinSyncVectorMethod {
	return {
		isBinaryMissing: () => opts.binaryMissing ?? false,
		retrieve: async (_query: string, options: { topK?: number; scope?: string }) =>
			rows.slice(0, options.topK ?? rows.length),
	} as unknown as MinSyncVectorMethod;
}

function result(id: string, source: string): RetrievalResult {
	return { id, source, content: `semantic content ${id}`, score: 0.9, metadata: { method: "minsync" } };
}

describe("search_minsync_documents tool", () => {
	it("exposes the tool name and path-opaque-only schema fields", () => {
		const tool = createSearchMinSyncDocumentsTool(() => undefined);
		expect(tool.name).toBe(SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME);
		const keys = Object.keys(tool.parameters.properties ?? {});
		expect(keys.sort()).toEqual(["query", "scope", "topK"]);
	});

	it("returns a path-free unavailable message when the method is missing", async () => {
		const tool = createSearchMinSyncDocumentsTool(() => undefined);
		const out = await tool.execute("call-1", { query: "meaning" });

		expect(out.details.method).toBe("search_minsync_documents");
		expect(out.details.resultCount).toBe(0);
		expect(out.details.available).toBe(false);
		const text = textOf(out);
		expect(text).toContain("not configured");
		expect(text).not.toContain(tmpDir);
	});

	it("returns a path-free unavailable message when the binary is missing", async () => {
		// Real MinSyncVectorMethod with a binaryPath that does not exist.
		const method = new MinSyncVectorMethod({
			root: tmpDir,
			binaryPath: join(tmpDir, "does-not-exist", "minsync"),
		});
		expect(method.isBinaryMissing()).toBe(true);
		expect(existsSync(join(tmpDir, "does-not-exist", "minsync"))).toBe(false);

		const tool = createSearchMinSyncDocumentsTool(() => method);
		const out = await tool.execute("call-2", { query: "meaning" });

		expect(out.details.method).toBe("search_minsync_documents");
		expect(out.details.resultCount).toBe(0);
		expect(out.details.available).toBe(false);
		const text = textOf(out);
		expect(text).toContain("unavailable");
		// No binary path leaks into the model-facing message.
		expect(text).not.toContain(tmpDir);
	});

	it("returns a path-free zero-result message for an empty query when available", async () => {
		const tool = createSearchMinSyncDocumentsTool(() => stubMethod([]));
		const out = await tool.execute("call-3", { query: "  " });

		expect(out.details.method).toBe("search_minsync_documents");
		expect(out.details.resultCount).toBe(0);
		expect(out.details.available).toBe(true);
		expect(textOf(out)).toContain("empty");
	});

	it("formats successful rows using opaque sources only", async () => {
		const tool = createSearchMinSyncDocumentsTool(() =>
			stubMethod([result("a", "/docs/notes"), result("b", "/docs/guide")]),
		);
		const out = await tool.execute("call-4", { query: "concept", topK: 2 });

		expect(out.details.method).toBe("search_minsync_documents");
		expect(out.details.resultCount).toBe(2);
		expect(out.details.sources).toEqual(["/docs/notes", "/docs/guide"]);
		expect(out.details.available).toBe(true);
		const text = textOf(out);
		expect(text).toContain("/docs/notes");
		expect(text).toContain("/docs/guide");
		expect(text).not.toContain(tmpDir);
	});

	it("reports a path-free unavailable message when retrieval throws", async () => {
		const throwing: StubMethod = {
			isBinaryMissing: () => false,
			retrieve(): Promise<never> {
				return Promise.reject(new Error(`workspace ${tmpDir}/.minsync blew up`));
			},
		};
		const tool = createSearchMinSyncDocumentsTool(() => throwing as never);
		const out = await tool.execute("call-5", { query: "concept" });

		expect(out.details.method).toBe("search_minsync_documents");
		expect(out.details.resultCount).toBe(0);
		expect(out.details.available).toBe(false);
		const text = textOf(out);
		expect(text).toContain("unavailable");
		expect(text).not.toContain(tmpDir);
		expect(text).not.toContain("blew up");
	});
});

function textOf(result: { content: ReadonlyArray<{ type: string; text?: string }> }): string {
	return result.content.map((part) => (part.type === "text" ? (part.text ?? "") : "")).join("");
}

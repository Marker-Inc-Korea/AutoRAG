import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	createSearchPosixDocumentsTool,
	SEARCH_POSIX_DOCUMENTS_TOOL_NAME,
	type SearchPosixDocumentsDetails,
} from "../../src/agent/search-posix-tool.ts";
import { PosixMethod } from "../../src/retrieval/methods/posix.ts";

let tmpDir: string;

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-posix-tool-test-"));
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
});

function writeFile(relative: string, content: string): void {
	const absolute = join(tmpDir, relative);
	mkdirSync(join(absolute, ".."), { recursive: true });
	writeFileSync(absolute, content, "utf8");
}

describe("search_posix_documents tool", () => {
	it("exposes the tool name and path-opaque-only schema fields", () => {
		const tool = createSearchPosixDocumentsTool(() => undefined);
		expect(tool.name).toBe(SEARCH_POSIX_DOCUMENTS_TOOL_NAME);
		const keys = Object.keys(searchPosixSchemaKeys(tool));
		expect(keys.sort()).toEqual(["query", "scope", "topK"]);
	});

	it("returns a path-free zero-result message when the method is missing", async () => {
		const tool = createSearchPosixDocumentsTool(() => undefined);
		const result = await tool.execute("call-1", { query: "anything" });

		expect(result.details.method).toBe("search_posix_documents");
		expect(result.details.resultCount).toBe(0);
		expect(result.details.sources).toEqual([]);
		const text = textOf(result);
		expect(text).toContain("not configured");
		expect(text).not.toContain(tmpDir);
	});

	it("returns a path-free zero-result message for an empty query", async () => {
		const method = new PosixMethod({ root: tmpDir, searchPaths: [tmpDir] });
		const tool = createSearchPosixDocumentsTool(() => method);
		const result = await tool.execute("call-2", { query: "   " });

		expect(result.details).toEqual({
			method: "search_posix_documents",
			resultCount: 0,
			sources: [],
		});
		expect(textOf(result)).toContain("empty");
	});

	it("formats successful results using opaque sources only", async () => {
		writeFile("docs/notes.md", "refund policy details here\nshipping rules");
		writeFile("src/index.ts", "export const refund = 1;\n");
		const method = new PosixMethod({ root: tmpDir, searchPaths: [tmpDir] });
		const tool = createSearchPosixDocumentsTool(() => method);

		const result = await tool.execute("call-3", { query: "refund", topK: 5 });

		const details = result.details as SearchPosixDocumentsDetails;
		expect(details.method).toBe("search_posix_documents");
		expect(details.resultCount).toBeGreaterThan(0);
		const text = textOf(result);
		// Opaque sources only — never the real tmpDir filesystem path.
		expect(text).not.toContain(tmpDir);
		for (const source of details.sources) {
			expect(source.startsWith("/")).toBe(true);
			expect(source).not.toContain(tmpDir);
		}
		expect(text).toMatch(/\[1\] \/.*score=/u);
	});

	it("reports a path-free message when retrieval throws", async () => {
		const throwing = {
			retrieve(): Promise<never> {
				return Promise.reject(new Error(`boom at ${tmpDir}/secret`));
			},
		};
		const tool = createSearchPosixDocumentsTool(() => throwing as never);
		const result = await tool.execute("call-4", { query: "anything" });

		expect(result.details.method).toBe("search_posix_documents");
		expect(result.details.resultCount).toBe(0);
		const text = textOf(result);
		expect(text).toContain("Posix search failed");
		expect(text).not.toContain(tmpDir);
		expect(text).not.toContain("secret");
	});
});

function textOf(result: { content: ReadonlyArray<{ type: string; text?: string }> }): string {
	return result.content.map((part) => (part.type === "text" ? (part.text ?? "") : "")).join("");
}

function searchPosixSchemaKeys(tool: {
	parameters: { properties?: Record<string, unknown> };
}): Record<string, unknown> {
	return tool.parameters.properties ?? {};
}

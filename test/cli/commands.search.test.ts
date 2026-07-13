import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Model } from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { SearchDocumentsResponse } from "../../src/agent/search-documents.ts";
import { runSearch, type SearchDeps } from "../../src/cli/commands/search.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";

let tmpDir: string;

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-cli-search-"));
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
});

interface Captured {
	readonly stdout: string[];
	readonly stderr: string[];
	readonly ctx: CommandContext;
}

function makeCtx(opts: {
	positionals?: string[];
	flags?: Record<string, string | boolean | undefined>;
	json?: boolean;
	debug?: boolean;
	cwd?: string;
}): Captured {
	const stdout: string[] = [];
	const stderr: string[] = [];
	const ctx: CommandContext = {
		positionals: opts.positionals ?? [],
		flags: opts.flags ?? {},
		json: opts.json ?? false,
		debug: opts.debug ?? false,
		cwd: opts.cwd ?? tmpDir,
		stdout: (line: string) => {
			stdout.push(line);
		},
		stderr: (line: string) => {
			stderr.push(line);
		},
	};
	return { stdout, stderr, ctx };
}

function cannedResponse(): SearchDocumentsResponse {
	return {
		sessionId: "sess-canned-123",
		query: "how does retrieval work",
		results: [
			{
				number: 1,
				title: "Retrieval Overview",
				summary: "Explains the parallel retriever pipeline.",
				evidence: [{ excerpt: "The retriever merges via min-max normalization." }],
				confidence: 0.82,
				feedbackId: "sess-canned-123:1",
			},
			{
				number: 2,
				title: "BM25 Fallback",
				summary: "Describes the TypeScript lexical fallback.",
				evidence: [{ excerpt: "BM25 falls back when the binding is missing." }],
				confidence: 0.61,
				feedbackId: "sess-canned-123:2",
			},
		],
		answer: "Retrieval merges results from posix, BM25, and MinSync methods.",
		searched: 2,
		warnings: [],
		diagnostics: [],
	};
}

describe("runSearch", () => {
	it("returns exit 2 with a usage error when the query is empty", async () => {
		const { ctx, stderr } = makeCtx({ positionals: [] });
		const code = await runSearch(ctx, { agentFactory: () => ({ searchDocuments: async () => cannedResponse() }) });
		expect(code).toBe(2);
		expect(stderr.join("\n")).toContain("Usage");
	});

	it("passes the default Sol model and API key into the production agent factory", async () => {
		const model: Model<"openai-responses"> = {
			id: "gpt-5.6-sol",
			name: "GPT-5.6 Sol",
			api: "openai-responses",
			provider: "myproxy",
			baseUrl: "https://proxy.example/v1",
			reasoning: true,
			input: ["text", "image"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 400_000,
			maxTokens: 128_000,
		};
		let received: { modelId?: string; apiKey?: string } | undefined;
		const { ctx } = makeCtx({ positionals: ["anything"], cwd: tmpDir });
		const code = await runSearch(ctx, {
			modelResolver: () => ({ model, apiKey: "secret" }),
			agentFactory: (options) => {
				received = { modelId: options.model?.id, apiKey: options.apiKey };
				return { searchDocuments: async () => cannedResponse() };
			},
		});

		expect(code).toBe(0);
		expect(received).toEqual({ modelId: "gpt-5.6-sol", apiKey: "secret" });
	});

	it("renders the documented search envelope as --json via an injected agentFactory", async () => {
		const canned = cannedResponse();
		const deps: SearchDeps = {
			agentFactory: () => ({
				searchDocuments: async () => canned,
			}),
		};
		const { ctx, stdout } = makeCtx({
			positionals: ["how", "does", "retrieval", "work"],
			json: true,
		});

		const code = await runSearch(ctx, deps);
		expect(code).toBe(0);
		expect(stdout.length).toBe(1);

		const parsed = JSON.parse(stdout[0]);
		expect(parsed.answer).toBe(canned.answer);
		expect(Array.isArray(parsed.results)).toBe(true);
		expect(parsed.results).toHaveLength(2);
		expect(parsed.results[0]).toEqual({
			number: 1,
			title: "Retrieval Overview",
			summary: "Explains the parallel retriever pipeline.",
		});
		expect(parsed.results[1]).toEqual({
			number: 2,
			title: "BM25 Fallback",
			summary: "Describes the TypeScript lexical fallback.",
		});
		// Non-debug json envelope must not leak path-bearing fields.
		expect(parsed.sessionId).toBeUndefined();
		expect(parsed.results[0].evidence).toBeUndefined();
	});

	it("renders path-opaque human output via an injected agentFactory", async () => {
		const canned = cannedResponse();
		const deps: SearchDeps = {
			agentFactory: () => ({
				searchDocuments: async () => canned,
			}),
		};
		const { ctx, stdout } = makeCtx({
			positionals: ["how does retrieval work"],
			json: false,
		});

		const code = await runSearch(ctx, deps);
		expect(code).toBe(0);
		const text = stdout.join("\n");
		// Number + title + summary present.
		expect(text).toContain("1. Retrieval Overview");
		expect(text).toContain("Explains the parallel retriever pipeline.");
		expect(text).toContain("2. BM25 Fallback");
		expect(text).toContain(canned.answer);
		// Path opacity: no filesystem paths leak (tmp dir, no "indexPath").
		expect(text).not.toContain(tmpDir);
		expect(text).not.toContain("indexPath");
		expect(text).not.toContain("/");
	});

	it("forwards --top-k, --scope, and --tags to the agent's searchDocuments", async () => {
		const canned = cannedResponse();
		let received: { query: string; options: unknown } | undefined;
		const deps: SearchDeps = {
			agentFactory: () => ({
				searchDocuments: async (query: string, options?: unknown) => {
					received = { query, options };
					return canned;
				},
			}),
		};
		const { ctx } = makeCtx({
			positionals: ["semantic search"],
			flags: { "top-k": "5", scope: "src/lib", tags: "a,b" },
			json: true,
		});

		const code = await runSearch(ctx, deps);
		expect(code).toBe(0);
		expect(received?.query).toBe("semantic search");
		expect(received?.options).toEqual({ topK: 5, scope: "src/lib", allowedTags: ["a", "b"] });
	});

	it("returns exit 1 when the agent throws a runtime error", async () => {
		const deps: SearchDeps = {
			agentFactory: () => ({
				searchDocuments: async () => {
					throw new Error("boom at runtime");
				},
			}),
		};
		const { ctx, stderr } = makeCtx({ positionals: ["query"], json: true });
		const code = await runSearch(ctx, deps);
		expect(code).toBe(1);
		const parsed = JSON.parse(stderr[0]);
		expect(parsed.ok).toBe(false);
		expect(parsed.error).toContain("boom at runtime");
	});
});

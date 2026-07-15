import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Model } from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { SearchDocumentsResponse } from "../../src/agent/search-documents.ts";
import { classifySearchHealthHint, runSearch, type SearchDeps } from "../../src/cli/commands/search.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import { ConfigError, resolveAgentModel } from "../../src/cli/config.ts";

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
			provider: "test-proxy",
			baseUrl: "https://proxy.example/v1",
			reasoning: true,
			input: ["text", "image"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 400_000,
			maxTokens: 128_000,
		};
		let received: { modelId?: string; explorerModelId?: string; apiKey?: string } | undefined;
		const { ctx } = makeCtx({ positionals: ["anything"], cwd: tmpDir });
		const code = await runSearch(ctx, {
			modelResolver: () => ({
				model,
				explorerModel: { ...model, id: "gpt-5.6-luna", name: "GPT-5.6 Luna" },
				apiKey: "secret",
			}),
			agentFactory: (options) => {
				received = {
					modelId: options.model?.id,
					explorerModelId: options.explorerModel?.id,
					apiKey: options.apiKey,
				};
				return { searchDocuments: async () => cannedResponse() };
			},
		});

		expect(code).toBe(0);
		expect(received).toEqual({
			modelId: "gpt-5.6-sol",
			explorerModelId: "gpt-5.6-luna",
			apiKey: "secret",
		});
	});

	it("forwards provider-scoped credentials without assigning the explorer key to apiKey", async () => {
		const model: Model<"openai-responses"> = {
			id: "gpt-5.6-sol",
			name: "GPT-5.6 Sol",
			api: "openai-responses",
			provider: "openai",
			baseUrl: "https://api.openai.com/v1",
			reasoning: true,
			input: ["text", "image"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 400_000,
			maxTokens: 128_000,
		};
		const explorerModel = { ...model, id: "gpt-5.6-luna", name: "GPT-5.6 Luna", provider: "test-proxy" };
		let received: { apiKey?: string; providerApiKeys?: Readonly<Record<string, string>> } | undefined;
		const { ctx } = makeCtx({ positionals: ["anything"], cwd: tmpDir });

		const code = await runSearch(ctx, {
			modelResolver: () => ({
				model,
				explorerModel,
				providerApiKeys: { "test-proxy": "explorer-secret" },
			}),
			agentFactory: (options) => {
				received = { apiKey: options.apiKey, providerApiKeys: options.providerApiKeys };
				return { searchDocuments: async () => cannedResponse() };
			},
		});

		expect(code).toBe(0);
		expect(received?.apiKey).toBeUndefined();
		expect(received?.providerApiKeys).toEqual({ "test-proxy": "explorer-secret" });
	});

	it("rejects an unknown configured model before constructing the agent", async () => {
		const configPath = join(tmpDir, "unknown-model.json");
		writeFileSync(
			configPath,
			JSON.stringify({
				searchPaths: [tmpDir],
				workspacePath: tmpDir,
				memoryPath: join(tmpDir, "memory.json"),
				agents: {
					orchestrator: { provider: "openai", id: "missing-before-construction" },
					explorer: { provider: "openai", id: "gpt-4o" },
				},
			}),
			"utf8",
		);
		let constructed = false;
		const { ctx, stderr } = makeCtx({
			positionals: ["anything"],
			flags: { config: configPath },
			json: true,
			cwd: tmpDir,
		});

		const code = await runSearch(ctx, {
			modelResolver: resolveAgentModel,
			agentFactory: () => {
				constructed = true;
				return { searchDocuments: async () => cannedResponse() };
			},
		});

		expect(code).toBe(2);
		expect(constructed).toBe(false);
		const parsed = JSON.parse(stderr[0]);
		expect(parsed.error).toContain("openai/missing-before-construction");
		// Model resolution failure surfaces a health hint.
		expect(parsed.hint).toEqual({
			command: "autorag health",
			reason: "model_resolution",
			message: "Run autorag health to diagnose model/provider and explorer subagent setup.",
		});
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

	it("returns exit 1 when the agent throws a generic runtime error with no hint", async () => {
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
		// Generic runtime error must not carry a health hint.
		expect(parsed.hint).toBeUndefined();
	});

	it("surfaces a subagent_failed hint when searchDocuments throws a pi-subagents error", async () => {
		const deps: SearchDeps = {
			agentFactory: () => ({
				searchDocuments: async () => {
					throw new Error("Mandatory pi-subagents extension failed to load: missing subagent capability");
				},
			}),
		};
		const { ctx, stderr } = makeCtx({ positionals: ["query"], json: true });
		const code = await runSearch(ctx, deps);
		expect(code).toBe(1);
		const parsed = JSON.parse(stderr[0]);
		expect(parsed.ok).toBe(false);
		expect(parsed.hint).toEqual({
			command: "autorag health",
			reason: "subagent_failed",
			message: "Run autorag health to diagnose model/provider and explorer subagent setup.",
		});
	});

	it("surfaces a subagent_failed hint when searchDocuments throws an autorag-explorer error", async () => {
		const deps: SearchDeps = {
			agentFactory: () => ({
				searchDocuments: async () => {
					throw new Error("AutoRAG requires a successful autorag-explorer subagent call before curation");
				},
			}),
		};
		const { ctx, stderr } = makeCtx({ positionals: ["query"], json: true });
		const code = await runSearch(ctx, deps);
		expect(code).toBe(1);
		const parsed = JSON.parse(stderr[0]);
		expect(parsed.hint).toEqual({
			command: "autorag health",
			reason: "subagent_failed",
			message: "Run autorag health to diagnose model/provider and explorer subagent setup.",
		});
	});

	it("surfaces an auth_missing hint when searchDocuments throws a 401 error", async () => {
		const deps: SearchDeps = {
			agentFactory: () => ({
				searchDocuments: async () => {
					throw new Error("Request failed with status 401 Unauthorized");
				},
			}),
		};
		const { ctx, stderr } = makeCtx({ positionals: ["query"], json: true });
		const code = await runSearch(ctx, deps);
		expect(code).toBe(1);
		const parsed = JSON.parse(stderr[0]);
		expect(parsed.hint).toEqual({
			command: "autorag health",
			reason: "auth_missing",
			message: "Run autorag health to diagnose model/provider and explorer subagent setup.",
		});
	});

	it("surfaces a provider_unreachable hint when searchDocuments throws an ENOTFOUND error", async () => {
		const deps: SearchDeps = {
			agentFactory: () => ({
				searchDocuments: async () => {
					const err = new Error("getaddrinfo ENOTFOUND api.example.com");
					(err as { code?: string }).code = "ENOTFOUND";
					throw err;
				},
			}),
		};
		const { ctx, stderr } = makeCtx({ positionals: ["query"], json: true });
		const code = await runSearch(ctx, deps);
		expect(code).toBe(1);
		const parsed = JSON.parse(stderr[0]);
		expect(parsed.hint).toEqual({
			command: "autorag health",
			reason: "provider_unreachable",
			message: "Run autorag health to diagnose model/provider and explorer subagent setup.",
		});
	});

	it("surfaces a timeout hint when searchDocuments throws an AbortError", async () => {
		const deps: SearchDeps = {
			agentFactory: () => ({
				searchDocuments: async () => {
					const err = new Error("The operation was aborted");
					err.name = "AbortError";
					throw err;
				},
			}),
		};
		const { ctx, stderr } = makeCtx({ positionals: ["query"], json: true });
		const code = await runSearch(ctx, deps);
		expect(code).toBe(1);
		const parsed = JSON.parse(stderr[0]);
		expect(parsed.hint).toEqual({
			command: "autorag health",
			reason: "timeout",
			message: "Run autorag health to diagnose model/provider and explorer subagent setup.",
		});
	});

	it("renders the hint message as a second line in human output", async () => {
		const deps: SearchDeps = {
			agentFactory: () => ({
				searchDocuments: async () => {
					throw new Error("Mandatory pi-subagents extension failed to load");
				},
			}),
		};
		const { ctx, stderr } = makeCtx({ positionals: ["query"], json: false });
		const code = await runSearch(ctx, deps);
		expect(code).toBe(1);
		const text = stderr.join("\n");
		expect(text).toContain("error: Mandatory pi-subagents extension failed to load");
		expect(text).toContain("Run autorag health to diagnose model/provider and explorer subagent setup.");
	});

	it("does not add a hint line for empty query in human output", async () => {
		const { ctx, stderr } = makeCtx({ positionals: [], json: false });
		const code = await runSearch(ctx, { agentFactory: () => ({ searchDocuments: async () => cannedResponse() }) });
		expect(code).toBe(2);
		const text = stderr.join("\n");
		expect(text).toContain("Usage: autorag search");
		expect(text).not.toContain("Run autorag health");
	});

	it("does not add a hint field for empty query in JSON output", async () => {
		const { ctx, stderr } = makeCtx({ positionals: [], json: true });
		const code = await runSearch(ctx, { agentFactory: () => ({ searchDocuments: async () => cannedResponse() }) });
		expect(code).toBe(2);
		const parsed = JSON.parse(stderr[0]);
		expect(parsed.ok).toBe(false);
		expect(parsed.hint).toBeUndefined();
	});
});

// ---------------------------------------------------------------------------
// classifySearchHealthHint unit tests
// ---------------------------------------------------------------------------

describe("classifySearchHealthHint", () => {
	const HINT_MESSAGE = "Run autorag health to diagnose model/provider and explorer subagent setup.";

	it("classifies ConfigError as model_resolution", () => {
		const hint = classifySearchHealthHint(new ConfigError("Unknown configured model: openai/missing"));
		expect(hint).toEqual({ command: "autorag health", reason: "model_resolution", message: HINT_MESSAGE });
	});

	it("classifies 'no model configured' as model_resolution", () => {
		const hint = classifySearchHealthHint(new Error("No model configured. Provide --model-provider and --model-id."));
		expect(hint?.reason).toBe("model_resolution");
	});

	it("classifies 'unknown configured' as model_resolution", () => {
		const hint = classifySearchHealthHint(new Error("Unknown configured explorer model: openai/missing"));
		expect(hint?.reason).toBe("model_resolution");
	});

	it("classifies pi-subagents messages as subagent_failed", () => {
		const hint = classifySearchHealthHint(new Error("Mandatory pi-subagents extension failed to load"));
		expect(hint?.reason).toBe("subagent_failed");
	});

	it("classifies autorag-explorer messages as subagent_failed", () => {
		const hint = classifySearchHealthHint(new Error("AutoRAG requires a successful autorag-explorer subagent call"));
		expect(hint?.reason).toBe("subagent_failed");
	});

	it("classifies concurrent Pi sessions as subagent_failed", () => {
		const hint = classifySearchHealthHint(
			new Error("AutoRAG cannot create concurrent Pi sessions with different child environment routing"),
		);
		expect(hint?.reason).toBe("subagent_failed");
	});

	it("classifies concurrent AutoRAG session busy as subagent_failed", () => {
		const hint = classifySearchHealthHint(new Error("concurrent AutoRAG session busy"));
		expect(hint?.reason).toBe("subagent_failed");
	});

	it("classifies 401 as auth_missing", () => {
		const hint = classifySearchHealthHint(new Error("Request failed with 401 Unauthorized"));
		expect(hint?.reason).toBe("auth_missing");
	});

	it("classifies 403 as auth_missing", () => {
		const hint = classifySearchHealthHint(new Error("Request failed with 403 Forbidden"));
		expect(hint?.reason).toBe("auth_missing");
	});

	it("classifies 'api key' as auth_missing", () => {
		const hint = classifySearchHealthHint(new Error("Missing API key for provider"));
		expect(hint?.reason).toBe("auth_missing");
	});

	it("classifies ENOTFOUND error code as provider_unreachable", () => {
		const err = new Error("getaddrinfo ENOTFOUND api.example.com");
		(err as { code?: string }).code = "ENOTFOUND";
		const hint = classifySearchHealthHint(err);
		expect(hint?.reason).toBe("provider_unreachable");
	});

	it("classifies ECONNREFUSED error code as provider_unreachable", () => {
		const err = new Error("connect ECONNREFUSED 127.0.0.1:443");
		(err as { code?: string }).code = "ECONNREFUSED";
		const hint = classifySearchHealthHint(err);
		expect(hint?.reason).toBe("provider_unreachable");
	});

	it("classifies ETIMEDOUT error code as provider_unreachable", () => {
		const err = new Error("connect ETIMEDOUT");
		(err as { code?: string }).code = "ETIMEDOUT";
		const hint = classifySearchHealthHint(err);
		expect(hint?.reason).toBe("provider_unreachable");
	});

	it("classifies 'provider unreachable' as provider_unreachable", () => {
		const hint = classifySearchHealthHint(new Error("Provider unreachable: connection refused"));
		expect(hint?.reason).toBe("provider_unreachable");
	});

	it("classifies AbortError as timeout", () => {
		const err = new Error("The operation was aborted");
		err.name = "AbortError";
		const hint = classifySearchHealthHint(err);
		expect(hint?.reason).toBe("timeout");
	});

	it("classifies 'timed out' as timeout", () => {
		const hint = classifySearchHealthHint(new Error("The request timed out after 10000ms"));
		expect(hint?.reason).toBe("timeout");
	});

	it("returns undefined for a generic Error", () => {
		const hint = classifySearchHealthHint(new Error("boom at runtime"));
		expect(hint).toBeUndefined();
	});

	it("returns undefined for retrieval/index errors", () => {
		const hint = classifySearchHealthHint(new Error("BM25 index missing"));
		expect(hint).toBeUndefined();
	});

	it("returns undefined for datasource errors", () => {
		const hint = classifySearchHealthHint(new Error("datasource katok failed to poll"));
		expect(hint).toBeUndefined();
	});

	it("returns undefined for empty-string error", () => {
		const hint = classifySearchHealthHint(new Error(""));
		expect(hint).toBeUndefined();
	});
});

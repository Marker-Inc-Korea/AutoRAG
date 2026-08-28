import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Model } from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { SearchDocumentsResponse } from "../../src/agent/search-documents.ts";
import { classifySearchHealthHint, runSearch } from "../../src/cli/commands/search.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import { ConfigError } from "../../src/cli/config.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-search-cli-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function context(positionals: string[], flags: CommandContext["flags"] = {}) {
	const stdout: string[] = [];
	const stderr: string[] = [];
	const ctx: CommandContext = {
		positionals,
		flags,
		json: true,
		debug: false,
		cwd: root,
		stdout: (line) => stdout.push(line),
		stderr: (line) => stderr.push(line),
	};
	return { ctx, stdout, stderr };
}

const response: SearchDocumentsResponse = {
	sessionId: "session",
	query: "query",
	answer: "[1] answer",
	results: [
		{
			number: 1,
			title: "A",
			summary: "answer",
			evidence: [{ excerpt: "answer" }],
			confidence: 1,
			feedbackId: "session:1",
		},
	],
	searched: 1,
	warnings: [],
	diagnostics: [],
};

function model(): Model<"openai-responses"> {
	return {
		id: "single-agent",
		name: "Single Agent",
		api: "openai-responses",
		provider: "test",
		baseUrl: "https://example.test/v1",
		reasoning: true,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 1000,
		maxTokens: 100,
	};
}

describe("runSearch", () => {
	it("reports usage for an empty query", async () => {
		const { ctx, stderr } = context([]);
		expect(await runSearch(ctx, { agentFactory: () => ({ searchDocuments: async () => response }) })).toBe(2);
		expect(stderr.join("\n")).toContain("Usage");
	});

	it("constructs the agent with one model and credentials", async () => {
		let received: { model?: string; apiKey?: string } = {};
		const { ctx, stdout } = context(["query"]);
		expect(
			await runSearch(ctx, {
				modelResolver: () => ({ model: model(), apiKey: "secret" }),
				agentFactory: (options) => {
					received = { model: options.model?.id, apiKey: options.apiKey };
					return { searchDocuments: async () => response };
				},
			}),
		).toBe(0);
		expect(received).toEqual({ model: "single-agent", apiKey: "secret" });
		expect(JSON.parse(stdout[0]).answer).toBe("[1] answer");
	});

	it("forwards retrieval options", async () => {
		let options: unknown;
		const { ctx } = context(["query"], { "top-k": "3", scope: "/docs" });
		await runSearch(ctx, {
			agentFactory: () => ({
				searchDocuments: async (_query, received) => {
					options = received;
					return response;
				},
			}),
		});
		expect(options).toMatchObject({ topK: 3, scope: "/docs" });
	});

	it("does not fail agent construction for unknown datasource names", async () => {
		const configPath = join(root, "config.json");
		writeFileSync(
			configPath,
			JSON.stringify({
				searchPaths: [root],
				workspacePath: root,
				datasources: { "discord-nomadamas": {}, "slack-local": {} },
			}),
		);
		let received: { startupDiagnostics?: unknown; datasourceSkills?: unknown } = {};
		const { ctx, stdout, stderr } = context(["find a local document"], { config: configPath });
		const code = await runSearch(ctx, {
			agentFactory: (options) => {
				received = {
					startupDiagnostics: options.startupDiagnostics,
					datasourceSkills: options.datasourceSkills,
				};
				return { searchDocuments: async () => response };
			},
		});
		expect(code).toBe(0);
		expect(stderr.join("\n")).not.toMatch(/Unknown datasource/);
		expect(received.datasourceSkills ?? []).toEqual([]);
		expect(received.startupDiagnostics).toEqual([
			expect.objectContaining({
				code: "unknown-datasource-skill",
				severity: "warning",
				source: "datasources",
			}),
		]);
		expect(JSON.parse(stdout[0]).answer).toBe("[1] answer");
	});
});

describe("classifySearchHealthHint", () => {
	it("classifies model, auth, provider, and timeout failures", () => {
		expect(classifySearchHealthHint(new ConfigError("bad config"))?.reason).toBe("model_resolution");
		expect(classifySearchHealthHint(new Error("401 unauthorized"))?.reason).toBe("auth_missing");
		expect(classifySearchHealthHint(new Error("ENOTFOUND provider"))?.reason).toBe("provider_unreachable");
		expect(classifySearchHealthHint(new Error("request timed out"))?.reason).toBe("timeout");
	});
});

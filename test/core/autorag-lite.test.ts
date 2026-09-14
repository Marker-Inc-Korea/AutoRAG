import { mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { AutoRAGResultsDetails } from "../../src/agent/emit-results-tool.ts";
import { createAutoRAGLite } from "../../src/core.ts";
import * as publicApi from "../../src/index.ts";

type LiteFactoryProbe = (options: unknown) => unknown;

type LiteProbe = {
	readonly config: unknown;
	readonly refresh: unknown;
	readonly getRefreshStatus: unknown;
	readonly getRetrievalEngine: unknown;
	readonly retrieve: unknown;
	readonly recordStructuredResultsSession: unknown;
	readonly getResultRegistry: unknown;
	readonly recordFeedbackByNumbers: unknown;
};

function isLiteFactory(value: unknown): value is LiteFactoryProbe {
	return typeof value === "function";
}

function hasLiteSurface(value: unknown): value is LiteProbe {
	if (typeof value !== "object" || value === null) return false;
	return [
		"config",
		"refresh",
		"getRefreshStatus",
		"getRetrievalEngine",
		"retrieve",
		"recordStructuredResultsSession",
		"getResultRegistry",
		"recordFeedbackByNumbers",
	].every((key) => key in value);
}

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-lite-core-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

describe("public AutoRAG-lite core facade", () => {
	it("exports a model-free lifecycle, retrieval, and persistence surface", () => {
		const factory = Reflect.get(publicApi, "createAutoRAGLite");
		expect(isLiteFactory(factory)).toBe(true);
		if (!isLiteFactory(factory)) return;

		const configPath = join(root, "config.json");
		writeFileSync(
			configPath,
			JSON.stringify({
				searchPaths: [root],
				workspacePath: root,
				memoryPath: join(root, "memory.json"),
				minSync: false,
				jikji: false,
			}),
		);

		const lite = factory({ flags: { config: configPath }, cwd: root });
		expect(hasLiteSurface(lite)).toBe(true);
	});

	it("preserves flag over environment over file config precedence", () => {
		const fileSearch = join(root, "file-search");
		const envSearch = join(root, "env-search");
		const flagSearch = join(root, "flag-search");
		const fileWorkspace = join(root, "file-workspace");
		const envWorkspace = join(root, "env-workspace");
		const flagWorkspace = join(root, "flag-workspace");
		for (const path of [fileSearch, envSearch, flagSearch, fileWorkspace, envWorkspace, flagWorkspace]) {
			mkdirSync(path, { recursive: true });
		}
		const configPath = join(root, "config.json");
		writeFileSync(
			configPath,
			JSON.stringify({ searchPaths: [fileSearch], workspacePath: fileWorkspace, minSync: false, jikji: false }),
		);

		const fromEnvironment = createAutoRAGLite({
			flags: { config: configPath },
			env: {
				AUTORAG_SEARCH_PATHS: envSearch,
				AUTORAG_WORKSPACE: envWorkspace,
				HOME: root,
			},
			cwd: root,
		});
		expect(fromEnvironment.config.searchPaths).toEqual([envSearch]);
		expect(fromEnvironment.config.workspacePath).toBe(envWorkspace);

		const fromFlags = createAutoRAGLite({
			flags: { config: configPath, "search-paths": flagSearch, workspace: flagWorkspace },
			env: {
				AUTORAG_SEARCH_PATHS: envSearch,
				AUTORAG_WORKSPACE: envWorkspace,
				HOME: root,
			},
			cwd: root,
		});
		expect(fromFlags.config.searchPaths).toEqual([flagSearch]);
		expect(fromFlags.config.workspacePath).toBe(flagWorkspace);
	});

	it("pins symlinked source roots and delegates deterministic retrieval", async () => {
		const docs = join(root, "docs");
		const alias = join(root, "docs-alias");
		mkdirSync(docs, { recursive: true });
		writeFileSync(join(docs, "note.txt"), "pinned source\n");
		symlinkSync(docs, alias);
		const configPath = join(root, "config.json");
		writeFileSync(
			configPath,
			JSON.stringify({
				searchPaths: [alias],
				workspacePath: root,
				memoryPath: join(root, "memory.json"),
				minSync: false,
				jikji: false,
			}),
		);
		const lite = createAutoRAGLite({ flags: { config: configPath }, cwd: root });
		const engine = lite.getRetrievalEngine();
		engine.register({
			describe: () => ({
				name: "fixture",
				type: "posix",
				description: "fixture",
				status: "active",
				capabilities: [],
			}),
			retrieve: async () => [{ id: "fixture:1", content: "pinned source", source: docs, score: 1, metadata: {} }],
		});
		const retrieved = await lite.retrieve("pinned");
		expect(retrieved.results[0]?.source).toBe(docs);
	});

	it("persists an opaque report and numbered feedback without reading its source", () => {
		const configPath = join(root, "config.json");
		writeFileSync(
			configPath,
			JSON.stringify({
				searchPaths: [root],
				workspacePath: root,
				memoryPath: join(root, "memory.json"),
				minSync: false,
				jikji: false,
			}),
		);
		const lite = createAutoRAGLite({ flags: { config: configPath }, cwd: root });
		const details: AutoRAGResultsDetails = {
			answer: "[1] opaque result",
			results: [
				{
					number: 1,
					title: "Opaque result",
					summary: "Submitted as data",
					evidence: [{ excerpt: "opaque evidence" }],
					confidence: 0.75,
				},
			],
			mapping: [
				{
					number: 1,
					source: "file:///do-not-read",
					method: "fixture",
					content: "opaque source content",
					evidenceRefs: [],
				},
			],
			warnings: [],
		};
		const response = lite.recordReport("opaque query", details);
		lite.recordFeedbackByNumbers(response.sessionId, [1]);
		const memory = lite.getMemorySchema();
		expect(response.results[0]?.source).toBe("file:///do-not-read");
		expect(memory.curatedResults[0]?.sessionId).toBe(response.sessionId);
		expect(memory.evidenceChunks[0]?.source).toBe("file:///do-not-read");
		expect(memory.feedbackSignals.length).toBeGreaterThan(0);
	});

	it("resolves trusted config without persisting credentials", () => {
		const factory = Reflect.get(publicApi, "createAutoRAGLite");
		expect(isLiteFactory(factory)).toBe(true);
		if (!isLiteFactory(factory)) return;

		const configPath = join(root, "config.json");
		const original = JSON.stringify({
			searchPaths: [root],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			model: { provider: "fixture", id: "model", baseUrl: "https://model.invalid", apiKeyEnv: "FIXTURE_SECRET" },
			minSync: false,
			jikji: false,
		});
		writeFileSync(configPath, original);

		const lite = factory({
			flags: { config: configPath },
			env: { FIXTURE_SECRET: "do-not-persist", HOME: root },
			cwd: root,
		});
		expect(hasLiteSurface(lite)).toBe(true);
		expect(readFileSync(configPath, "utf8")).toBe(original);
	});
});

describe("public config exports", () => {
	it("exports config resolution and atomic write entry points", () => {
		expect(typeof Reflect.get(publicApi, "resolveConfig")).toBe("function");
		expect(typeof Reflect.get(publicApi, "resolveConfigReadOnly")).toBe("function");
		expect(typeof Reflect.get(publicApi, "buildAgentOptions")).toBe("function");
		expect(typeof Reflect.get(publicApi, "writeConfigObject")).toBe("function");
	});
});

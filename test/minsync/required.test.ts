import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { buildAgentOptions, ConfigError, normalizeIndexingConfig } from "../../src/cli/config.ts";
import { MinSyncRequiredError } from "../../src/minsync/errors.ts";
import { MinSyncVectorMethod } from "../../src/minsync/method.ts";
import { RetrievalEngine } from "../../src/retrieval/engine.ts";
import { ParallelRetriever } from "../../src/retrieval/merger.ts";
import type { RetrievalMethod } from "../../src/retrieval/types.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-minsync-required-"));
	mkdirSync(join(root, "docs"), { recursive: true });
	writeFileSync(join(root, "docs", "budget.md"), "# Budget\n\nThe Q3 budget was approved by Mina Park.\n");
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function agentWithMissingBinary(): AutoRAGAgent {
	return new AutoRAGAgent({
		searchPaths: [join(root, "docs")],
		memoryPath: join(root, "memory.json"),
		workspacePath: root,
		jikji: false,
		everything: false,
		fsearch: false,
		minSync: { binaryPath: join(root, "missing-minsync"), autoInstall: false },
	});
}

function throwingMethod(name: string, error: Error): RetrievalMethod {
	return {
		describe: () => ({ name, type: "vector", description: "", status: "active", capabilities: [] }),
		retrieve: async () => {
			throw error;
		},
	};
}

describe("config: MinSync cannot be turned off", () => {
	it("rejects `minSync: false`", () => {
		expect(() => normalizeIndexingConfig({ minSync: false as never })).toThrow(ConfigError);
		expect(() => normalizeIndexingConfig({ minSync: false as never })).toThrow(/MinSync is required/);
	});

	it("rejects `minSync.enabled: false`", () => {
		expect(() => normalizeIndexingConfig({ minSync: { enabled: false as never } })).toThrow(/MinSync is required/);
	});

	it("still accepts `enabled: true` so existing configs keep loading", () => {
		expect(normalizeIndexingConfig({ minSync: { enabled: true } }).minSync.enabled).toBe(true);
	});

	it("always hands MinSync options to the agent", () => {
		const options = buildAgentOptions({
			searchPaths: ["."],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: { enabled: true, autoInstall: false },
		});
		expect(options.minSync).toEqual({ autoInstall: false });
	});
});

describe("agent: a missing MinSync is an error, not a degraded answer", () => {
	it("refuses to construct with `minSync: false`", () => {
		expect(
			() =>
				new AutoRAGAgent({
					searchPaths: [join(root, "docs")],
					memoryPath: join(root, "memory.json"),
					workspacePath: root,
					minSync: false as never,
				}),
		).toThrow(MinSyncRequiredError);
	});

	it("fails searchDocuments when the binary cannot be resolved", async () => {
		await expect(agentWithMissingBinary().searchDocuments("who approved the budget")).rejects.toBeInstanceOf(
			MinSyncRequiredError,
		);
	});

	it("fails refresh when the binary cannot be resolved", async () => {
		await expect(agentWithMissingBinary().refresh(true, { methods: ["minsync"] })).rejects.toBeInstanceOf(
			MinSyncRequiredError,
		);
	});

	it("fails syncMinSync when the binary cannot be resolved", async () => {
		await expect(agentWithMissingBinary().syncMinSync()).rejects.toBeInstanceOf(MinSyncRequiredError);
	});

	it("does not need the binary to refresh only the parsed mirrors", async () => {
		const result = await agentWithMissingBinary().refresh(true, { methods: ["parsed"] });
		expect(result.minsync).toBeUndefined();
	});
});

describe("method: no empty result stands in for a missing binary", () => {
	it("rejects retrieve with MinSyncRequiredError", async () => {
		const method = new MinSyncVectorMethod({
			root,
			workspacePath: join(root, "minsync"),
			binaryPath: join(root, "missing-minsync"),
			autoInstall: false,
		});
		await expect(method.retrieve("anything", { topK: 3 })).rejects.toBeInstanceOf(MinSyncRequiredError);
	});

	it("rejects sync with MinSyncRequiredError instead of returning ok:false", async () => {
		const method = new MinSyncVectorMethod({
			root,
			workspacePath: join(root, "minsync"),
			binaryPath: join(root, "missing-minsync"),
			autoInstall: false,
		});
		await expect(method.sync()).rejects.toBeInstanceOf(MinSyncRequiredError);
	});

	it("reports a failed auto-install as MinSyncRequiredError carrying the installer's message", async () => {
		const savedPath = process.env.PATH;
		process.env.PATH = "/nonexistent";
		try {
			const method = new MinSyncVectorMethod({
				root,
				workspacePath: join(root, "minsync"),
				autoInstall: true,
				installer: {
					cargoInstaller: async () => {
						throw new Error("cargo exploded");
					},
					releaseProvider: async () => {
						throw new Error("release feed exploded");
					},
				},
			});
			await expect(method.sync()).rejects.toThrow(/installing the `minsync` binary failed/);
		} finally {
			process.env.PATH = savedPath;
		}
	});
});

describe("retrieval pipeline: only the absence of MinSync is fatal", () => {
	it("rethrows MinSyncRequiredError from a method instead of skipping it", async () => {
		const required = new MinSyncRequiredError("MinSync is required but missing");
		const retriever = new ParallelRetriever();
		await expect(retriever.retrieveWithDiagnostics([throwingMethod("minsync", required)], "q", {})).rejects.toBe(
			required,
		);
		await expect(retriever.retrieve([throwingMethod("minsync", required)], "q", {})).rejects.toBe(required);
	});

	it("still skips an ordinary MinSync query failure and reports it as a method failure", async () => {
		const retriever = new ParallelRetriever();
		const { diagnostics } = await retriever.retrieveWithDiagnostics(
			[throwingMethod("minsync", new Error("another sync is in progress"))],
			"q",
			{},
		);
		expect(diagnostics).toHaveLength(1);
		expect(diagnostics[0]?.code).toBe("retrieval-method-failed");
		expect(diagnostics[0]?.reason).toContain("another sync is in progress");
	});

	it("fails RetrievalEngine.retrieve when MinSync is missing", async () => {
		const engine = new RetrievalEngine();
		engine.register(throwingMethod("minsync", new MinSyncRequiredError("MinSync is required but missing")));
		await expect(engine.retrieve("q")).rejects.toBeInstanceOf(MinSyncRequiredError);
	});

	it("does not invent a diagnostic for a MinSync method that simply returned nothing", async () => {
		const engine = new RetrievalEngine();
		engine.register({
			describe: () => ({ name: "minsync", type: "vector", description: "", status: "active", capabilities: [] }),
			retrieve: async () => [],
		});
		const { diagnostics, unsearched } = await engine.retrieve("q");
		expect(diagnostics).toEqual([]);
		expect(unsearched).toEqual([]);
	});
});

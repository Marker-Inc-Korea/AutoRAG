import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { runInit } from "../../src/cli/commands/init.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import { buildAgentOptions, resolveConfig } from "../../src/cli/config.ts";

let root: string;
let previousHome: string | undefined;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-cli-init-"));
	previousHome = process.env.HOME;
	process.env.HOME = join(root, "home");
});

afterEach(() => {
	if (previousHome === undefined) delete process.env.HOME;
	else process.env.HOME = previousHome;
	rmSync(root, { recursive: true, force: true });
});

function homeConfigPath(): string {
	return join(process.env.HOME as string, ".autorag", "config.json");
}

function makeCtx(overrides: Partial<CommandContext> = {}): CommandContext {
	return {
		positionals: [],
		flags: {},
		json: false,
		debug: false,
		cwd: root,
		stdout: () => {},
		stderr: () => {},
		...overrides,
	};
}

describe("runInit", () => {
	it("writes one model setting", async () => {
		const code = await runInit(
			makeCtx({
				flags: {
					"model-provider": "test-provider",
					"model-id": "single-agent",
				},
			}),
		);
		expect(code).toBe(0);
		const config = JSON.parse(readFileSync(homeConfigPath(), "utf8"));
		expect(config.model).toEqual({ provider: "test-provider", id: "single-agent" });
	});

	it("does not write private model defaults without model flags", async () => {
		const code = await runInit(makeCtx());

		expect(code).toBe(0);
		const config = JSON.parse(readFileSync(homeConfigPath(), "utf8"));
		expect(config.model).toBeUndefined();
		expect(config.jikji).toEqual({});
		expect(config.excludeExactDuplicates).toBe(true);
	});

	it("preserves a migrated legacy model unless explicitly overridden", async () => {
		writeFileSync(
			join(root, "autorag.config.json"),
			JSON.stringify({
				model: { provider: "legacy-provider", id: "legacy-model" },
			}),
			"utf8",
		);

		const code = await runInit(
			makeCtx({
				flags: {
					"model-provider": "override-provider",
					"model-id": "override-model",
				},
			}),
		);

		expect(code).toBe(0);
		const config = JSON.parse(readFileSync(homeConfigPath(), "utf8"));
		expect(config.model).toEqual({ provider: "override-provider", id: "override-model" });
	});

	it("writes the default config under ~/.autorag and reports its absolute path", async () => {
		const home = join(root, "home");
		const stdout: string[] = [];
		const code = await runInit(makeCtx({ json: true, stdout: (line) => stdout.push(line) }));
		expect(code).toBe(0);
		const configPath = join(home, ".autorag", "config.json");
		expect(existsSync(configPath)).toBe(true);
		expect(JSON.parse(stdout[0])).toEqual({ ok: true, wrote: [configPath] });
	});

	it("keeps init-approved search roots stable when the library resolves config from another cwd", async () => {
		const initCwd = join(root, "project");
		const otherCwd = join(root, "other");
		const approvedRoot = join(initCwd, "docs");
		mkdirSync(approvedRoot, { recursive: true });
		mkdirSync(otherCwd, { recursive: true });

		const code = await runInit(
			makeCtx({
				cwd: initCwd,
				flags: { "search-paths": "docs" },
			}),
		);
		const persisted = JSON.parse(readFileSync(homeConfigPath(), "utf8"));
		const resolved = resolveConfig({
			flags: {},
			env: { HOME: process.env.HOME },
			cwd: otherCwd,
		});

		expect(code).toBe(0);
		expect(persisted.searchPaths).toEqual([approvedRoot]);
		expect(resolved.searchPaths).toEqual([approvedRoot]);
		expect(buildAgentOptions(resolved).searchPaths).toEqual([approvedRoot]);
	});

	it("matches automatic legacy migration when the legacy workspace differs from ctx.cwd", async () => {
		const legacyCwd = join(root, "legacy-cwd");
		const automaticHome = join(root, "automatic-home");
		const expectedWorkspace = join(legacyCwd, "workspace");
		const expectedSearch = join(expectedWorkspace, "docs");
		const expectedMemory = join(expectedWorkspace, "memory.json");
		mkdirSync(legacyCwd, { recursive: true });
		writeFileSync(
			join(legacyCwd, "autorag.config.json"),
			JSON.stringify({ workspacePath: "workspace", searchPaths: ["docs"], memoryPath: "memory.json" }),
			"utf8",
		);

		const initCode = await runInit(makeCtx({ cwd: legacyCwd }));
		const initPersisted = JSON.parse(readFileSync(homeConfigPath(), "utf8"));
		const automaticResolved = resolveConfig({ flags: {}, env: { HOME: automaticHome }, cwd: legacyCwd });
		const automaticPersisted = JSON.parse(readFileSync(join(automaticHome, ".autorag", "config.json"), "utf8"));
		const expected = {
			workspacePath: expectedWorkspace,
			searchPaths: [expectedSearch],
			memoryPath: expectedMemory,
		};

		expect(initCode).toBe(0);
		expect(initPersisted).toMatchObject(expected);
		expect(automaticResolved).toMatchObject(expected);
		expect(automaticPersisted).toMatchObject(expected);
	});

	it("keeps explicit init path flags over inherited legacy paths", async () => {
		const legacyCwd = join(root, "legacy-cwd");
		const explicitWorkspace = join(root, "explicit-workspace");
		const explicitSearch = join(root, "explicit-docs");
		const explicitMemory = join(root, "explicit-memory.json");
		mkdirSync(legacyCwd, { recursive: true });
		writeFileSync(
			join(legacyCwd, "autorag.config.json"),
			JSON.stringify({ workspacePath: "workspace", searchPaths: ["docs"], memoryPath: "memory.json" }),
			"utf8",
		);

		const code = await runInit(
			makeCtx({
				cwd: legacyCwd,
				flags: {
					workspace: explicitWorkspace,
					"search-paths": explicitSearch,
					"memory-path": explicitMemory,
				},
			}),
		);

		expect(code).toBe(0);
		expect(JSON.parse(readFileSync(homeConfigPath(), "utf8"))).toMatchObject({
			workspacePath: explicitWorkspace,
			searchPaths: [explicitSearch],
			memoryPath: explicitMemory,
		});
	});

	it("writes autorag.config.json with the provided keys folded in", async () => {
		const stdout: string[] = [];
		const code = await runInit(
			makeCtx({
				flags: {
					"search-paths": "docs, notes",
					workspace: root,
					"memory-path": join(root, "memory.json"),
					"model-provider": "openai",
					"model-id": "gpt-4o",
				},
				stdout: (line) => stdout.push(line),
			}),
		);

		expect(code).toBe(0);
		const file = homeConfigPath();
		expect(existsSync(file)).toBe(true);

		const config = JSON.parse(readFileSync(file, "utf-8"));
		expect(config.searchPaths).toEqual([join(root, "docs"), join(root, "notes")]);
		expect(config.workspacePath).toBe(root);
		expect(config.memoryPath).toBe(join(root, "memory.json"));
		expect(config.model).toEqual({ provider: "openai", id: "gpt-4o" });
	});

	it("returns 2 and writes an error when the config already exists without --force", async () => {
		mkdirSync(join(process.env.HOME as string, ".autorag"), { recursive: true });
		writeFileSync(homeConfigPath(), `${JSON.stringify({ existing: true })}\n`);

		const stderr: string[] = [];
		const code = await runInit(
			makeCtx({
				flags: { "search-paths": "docs" },
				stderr: (line) => stderr.push(line),
			}),
		);

		expect(code).toBe(2);
		expect(stderr.length).toBeGreaterThan(0);
		// The pre-existing file must be untouched.
		const config = JSON.parse(readFileSync(homeConfigPath(), "utf-8"));
		expect(config.existing).toBe(true);
	});

	it("overwrites the existing config when --force is set", async () => {
		mkdirSync(join(process.env.HOME as string, ".autorag"), { recursive: true });
		writeFileSync(homeConfigPath(), `${JSON.stringify({ old: true })}\n`);

		const code = await runInit(
			makeCtx({
				flags: { "search-paths": "new", force: true },
			}),
		);

		expect(code).toBe(0);
		const config = JSON.parse(readFileSync(homeConfigPath(), "utf-8"));
		expect(config.searchPaths).toEqual([join(root, "new")]);
		expect(config.old).toBeUndefined();
	});

	it("emits a JSON envelope with the written filename in --json mode", async () => {
		const stdout: string[] = [];
		const code = await runInit(
			makeCtx({
				flags: { "search-paths": "docs" },
				json: true,
				stdout: (line) => stdout.push(line),
			}),
		);

		expect(code).toBe(0);
		expect(stdout).toHaveLength(1);
		const envelope = JSON.parse(stdout[0]);
		expect(envelope.ok).toBe(true);
		expect(envelope.wrote).toEqual([homeConfigPath()]);
	});

	it("emits a human line in non-json mode", async () => {
		const stdout: string[] = [];
		const code = await runInit(
			makeCtx({
				flags: { "search-paths": "docs" },
				stdout: (line) => stdout.push(line),
			}),
		);

		expect(code).toBe(0);
		expect(stdout).toHaveLength(1);
		expect(stdout[0]).toContain(homeConfigPath());
		expect(stdout[0]).not.toContain("{");
	});
});

describe("runInit embedder flags", () => {
	it("writes embedder fields into minSync.embedder from flags", async () => {
		const code = await runInit(
			makeCtx({
				flags: {
					"search-paths": "docs",
					"embedder-id": "text-embedding-3-small",
					"embedder-base-url": "https://api.openai.com/v1",
					"embedder-api-key-env": "OPENAI_API_KEY",
					"embedder-dimension": "1536",
					"embedder-query-prefix": "",
					"embedder-passage-prefix": "passage: ",
					"embedder-timeout-ms": "30000",
					"embedder-batch-size": "64",
				},
			}),
		);
		expect(code).toBe(0);
		const config = JSON.parse(readFileSync(homeConfigPath(), "utf8"));
		expect(config.minSync).toBeDefined();
		expect(config.minSync.enabled).toBe(true);
		expect(config.minSync.autoInstall).toBe(true);
		expect(config.minSync.embedder).toEqual({
			id: "text-embedding-3-small",
			baseUrl: "https://api.openai.com/v1",
			apiKeyEnv: "OPENAI_API_KEY",
			dimension: 1536,
			queryPrefix: "",
			passagePrefix: "passage: ",
			timeoutMs: 30000,
			batchSize: 64,
		});
	});

	it("writes only provided embedder fields", async () => {
		const code = await runInit(
			makeCtx({
				flags: {
					"search-paths": "docs",
					"embedder-id": "bge-m3",
					"embedder-dimension": "1024",
				},
			}),
		);
		expect(code).toBe(0);
		const config = JSON.parse(readFileSync(homeConfigPath(), "utf8"));
		expect(config.minSync.embedder).toEqual({
			id: "bge-m3",
			dimension: 1024,
		});
	});

	it("writes the MinSync chunk size into method config", async () => {
		const code = await runInit(makeCtx({ flags: { "search-paths": "docs", "minsync-max-chunk-size": "1000" } }));
		expect(code).toBe(0);
		const config = JSON.parse(readFileSync(homeConfigPath(), "utf8"));
		expect(config.minSync.maxChunkSize).toBe(1000);
		expect(config.minSync.autoInstall).toBe(true);
	});

	it("does not write minSync.embedder when no embedder flags are given", async () => {
		const code = await runInit(
			makeCtx({
				flags: { "search-paths": "docs" },
			}),
		);
		expect(code).toBe(0);
		const config = JSON.parse(readFileSync(homeConfigPath(), "utf8"));
		// minSync is present (default ON) but no embedder key.
		expect(config.minSync).toBeDefined();
		expect(config.minSync.embedder).toBeUndefined();
	});

	it("rejects a non-positive embedder dimension", async () => {
		const stderr: string[] = [];
		const code = await runInit(
			makeCtx({
				flags: { "search-paths": "docs", "embedder-dimension": "0" },
				stderr: (line) => stderr.push(line),
			}),
		);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toContain("positive integer");
	});

	it("returns a config error for an invalid MinSync chunk size", async () => {
		const stderr: string[] = [];
		const code = await runInit(
			makeCtx({
				flags: { "search-paths": "docs", "minsync-max-chunk-size": "0" },
				stderr: (line) => stderr.push(line),
			}),
		);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toContain("positive integer");
		expect(existsSync(homeConfigPath())).toBe(false);
	});

	it("defaults bm25 and minSync enabled when no method flags are given", async () => {
		const code = await runInit(makeCtx({ flags: { "search-paths": "docs" } }));
		expect(code).toBe(0);
		const config = JSON.parse(readFileSync(homeConfigPath(), "utf8"));
		expect(config.bm25).toBeDefined();
		expect(config.bm25.enabled).toBe(true);
		expect(config.minSync).toBeDefined();
		expect(config.minSync.enabled).toBe(true);
		expect(config.minSync.autoInstall).toBe(true);
	});
});

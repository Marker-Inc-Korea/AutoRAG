import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	buildAgentOptions,
	type CliConfig,
	ConfigError,
	DEFAULT_CONFIG_FILENAME,
	resolveConfig,
	resolveModel,
	writeDefaultConfig,
} from "../../src/cli/config.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-cli-config-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function writeConfigFile(dir: string, config: Partial<CliConfig>): string {
	const path = join(dir, DEFAULT_CONFIG_FILENAME);
	writeFileSync(path, JSON.stringify(config, null, 2), "utf8");
	return path;
}

describe("resolveConfig defaults", () => {
	it("uses default searchPaths, workspacePath, and memoryPath when nothing is provided", () => {
		const config = resolveConfig({ flags: {}, cwd: root });
		expect(config.searchPaths).toEqual(["."]);
		expect(config.workspacePath).toBe(root);
		expect(config.memoryPath).toBe(join(root, ".autorag", "memory.json"));
		expect(config.model).toBeUndefined();
	});

	it("does not require a config file to exist at cwd", () => {
		expect(existsSync(join(root, DEFAULT_CONFIG_FILENAME))).toBe(false);
		const config = resolveConfig({ flags: {}, cwd: root });
		expect(config.searchPaths).toEqual(["."]);
	});
});

describe("resolveConfig precedence", () => {
	it("flag overrides env overrides file overrides defaults", () => {
		writeConfigFile(root, {
			searchPaths: ["./file"],
			workspacePath: "/file/workspace",
			memoryPath: "/file/memory.json",
			model: { provider: "fileprov", id: "fileid" },
		});

		const fileOnly = resolveConfig({
			flags: {},
			env: {},
			cwd: root,
		});
		expect(fileOnly.searchPaths).toEqual(["./file"]);
		expect(fileOnly.workspacePath).toBe("/file/workspace");
		expect(fileOnly.memoryPath).toBe("/file/memory.json");
		expect(fileOnly.model).toEqual({ provider: "fileprov", id: "fileid" });

		const envOnly = resolveConfig({
			flags: {},
			env: {
				AUTORAG_SEARCH_PATHS: "./env1,./env2",
				AUTORAG_WORKSPACE: "/env/workspace",
				AUTORAG_MEMORY_PATH: "/env/memory.json",
				AUTORAG_MODEL_PROVIDER: "envprov",
				AUTORAG_MODEL_ID: "envid",
			},
			cwd: root,
		});
		expect(envOnly.searchPaths).toEqual(["./env1", "./env2"]);
		expect(envOnly.workspacePath).toBe("/env/workspace");
		expect(envOnly.memoryPath).toBe("/env/memory.json");
		expect(envOnly.model).toEqual({ provider: "envprov", id: "envid" });

		const flagOnly = resolveConfig({
			flags: {
				"search-paths": "./flag1,./flag2",
				workspace: "/flag/workspace",
				"memory-path": "/flag/memory.json",
				"model-provider": "flagprov",
				"model-id": "flagid",
			},
			env: {
				AUTORAG_SEARCH_PATHS: "./env1,./env2",
				AUTORAG_WORKSPACE: "/env/workspace",
				AUTORAG_MEMORY_PATH: "/env/memory.json",
				AUTORAG_MODEL_PROVIDER: "envprov",
				AUTORAG_MODEL_ID: "envid",
			},
			cwd: root,
		});
		expect(flagOnly.searchPaths).toEqual(["./flag1", "./flag2"]);
		expect(flagOnly.workspacePath).toBe("/flag/workspace");
		expect(flagOnly.memoryPath).toBe("/flag/memory.json");
		expect(flagOnly.model).toEqual({ provider: "flagprov", id: "flagid" });
	});

	it("flag overrides env even when file is absent", () => {
		const config = resolveConfig({
			flags: { workspace: "/flag/workspace" },
			env: { AUTORAG_WORKSPACE: "/env/workspace" },
			cwd: root,
		});
		expect(config.workspacePath).toBe("/flag/workspace");
	});

	it("env overrides file when flag is absent", () => {
		writeConfigFile(root, { workspacePath: "/file/workspace" });
		const config = resolveConfig({
			flags: {},
			env: { AUTORAG_WORKSPACE: "/env/workspace" },
			cwd: root,
		});
		expect(config.workspacePath).toBe("/env/workspace");
	});

	it("file overrides defaults when flag and env are absent", () => {
		writeConfigFile(root, { workspacePath: "/file/workspace" });
		const config = resolveConfig({ flags: {}, env: {}, cwd: root });
		expect(config.workspacePath).toBe("/file/workspace");
	});
});

describe("resolveConfig env var mapping", () => {
	it("maps AUTORAG_SEARCH_PATHS csv, AUTORAG_MODEL_PROVIDER, and AUTORAG_MODEL_ID", () => {
		const config = resolveConfig({
			flags: {},
			env: {
				AUTORAG_SEARCH_PATHS: "src,test,docs",
				AUTORAG_MODEL_PROVIDER: "openai",
				AUTORAG_MODEL_ID: "gpt-4o",
			},
			cwd: root,
		});
		expect(config.searchPaths).toEqual(["src", "test", "docs"]);
		expect(config.model).toEqual({ provider: "openai", id: "gpt-4o" });
	});

	it("maps AUTORAG_CONFIG to an explicit config file path", () => {
		const alt = join(root, "alt.config.json");
		writeFileSync(alt, JSON.stringify({ workspacePath: "/alt/workspace" }), "utf8");
		const config = resolveConfig({
			flags: {},
			env: { AUTORAG_CONFIG: alt },
			cwd: root,
		});
		expect(config.workspacePath).toBe("/alt/workspace");
	});

	it("maps flags.config to an explicit config file path", () => {
		const alt = join(root, "alt.config.json");
		writeFileSync(alt, JSON.stringify({ workspacePath: "/alt/workspace" }), "utf8");
		const config = resolveConfig({
			flags: { config: alt },
			env: {},
			cwd: root,
		});
		expect(config.workspacePath).toBe("/alt/workspace");
	});

	it("throws ConfigError when an explicit config file does not exist", () => {
		expect(() =>
			resolveConfig({
				flags: { config: join(root, "missing.json") },
				env: {},
				cwd: root,
			}),
		).toThrow(ConfigError);
	});
});

describe("resolveConfig model partial handling", () => {
	it("does not set model when only provider is provided", () => {
		const config = resolveConfig({
			flags: { "model-provider": "openai" },
			env: {},
			cwd: root,
		});
		expect(config.model).toBeUndefined();
	});

	it("does not set model when only id is provided", () => {
		const config = resolveConfig({
			flags: { "model-id": "gpt-4o" },
			env: {},
			cwd: root,
		});
		expect(config.model).toBeUndefined();
	});
});

describe("buildAgentOptions", () => {
	it("always includes searchPaths and only includes optional keys when present", () => {
		const minimal = buildAgentOptions({
			searchPaths: ["."],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
		});
		expect(minimal.searchPaths).toEqual(["."]);
		expect(minimal.workspacePath).toBe(root);
		expect(minimal.memoryPath).toBe(join(root, "memory.json"));
		expect("minSync" in minimal).toBe(false);
		expect("bm25" in minimal).toBe(false);
		expect("jikji" in minimal).toBe(false);
		expect("parserOptions" in minimal).toBe(false);
		expect("model" in minimal).toBe(false);
	});

	it("includes optional keys when present", () => {
		const opts = buildAgentOptions({
			searchPaths: ["."],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: { foo: 1 },
			bm25: { bar: 2 },
			jikji: { baz: 3 },
			parserOptions: { qux: 4 },
		});
		expect(opts.minSync).toEqual({ foo: 1 });
		expect(opts.bm25).toEqual({ bar: 2 });
		expect(opts.jikji).toEqual({ baz: 3 });
		expect(opts.parserOptions).toEqual({ qux: 4 });
	});
});

describe("resolveModel", () => {
	it("throws ConfigError when model is absent and message names --model-provider and the model config key", () => {
		const config: CliConfig = {
			searchPaths: ["."],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
		};
		expect(() => resolveModel(config)).toThrow(ConfigError);
		expect(() => resolveModel(config)).toThrow(/--model-provider/);
		expect(() => resolveModel(config)).toThrow(/model/);
	});
});

describe("writeDefaultConfig", () => {
	it("writes autorag.config.json with defaults when partial is empty", () => {
		const path = join(root, DEFAULT_CONFIG_FILENAME);
		writeDefaultConfig(path, {});
		expect(existsSync(path)).toBe(true);
		const written = JSON.parse(readFileSync(path, "utf8")) as CliConfig;
		expect(written.searchPaths).toEqual(["."]);
		expect(typeof written.workspacePath).toBe("string");
		expect(typeof written.memoryPath).toBe("string");
	});

	it("throws ConfigError when the file already exists and force is not set", () => {
		const path = join(root, DEFAULT_CONFIG_FILENAME);
		writeDefaultConfig(path, {});
		expect(() => writeDefaultConfig(path, {})).toThrow(ConfigError);
	});

	it("overwrites the existing file when force is true", () => {
		const path = join(root, DEFAULT_CONFIG_FILENAME);
		writeDefaultConfig(path, {});
		writeDefaultConfig(path, { searchPaths: ["src"] }, { force: true });
		const written = JSON.parse(readFileSync(path, "utf8")) as CliConfig;
		expect(written.searchPaths).toEqual(["src"]);
	});

	it("preserves provided partial values", () => {
		const path = join(root, DEFAULT_CONFIG_FILENAME);
		writeDefaultConfig(path, {
			searchPaths: ["docs"],
			workspacePath: "/custom/workspace",
			memoryPath: "/custom/memory.json",
			model: { provider: "openai", id: "gpt-4o" },
		});
		const written = JSON.parse(readFileSync(path, "utf8")) as CliConfig;
		expect(written.searchPaths).toEqual(["docs"]);
		expect(written.workspacePath).toBe("/custom/workspace");
		expect(written.memoryPath).toBe("/custom/memory.json");
		expect(written.model).toEqual({ provider: "openai", id: "gpt-4o" });
	});
});

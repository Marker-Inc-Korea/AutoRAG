import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	buildAgentOptions,
	type CliConfig,
	ConfigError,
	normalizeEmbedder,
	normalizeIndexingConfig,
	resolveAgentModel,
	resolveConfig,
	writeDefaultConfig,
} from "../../src/cli/config.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-config-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

describe("single-model CLI config", () => {
	it("resolves model flags and ordinary feature settings", () => {
		const config = resolveConfig({
			flags: {
				"model-provider": "openai",
				"model-id": "gpt-4o",
				"search-paths": "docs,notes",
				workspace: root,
				"memory-path": join(root, "memory.json"),
			},
			env: { HOME: root },
			cwd: root,
		});
		expect(config.model).toEqual({ provider: "openai", id: "gpt-4o" });
		expect(config.searchPaths).toEqual(["docs", "notes"]);
		expect(config.minSync?.enabled).toBe(true);
		expect(config.minSync?.autoInstall).toBe(true);
		expect(config.jikji).toEqual({});
		expect(config.excludeExactDuplicates).toBe(true);
	});

	it("preserves explicit Jikji opt-out in resolved agent options", () => {
		const config = resolveConfig({
			flags: {},
			env: { HOME: root },
			cwd: root,
		});
		const optedOut = { ...config, jikji: false as const };
		expect(buildAgentOptions(optedOut).jikji).toBe(false);
	});

	it("rejects partial model flags", () => {
		expect(() => resolveConfig({ flags: { "model-provider": "openai" }, env: {}, cwd: root })).toThrow(
			/model requires both provider and id/i,
		);
	});

	it("writes and reads a config with retrieval options", () => {
		const path = join(root, "config.json");
		writeDefaultConfig(
			path,
			{
				searchPaths: ["docs"],
				workspacePath: root,
				memoryPath: join(root, "memory.json"),
				model: { provider: "openai", id: "gpt-4o" },
				minSync: { enabled: false },
				ui: {
					host: "localhost",
					port: 8787,
					corsOrigins: ["https://admin.example.test"],
				},
			},
			{ cwd: root },
		);
		const written = JSON.parse(readFileSync(path, "utf8")) as CliConfig;
		expect(written.model).toEqual({ provider: "openai", id: "gpt-4o" });
		expect(written.minSync?.enabled).toBe(false);
		expect(written.ui).toEqual({
			host: "localhost",
			port: 8787,
			corsOrigins: ["https://admin.example.test"],
		});
		expect(JSON.stringify(written)).not.toMatch(/explorer|orchestrator/i);
	});

	it("builds agent options for all non-model features", () => {
		const opts = buildAgentOptions({
			searchPaths: ["."],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: { enabled: false },
			jikji: {},
			parserOptions: { pdf: true },
			dupey: { enabled: false },
			excludeExactDuplicates: false,
		});
		expect(opts.minSync).toBe(false);
		expect(opts.jikji).toEqual({});
		expect(opts.parserOptions).toEqual({ pdf: true });
		expect(opts.dupey).toBe(false);
		expect(opts.excludeExactDuplicates).toBe(false);
	});

	it("resolves a configured catalog model without local runtime config", () => {
		const resolved = resolveAgentModel(
			{
				searchPaths: ["."],
				workspacePath: root,
				memoryPath: join(root, "memory.json"),
				model: { provider: "openai", id: "gpt-4o" },
			},
			{ configPath: join(root, "missing.toml") },
		);
		expect(resolved.model).toMatchObject({ provider: "openai", id: "gpt-4o" });
	});

	it("validates MinSync embedder boundaries", () => {
		expect(normalizeEmbedder({ id: "embed", dimension: 3 }, "minSync.embedder")).toEqual({
			id: "embed",
			dimension: 3,
		});
		expect(() => normalizeEmbedder({ dimension: 0 }, "minSync.embedder")).toThrow(ConfigError);
	});

	it("validates the MinSync chunk size", () => {
		expect(normalizeIndexingConfig({ minSync: { maxChunkSize: 1000 } }).minSync.maxChunkSize).toBe(1000);
		expect(() => normalizeIndexingConfig({ minSync: { maxChunkSize: 0 } })).toThrow(ConfigError);
	});

	it("ignores a legacy minSync.binaryPath instead of rejecting the config", () => {
		// Configs written before MinSync moved to PATH resolution still carry this
		// key; rejecting it would break every command for existing installs.
		const config = normalizeIndexingConfig({
			minSync: { enabled: true, binaryPath: "/usr/local/bin/minsync" } as never,
		}).minSync;
		expect(config.enabled).toBe(true);
		expect(config).not.toHaveProperty("binaryPath");
		expect(() => normalizeIndexingConfig({ minSync: { nope: true } as never })).toThrow(ConfigError);
	});

	it("rejects malformed JSON config", () => {
		const path = join(root, "bad.json");
		mkdirSync(root, { recursive: true });
		writeFileSync(path, "{");
		expect(() => resolveConfig({ flags: { config: path }, cwd: root })).toThrow(ConfigError);
	});

	it("resolves deployment UI settings without weakening local defaults", () => {
		const path = join(root, "ui.json");
		writeFileSync(
			path,
			JSON.stringify({
				searchPaths: ["."],
				workspacePath: root,
				memoryPath: join(root, "memory.json"),
				ui: {
					host: "0.0.0.0",
					port: 8080,
					allowRemote: true,
					publicOrigin: "https://admin.example.test",
					corsOrigins: ["https://admin.example.test"],
					tokenEnv: "AUTORAG_UI_TOKEN",
				},
			}),
		);

		const config = resolveConfig({ flags: { config: path }, cwd: root, env: {} });

		expect(config.ui).toEqual({
			host: "0.0.0.0",
			port: 8080,
			allowRemote: true,
			publicOrigin: "https://admin.example.test",
			corsOrigins: ["https://admin.example.test"],
			tokenEnv: "AUTORAG_UI_TOKEN",
		});
	});
});

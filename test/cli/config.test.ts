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
import { DEFAULT_LANGUAGES } from "../../src/language.ts";

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
				"model-id": "gpt-6-luna",
				"search-paths": "docs,notes",
				workspace: root,
				"memory-path": join(root, "memory.json"),
			},
			env: { HOME: root },
			cwd: root,
		});
		expect(config.model).toEqual({ provider: "openai", id: "gpt-6-luna" });
		expect(config.searchPaths).toEqual(["docs", "notes"]);
		expect(config.minSync?.enabled).toBe(true);
		expect(config.minSync?.autoInstall).toBe(true);
		expect(config.jikji).toEqual({});
		expect(config.excludeExactDuplicates).toBe(true);
	});

	it("reads and normalizes languages from config.json", () => {
		const path = join(root, "config.json");
		writeFileSync(path, JSON.stringify({ languages: [" KO ", "en", "ko"] }), "utf8");

		const config = resolveConfig({
			flags: { config: path },
			env: { HOME: root },
			cwd: root,
		});

		expect(config.languages).toEqual(["ko", "en"]);
	});

	it("resolves languages in flag, env, config, default precedence order", () => {
		const path = join(root, "config.json");
		writeFileSync(path, JSON.stringify({ languages: ["de", "fr"] }), "utf8");

		expect(
			resolveConfig({
				flags: { config: path },
				env: { HOME: root },
				cwd: root,
			}).languages,
		).toEqual(["de", "fr"]);
		expect(
			resolveConfig({
				flags: { config: path },
				env: { HOME: root, AUTORAG_LANGUAGES: "en,vi" },
				cwd: root,
			}).languages,
		).toEqual(["en", "vi"]);
		expect(
			resolveConfig({
				flags: { config: path, languages: "ja,ko" },
				env: { HOME: root, AUTORAG_LANGUAGES: "en,vi" },
				cwd: root,
			}).languages,
		).toEqual(["ja", "ko"]);
		const noLanguagesPath = join(root, "default-config.json");
		writeFileSync(noLanguagesPath, JSON.stringify({}), "utf8");
		expect(
			resolveConfig({
				flags: { config: noLanguagesPath },
				env: { HOME: root },
				cwd: root,
			}).languages,
		).toEqual(DEFAULT_LANGUAGES);
	});

	it("rejects an unsupported config language as ConfigError", () => {
		const path = join(root, "config.json");
		writeFileSync(path, JSON.stringify({ languages: ["ko", "kr"] }), "utf8");

		expect(() =>
			resolveConfig({
				flags: { config: path },
				env: { HOME: root },
				cwd: root,
			}),
		).toThrow(ConfigError);
		expect(() =>
			resolveConfig({
				flags: { config: path },
				env: { HOME: root },
				cwd: root,
			}),
		).toThrow(/Unsupported language/);
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
				model: { provider: "openai", id: "gpt-6-luna" },
				minSync: { enabled: false },
			},
			{ cwd: root },
		);
		const written = JSON.parse(readFileSync(path, "utf8")) as CliConfig;
		expect(written.model).toEqual({ provider: "openai", id: "gpt-6-luna" });
		expect(written.minSync?.enabled).toBe(false);
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

	it("includes resolved languages in agent options", () => {
		const opts = buildAgentOptions({
			searchPaths: ["."],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			languages: ["ja", "en"],
		});

		expect(opts.languages).toEqual(["ja", "en"]);
	});

	it("resolves a configured catalog model without local runtime config", async () => {
		const resolved = await resolveAgentModel(
			{
				searchPaths: ["."],
				workspacePath: root,
				memoryPath: join(root, "memory.json"),
				model: { provider: "openai", id: "gpt-6-luna" },
			},
			{ configPath: join(root, "missing.toml"), agentDir: join(root, "agent") },
		);
		expect(resolved.model).toMatchObject({ provider: "openai", id: "gpt-6-luna" });
	});

	it("uses stored pi credentials for a catalog model", async () => {
		const agentDir = join(root, "agent");
		mkdirSync(agentDir, { recursive: true });
		writeFileSync(join(agentDir, "auth.json"), JSON.stringify({ openai: { type: "api_key", key: "sk-stored" } }));
		const resolved = await resolveAgentModel(
			{
				searchPaths: ["."],
				workspacePath: root,
				memoryPath: join(root, "memory.json"),
				model: { provider: "openai", id: "gpt-6-luna" },
			},
			{ configPath: join(root, "missing.toml"), agentDir },
		);
		expect(resolved.model).toMatchObject({ provider: "openai", id: "gpt-6-luna" });
		expect(resolved.apiKey).toBe("sk-stored");
		expect(resolved.providerApiKeys).toEqual({ openai: "sk-stored" });
	});

	it("falls back to the pi settings default model when no model is configured", async () => {
		const agentDir = join(root, "agent");
		mkdirSync(agentDir, { recursive: true });
		writeFileSync(
			join(agentDir, "settings.json"),
			JSON.stringify({ defaultProvider: "openai", defaultModel: "gpt-6-luna" }),
		);
		writeFileSync(join(agentDir, "auth.json"), JSON.stringify({ openai: { type: "api_key", key: "sk-stored" } }));
		const resolved = await resolveAgentModel(
			{ searchPaths: ["."], workspacePath: root, memoryPath: join(root, "memory.json") },
			{ configPath: join(root, "missing.toml"), agentDir, cwd: root },
		);
		expect(resolved.model).toMatchObject({ provider: "openai", id: "gpt-6-luna" });
		expect(resolved.apiKey).toBe("sk-stored");
	});

	describe("catalog model with a configured endpoint (#1757)", () => {
		const base = () => ({ searchPaths: ["."], workspacePath: root, memoryPath: join(root, "memory.json") });
		const catalogRef = { provider: "openrouter", id: "deepseek/deepseek-v4.1-flash" };
		const resolveWith = (model: CliConfig["model"], env: NodeJS.ProcessEnv = {}) =>
			resolveAgentModel(
				{ ...base(), model },
				{ configPath: join(root, "missing.toml"), agentDir: join(root, "agent"), env },
			);

		it("keeps catalog reasoning, compat, and limits when baseUrl/api restate the catalog endpoint", async () => {
			const catalogOnly = (await resolveWith(catalogRef)).model;
			const withEndpoint = (
				await resolveWith({
					...catalogRef,
					baseUrl: "https://openrouter.ai/api/v1",
					api: "openai-completions",
					apiKeyEnv: "OPENROUTER_API_KEY",
				})
			).model;
			expect(catalogOnly.reasoning).toBe(true);
			expect(catalogOnly.compat).toBeDefined();
			expect(withEndpoint).toEqual(catalogOnly);
		});

		it("overrides only the explicitly configured fields on top of the catalog entry", async () => {
			const catalogOnly = (await resolveWith(catalogRef)).model;
			const overridden = (
				await resolveWith({
					...catalogRef,
					name: "Proxy Flash",
					baseUrl: "https://proxy.example.test/v1",
					api: "openai-responses",
					reasoning: false,
					input: ["text"],
					contextWindow: 64_000,
					maxTokens: 8_000,
				})
			).model;
			expect(overridden).toEqual({
				...catalogOnly,
				name: "Proxy Flash",
				baseUrl: "https://proxy.example.test/v1",
				api: "openai-responses",
				reasoning: false,
				input: ["text"],
				contextWindow: 64_000,
				maxTokens: 8_000,
			});
		});

		it("still passes the configured apiKeyEnv secret for a catalog model with baseUrl", async () => {
			const resolved = await resolveWith(
				{ ...catalogRef, baseUrl: "https://openrouter.ai/api/v1", apiKeyEnv: "MY_ROUTER_KEY" },
				{ MY_ROUTER_KEY: "sk-test" },
			);
			expect(resolved.apiKey).toBe("sk-test");
			expect(resolved.providerApiKeys).toEqual({ openrouter: "sk-test" });
		});

		it("builds a generic model when the catalog has no entry for the configured endpoint", async () => {
			const model = (
				await resolveWith({
					provider: "openrouter",
					id: "private/not-in-catalog",
					baseUrl: "https://openrouter.ai/api/v1",
				})
			).model;
			expect(model).toMatchObject({
				id: "private/not-in-catalog",
				api: "openai-completions",
				reasoning: false,
				input: ["text"],
				contextWindow: 128_000,
				maxTokens: 16_384,
			});
			expect(model.compat).toBeUndefined();
		});
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
});

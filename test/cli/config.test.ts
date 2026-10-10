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
import { buildStoreEntry, mapServerModels } from "../../src/cloud/models.ts";
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
				minSync: { autoInstall: false },
			},
			{ cwd: root },
		);
		const written = JSON.parse(readFileSync(path, "utf8")) as CliConfig;
		expect(written.model).toEqual({ provider: "openai", id: "gpt-6-luna" });
		expect(written.minSync?.enabled).toBe(true);
		expect(written.minSync?.autoInstall).toBe(false);
		expect(JSON.stringify(written)).not.toMatch(/explorer|orchestrator/i);
	});

	it("builds agent options for all non-model features", () => {
		const opts = buildAgentOptions({
			searchPaths: ["."],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: { autoInstall: false },
			jikji: {},
			parserOptions: { pdf: true },
			dupey: { enabled: false },
			excludeExactDuplicates: false,
		});
		expect(opts.minSync).toEqual({ autoInstall: false });
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

	describe("no model configured and no usable local Codex runtime", () => {
		const bareConfig = (): CliConfig => ({
			searchPaths: ["."],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
		});

		it("explains how to pick a model instead of surfacing a bare ENOENT", async () => {
			const error = await resolveAgentModel(bareConfig(), {
				configPath: join(root, "missing.toml"),
				agentDir: join(root, "agent"),
				cwd: root,
				env: {},
			}).catch((caught: unknown) => caught);

			expect(error).toBeInstanceOf(ConfigError);
			const message = (error as Error).message;
			expect(message).toMatch(/^No model configured\./);
			expect(message).toContain("autorag tui");
			expect(message).toContain("/login");
			expect(message).toContain('"model"');
			expect(message).toContain("autorag models list");
			expect(message).toContain("ENOENT");
			expect(message).toContain("missing.toml");
		});

		it("keeps the local runtime's own reason when its config is incomplete", async () => {
			const codexConfig = join(root, "codex.toml");
			writeFileSync(codexConfig, "# No provider configured\n");
			const error = await resolveAgentModel(bareConfig(), {
				configPath: codexConfig,
				agentDir: join(root, "agent"),
				cwd: root,
				env: {},
			}).catch((caught: unknown) => caught);

			expect(error).toBeInstanceOf(ConfigError);
			expect((error as Error).message).toContain(`AutoRAG requires model_provider in ${codexConfig}`);
		});

		it("still resolves a complete local Codex runtime", async () => {
			const codexConfig = join(root, "codex.toml");
			writeFileSync(
				codexConfig,
				[
					'model_provider = "proxy"',
					"[model_providers.proxy]",
					'base_url = "http://127.0.0.1:9/v1"',
					'wire_api = "responses"',
					'env_key = "PROXY_KEY"',
					"",
				].join("\n"),
			);
			const resolved = await resolveAgentModel(bareConfig(), {
				configPath: codexConfig,
				agentDir: join(root, "agent"),
				cwd: root,
				env: { PROXY_KEY: "sk-local" },
			});

			expect(resolved.model).toMatchObject({ provider: "proxy", baseUrl: "http://127.0.0.1:9/v1" });
			expect(resolved.apiKey).toBe("sk-local");
		});
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

	describe("hosted autorag provider", () => {
		const MODEL_ID = "anthropic/claude-haiku-5.5";
		const base = () => ({ searchPaths: ["."], workspacePath: root, memoryPath: join(root, "memory.json") });
		const options = (agentDir: string, env: NodeJS.ProcessEnv = {}) => ({
			configPath: join(root, "missing.toml"),
			agentDir,
			env,
		});

		/** Persist a catalog snapshot the way a prior /v1/models refresh would. */
		function seedSnapshot(agentDir: string): void {
			mkdirSync(agentDir, { recursive: true });
			const models = mapServerModels({
				object: "list",
				data: [
					{
						id: MODEL_ID,
						name: "Claude Haiku 5.5",
						context_window: 200_000,
						max_output_tokens: 8_192,
						input_modalities: ["text", "image"],
						reasoning: true,
						pricing: { input: 1, output: 2 },
					},
				],
			});
			writeFileSync(
				join(agentDir, "models-store.json"),
				JSON.stringify({ autorag: buildStoreEntry(models, "autorag", "https://api.dazziapp.com/v1") }),
			);
		}

		it("resolves the model from the persisted catalog with a stored OAuth credential", async () => {
			const agentDir = join(root, "agent");
			seedSnapshot(agentDir);
			writeFileSync(
				join(agentDir, "auth.json"),
				JSON.stringify({
					autorag: { type: "oauth", access: "dz_stored", refresh: "", expires: Number.MAX_SAFE_INTEGER },
				}),
			);

			const resolved = await resolveAgentModel(
				{ ...base(), model: { provider: "autorag", id: MODEL_ID } },
				options(agentDir),
			);

			expect(resolved.model).toMatchObject({
				provider: "autorag",
				id: MODEL_ID,
				api: "openai-responses",
			});
			expect(resolved.apiKey).toBe("dz_stored");
			expect(resolved.providerApiKeys).toEqual({ autorag: "dz_stored" });
		});

		it("resolves the model from the persisted catalog with AUTORAG_API_KEY", async () => {
			const agentDir = join(root, "agent");
			seedSnapshot(agentDir);
			const previous = process.env.AUTORAG_API_KEY;
			process.env.AUTORAG_API_KEY = "dz_env_key";
			try {
				const resolved = await resolveAgentModel(
					{ ...base(), model: { provider: "autorag", id: MODEL_ID } },
					options(agentDir),
				);
				expect(resolved.model).toMatchObject({ provider: "autorag", id: MODEL_ID });
				expect(resolved.apiKey).toBe("dz_env_key");
			} finally {
				if (previous === undefined) delete process.env.AUTORAG_API_KEY;
				else process.env.AUTORAG_API_KEY = previous;
			}
		});

		it("keeps an explicit configured endpoint resolving as before, with baseUrl winning", async () => {
			const agentDir = join(root, "agent");
			const resolved = await resolveAgentModel(
				{
					...base(),
					model: {
						provider: "autorag",
						id: MODEL_ID,
						baseUrl: "https://api.dazziapp.com/v1",
						api: "openai-responses",
						apiKeyEnv: "AUTORAG_API_KEY",
					},
				},
				options(agentDir, { AUTORAG_API_KEY: "dz_app" }),
			);

			expect(resolved.model).toMatchObject({
				provider: "autorag",
				id: MODEL_ID,
				api: "openai-responses",
				baseUrl: "https://api.dazziapp.com/v1",
			});
			expect(resolved.apiKey).toBe("dz_app");
		});

		it("keeps the explicit baseUrl when a catalog snapshot also knows the model", async () => {
			const agentDir = join(root, "agent");
			seedSnapshot(agentDir);
			const resolved = await resolveAgentModel(
				{
					...base(),
					model: {
						provider: "autorag",
						id: MODEL_ID,
						baseUrl: "https://staging.example.com/v1",
						apiKeyEnv: "AUTORAG_API_KEY",
					},
				},
				options(agentDir, { AUTORAG_API_KEY: "dz_app" }),
			);

			expect(resolved.model.baseUrl).toBe("https://staging.example.com/v1");
			expect(resolved.model.contextWindow).toBe(200_000);
		});

		it("fails clearly when the plan is unknown (not signed in, no catalog snapshot)", async () => {
			const agentDir = join(root, "agent");
			const error = await resolveAgentModel(
				{ ...base(), model: { provider: "autorag", id: MODEL_ID } },
				options(agentDir),
			).catch((caught: unknown) => caught);

			expect(error).toBeInstanceOf(ConfigError);
			expect((error as Error).message).toContain(`Unknown configured model: autorag/${MODEL_ID}`);
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

import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	buildAgentOptions,
	type CliConfig,
	ConfigError,
	normalizeJevConfig,
	resolveConfig,
	resolveDecisionFallbackModel,
} from "../../src/cli/config.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-config-jev-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function baseConfig(jev?: CliConfig["jev"]): CliConfig {
	return {
		searchPaths: ["."],
		workspacePath: root,
		memoryPath: join(root, "memory.json"),
		minSync: { autoInstall: false },
		...(jev === undefined ? {} : { jev }),
	};
}

describe("jev CLI config", () => {
	it("enables Jev on OpenRouter by default when unconfigured and not signed in", () => {
		expect(buildAgentOptions(baseConfig(), join(root, "agent")).jev).toEqual({ backend: "openrouter" });
	});

	it("defaults Jev to Dazzi Models when signed in", () => {
		const agentDir = join(root, "agent");
		mkdirSync(agentDir, { recursive: true });
		writeFileSync(
			join(agentDir, "auth.json"),
			JSON.stringify({
				dazzi: { type: "oauth", access: "dz_login", refresh: "", expires: Number.MAX_SAFE_INTEGER },
			}),
		);
		expect(buildAgentOptions(baseConfig(), agentDir).jev).toEqual({ backend: "dazzi" });
		expect(buildAgentOptions(baseConfig({}), agentDir).jev).toEqual({ backend: "dazzi" });
	});

	it("lets an explicit backend win over the signed-in default", () => {
		const agentDir = join(root, "agent");
		mkdirSync(agentDir, { recursive: true });
		writeFileSync(
			join(agentDir, "auth.json"),
			JSON.stringify({
				dazzi: { type: "oauth", access: "dz_login", refresh: "", expires: Number.MAX_SAFE_INTEGER },
			}),
		);
		expect(buildAgentOptions(baseConfig({ backend: "openrouter" }), agentDir).jev).toEqual({ backend: "openrouter" });
	});

	it("passes backend, model, and confidence threshold through", () => {
		const opts = buildAgentOptions(
			baseConfig({ backend: "openrouter", model: "jev-1.13", confidenceThreshold: 0.7 }),
		);
		expect(opts.jev).toEqual({ backend: "openrouter", model: "jev-1.13", confidenceThreshold: 0.7 });
	});

	it("accepts the hosted Dazzi Models backend", () => {
		expect(buildAgentOptions(baseConfig({ backend: "dazzi", model: "jev-1.13" })).jev).toEqual({
			backend: "dazzi",
			model: "jev-1.13",
		});
	});

	it("rejects turning the decision model off, in the file and in the section", () => {
		expect(() => normalizeJevConfig(false)).toThrow(/jev' cannot be false.*requires its decision model/u);
		expect(() => normalizeJevConfig({ enabled: false })).toThrow(/jev\.enabled is no longer supported/u);
		expect(() => normalizeJevConfig({ enabled: true, backend: "openrouter" })).toThrow(ConfigError);
		const configPath = join(root, "config.json");
		writeFileSync(configPath, JSON.stringify({ searchPaths: [root], jev: false }));
		expect(() => resolveConfig({ flags: { config: configPath }, env: {}, readOnly: true })).toThrow(ConfigError);
	});

	it("keeps the OpenRouter default for an empty section", () => {
		expect(buildAgentOptions(baseConfig({})).jev).toEqual({ backend: "openrouter" });
	});

	it("accepts every pi provider that serves decision models, and custom provider ids", () => {
		for (const backend of [
			"openai",
			"openrouter",
			"typesafe",
			"vercel-ai-gateway",
			"cloudflare-workers-ai",
			"opencode",
		]) {
			expect(normalizeJevConfig({ backend })).toEqual({ backend });
		}
		expect(normalizeJevConfig({ backend: "openai", model: "gpt-6-luna" })).toEqual({
			backend: "openai",
			model: "gpt-6-luna",
		});
		// pi's catalog is read at run time: a provider without decision models fails the session check, not the config.
		expect(normalizeJevConfig({ backend: "my-local-decider" })).toEqual({ backend: "my-local-decider" });
	});

	it("rejects the retired `vercel` name with the pi provider id to use", () => {
		expect(() => normalizeJevConfig({ backend: "vercel" })).toThrow(/vercel-ai-gateway/u);
	});

	it("rejects unknown fields and malformed sections", () => {
		expect(() => normalizeJevConfig({ backend: "" })).toThrow(ConfigError);
		expect(() => normalizeJevConfig({ provider: "openrouter" })).toThrow(ConfigError);
		expect(() => normalizeJevConfig("openrouter")).toThrow(ConfigError);
	});

	it("validates the sLLM fallback: a model reference, or false", () => {
		expect(normalizeJevConfig({ fallback: false })).toEqual({ fallback: false });
		expect(normalizeJevConfig({ fallback: {} })).toEqual({ fallback: {} });
		expect(normalizeJevConfig({ fallback: { model: { provider: "openrouter", id: "qwen/qwen3.8-flash" } } })).toEqual(
			{
				fallback: { model: { provider: "openrouter", id: "qwen/qwen3.8-flash" } },
			},
		);
		expect(() => normalizeJevConfig({ fallback: true })).toThrow(ConfigError);
		expect(() => normalizeJevConfig({ fallback: { provider: "openrouter" } })).toThrow(ConfigError);
	});

	it("keeps the fallback out of the synchronous agent option: it is resolved with its credential later", () => {
		expect(buildAgentOptions(baseConfig({ backend: "openai", fallback: false })).jev).toEqual({ backend: "openai" });
	});

	it("rejects an out-of-range confidence threshold", () => {
		expect(() => buildAgentOptions(baseConfig({ confidenceThreshold: -0.1 }))).toThrow(ConfigError);
		expect(() => buildAgentOptions(baseConfig({ confidenceThreshold: 1.5 }))).toThrow(ConfigError);
	});
});

describe("resolveDecisionFallbackModel", () => {
	async function runtime() {
		return ModelRuntime.create({
			authPath: join(root, "auth.json"),
			modelsPath: join(root, "models.json"),
			allowModelNetwork: false,
		});
	}

	it("defaults to the question-decomposition model: its configured one, else the default", async () => {
		const options = { runtime: await runtime(), agentDir: join(root, "agent"), env: {} };
		const byDefault = await resolveDecisionFallbackModel(baseConfig(), options);
		expect(`${byDefault?.model.provider}/${byDefault?.model.id}`).toBe("openrouter/qwen/qwen3.7-flash");
		const configured = await resolveDecisionFallbackModel(
			{ ...baseConfig(), queryDecomposition: { model: { provider: "openrouter", id: "qwen/qwen3.8-flash" } } },
			options,
		);
		expect(configured?.model.id).toBe("qwen/qwen3.8-flash");
	});

	it("uses the configured model, and none when the fallback is off", async () => {
		const options = { runtime: await runtime(), agentDir: join(root, "agent"), env: {} };
		const named = await resolveDecisionFallbackModel(
			baseConfig({ fallback: { model: { provider: "openrouter", id: "qwen/qwen3.8-flash" } } }),
			options,
		);
		expect(named?.model.id).toBe("qwen/qwen3.8-flash");
		expect(await resolveDecisionFallbackModel(baseConfig({ fallback: false }), options)).toBeUndefined();
	});
});

import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	buildAgentOptions,
	type CliConfig,
	ConfigError,
	defaultRerankConfig,
	normalizeRerankConfig,
} from "../../src/cli/config.ts";
import {
	DEFAULT_DAZZI_RERANK_MODEL,
	DEFAULT_RERANK_API_KEY_ENV,
	DEFAULT_RERANK_MODEL,
	DEFAULT_RERANK_PROVIDER,
	DEFAULT_RERANK_TOP_N,
} from "../../src/retrieval/rerank.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-config-rerank-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function baseConfig(rerank?: CliConfig["rerank"]): CliConfig {
	return {
		searchPaths: ["."],
		workspacePath: root,
		memoryPath: join(root, "memory.json"),
		minSync: { autoInstall: false },
		...(rerank === undefined ? {} : { rerank }),
	};
}

describe("defaultRerankConfig (what `autorag init` writes)", () => {
	it("defaults to OpenRouter when not signed in to Dazzi Models", () => {
		expect(defaultRerankConfig(join(root, "agent"))).toEqual({
			provider: DEFAULT_RERANK_PROVIDER,
			model: DEFAULT_RERANK_MODEL,
			apiKeyEnv: DEFAULT_RERANK_API_KEY_ENV,
			topN: DEFAULT_RERANK_TOP_N,
		});
	});

	it("defaults to the Dazzi Models reranker when signed in", () => {
		const agentDir = join(root, "agent");
		mkdirSync(agentDir, { recursive: true });
		writeFileSync(
			join(agentDir, "auth.json"),
			JSON.stringify({
				dazzi: { type: "oauth", access: "dz_login", refresh: "", expires: Number.MAX_SAFE_INTEGER },
			}),
		);
		expect(defaultRerankConfig(agentDir)).toEqual({
			provider: "dazzi",
			model: DEFAULT_DAZZI_RERANK_MODEL,
			topN: DEFAULT_RERANK_TOP_N,
		});
	});
});

describe("normalizeRerankConfig", () => {
	it("fills provider, model, apiKeyEnv, and topN defaults", () => {
		expect(normalizeRerankConfig(undefined, "rerank")).toEqual({
			provider: DEFAULT_RERANK_PROVIDER,
			model: DEFAULT_RERANK_MODEL,
			apiKeyEnv: DEFAULT_RERANK_API_KEY_ENV,
			topN: DEFAULT_RERANK_TOP_N,
		});
	});

	it("keeps an explicit model and provider", () => {
		expect(normalizeRerankConfig({ provider: "openrouter", model: "voyageai/rerank-3" }, "rerank")).toEqual({
			provider: "openrouter",
			model: "voyageai/rerank-3",
			apiKeyEnv: DEFAULT_RERANK_API_KEY_ENV,
			topN: DEFAULT_RERANK_TOP_N,
		});
	});

	it("fills the dazzi provider's own model and no key env (it uses the /login credential)", () => {
		expect(normalizeRerankConfig({ provider: "dazzi" }, "rerank")).toEqual({
			provider: "dazzi",
			model: DEFAULT_DAZZI_RERANK_MODEL,
			topN: DEFAULT_RERANK_TOP_N,
		});
	});

	it("rejects apiKeyEnv for the dazzi provider instead of silently ignoring it", () => {
		expect(() => normalizeRerankConfig({ provider: "dazzi", apiKeyEnv: "DAZZI_KEY" }, "rerank")).toThrow(
			/apiKeyEnv is not supported for provider "dazzi"/,
		);
	});

	it("returns false for an explicit disable", () => {
		expect(normalizeRerankConfig(false, "rerank")).toBe(false);
	});

	it("rejects unknown fields", () => {
		expect(() => normalizeRerankConfig({ nope: 1 }, "rerank")).toThrow(ConfigError);
	});

	it("rejects an invalid apiKeyEnv", () => {
		expect(() => normalizeRerankConfig({ apiKeyEnv: "1bad" }, "rerank")).toThrow(ConfigError);
	});

	it("rejects an unsupported provider", () => {
		expect(() => normalizeRerankConfig({ provider: "cohere" }, "rerank")).toThrow(ConfigError);
	});

	it("rejects non-positive topN and timeoutMs", () => {
		expect(() => normalizeRerankConfig({ topN: 0 }, "rerank")).toThrow(ConfigError);
		expect(() => normalizeRerankConfig({ timeoutMs: -1 }, "rerank")).toThrow(ConfigError);
	});
});

describe("rerank agent options", () => {
	it("leaves rerank undefined when unconfigured", () => {
		expect(buildAgentOptions(baseConfig()).rerank).toBeUndefined();
	});

	it("maps enabled:false to the agent opt-out", () => {
		expect(buildAgentOptions(baseConfig({ enabled: false })).rerank).toBe(false);
	});

	it("passes the configured provider, model, and topN through", () => {
		const opts = buildAgentOptions(baseConfig({ provider: "openrouter", model: "voyageai/rerank-3-lite", topN: 5 }));
		expect(opts.rerank).toEqual({ provider: "openrouter", model: "voyageai/rerank-3-lite", topN: 5 });
	});
});

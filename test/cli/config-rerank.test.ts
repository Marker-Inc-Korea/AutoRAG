import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	buildAgentOptions,
	type CliConfig,
	ConfigError,
	normalizeRerankConfig,
	resolveConfig,
	writeDefaultConfig,
} from "../../src/cli/config.ts";
import {
	DEFAULT_RERANK_API_KEY_ENV,
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

describe("autorag init (reranking is opt-in)", () => {
	function initWith(rerank?: CliConfig["rerank"]): CliConfig {
		const path = join(root, "config.json");
		writeDefaultConfig(
			path,
			{ workspacePath: root, ...(rerank === undefined ? {} : { rerank }) },
			{ cwd: root, env: {} },
		);
		return resolveConfig({ flags: { config: path }, cwd: root, env: {} });
	}

	it("writes no rerank block, so a new setup never sends text to a remote reranker", () => {
		expect(initWith().rerank).toBeUndefined();
	});

	it("keeps an explicit rerank block together with its model", () => {
		expect(initWith({ provider: "dazzi", model: "qwen/qwen3-reranker-8b" }).rerank).toEqual({
			provider: "dazzi",
			model: "qwen/qwen3-reranker-8b",
			topN: DEFAULT_RERANK_TOP_N,
		});
	});
});

describe("normalizeRerankConfig", () => {
	it("fills provider, apiKeyEnv, and topN defaults but never invents a model", () => {
		const out = normalizeRerankConfig(undefined, "rerank");
		expect(out).toEqual({
			provider: DEFAULT_RERANK_PROVIDER,
			apiKeyEnv: DEFAULT_RERANK_API_KEY_ENV,
			topN: DEFAULT_RERANK_TOP_N,
		});
		expect(out !== false && "model" in out).toBe(false);
	});

	it("keeps an explicit model and provider", () => {
		expect(normalizeRerankConfig({ provider: "openrouter", model: "voyageai/rerank-3" }, "rerank")).toEqual({
			provider: "openrouter",
			model: "voyageai/rerank-3",
			apiKeyEnv: DEFAULT_RERANK_API_KEY_ENV,
			topN: DEFAULT_RERANK_TOP_N,
		});
	});

	it("does not invent a model for the dazzi provider either (it uses the /login credential)", () => {
		const out = normalizeRerankConfig({ provider: "dazzi" }, "rerank");
		expect(out).toEqual({ provider: "dazzi", topN: DEFAULT_RERANK_TOP_N });
		expect(out !== false && "model" in out).toBe(false);
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

	it("passes a block without a model through as is and never invents one", () => {
		expect(buildAgentOptions(baseConfig({ provider: "openrouter" })).rerank).toEqual({ provider: "openrouter" });
		expect(buildAgentOptions(baseConfig({ provider: "dazzi", topN: 5 })).rerank).toEqual({
			provider: "dazzi",
			topN: 5,
		});
	});
});

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { buildAgentOptions, type CliConfig, ConfigError, normalizeJevConfig } from "../../src/cli/config.ts";

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
		minSync: { enabled: false },
		...(jev === undefined ? {} : { jev }),
	};
}

describe("jev CLI config", () => {
	it("leaves the decision tool disabled when unconfigured", () => {
		expect(buildAgentOptions(baseConfig()).jev).toBeUndefined();
	});

	it("passes backend, model, and confidence threshold through", () => {
		const opts = buildAgentOptions(
			baseConfig({ backend: "openrouter", model: "jev-1.13", confidenceThreshold: 0.7 }),
		);
		expect(opts.jev).toEqual({ backend: "openrouter", model: "jev-1.13", confidenceThreshold: 0.7 });
	});

	it("maps false and enabled:false to the agent opt-out", () => {
		expect(buildAgentOptions(baseConfig(false)).jev).toBe(false);
		expect(buildAgentOptions(baseConfig({ enabled: false })).jev).toBe(false);
	});

	it("enables the tool with an empty section", () => {
		expect(buildAgentOptions(baseConfig({})).jev).toEqual({});
	});

	it("rejects unknown backends and fields", () => {
		expect(() => buildAgentOptions(baseConfig({ backend: "not-a-backend" as never }))).toThrow(ConfigError);
		expect(() => normalizeJevConfig({ provider: "openrouter" })).toThrow(ConfigError);
		expect(() => normalizeJevConfig("openrouter")).toThrow(ConfigError);
	});

	it("rejects an out-of-range confidence threshold", () => {
		expect(() => buildAgentOptions(baseConfig({ confidenceThreshold: -0.1 }))).toThrow(ConfigError);
		expect(() => buildAgentOptions(baseConfig({ confidenceThreshold: 1.5 }))).toThrow(ConfigError);
	});
});

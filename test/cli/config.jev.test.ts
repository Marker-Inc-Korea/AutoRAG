import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	buildAgentOptions,
	type CliConfig,
	ConfigError,
	normalizeJevConfig,
	resolveConfig,
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

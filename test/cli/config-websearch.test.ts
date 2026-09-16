import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { buildAgentOptions, type CliConfig, ConfigError } from "../../src/cli/config.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-config-websearch-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function baseConfig(webSearch?: CliConfig["webSearch"]): CliConfig {
	return {
		searchPaths: ["."],
		workspacePath: root,
		memoryPath: join(root, "memory.json"),
		minSync: { enabled: false },
		...(webSearch === undefined ? {} : { webSearch }),
	};
}

describe("webSearch CLI config", () => {
	it("leaves web tools at the credential-free default when unconfigured", () => {
		const opts = buildAgentOptions(baseConfig());
		expect(opts.webSearch).toBeUndefined();
	});

	it("maps enabled:false to the agent opt-out", () => {
		const opts = buildAgentOptions(baseConfig({ enabled: false }));
		expect(opts.webSearch).toBe(false);
	});

	it("passes provider, order, exclusion, and timeouts through", () => {
		const opts = buildAgentOptions(
			baseConfig({
				provider: "duckduckgo",
				order: ["google", "duckduckgo"],
				exclude: ["google"],
				timeoutSeconds: 45,
				fetch: { timeoutSeconds: 20 },
			}),
		);
		expect(opts.webSearch).toEqual({
			provider: "duckduckgo",
			order: ["google", "duckduckgo"],
			exclude: ["google"],
			timeoutSeconds: 45,
			fetch: { timeoutSeconds: 20 },
		});
	});

	it("maps fetch:false to the web_fetch opt-out", () => {
		const opts = buildAgentOptions(baseConfig({ fetch: false }));
		expect(opts.webSearch).toEqual({ fetch: false });
	});

	it("rejects unknown provider ids", () => {
		expect(() => buildAgentOptions(baseConfig({ provider: "not-a-provider" }))).toThrow(ConfigError);
		expect(() => buildAgentOptions(baseConfig({ order: ["duckduckgo", "nope"] }))).toThrow(ConfigError);
		expect(() => buildAgentOptions(baseConfig({ exclude: ["nope"] }))).toThrow(ConfigError);
	});

	it("rejects out-of-range timeouts", () => {
		expect(() => buildAgentOptions(baseConfig({ timeoutSeconds: 0 }))).toThrow(ConfigError);
		expect(() => buildAgentOptions(baseConfig({ timeoutSeconds: 301 }))).toThrow(ConfigError);
		expect(() => buildAgentOptions(baseConfig({ fetch: { timeoutSeconds: 0 } }))).toThrow(ConfigError);
	});
});

import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	buildAgentOptions,
	type CliConfig,
	ConfigError,
	resolveConfig,
	writeDefaultConfig,
} from "../../src/cli/config.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-refresh-config-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function resolveFrom(extra: Record<string, unknown>): CliConfig {
	const path = join(root, "config.json");
	mkdirSync(root, { recursive: true });
	writeFileSync(
		path,
		JSON.stringify({
			searchPaths: ["."],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			...extra,
		}),
	);
	return resolveConfig({ flags: { config: path }, cwd: root, env: {} });
}

describe("refresh config normalization", () => {
	it("stays undefined when the key is absent", () => {
		expect(resolveFrom({}).refresh).toBeUndefined();
	});

	it("accepts every refresh field and maps it onto the agent option", () => {
		const all = { maxDeleteRatio: 0.5, maxScannedEntries: 1000, maxDepth: 8, transientRetries: 2 };
		const config = resolveFrom({ refresh: all });

		expect(config.refresh).toEqual(all);
		expect(buildAgentOptions(config).refresh).toEqual(all);
	});

	it("accepts transientRetries including 0", () => {
		expect(resolveFrom({ refresh: { transientRetries: 5 } }).refresh).toEqual({ transientRetries: 5 });
		const zero = resolveFrom({ refresh: { transientRetries: 0 } });
		expect(zero.refresh).toEqual({ transientRetries: 0 });
		expect(buildAgentOptions(zero).refresh).toEqual({ transientRetries: 0 });
	});

	it("accepts the cloud download budget and eviction switch and maps them onto the agent option", () => {
		const cloud = { cloudMaxFiles: 50, cloudMaxBytes: 1048576, cloudEvictAfterParse: false };
		const config = resolveFrom({ refresh: cloud });

		expect(config.refresh).toEqual(cloud);
		expect(buildAgentOptions(config).refresh).toEqual(cloud);
	});

	it("rejects invalid cloud refresh values", () => {
		expect(() => resolveFrom({ refresh: { cloudMaxFiles: 0 } })).toThrow(ConfigError);
		expect(() => resolveFrom({ refresh: { cloudMaxFiles: 1.5 } })).toThrow(ConfigError);
		expect(() => resolveFrom({ refresh: { cloudMaxBytes: -1 } })).toThrow(ConfigError);
		expect(() => resolveFrom({ refresh: { cloudEvictAfterParse: "no" } })).toThrow(ConfigError);
	});

	it("accepts the ratio boundaries 0 and 1", () => {
		expect(resolveFrom({ refresh: { maxDeleteRatio: 0 } }).refresh).toEqual({ maxDeleteRatio: 0 });
		expect(resolveFrom({ refresh: { maxDeleteRatio: 1 } }).refresh).toEqual({ maxDeleteRatio: 1 });
	});

	it("rejects an unknown refresh key", () => {
		expect(() => resolveFrom({ refresh: { bogus: 1 } })).toThrow(ConfigError);
	});

	it("rejects a non-object refresh section", () => {
		expect(() => resolveFrom({ refresh: 5 })).toThrow(ConfigError);
		expect(() => resolveFrom({ refresh: [] })).toThrow(ConfigError);
	});

	it("rejects out-of-range values", () => {
		expect(() => resolveFrom({ refresh: { maxDeleteRatio: 1.5 } })).toThrow(ConfigError);
		expect(() => resolveFrom({ refresh: { maxDeleteRatio: -0.1 } })).toThrow(ConfigError);
		expect(() => resolveFrom({ refresh: { maxDeleteRatio: "0.3" } })).toThrow(ConfigError);
		expect(() => resolveFrom({ refresh: { maxScannedEntries: 0 } })).toThrow(ConfigError);
		expect(() => resolveFrom({ refresh: { maxDepth: -1 } })).toThrow(ConfigError);
		expect(() => resolveFrom({ refresh: { maxDepth: 1.5 } })).toThrow(ConfigError);
		expect(() => resolveFrom({ refresh: { transientRetries: -1 } })).toThrow(ConfigError);
		expect(() => resolveFrom({ refresh: { transientRetries: 1.5 } })).toThrow(ConfigError);
		expect(() => resolveFrom({ refresh: { transientRetries: "3" } })).toThrow(ConfigError);
	});

	it("persists refresh through writeDefaultConfig and reads it back", () => {
		const path = join(root, "written.json");
		writeDefaultConfig(path, { workspacePath: root, refresh: { maxDeleteRatio: 0.2 } }, { cwd: root, env: {} });

		const config = resolveConfig({ flags: { config: path }, cwd: root, env: {} });

		expect(config.refresh).toEqual({ maxDeleteRatio: 0.2 });
	});
});

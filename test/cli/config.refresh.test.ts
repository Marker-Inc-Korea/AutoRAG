import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { buildAgentOptions, type CliConfig, ConfigError, resolveConfig } from "../../src/cli/config.ts";

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

	it("accepts transientRetries including 0 and carries it into the agent options", () => {
		expect(resolveFrom({ refresh: { transientRetries: 5 } }).refresh).toEqual({ transientRetries: 5 });
		const zero = resolveFrom({ refresh: { transientRetries: 0 } });
		expect(zero.refresh).toEqual({ transientRetries: 0 });
		expect(buildAgentOptions(zero).refresh).toEqual({ transientRetries: 0 });
	});

	it("rejects unknown fields and invalid retry counts", () => {
		expect(() => resolveFrom({ refresh: { bogus: 1 } })).toThrow(ConfigError);
		expect(() => resolveFrom({ refresh: { transientRetries: -1 } })).toThrow(ConfigError);
		expect(() => resolveFrom({ refresh: { transientRetries: 1.5 } })).toThrow(ConfigError);
		expect(() => resolveFrom({ refresh: { transientRetries: "3" } })).toThrow(ConfigError);
		expect(() => resolveFrom({ refresh: [] })).toThrow(ConfigError);
	});
});

import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { buildAgentOptions, ConfigError, resolveConfig, writeDefaultConfig } from "../../src/cli/config.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-refresh-config-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function resolveFrom(extra: Record<string, unknown>) {
	const path = join(root, "config.json");
	mkdirSync(root, { recursive: true });
	writeFileSync(path, JSON.stringify({ workspacePath: root, searchPaths: [root], ...extra }));
	return resolveConfig({ flags: { config: path }, cwd: root, env: {} });
}

describe("refresh config normalization", () => {
	it("stays undefined when the key is absent", () => {
		expect(resolveFrom({}).refresh).toBeUndefined();
	});

	it("accepts every refresh safety field and maps it onto the agent option", () => {
		const config = resolveFrom({ refresh: { maxDeleteRatio: 0.5, maxScannedEntries: 1000, maxDepth: 8 } });

		expect(config.refresh).toEqual({ maxDeleteRatio: 0.5, maxScannedEntries: 1000, maxDepth: 8 });
		expect(buildAgentOptions(config).refresh).toEqual({ maxDeleteRatio: 0.5, maxScannedEntries: 1000, maxDepth: 8 });
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
	});

	it("persists refresh through writeDefaultConfig and reads it back", () => {
		const path = join(root, "written.json");
		writeDefaultConfig(path, { workspacePath: root, refresh: { maxDeleteRatio: 0.2 } }, { cwd: root, env: {} });

		const config = resolveConfig({ flags: { config: path }, cwd: root, env: {} });

		expect(config.refresh).toEqual({ maxDeleteRatio: 0.2 });
	});
});

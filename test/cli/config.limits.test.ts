import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { buildAgentOptions, ConfigError, resolveConfig } from "../../src/cli/config.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-limits-config-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function writeConfig(extra: Record<string, unknown>): string {
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
	return path;
}

function resolveFrom(extra: Record<string, unknown>) {
	const path = writeConfig(extra);
	return resolveConfig({ flags: { config: path }, cwd: root, env: {} });
}

describe("limits config normalization", () => {
	it("stays undefined when the key is absent", () => {
		expect(resolveFrom({}).limits).toBeUndefined();
	});

	it("accepts every limit field including nested prefetch", () => {
		const config = resolveFrom({
			limits: {
				mergedEvidenceCeiling: 1000,
				singleDatasourceTopK: 80,
				minSyncTopK: 40,
				minSyncScopedQueryTopK: 200,
				toolDescriptionInstanceScopes: 3,
				prefetch: { jikjiTopK: 12, minSyncTopK: 250, jikjiPathLimit: 40, sectionLimit: 30 },
			},
		});

		expect(config.limits).toEqual({
			mergedEvidenceCeiling: 1000,
			singleDatasourceTopK: 80,
			minSyncTopK: 40,
			minSyncScopedQueryTopK: 200,
			toolDescriptionInstanceScopes: 3,
			prefetch: { jikjiTopK: 12, minSyncTopK: 250, jikjiPathLimit: 40, sectionLimit: 30 },
		});
	});

	it("rejects an unknown top-level limit key", () => {
		expect(() => resolveFrom({ limits: { bogus: 1 } })).toThrow(ConfigError);
	});

	it("rejects an unknown prefetch key", () => {
		expect(() => resolveFrom({ limits: { prefetch: { bogus: 1 } } })).toThrow(ConfigError);
	});

	it("rejects non-positive and non-integer values", () => {
		expect(() => resolveFrom({ limits: { mergedEvidenceCeiling: 0 } })).toThrow(ConfigError);
		expect(() => resolveFrom({ limits: { minSyncTopK: -1 } })).toThrow(ConfigError);
		expect(() => resolveFrom({ limits: { prefetch: { sectionLimit: 2.5 } } })).toThrow(ConfigError);
	});

	it("maps limits onto the agent option", () => {
		const config = resolveFrom({ limits: { prefetch: { sectionLimit: 30 } } });

		expect(buildAgentOptions(config).limits).toEqual({ prefetch: { sectionLimit: 30 } });
	});
});

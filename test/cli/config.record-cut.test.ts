import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { ConfigError, resolveConfig } from "../../src/cli/config.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-record-cut-config-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function resolveFrom(limits: unknown) {
	const path = join(root, "config.json");
	mkdirSync(root, { recursive: true });
	writeFileSync(
		path,
		JSON.stringify({ searchPaths: ["."], workspacePath: root, memoryPath: join(root, "memory.json"), limits }),
	);
	return resolveConfig({ flags: { config: path }, cwd: root, env: {} });
}

describe("limits.recordCut config", () => {
	it("stays absent in the resolved config when not written: the default lives in one place, the cut itself", () => {
		expect(resolveFrom({ minSyncTopK: 40 }).limits?.recordCut).toBeUndefined();
	});

	it("accepts enabled:false as the way to turn the cut off", () => {
		expect(resolveFrom({ recordCut: { enabled: false } }).limits?.recordCut).toEqual({ enabled: false });
	});

	it("accepts the switch and every tuning field", () => {
		const config = resolveFrom({
			recordCut: {
				enabled: true,
				depth: 4,
				newRecordShare: 0.25,
				perRecordFloor: 3,
				perRecordDivisor: 5,
				resultChars: 90_000,
				minReadChars: 1_500,
			},
		});
		expect(config.limits?.recordCut).toEqual({
			enabled: true,
			depth: 4,
			newRecordShare: 0.25,
			perRecordFloor: 3,
			perRecordDivisor: 5,
			resultChars: 90_000,
			minReadChars: 1_500,
		});
	});

	it.each([
		[{ enabled: "yes" }, /limits\.recordCut\.enabled must be a boolean/u],
		[{ depth: 0 }, /limits\.recordCut\.depth must be an integer from 1 to 10/u],
		[{ depth: 11 }, /limits\.recordCut\.depth must be an integer from 1 to 10/u],
		[{ depth: 2.5 }, /limits\.recordCut\.depth/u],
		[{ newRecordShare: 1.1 }, /limits\.recordCut\.newRecordShare must be a number from 0 to 1/u],
		[{ newRecordShare: -0.1 }, /limits\.recordCut\.newRecordShare/u],
		[{ perRecordFloor: 0 }, /limits\.recordCut\.perRecordFloor must be an integer from 1 to 300/u],
		[{ perRecordDivisor: 301 }, /limits\.recordCut\.perRecordDivisor/u],
		[{ resultChars: 0 }, /limits\.recordCut\.resultChars must be a positive integer/u],
		[{ minReadChars: -1 }, /limits\.recordCut\.minReadChars must be a positive integer/u],
		[{ typo: true }, /limits\.recordCut\.typo is not a recognized field/u],
	])("rejects %j", (recordCut, message) => {
		expect(() => resolveFrom({ recordCut })).toThrow(ConfigError);
		expect(() => resolveFrom({ recordCut })).toThrow(message);
	});

	it("rejects a non-object section", () => {
		expect(() => resolveFrom({ recordCut: true })).toThrow(/limits\.recordCut must be an object/u);
	});
});

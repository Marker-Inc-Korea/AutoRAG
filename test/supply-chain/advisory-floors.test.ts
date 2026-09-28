import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), "../..");

const ADVISORY_FLOORS = {
	xlsx: "0.20.2",
	unstructured: "0.24.0",
	aiohttp: "3.14.3",
} as const;

function compareVersions(left: string, right: string): number {
	const leftParts = left.split(".").map((part) => Number(part));
	const rightParts = right.split(".").map((part) => Number(part));
	const length = Math.max(leftParts.length, rightParts.length);
	for (let index = 0; index < length; index += 1) {
		const leftPart = leftParts[index] ?? 0;
		const rightPart = rightParts[index] ?? 0;
		if (!Number.isInteger(leftPart) || !Number.isInteger(rightPart)) {
			throw new Error(`non-numeric version ${left} vs ${right}`);
		}
		if (leftPart !== rightPart) return leftPart < rightPart ? -1 : 1;
	}
	return 0;
}

function xlsxResolvedVersion(lock: string): string {
	const match = /"xlsx": \["xlsx@([^"]+)"/.exec(lock);
	const spec = match?.[1];
	if (spec === undefined) throw new Error("xlsx is missing from bun.lock");
	const version = /(\d+\.\d+\.\d+)/.exec(spec)?.[1];
	if (version === undefined) throw new Error(`unparsed xlsx spec ${spec}`);
	return version;
}

function uvPackageVersions(lock: string, name: string): readonly string[] {
	const pattern = new RegExp(`\\[\\[package\\]\\]\\r?\\nname = "${name}"\\r?\\nversion = "([^"]+)"`, "g");
	const versions: string[] = [];
	for (const match of lock.matchAll(pattern)) {
		const version = match[1];
		if (version === undefined) throw new Error(`missing version for ${name}`);
		versions.push(version);
	}
	if (versions.length === 0) throw new Error(`${name} is missing from legacy/uv.lock`);
	return versions;
}

describe("issue 1677 advisory floors", () => {
	const bunLock = readFileSync(join(repoRoot, "bun.lock"), "utf8");
	const uvLock = readFileSync(join(repoRoot, "legacy/uv.lock"), "utf8");

	it("resolves xlsx outside the prototype-pollution and ReDoS ranges", () => {
		expect(compareVersions(xlsxResolvedVersion(bunLock), ADVISORY_FLOORS.xlsx) >= 0).toBe(true);
	});

	it("locks unstructured at the patched 0.24.0 floor", () => {
		const versions = uvPackageVersions(uvLock, "unstructured");
		expect(versions.length).toBeGreaterThan(0);
		for (const version of versions) {
			expect(compareVersions(version, ADVISORY_FLOORS.unstructured) >= 0).toBe(true);
		}
	});

	it("locks accelerate outside GHSA-4j2p-28q2-5m79", () => {
		for (const version of uvPackageVersions(uvLock, "accelerate")) {
			expect(compareVersions(version, "1.14.0") > 0).toBe(true);
		}
	});

	it("locks aiohttp at the 3.14.3 floor", () => {
		for (const version of uvPackageVersions(uvLock, "aiohttp")) {
			expect(compareVersions(version, ADVISORY_FLOORS.aiohttp) >= 0).toBe(true);
		}
	});

	it("does not invent a chromadb or nltk bump past an unpatched range", () => {
		expect(uvPackageVersions(uvLock, "chromadb")).toEqual(["1.5.9"]);
		expect(uvPackageVersions(uvLock, "nltk")).toEqual(["3.10.3"]);
	});
});

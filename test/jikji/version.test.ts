import { describe, expect, it } from "vitest";
import {
	isSupportedJikjiVersion,
	JIKJI_INSTALL_VERSION,
	JIKJI_MIN_VERSION,
	parseJikjiVersion,
} from "../../src/jikji/version.ts";

describe("parseJikjiVersion", () => {
	it("reads the version from `jikji --version` output", () => {
		expect(parseJikjiVersion("jikji 0.2.1\n")).toBe("0.2.1");
		expect(parseJikjiVersion("jikji-cli 10.20.30")).toBe("10.20.30");
	});

	it("drops a pre-release or build suffix", () => {
		expect(parseJikjiVersion("jikji 0.2.1-dev+abc123")).toBe("0.2.1");
	});

	it("returns undefined for output that is not a jikji version line", () => {
		expect(parseJikjiVersion("")).toBeUndefined();
		expect(parseJikjiVersion('{"not":"a pack"}')).toBeUndefined();
		expect(parseJikjiVersion("jikji unknown")).toBeUndefined();
		expect(parseJikjiVersion("minsync 0.5.1")).toBeUndefined();
	});
});

describe("isSupportedJikjiVersion", () => {
	it("rejects releases below the floor", () => {
		// 0.1.1 has no `find --no-background-refresh`, so every AutoRAG find exits 2 (clap: unexpected argument).
		expect(isSupportedJikjiVersion("0.1.1")).toBe(false);
		expect(isSupportedJikjiVersion("0.1.99")).toBe(false);
	});

	it("puts the floor at the first release whose find accepts every flag AutoRAG passes", () => {
		expect(JIKJI_MIN_VERSION).toBe("0.2.0");
		expect(isSupportedJikjiVersion("0.2.0")).toBe(true);
	});

	it("accepts the floor and anything above it, comparing numerically not lexically", () => {
		expect(isSupportedJikjiVersion(JIKJI_MIN_VERSION)).toBe(true);
		expect(isSupportedJikjiVersion("0.2.10")).toBe(true);
		expect(isSupportedJikjiVersion("0.2.1")).toBe(true);
		expect(isSupportedJikjiVersion("0.10.0")).toBe(true);
		expect(isSupportedJikjiVersion("1.0.0")).toBe(true);
	});

	it("only ever installs a version that satisfies its own floor", () => {
		expect(isSupportedJikjiVersion(JIKJI_INSTALL_VERSION)).toBe(true);
	});

	it("installs 0.2.1, which carries the Windows rooted-cache-path fix 0.2.0 lacks", () => {
		expect(JIKJI_INSTALL_VERSION).toBe("0.2.1");
	});
});

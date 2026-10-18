import { describe, expect, it } from "vitest";
import { type KiwiProbeEnvironment, kiwiAvailable } from "../../src/minsync/kiwi.ts";

/** A probe environment where exactly `present` exist on disk. */
function probe(
	platform: NodeJS.Platform,
	present: readonly string[],
	extra: Partial<KiwiProbeEnvironment> = {},
): KiwiProbeEnvironment {
	return {
		env: {},
		platform,
		home: platform === "win32" ? "C:\\Users\\u" : "/home/u",
		exists: (path) => present.includes(path),
		...extra,
	};
}

describe("kiwiAvailable", () => {
	it("is false when neither the library nor the model is installed", () => {
		expect(kiwiAvailable(probe("linux", []))).toBe(false);
	});

	it("needs both the library and the model", () => {
		const library = "/home/u/.local/kiwi/lib/libkiwi.so";
		const model = "/home/u/.local/kiwi/models/cong/base";
		expect(kiwiAvailable(probe("linux", [library, model]))).toBe(true);
		expect(kiwiAvailable(probe("linux", [library]))).toBe(false);
		expect(kiwiAvailable(probe("linux", [model]))).toBe(false);
	});

	it("finds system-wide installs", () => {
		expect(kiwiAvailable(probe("linux", ["/usr/local/lib/libkiwi.so", "/usr/share/kiwi/models/cong/base"]))).toBe(
			true,
		);
		expect(
			kiwiAvailable(probe("darwin", ["/opt/homebrew/lib/libkiwi.dylib", "/opt/homebrew/models/cong/base"])),
		).toBe(true);
	});

	it("looks for the library file name of the running platform", () => {
		const present = ["/home/u/.local/kiwi/lib/libkiwi.so", "/home/u/.local/kiwi/models/cong/base"];
		expect(kiwiAvailable(probe("darwin", present))).toBe(false);
	});

	it("finds the per-user Windows install", () => {
		const env = { LOCALAPPDATA: "C:\\Users\\u\\AppData\\Local" };
		const present = [
			"C:\\Users\\u\\AppData\\Local\\kiwi\\lib\\kiwi.dll",
			"C:\\Users\\u\\AppData\\Local\\kiwi\\models\\cong\\base",
		];
		expect(kiwiAvailable(probe("win32", present, { env }))).toBe(true);
	});

	it("honors KIWI_LIBRARY_PATH and KIWI_MODEL_PATH", () => {
		const env = { KIWI_LIBRARY_PATH: "/opt/kiwi/libkiwi.so", KIWI_MODEL_PATH: "/opt/kiwi/model" };
		expect(kiwiAvailable(probe("linux", ["/opt/kiwi/libkiwi.so", "/opt/kiwi/model"], { env }))).toBe(true);
	});

	it("does not fall back to a default install when an explicit path is wrong", () => {
		// Kiwi's own loader treats an explicit path as authoritative, so the default location cannot rescue it.
		const env = { KIWI_LIBRARY_PATH: "/opt/kiwi/missing.so" };
		const present = ["/home/u/.local/kiwi/lib/libkiwi.so", "/home/u/.local/kiwi/models/cong/base"];
		expect(kiwiAvailable(probe("linux", present, { env }))).toBe(false);
	});
});

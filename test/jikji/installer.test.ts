import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { JikjiClient } from "../../src/jikji/client.ts";
import {
	cachedJikjiBinaryPath,
	ensureJikjiBinary,
	jikjiExecutableName,
	lookupExecutableInPath,
} from "../../src/jikji/installer.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-jikji-installer-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

describe("lookupExecutableInPath", () => {
	it("finds an executable in a PATH directory", () => {
		const binDir = join(root, "bin");
		mkdirSync(binDir, { recursive: true });
		writeFileSync(join(binDir, "jikji"), "#!/bin/sh\n");
		expect(lookupExecutableInPath("jikji", { PATH: binDir })).toBe(join(binDir, "jikji"));
	});

	it("returns undefined when PATH is empty or missing", () => {
		expect(lookupExecutableInPath("jikji", {})).toBeUndefined();
		expect(lookupExecutableInPath("jikji", { PATH: "" })).toBeUndefined();
	});
});

describe("ensureJikjiBinary", () => {
	it("returns the cached binary without invoking cargo", async () => {
		const cached = cachedJikjiBinaryPath(root);
		mkdirSync(join(root, ".autorag", "bin"), { recursive: true });
		writeFileSync(cached, "binary");
		const result = await ensureJikjiBinary({
			root,
			runner: () => {
				throw new Error("runner must not be called for a cached binary");
			},
		});
		expect(result).toEqual({ ok: true, binaryPath: cached, source: "cached" });
	});

	it("degrades with missing-cargo when no Rust toolchain is on PATH", async () => {
		const result = await ensureJikjiBinary({
			root,
			env: { PATH: join(root, "empty") },
			runner: () => {
				throw new Error("runner must not be called without cargo");
			},
		});
		expect(result.ok).toBe(false);
		if (!result.ok) expect(result.reason).toBe("missing-cargo");
	});

	it("installs through cargo and reports the cached binary path", async () => {
		const calls: string[][] = [];
		const result = await ensureJikjiBinary({
			root,
			env: { PATH: join(root, "cargo-bin") },
			cargoLocator: () => "/usr/bin/cargo",
			runner: async (args) => {
				calls.push([...args]);
				mkdirSync(join(root, ".autorag", "bin"), { recursive: true });
				writeFileSync(cachedJikjiBinaryPath(root), "binary");
				return { code: 0, stderr: "" };
			},
		});
		expect(result).toEqual({ ok: true, binaryPath: cachedJikjiBinaryPath(root), source: "installed" });
		expect(calls).toEqual([["install", "jikji-cli", "--locked", "--root", join(root, ".autorag")]]);
	});

	it("pins the version when requested", async () => {
		const calls: string[][] = [];
		await ensureJikjiBinary({
			root,
			version: "0.1.1",
			cargoLocator: () => "/usr/bin/cargo",
			runner: async (args) => {
				calls.push([...args]);
				mkdirSync(join(root, ".autorag", "bin"), { recursive: true });
				writeFileSync(cachedJikjiBinaryPath(root), "binary");
				return { code: 0, stderr: "" };
			},
		});
		expect(calls[0]).toContain("--version");
		expect(calls[0]).toContain("0.1.1");
	});

	it("degrades with install-failed when cargo exits nonzero", async () => {
		const result = await ensureJikjiBinary({
			root,
			cargoLocator: () => "/usr/bin/cargo",
			runner: async () => ({ code: 101, stderr: "error: could not compile" }),
		});
		expect(result.ok).toBe(false);
		if (!result.ok) {
			expect(result.reason).toBe("install-failed");
			expect(result.message).toContain("could not compile");
		}
	});
});

describe("JikjiClient binary resolution", () => {
	it("uses the cached .autorag/bin binary when PATH has no jikji", async () => {
		const cached = cachedJikjiBinaryPath(root);
		mkdirSync(join(root, ".autorag", "bin"), { recursive: true });
		writeFileSync(cached, '#!/bin/sh\necho \'{"not":"a pack"}\'\n');
		chmodSync(cached, 0o755);
		const client = new JikjiClient({ root, autoInstall: false, timeoutMs: 5_000 });
		const result = await client.find(root, "query");
		// The fixture emits invalid JSON, proving the cached binary ran.
		expect(result.ok).toBe(false);
		if (!result.ok) expect(result.reason).toBe("bad-answer-pack");
	});

	it("falls back to the bare jikji command when autoInstall is disabled and nothing is installed", async () => {
		const client = new JikjiClient({ root, autoInstall: false, timeoutMs: 5_000 });
		// No binary anywhere: resolution falls back to `jikji`, which spawns or
		// fails exactly as before this change (no throw either way).
		const result = await client.find(root, "query");
		expect(typeof result.ok).toBe("boolean");
	});

	it("names the executable jikji.exe on win32", () => {
		expect(jikjiExecutableName("win32")).toBe("jikji.exe");
		expect(jikjiExecutableName("darwin")).toBe("jikji");
	});
});

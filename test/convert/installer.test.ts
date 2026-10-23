import { createHash } from "node:crypto";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
	ensureFileeBinary,
	FILEE_CLI_ASSETS,
	FILEE_CLI_VERSION,
	FileeCliError,
	FileeInstallError,
	resolveFileeBinary,
	resolveFileeOptions,
	selectFileeAsset,
	warmFileeBinary,
} from "../../src/convert/index.ts";

const roots: string[] = [];
afterEach(() => {
	for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});
function tempRoot(): string {
	const root = mkdtempSync(join(tmpdir(), "filee-install-"));
	roots.push(root);
	return root;
}

/** Archive bytes the fake downloader writes; the fake extractor lays out the real package shape. */
const ARCHIVE_BYTES = Buffer.from("fake filee-cli archive");
const ARCHIVE_SHA = createHash("sha256").update(ARCHIVE_BYTES).digest("hex");

function seams(platform: NodeJS.Platform, arch: NodeJS.Architecture, calls: string[] = []) {
	const asset = selectFileeAsset(platform, arch);
	return {
		platform,
		arch,
		assets: { ...FILEE_CLI_ASSETS, [`${platform}-${arch}`]: { ...asset, sha256: ARCHIVE_SHA } },
		async download(url: string, destination: string) {
			calls.push(`download ${url}`);
			writeFileSync(destination, ARCHIVE_BYTES);
		},
		async extract(_archive: string, directory: string) {
			calls.push("extract");
			const folder = join(directory, asset.name.replace(/\.(tar\.gz|zip)$/, ""));
			mkdirSync(join(folder, "engines"), { recursive: true });
			const binary = join(folder, platform === "win32" ? "filee-cli.exe" : "filee-cli");
			writeFileSync(binary, "#!/bin/sh\n");
			chmodSync(binary, 0o755);
		},
	};
}

describe("selectFileeAsset", () => {
	it("pins one release asset per supported platform, with a sha256 and the release url", () => {
		const targets: [NodeJS.Platform, NodeJS.Architecture, string][] = [
			["darwin", "arm64", "osx-arm64.tar.gz"],
			["darwin", "x64", "osx-x64.tar.gz"],
			["linux", "x64", "linux-x64.tar.gz"],
			["linux", "arm64", "linux-arm64.tar.gz"],
			["win32", "x64", "win-x64.zip"],
		];
		for (const [platform, arch, suffix] of targets) {
			const asset = selectFileeAsset(platform, arch);
			expect(asset.name).toBe(`filee-cli-${FILEE_CLI_VERSION}-${suffix}`);
			expect(asset.sha256).toMatch(/^[0-9a-f]{64}$/);
			expect(asset.url).toBe(
				`https://github.com/NomaDamas/filee-cli/releases/download/v${FILEE_CLI_VERSION}/${asset.name}`,
			);
		}
	});

	it("rejects a platform the project does not publish", () => {
		expect(() => selectFileeAsset("win32", "arm64")).toThrow(FileeInstallError);
		expect(() => selectFileeAsset("freebsd", "x64")).toThrow(/Unsupported filee-cli platform: freebsd\/x64/);
	});
});

describe("ensureFileeBinary", () => {
	it("downloads, verifies the pinned sha256, extracts, and returns the binary inside the cache", async () => {
		const home = tempRoot();
		const calls: string[] = [];
		const installed = await ensureFileeBinary({ home, ...seams("darwin", "arm64", calls) });
		expect(installed.source).toBe("installed");
		expect(installed.binaryPath).toBe(join(home, "tools", "filee-cli", FILEE_CLI_VERSION, "filee-cli"));
		expect(existsSync(join(home, "tools", "filee-cli", FILEE_CLI_VERSION, "engines"))).toBe(true);
		expect(calls[0]).toContain("/releases/download/v");
		expect(calls).toContain("extract");
	});

	it("uses .exe on Windows", async () => {
		const home = tempRoot();
		const installed = await ensureFileeBinary({ home, ...seams("win32", "x64") });
		expect(installed.binaryPath.endsWith("filee-cli.exe")).toBe(true);
	});

	it("is a no-op the second time: the cache answers without downloading", async () => {
		const home = tempRoot();
		await ensureFileeBinary({ home, ...seams("linux", "x64") });
		const calls: string[] = [];
		const again = await ensureFileeBinary({ home, ...seams("linux", "x64", calls) });
		expect(again.source).toBe("cached");
		expect(calls).toEqual([]);
	});

	it("refuses an archive whose sha256 differs from the pin and leaves nothing behind", async () => {
		const home = tempRoot();
		const base = seams("linux", "x64");
		await expect(
			ensureFileeBinary({
				home,
				...base,
				async download(_url, destination) {
					writeFileSync(destination, "tampered");
				},
			}),
		).rejects.toThrow(/sha256 mismatch/);
		const dir = join(home, "tools", "filee-cli");
		expect(existsSync(join(dir, FILEE_CLI_VERSION))).toBe(false);
		expect(existsSync(dir) ? readdirSync(dir) : []).toEqual([]);
	});

	it("fails clearly when the archive does not contain the binary", async () => {
		const home = tempRoot();
		await expect(
			ensureFileeBinary({
				home,
				...seams("linux", "x64"),
				async extract(_archive, directory) {
					mkdirSync(join(directory, "empty"), { recursive: true });
				},
			}),
		).rejects.toThrow(/did not contain/);
		expect(existsSync(join(home, "tools", "filee-cli", FILEE_CLI_VERSION))).toBe(false);
	});

	it("survives two concurrent installers: both resolve to one complete install", async () => {
		const home = tempRoot();
		const [a, b] = await Promise.all([
			ensureFileeBinary({ home, ...seams("linux", "x64") }),
			ensureFileeBinary({ home, ...seams("linux", "x64") }),
		]);
		expect(a.binaryPath).toBe(b.binaryPath);
		expect(existsSync(a.binaryPath)).toBe(true);
		expect(readdirSync(join(home, "tools", "filee-cli"))).toEqual([FILEE_CLI_VERSION]);
	});
});

describe("resolveFileeBinary", () => {
	function pathWith(name: string): { dir: string; env: NodeJS.ProcessEnv; binary: string } {
		const dir = tempRoot();
		const binary = join(dir, name);
		writeFileSync(binary, "#!/bin/sh\n");
		chmodSync(binary, 0o755);
		return { dir, binary, env: { PATH: [dir, "/nonexistent"].join(delimiter) } };
	}

	it("prefers the configured executable, then PATH, then the managed cache", async () => {
		const home = tempRoot();
		const onPath = pathWith("filee-cli");
		const configured = join(tempRoot(), "mine");
		writeFileSync(configured, "x");

		const first = await resolveFileeBinary({ home, env: onPath.env, executable: configured });
		expect(first).toEqual({ path: configured, source: "configured" });

		const second = await resolveFileeBinary({ home, env: onPath.env });
		expect(second).toEqual({ path: onPath.binary, source: "path" });

		const third = await resolveFileeBinary({ home, env: { PATH: "/nonexistent" }, installer: seams("linux", "x64") });
		expect(third.source).toBe("installed");
		const fourth = await resolveFileeBinary({
			home,
			env: { PATH: "/nonexistent" },
			installer: seams("linux", "x64"),
		});
		expect(fourth).toEqual({ path: third.path, source: "cached" });
	});

	it("does not download when autoInstall is off, and says how to fix it", async () => {
		const home = tempRoot();
		const calls: string[] = [];
		const error = await resolveFileeBinary({
			home,
			env: { PATH: "/nonexistent" },
			autoInstall: false,
			installer: seams("linux", "x64", calls),
		}).catch((e: unknown) => e);
		expect(error).toBeInstanceOf(FileeCliError);
		expect((error as Error).message).toMatch(/filee-cli/);
		expect((error as Error).message).toMatch(/autoInstall/);
		expect(calls).toEqual([]);
	});

	it("reports a failed install as a FileeCliError the tool can show", async () => {
		const home = tempRoot();
		const error = await resolveFileeBinary({
			home,
			env: { PATH: "/nonexistent" },
			installer: {
				...seams("linux", "x64"),
				async download() {
					throw new Error("network down");
				},
			},
		}).catch((e: unknown) => e);
		expect(error).toBeInstanceOf(FileeCliError);
		expect((error as Error).message).toMatch(/network down/);
	});
});

describe("resolveFileeOptions", () => {
	it("fills in the managed executable so convert_files can spawn it", async () => {
		const home = tempRoot();
		const options = await resolveFileeOptions(
			{ timeoutMs: 5, dataDir: "/d" },
			{ home, env: { PATH: "/nonexistent" }, installer: seams("linux", "x64") },
		);
		expect(options.executable).toBe(join(home, "tools", "filee-cli", FILEE_CLI_VERSION, "filee-cli"));
		expect(options.timeoutMs).toBe(5);
		expect(options.dataDir).toBe("/d");
	});

	it("leaves a test seam untouched and never installs for it", async () => {
		const run = async () => ({ stdout: "", stderr: "", exitCode: 0 });
		const options = await resolveFileeOptions({ run }, { env: { PATH: "/nonexistent" } });
		expect(options.run).toBe(run);
		expect(options.executable).toBeUndefined();
	});
});

describe("warmFileeBinary", () => {
	it("installs in the background and resolves to the outcome, never throwing", async () => {
		const home = tempRoot();
		const ok = await warmFileeBinary({}, { home, env: { PATH: "/nonexistent" }, installer: seams("linux", "x64") });
		expect(ok).toMatchObject({ state: "installed" });
		const failed = await warmFileeBinary(
			{},
			{
				home: tempRoot(),
				env: { PATH: "/nonexistent" },
				installer: {
					...seams("linux", "x64"),
					async download() {
						throw new Error("offline");
					},
				},
			},
		);
		expect(failed).toMatchObject({ state: "failed" });
	});

	it("does nothing when conversion is off or auto-install is off", async () => {
		const calls: string[] = [];
		const installer = seams("linux", "x64", calls);
		const deps = { home: tempRoot(), env: { PATH: "/nonexistent" }, installer };
		expect(await warmFileeBinary(false, deps)).toEqual({ state: "skipped" });
		expect(await warmFileeBinary({ autoInstall: false }, deps)).toEqual({ state: "skipped" });
		expect(calls).toEqual([]);
	});
});

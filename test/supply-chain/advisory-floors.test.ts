import { once } from "node:events";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { createServer as createRawServer } from "node:net";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), "../..");

const ADVISORY_FLOORS = {
	xlsx: "0.20.2",
	unstructured: "0.24.0",
	aiohttp: "3.14.3",
	undici6: "6.29.1",
	undici7: "7.30.1",
	undici8: "8.12.2",
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

function bunPackageVersions(lock: string, name: string): readonly string[] {
	const pattern = new RegExp(`"(?:[^"]+/)?${name}": \\["${name}@(\\d+\\.\\d+\\.\\d+)`, "g");
	const versions: string[] = [];
	for (const match of lock.matchAll(pattern)) {
		const version = match[1];
		if (version === undefined) throw new Error(`missing version for ${name}`);
		versions.push(version);
	}
	return versions;
}

function undiciMinimumFloorForMajor(version: string): string {
	const major = Number(version.split(".")[0]);
	if (major === 6) return ADVISORY_FLOORS.undici6;
	if (major === 7) return ADVISORY_FLOORS.undici7;
	if (major === 8) return ADVISORY_FLOORS.undici8;
	if (major > 8) return `${major}.0.0`;
	throw new Error(`unsupported undici major version: ${version}`);
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
		expect(uvPackageVersions(uvLock, "nltk")).toEqual(["3.10.4"]);
	});
});

describe("undici 2026 CVE advisory floors", () => {
	type OverrideEntry = string | Record<string, string>;
	const manifest = JSON.parse(readFileSync(join(repoRoot, "package.json"), "utf8")) as {
		overrides?: Record<string, OverrideEntry>;
	};
	const bunLock = readFileSync(join(repoRoot, "bun.lock"), "utf8");

	function readScopedOverride(
		overrides: Record<string, OverrideEntry>,
		parent: string,
		child: string,
	): string | undefined {
		const flat = overrides[`${parent}/${child}`];
		if (typeof flat === "string") return flat;
		const nested = overrides[parent];
		if (nested && typeof nested === "object" && typeof nested[child] === "string") return nested[child];
		return typeof overrides[child] === "string" ? overrides[child] : undefined;
	}

	it("pins every resolved undici version in bun.lock outside the 8 2026 CVE ranges", () => {
		const versions = bunPackageVersions(bunLock, "undici");
		expect(versions.length).toBeGreaterThan(0);
		for (const version of versions) {
			const floor = undiciMinimumFloorForMajor(version);
			expect(
				compareVersions(version, floor) >= 0,
				`undici@${version} is below patched floor ${floor} (CVE-2026-109383/109869/109924/109932/19534/87018/85024/18540)`,
			).toBe(true);
		}
	});

	it("enforces an explicit undici 6.x floor override in package.json for transitive discord.js / @discordjs/rest consumers", () => {
		const overrides = manifest.overrides ?? {};
		const discordJsOverride = readScopedOverride(overrides, "discord.js", "undici");
		const restOverride = readScopedOverride(overrides, "@discordjs/rest", "undici");
		expect(discordJsOverride, "package.json must override discord.js transitive undici >= 6.29.1").toBeDefined();
		expect(restOverride, "package.json must override @discordjs/rest transitive undici >= 6.29.1").toBeDefined();
		for (const spec of [discordJsOverride, restOverride]) {
			const version = /(\d+\.\d+\.\d+)/.exec(spec ?? "")?.[1];
			expect(version, `override spec ${spec} must contain a semver floor`).toBeDefined();
			expect(compareVersions(version as string, undiciMinimumFloorForMajor(version as string)) >= 0).toBe(true);
		}
	});

	it("resolves patched undici at runtime for both root and discord.js / @discordjs/rest", () => {
		const rootRequire = createRequire(join(repoRoot, "package.json"));
		const discordRequire = createRequire(join(repoRoot, "node_modules/discord.js/package.json"));
		const restRequire = createRequire(join(repoRoot, "node_modules/@discordjs/rest/package.json"));

		const resolved = [
			{ label: "root", pkg: rootRequire("undici/package.json") as { version: string } },
			{ label: "discord.js", pkg: discordRequire("undici/package.json") as { version: string } },
			{ label: "@discordjs/rest", pkg: restRequire("undici/package.json") as { version: string } },
		];
		for (const { label, pkg } of resolved) {
			const floor = undiciMinimumFloorForMajor(pkg.version);
			expect(
				compareVersions(pkg.version, floor) >= 0,
				`${label} resolved undici@${pkg.version}, below floor ${floor}`,
			).toBe(true);
		}
	});

	it("preserves Agent tls.checkServerIdentity callback in discord.js undici (CVE-2026-109383)", () => {
		const discordRequire = createRequire(join(repoRoot, "node_modules/discord.js/package.json"));
		const undici6 = discordRequire("undici") as {
			Agent: new (opts: { tls?: { checkServerIdentity?: () => undefined } }) => { close: () => Promise<void> };
		};
		// Callback identity is verified via strict reference equality against Agent's stored tls option.
		const checkServerIdentity = () => undefined;
		const agent = new undici6.Agent({ tls: { checkServerIdentity } });
		try {
			const optionSymbol = Object.getOwnPropertySymbols(agent).find((symbol) => symbol.description === "options");
			expect(optionSymbol, "undici Agent should store constructor options").toBeDefined();
			const stored = (agent as unknown as Record<symbol, { tls?: { checkServerIdentity?: unknown } }>)[
				optionSymbol as symbol
			];
			expect(stored?.tls?.checkServerIdentity).toBe(checkServerIdentity);
		} finally {
			void agent.close();
		}
	});

	it("rejects excessive Content-Encoding with early socket close cleanly without unhandled error (CVE-2026-109869)", async () => {
		const discordRequire = createRequire(join(repoRoot, "node_modules/discord.js/package.json"));
		const undici6 = discordRequire("undici") as { fetch: typeof fetch };
		const server = createRawServer((socket) => {
			socket.once("data", () => {
				socket.write(
					"HTTP/1.1 200 OK\r\n" +
						"Content-Type: text/plain\r\n" +
						"Content-Encoding: gzip, gzip, gzip, gzip, gzip, gzip\r\n" +
						"Content-Length: 100\r\n" +
						"\r\n",
				);
				socket.destroy();
			});
		});
		server.listen(0, "127.0.0.1");
		await once(server, "listening");
		try {
			const address = server.address();
			if (!address || typeof address === "string") throw new Error("server did not bind");
			await expect(undici6.fetch(`http://127.0.0.1:${address.port}`)).rejects.toThrow();
		} finally {
			server.close();
			await once(server, "close");
		}
	});
});

import { execFileSync } from "node:child_process";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CrawlerCliClient } from "../../src/datasource/crawler-client.ts";
import type { CrawlerProfile } from "../../src/datasource/crawler-types.ts";

const PROFILE: CrawlerProfile = {
	binaryName: "fakecrawl",
	allowedEnvPrefixes: ["FAKECRAWL_"],
	syncArgs: () => ["sync"],
	searchArgs: (_options, query) => ["search", query],
	parseSyncCount: (stdout) => {
		const parsed: unknown = JSON.parse(stdout);
		return typeof parsed === "object" && parsed !== null && "synced" in parsed ? Number(parsed.synced) : undefined;
	},
	parseHits: () => [],
};

let root: string;
let binaryPath: string;
const spawnedPids: number[] = [];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-crawler-client-"));
	mkdirSync(join(root, "bin"), { recursive: true });
	binaryPath = join(root, "bin", "fakecrawl");
});

afterEach(() => {
	for (const pid of spawnedPids.splice(0)) {
		try {
			process.kill(pid, "SIGKILL");
		} catch {}
	}
	rmSync(root, { recursive: true, force: true });
});

/** kill(pid, 0) succeeds for zombies until init reaps them; those are not running. */
function isRunning(pid: number): boolean {
	try {
		process.kill(pid, 0);
	} catch {
		return false;
	}
	try {
		return !/^Z/.test(execFileSync("ps", ["-o", "stat=", "-p", String(pid)], { encoding: "utf8" }).trim());
	} catch {
		return false;
	}
}

function writeScript(body: string): void {
	writeFileSync(binaryPath, `#!/usr/bin/env node\n${body}`);
	chmodSync(binaryPath, 0o755);
}

// Real wall-clock bound on purpose: the failure under test is a promise that never settles,
// which fake timers cannot observe. The tsconfig lib predates Promise.withResolvers.
async function settleWithin<T>(promise: Promise<T>, ms: number, what: string): Promise<T> {
	let timer: ReturnType<typeof setTimeout> | undefined;
	const bound = new Promise<never>((_, reject) => {
		timer = setTimeout(() => reject(new Error(`${what} stayed pending for ${ms}ms`)), ms);
	});
	try {
		return await Promise.race([promise, bound]);
	} finally {
		clearTimeout(timer);
	}
}

describe.skipIf(process.platform === "win32")("CrawlerCliClient process lifecycle", () => {
	it("settles and reaps a descendant that keeps the inherited stdio open after the direct child exits", {
		timeout: 15_000,
	}, async () => {
		writeScript(`import { spawn } from "node:child_process";
const descendant = spawn(process.execPath, ["-e", "setInterval(() => undefined, 1000)"], { stdio: "inherit" });
process.stdout.write(JSON.stringify({ synced: 1, grandchildPid: descendant.pid }));
process.exit(0);
`);
		const client = new CrawlerCliClient(PROFILE, { binaryPath, timeoutMs: 30_000, indexTimeoutMs: 30_000 });

		const result = await settleWithin(client.sync(), 4_000, "crawler sync");

		expect(result.ok).toBe(true);
		const pid = Number(/"grandchildPid":(\d+)/.exec(result.stdout)?.[1]);
		expect(pid).toBeGreaterThan(0);
		spawnedPids.push(pid);
		await vi.waitFor(() => expect(isRunning(pid)).toBe(false), { timeout: 3_000 });
	});

	it("escalates past a SIGTERM-ignoring child and reaps its descendant on timeout", { timeout: 15_000 }, async () => {
		const pidFile = join(root, "descendant.pid");
		writeScript(`import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";
process.on("SIGTERM", () => undefined);
const descendant = spawn(process.execPath, ["-e", "process.on('SIGTERM', () => undefined); setInterval(() => undefined, 1000)"], { stdio: "inherit" });
writeFileSync(${JSON.stringify(pidFile)}, String(descendant.pid));
setInterval(() => undefined, 1000);
`);
		// The wall must outlive Node startup under load, or the fixture is killed before it records the
		// descendant pid. A fixed 500ms raced that and failed intermittently with the pid file missing.
		const client = new CrawlerCliClient(PROFILE, { binaryPath, timeoutMs: 3_000 });

		const pending = client.search("anything");
		await vi.waitFor(() => expect(existsSync(pidFile)).toBe(true), { timeout: 10_000 });
		const pid = Number(readFileSync(pidFile, "utf8"));
		spawnedPids.push(pid);

		expect(await settleWithin(pending, 8_000, "crawler search")).toMatchObject({ ok: false, reason: "timeout" });
		await vi.waitFor(() => expect(isRunning(pid)).toBe(false), { timeout: 3_000 });
	});

	it("still honors abort after an output-cap termination was already requested", { timeout: 15_000 }, async () => {
		const pidFile = join(root, "descendant.pid");
		writeScript(`import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";
process.on("SIGTERM", () => undefined);
const descendant = spawn(process.execPath, ["-e", "process.on('SIGTERM', () => undefined); setInterval(() => undefined, 1000)"], { stdio: "inherit" });
writeFileSync(${JSON.stringify(pidFile)}, String(descendant.pid));
process.stdout.write("x".repeat(64));
setInterval(() => undefined, 1000);
`);
		const controller = new AbortController();
		const client = new CrawlerCliClient(PROFILE, { binaryPath, maxBufferBytes: 8, timeoutMs: 30_000 });

		const pending = client.search("anything", { signal: controller.signal });
		await vi.waitFor(() => expect(existsSync(pidFile)).toBe(true), { timeout: 5_000 });
		controller.abort();

		expect(await settleWithin(pending, 5_000, "crawler search")).toMatchObject({ ok: false });
		const pid = Number(readFileSync(pidFile, "utf8"));
		spawnedPids.push(pid);
		await vi.waitFor(() => expect(isRunning(pid)).toBe(false), { timeout: 3_000 });
	});
});

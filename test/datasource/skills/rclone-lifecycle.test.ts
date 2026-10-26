import { execFileSync } from "node:child_process";
import { chmodSync, existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RcloneConnector } from "../../../src/datasource/skills/cloud-drive/rclone-connector.ts";

let root: string;
let binaryPath: string;
const spawnedPids: number[] = [];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-rclone-lifecycle-"));
	binaryPath = join(root, "rclone");
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

const HOLDING_DESCENDANT = (pidFile: string): string => `import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";
const descendant = spawn(process.execPath, ["-e", "process.on('SIGTERM', () => undefined); setInterval(() => undefined, 1000)"], { stdio: "inherit" });
writeFileSync(${JSON.stringify(pidFile)}, String(descendant.pid));
`;

describe.skipIf(process.platform === "win32")("RcloneConnector process lifecycle", () => {
	it("settles and reaps a descendant that keeps the inherited stdio open after the direct child exits", {
		timeout: 15_000,
	}, async () => {
		const pidFile = join(root, "descendant.pid");
		writeScript(`${HOLDING_DESCENDANT(pidFile)}process.stdout.write("[]");\nprocess.exit(0);\n`);
		const connector = new RcloneConnector({ binaryPath, remote: "drive:", timeoutMs: 30_000 });

		const result = await settleWithin(connector.fetch(), 4_000, "rclone fetch");

		expect(result.ok).toBe(true);
		const pid = Number(readFileSync(pidFile, "utf8"));
		spawnedPids.push(pid);
		await vi.waitFor(() => expect(isRunning(pid)).toBe(false), { timeout: 3_000 });
	});

	it("reaps a descendant that holds the pipes when the timeout fires", { timeout: 15_000 }, async () => {
		const pidFile = join(root, "descendant.pid");
		writeScript(`${HOLDING_DESCENDANT(pidFile)}setInterval(() => undefined, 1000);\n`);
		// The wall must outlive Node startup under load, or the fixture is killed before it records the
		// descendant pid.
		const connector = new RcloneConnector({ binaryPath, remote: "drive:", timeoutMs: 3_000 });

		const pending = connector.fetch();
		await vi.waitFor(() => expect(existsSync(pidFile)).toBe(true), { timeout: 10_000 });
		const pid = Number(readFileSync(pidFile, "utf8"));
		spawnedPids.push(pid);

		expect(await settleWithin(pending, 8_000, "rclone fetch")).toMatchObject({ ok: false });
		await vi.waitFor(() => expect(isRunning(pid)).toBe(false), { timeout: 3_000 });
	});
});

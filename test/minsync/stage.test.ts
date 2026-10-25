import { type ChildProcess, spawn } from "node:child_process";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MinSyncStageError, type MinSyncStageOpenInput, MinSyncStageSession } from "../../src/minsync/stage.ts";
import { writeFakeMinSync } from "../helpers/fake-minsync.ts";

const posixOnly = process.platform === "win32" ? describe.skip : describe;

/** The host script imports the transport by absolute URL, so it can live outside the repo. */
const STAGE_MODULE = new URL("../../src/minsync/stage.ts", import.meta.url).href;

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-stage-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function stageSession(binary: string, stateDir: string, signal?: AbortSignal): MinSyncStageSession {
	return new MinSyncStageSession(binary, stateDir, root, signal);
}

/** Run a session and always close it, so no stage child outlives its test. */
async function withSession<T>(
	binary: string,
	run: (session: MinSyncStageSession, stateDir: string) => Promise<T>,
	options: { readonly stateDir?: string; readonly signal?: AbortSignal } = {},
): Promise<T> {
	const stateDir = options.stateDir ?? join(root, "state");
	const session = stageSession(binary, stateDir, options.signal);
	try {
		return await run(session, stateDir);
	} finally {
		await session.close();
	}
}

function openInput(overrides: Partial<MinSyncStageOpenInput> = {}): MinSyncStageOpenInput {
	return {
		stage: "parse",
		source_id: "corpus",
		fingerprint: "parser-1",
		output_root: join(root, "out"),
		items: [{ key: "docs/a.pdf", version: "sha256:x" }],
		...overrides,
	};
}

function alive(pid: number): boolean {
	try {
		process.kill(pid, 0);
		return true;
	} catch {
		return false;
	}
}

/** A reaped child can still answer `kill(pid, 0)` for a moment as an exiting process under load. */
async function expectDead(pid: number): Promise<void> {
	await vi.waitFor(() => expect(alive(pid)).toBe(false), { timeout: 10_000 });
}

async function stubPid(pidFile: string): Promise<number> {
	await vi.waitFor(() => expect(existsSync(pidFile)).toBe(true), { timeout: 15_000 });
	return Number(readFileSync(pidFile, "utf8"));
}

/**
 * How a stub answers the stage protocol. Canned lines stand in for MinSync so a
 * transport case (a refusal, a response that cannot be read, an exit code, a
 * request that is never answered) is exact instead of incidental.
 */
type StubKind =
	| { readonly kind: "responses"; readonly lines: readonly string[]; readonly exitCode?: number }
	| { readonly kind: "exit-on-request"; readonly code: number }
	| { readonly kind: "ignore-eof" };

/** A `minsync` stand-in that speaks the line protocol without implementing any of it. */
function writeStub(
	path: string,
	stub: StubKind,
	options: { readonly pidFile?: string; readonly logPath?: string; readonly stderrText?: string } = {},
): void {
	const config = { ...stub, pidFile: options.pidFile, logPath: options.logPath, stderrText: options.stderrText };
	writeFileSync(
		path,
		`#!/usr/bin/env node
import { appendFileSync, writeFileSync } from "node:fs";
const config = ${JSON.stringify(config)};
if (config.pidFile) writeFileSync(config.pidFile, String(process.pid));
if (config.stderrText) process.stderr.write(config.stderrText);
if (config.kind === "ignore-eof") {
  setInterval(() => {}, 1 << 30);
  process.stdin.resume();
} else if (config.kind === "exit-on-request") {
  process.stdin.resume();
  process.stdin.on("data", () => process.exit(config.code));
} else {
  let buffer = "";
  let index = 0;
  process.stdin.setEncoding("utf8");
  process.stdin.on("data", (chunk) => {
    buffer += chunk;
    for (;;) {
      const newline = buffer.indexOf("\\n");
      if (newline < 0) break;
      const line = buffer.slice(0, newline);
      buffer = buffer.slice(newline + 1);
      if (line.trim().length === 0) continue;
      if (config.logPath) appendFileSync(config.logPath, line + "\\n");
      const response = config.lines[index++];
      if (response === undefined) continue;
      const last = index === config.lines.length;
      process.stdout.write(response + "\\n", () => { if (last) process.exit(config.exitCode ?? 0); });
    }
  });
  process.stdin.on("end", () => process.exit(1));
}
`,
	);
	chmodSync(path, 0o755);
}

const OPEN_OK = JSON.stringify({
	ok: true,
	op: "open",
	run_token: "run-1",
	resumed: false,
	counts: { process: 1, retry: 0, skip: 0, resumed: 0, delete: 0, preserve: 0 },
	decisions: [{ key: "docs/a.pdf", action: "process", version: "sha256:x" }],
});
const FINISH_OK = JSON.stringify({ ok: true, op: "finish", committed: 1, deleted: 0, failed: 0, pending: 0 });
const REFUSAL = JSON.stringify({
	ok: false,
	op: "commit",
	error: { code: "output_missing", message: "output file is missing" },
});

describe("MinSyncStageSession against the stage protocol", () => {
	it("opens, commits and finishes a run", async () => {
		const binary = join(root, "fake-minsync.mjs");
		writeFakeMinSync(binary);
		const output = join(root, "out");
		mkdirSync(output, { recursive: true });
		writeFileSync(join(output, "a.md"), "alpha");
		const stateDir = join(root, "state");

		const first = stageSession(binary, stateDir);
		const report = await first.open(openInput({ output_root: output, items: [{ key: "a.md", version: "v1" }] }));
		expect(report.resumed).toBe(false);
		expect(report.run_token.length).toBeGreaterThan(0);
		expect(report.decisions).toEqual([{ key: "a.md", action: "process", version: "v1" }]);
		expect(report.counts.process).toBe(1);

		const recorded = await first.commit("a.md", "v1", { path: "a.md" });
		expect(recorded.path).toBe("a.md");
		expect(recorded.size).toBe(5);
		expect(recorded.digest.startsWith("sha256:")).toBe(true);

		expect(await first.finish()).toEqual({ committed: 1, deleted: 0, failed: 0, pending: 0 });
		await first.close();

		await withSession(
			binary,
			async (second) => {
				const again = await second.open(
					openInput({ output_root: output, items: [{ key: "a.md", version: "v1" }] }),
				);
				expect(again.decisions).toEqual([]);
				expect(again.counts.skip).toBe(1);
				expect(await second.finish()).toEqual({ committed: 0, deleted: 0, failed: 0, pending: 0 });
			},
			{ stateDir },
		);
	});

	it("resumes the run a closed session leaves behind", async () => {
		const binary = join(root, "fake-minsync.mjs");
		writeFakeMinSync(binary);
		const output = join(root, "out");
		mkdirSync(output, { recursive: true });
		writeFileSync(join(output, "a.md"), "alpha");
		const items = [
			{ key: "a.md", version: "v1" },
			{ key: "b.md", version: "v1" },
		];

		await withSession(binary, async (interrupted, stateDir) => {
			const opened = await interrupted.open(openInput({ output_root: output, items }));
			expect(opened.decisions.map((decision) => decision.key)).toEqual(["a.md", "b.md"]);
			await interrupted.commit("a.md", "v1", { path: "a.md" });
			// End without `finish`: stdin EOF, exit 1, and the run stays resumable.
			await interrupted.close();

			await withSession(
				binary,
				async (resumed) => {
					const report = await resumed.open(openInput({ output_root: output, items }));
					expect(report.resumed).toBe(true);
					expect(report.counts.resumed).toBe(1);
					expect(report.decisions.map((decision) => decision.key)).toEqual(["b.md"]);
					await resumed.close();
				},
				{ stateDir },
			);
		});
	});
});

describe("MinSyncStageSession transport", () => {
	it("rejects a refusal with its protocol code and keeps the session usable", async () => {
		const binary = join(root, "stub.mjs");
		writeStub(binary, { kind: "responses", lines: [OPEN_OK, REFUSAL, FINISH_OK] });

		await withSession(binary, async (session) => {
			await session.open(openInput());
			const failure = await session
				.commit("docs/a.pdf", "sha256:x", { path: "docs/a.pdf.md" })
				.catch((cause: unknown) => cause);
			expect(failure).toBeInstanceOf(MinSyncStageError);
			expect(failure).toMatchObject({
				name: "MinSyncStageError",
				code: "output_missing",
				op: "commit",
				message: "output file is missing",
			});
			expect(await session.finish()).toEqual({ committed: 1, deleted: 0, failed: 0, pending: 0 });
		});
	});

	it("rejects a response it cannot read", async () => {
		const unreadable = join(root, "unreadable.mjs");
		writeStub(unreadable, { kind: "responses", lines: ["not json"] });
		await withSession(
			unreadable,
			async (session) => {
				await expect(session.open(openInput())).rejects.toMatchObject({
					code: "invalid_response",
					op: "open",
				});
			},
			{ stateDir: join(root, "state-a") },
		);

		const incomplete = join(root, "incomplete.mjs");
		writeStub(incomplete, { kind: "responses", lines: [JSON.stringify({ ok: true, op: "open" })] });
		await withSession(
			incomplete,
			async (session) => {
				await expect(session.open(openInput())).rejects.toMatchObject({
					code: "invalid_response",
					op: "open",
				});
			},
			{ stateDir: join(root, "state-b") },
		);
	});

	it("keeps a bounded stderr tail for a session that dies", async () => {
		const binary = join(root, "noisy.mjs");
		writeStub(binary, { kind: "exit-on-request", code: 1 }, { stderrText: "e".repeat(12_000) });

		await withSession(binary, async (session) => {
			const failure = await session.open(openInput()).catch((cause: unknown) => cause);
			expect(failure).toBeInstanceOf(MinSyncStageError);
			const { stderr } = failure as MinSyncStageError;
			expect(stderr.length).toBeLessThanOrEqual(8_200);
			expect(stderr.startsWith("…")).toBe(true);
			expect(stderr.endsWith("e")).toBe(true);
		});
	});

	it("requires exit code 0 to finish", async () => {
		const binary = join(root, "exit3.mjs");
		writeStub(binary, { kind: "responses", lines: [OPEN_OK, FINISH_OK], exitCode: 3 });

		await withSession(binary, async (session) => {
			await session.open(openInput());
			await expect(session.finish()).rejects.toMatchObject({ code: "process_failed", op: "finish" });
		});
	});

	it("rejects requests MinSync never answers", async () => {
		const binary = join(root, "dying.mjs");
		writeStub(binary, { kind: "exit-on-request", code: 1 });

		await withSession(binary, async (session) => {
			const failure = await session.open(openInput()).catch((cause: unknown) => cause);
			expect(failure).toBeInstanceOf(MinSyncStageError);
			expect(failure).toMatchObject({ code: "eof", op: "open" });
			expect((failure as MinSyncStageError).message).toContain("before answering `open`");
			// The session is gone for good, and says so with the same reason.
			await expect(session.finish()).rejects.toBe(failure);
		});
	});

	it("serializes requests that are not awaited", async () => {
		const binary = join(root, "serial.mjs");
		writeStub(binary, { kind: "responses", lines: [OPEN_OK, FINISH_OK] });

		await withSession(binary, async (session) => {
			const opening = session.open(openInput());
			const finishing = session.finish();
			expect((await opening).run_token).toBe("run-1");
			expect(await finishing).toEqual({ committed: 1, deleted: 0, failed: 0, pending: 0 });
		});
	});
});

posixOnly("MinSyncStageSession process lifecycle", () => {
	it("rejects an in-flight request with the caller's abort reason and stops the child", async () => {
		const binary = join(root, "unanswered.mjs");
		const pidFile = join(root, "stub.pid");
		// Never answers, and only exits when something kills it.
		writeStub(binary, { kind: "responses", lines: [] }, { pidFile });
		const controller = new AbortController();
		const reason = new Error("stage aborted by the caller");

		await withSession(
			binary,
			async (session) => {
				const opening = session.open(openInput());
				const pid = await stubPid(pidFile);
				expect(alive(pid)).toBe(true);
				controller.abort(reason);
				await expect(opening).rejects.toBe(reason);
				await expectDead(pid);
			},
			{ signal: controller.signal },
		);
	});

	it("closes stdin without finishing and reaps a child that exits on EOF", async () => {
		const binary = join(root, "eof-exit.mjs");
		const pidFile = join(root, "stub.pid");
		writeStub(binary, { kind: "responses", lines: [] }, { pidFile });

		const session = stageSession(binary, join(root, "state"));
		const pid = await stubPid(pidFile);
		const started = Date.now();
		await session.close();
		await session.close();
		// EOF alone ended it, so the SIGTERM grace was never needed.
		expect(Date.now() - started).toBeLessThan(5_000);
		await expectDead(pid);
	});

	it("escalates past an ignored stdin EOF", async () => {
		const binary = join(root, "ignores-eof.mjs");
		const pidFile = join(root, "stub.pid");
		writeStub(binary, { kind: "ignore-eof" }, { pidFile });

		const session = stageSession(binary, join(root, "state"));
		const pid = await stubPid(pidFile);
		const started = Date.now();
		await session.close();
		expect(Date.now() - started).toBeGreaterThanOrEqual(4_000);
		await expectDead(pid);
	});

	it("exits the child when the host is killed", async () => {
		const binary = join(root, "eof-exit.mjs");
		const pidFile = join(root, "stub.pid");
		writeStub(binary, { kind: "responses", lines: [] }, { pidFile });
		const hostPath = join(root, "host.ts");
		writeFileSync(
			hostPath,
			`import { MinSyncStageSession } from ${JSON.stringify(STAGE_MODULE)};
const session = new MinSyncStageSession(String(process.env.STAGE_BINARY), String(process.env.STAGE_STATE_DIR), process.cwd());
if (session === undefined) process.exit(2);
await new Promise(() => {});
`,
		);

		const host = spawn(process.execPath, [hostPath], {
			cwd: root,
			stdio: ["pipe", "pipe", "pipe"],
			env: { ...process.env, STAGE_BINARY: binary, STAGE_STATE_DIR: join(root, "state") },
		});
		try {
			const pid = await stubPid(pidFile);
			expect(alive(pid)).toBe(true);
			host.kill("SIGKILL");
			await expectDead(pid);
			expect(await exited(host)).toBe(true);
		} finally {
			host.kill("SIGKILL");
		}
	});
});

function exited(child: ChildProcess): Promise<boolean> {
	if (child.exitCode !== null || child.signalCode !== null) return Promise.resolve(true);
	return new Promise<boolean>((resolve) => {
		child.once("close", () => resolve(true));
	});
}

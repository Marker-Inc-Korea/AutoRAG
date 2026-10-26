import { type ChildProcess, spawn } from "node:child_process";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
	type MinSyncStageDecision,
	MinSyncStageError,
	type MinSyncStageOpenInput,
	MinSyncStageSession,
	readMinSyncStage,
} from "../../src/minsync/stage.ts";
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
const CAPABILITIES_OK = JSON.stringify({
	ok: true,
	op: "capabilities",
	protocol: 2,
	features: [
		"retry_fingerprint",
		"retry_failed",
		"deferred_failure",
		"discard_output",
		"deliberate_skip",
		"plan",
		"inspect",
	],
	journal_format: 3,
	snapshot_format: 2,
});
const FINISH_OK = JSON.stringify({ ok: true, op: "finish", committed: 1, deleted: 0, failed: 0, pending: 0 });
const REFUSAL = JSON.stringify({
	ok: false,
	op: "commit",
	error: { code: "output_missing", message: "output file is missing" },
});
const INSPECT_OK = JSON.stringify({
	ok: true,
	op: "inspect",
	items: [
		{
			key: "docs/a.pdf",
			version: "sha256:x",
			fingerprint: "parser-1",
			output: { path: "docs/a.pdf.md", size: 5, digest: "sha256:abc", mtime_ns: "1700000000000000000" },
			updated_at: "2024-01-01T00:00:00.000Z",
		},
	],
	failed: [{ key: "docs/b.pdf", version: "sha256:y", error: "parser crashed", deferred: true }],
	skipped: [{ key: "docs/c.pdf", version: "sha256:z", reason: "unsupported type", fingerprint: "parser-1" }],
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
		writeStub(binary, { kind: "responses", lines: [CAPABILITIES_OK, OPEN_OK, REFUSAL, FINISH_OK] });

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
		writeStub(unreadable, { kind: "responses", lines: [CAPABILITIES_OK, "not json"] });
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
		writeStub(incomplete, {
			kind: "responses",
			lines: [CAPABILITIES_OK, JSON.stringify({ ok: true, op: "open" })],
		});
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
		writeStub(binary, { kind: "responses", lines: [CAPABILITIES_OK, OPEN_OK, FINISH_OK], exitCode: 3 });

		await withSession(binary, async (session) => {
			await session.open(openInput());
			await expect(session.finish()).rejects.toMatchObject({ code: "process_failed", op: "finish" });
		});
	});

	it("rejects a handshake MinSync never answers", async () => {
		const binary = join(root, "dying.mjs");
		writeStub(binary, { kind: "exit-on-request", code: 1 });

		await withSession(binary, async (session) => {
			const failure = await rejection(session.open(openInput()));
			// The handshake is the first request of a session, and no run can
			// start without it.
			expect(failure).toMatchObject({ code: "eof", op: "capabilities" });
			expect(failure.message).toContain("before answering `capabilities`");
			// The session is gone for good, and says so with the same reason.
			await expect(session.finish()).rejects.toBe(failure);
		});
	});

	it("refuses a build that does not answer the capability handshake", async () => {
		const binary = join(root, "old.mjs");
		writeStub(binary, {
			kind: "responses",
			lines: [
				JSON.stringify({
					ok: false,
					op: "unknown",
					error: { code: "invalid_request", message: "unknown variant `capabilities`" },
				}),
			],
		});

		await withSession(binary, async (session) => {
			const failure = await rejection(session.open(openInput()));
			expect(failure).toMatchObject({ code: "unsupported_protocol", op: "capabilities" });
			expect(failure.message).toContain("`capabilities`");
			// The handshake is answered once and remembered.
			await expect(session.open(openInput())).rejects.toBe(failure);
		});
	});

	it("refuses a build that reports an older protocol or is missing a feature", async () => {
		const older = join(root, "protocol-1.mjs");
		writeStub(older, {
			kind: "responses",
			lines: [
				JSON.stringify({
					ok: true,
					op: "capabilities",
					protocol: 1,
					features: [
						"retry_fingerprint",
						"retry_failed",
						"deferred_failure",
						"discard_output",
						"deliberate_skip",
						"plan",
						"inspect",
					],
				}),
			],
		});
		await withSession(
			older,
			async (session) => {
				const failure = await rejection(session.open(openInput()));
				expect(failure).toMatchObject({ code: "unsupported_protocol", op: "capabilities" });
				expect(failure.message).toContain("protocol 2");
			},
			{ stateDir: join(root, "state-older") },
		);

		const partial = join(root, "partial.mjs");
		writeStub(partial, {
			kind: "responses",
			lines: [
				JSON.stringify({
					ok: true,
					op: "capabilities",
					protocol: 2,
					features: ["inspect", "plan", "deliberate_skip"],
				}),
			],
		});
		await withSession(
			partial,
			async (session) => {
				const failure = await rejection(session.open(openInput()));
				expect(failure).toMatchObject({ code: "unsupported_protocol", op: "capabilities" });
				expect(failure.message).toContain(
					"missing retry_fingerprint, retry_failed, deferred_failure, discard_output",
				);
			},
			{ stateDir: join(root, "state-partial") },
		);
	});

	it("performs the capability handshake once per session", async () => {
		const binary = join(root, "handshake.mjs");
		const logPath = join(root, "handshake.log");
		writeStub(binary, { kind: "responses", lines: [CAPABILITIES_OK, OPEN_OK, OPEN_OK] }, { logPath });

		await withSession(binary, async (session) => {
			await session.open(openInput());
			// A second open is the protocol's own business, not another handshake.
			await session.open(openInput());
		});

		const ops = readFileSync(logPath, "utf8")
			.trim()
			.split("\n")
			.map((line) => {
				const request: unknown = JSON.parse(line);
				return request !== null && typeof request === "object" && "op" in request ? request.op : undefined;
			});
		expect(ops).toEqual(["capabilities", "open", "open"]);
	});

	it("reads an inspect report and insists on a text mtime", async () => {
		const binary = join(root, "inspect.mjs");
		writeStub(binary, { kind: "responses", lines: [INSPECT_OK] });

		await withSession(binary, async (session) => {
			const state = await session.inspect({ stage: "parse", source_id: "corpus", output_root: join(root, "out") });
			expect(state.items[0]?.output.mtime_ns).toBe("1700000000000000000");
			expect(state.failed).toEqual([
				{ key: "docs/b.pdf", version: "sha256:y", error: "parser crashed", deferred: true },
			]);
			expect(state.skipped).toEqual([
				{ key: "docs/c.pdf", version: "sha256:z", reason: "unsupported type", fingerprint: "parser-1" },
			]);
		});

		const numeric = join(root, "inspect-numeric.mjs");
		writeStub(numeric, {
			kind: "responses",
			lines: [
				JSON.stringify({
					ok: true,
					op: "inspect",
					items: [
						{
							key: "docs/a.pdf",
							version: "sha256:x",
							fingerprint: "parser-1",
							output: {
								path: "docs/a.pdf.md",
								size: 5,
								digest: "sha256:abc",
								mtime_ns: 1_700_000_000_000_000_000,
							},
							updated_at: "2024-01-01T00:00:00.000Z",
						},
					],
					failed: [],
					skipped: [],
				}),
			],
		});
		await withSession(
			numeric,
			async (session) => {
				await expect(
					session.inspect({ stage: "parse", source_id: "corpus", output_root: join(root, "out") }),
				).rejects.toMatchObject({ code: "invalid_response", op: "inspect" });
			},
			{ stateDir: join(root, "numeric-state") },
		);
	});

	it("reads a plan report with every decision on it", async () => {
		const binary = join(root, "plan.mjs");
		writeStub(binary, {
			kind: "responses",
			lines: [
				JSON.stringify({
					ok: true,
					op: "plan",
					counts: { process: 1, retry: 0, skip: 2, resumed: 1, delete: 0, preserve: 0 },
					decisions: [
						{ key: "docs/a.pdf", action: "skip" },
						{ key: "docs/b.pdf", action: "skip", version: "sha256:y", deferred: true, error: "parser crashed" },
						{ key: "docs/c.pdf", action: "process", version: "sha256:z" },
					],
				}),
			],
		});

		await withSession(binary, async (session) => {
			const planned = await session.plan(openInput());
			expect(planned.counts).toEqual({ process: 1, retry: 0, skip: 2, resumed: 1, delete: 0, preserve: 0 });
			expect(planned.decisions).toEqual([
				{ key: "docs/a.pdf", action: "skip" },
				{ key: "docs/b.pdf", action: "skip", version: "sha256:y", deferred: true, error: "parser crashed" },
				{ key: "docs/c.pdf", action: "process", version: "sha256:z" },
			]);
		});
	});

	it("sends the extended open, fail and skip requests", async () => {
		const binary = join(root, "wire.mjs");
		const logPath = join(root, "wire.log");
		writeStub(
			binary,
			{
				kind: "responses",
				lines: [
					CAPABILITIES_OK,
					OPEN_OK,
					JSON.stringify({ ok: true, op: "fail", key: "docs/a.pdf" }),
					JSON.stringify({ ok: true, op: "fail", key: "docs/a.pdf" }),
					JSON.stringify({ ok: true, op: "skip", key: "docs/a.pdf" }),
				],
			},
			{ logPath },
		);

		await withSession(binary, async (session) => {
			await session.open(openInput({ retry_fingerprint: "rf-1", retry_failed: true }));
			await session.fail("docs/a.pdf", "sha256:x", "parser crashed");
			await session.fail("docs/a.pdf", "sha256:x", "parser crashed", { defer: true, discard_output: true });
			await session.skip("docs/a.pdf", "sha256:x", "unsupported type");
		});

		const sent: unknown[] = readFileSync(logPath, "utf8")
			.trim()
			.split("\n")
			.map((line) => JSON.parse(line));
		expect(sent[0]).toEqual({ op: "capabilities" });
		expect(sent[1]).toMatchObject({ op: "open", retry_fingerprint: "rf-1", retry_failed: true });
		expect(sent[2]).toEqual({ op: "fail", key: "docs/a.pdf", version: "sha256:x", error: "parser crashed" });
		expect(sent[3]).toEqual({
			op: "fail",
			key: "docs/a.pdf",
			version: "sha256:x",
			error: "parser crashed",
			defer: true,
			discard_output: true,
		});
		expect(sent[4]).toEqual({ op: "skip", key: "docs/a.pdf", version: "sha256:x", reason: "unsupported type" });
	});

	it("serializes requests that are not awaited", async () => {
		const binary = join(root, "serial.mjs");
		writeStub(binary, { kind: "responses", lines: [CAPABILITIES_OK, OPEN_OK, FINISH_OK] });

		await withSession(binary, async (session) => {
			const opening = session.open(openInput());
			const finishing = session.finish();
			// Both are issued without awaiting; each still gets its own answer, in
			// order, and a failure in one is not left unhandled while the other waits.
			const [report, finished] = await Promise.all([opening, finishing]);
			expect(report.run_token).toBe("run-1");
			expect(finished).toEqual({ committed: 1, deleted: 0, failed: 0, pending: 0 });
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

describe("MinSyncStageSession durable outcomes", () => {
	it("records a deliberate omission and stops asking for the item", async () => {
		const binary = join(root, "fake-minsync.mjs");
		writeFakeMinSync(binary);
		const output = join(root, "out");
		mkdirSync(output, { recursive: true });
		const stateDir = join(root, "state");
		const items = [{ key: "a.md", version: "v1" }];

		await withSession(
			binary,
			async (session) => {
				expect((await session.open(openInput({ output_root: output, items }))).counts.process).toBe(1);
				await session.skip("a.md", "v1", "unsupported type");
				expect(await session.finish()).toEqual({ committed: 0, deleted: 0, failed: 0, pending: 0 });
			},
			{ stateDir },
		);

		await withSession(
			binary,
			async (session) => {
				const report = await session.open(openInput({ output_root: output, items }));
				expect(report.decisions).toEqual([]);
				expect(report.counts.skip).toBe(1);
				const planned = await session.plan(openInput({ output_root: output, items }));
				expect(planned.decisions).toEqual([{ key: "a.md", action: "skip" }]);
				const state = await session.inspect({ stage: "parse", source_id: "corpus", output_root: output });
				expect(state.items).toEqual([]);
				expect(state.failed).toEqual([]);
				expect(state.skipped).toEqual([
					{ key: "a.md", version: "v1", reason: "unsupported type", fingerprint: "parser-1" },
				]);
				await session.finish();
			},
			{ stateDir },
		);
	});

	it("removes a previous output when the item is deliberately omitted", async () => {
		const binary = join(root, "fake-minsync.mjs");
		writeFakeMinSync(binary);
		const output = join(root, "out");
		mkdirSync(output, { recursive: true });
		writeFileSync(join(output, "a.md"), "alpha");
		const stateDir = join(root, "state");
		const items = [{ key: "a.md", version: "v1" }];

		await withSession(
			binary,
			async (session) => {
				await session.open(openInput({ output_root: output, items }));
				await session.commit("a.md", "v1", { path: "a.md" });
				await session.finish();
			},
			{ stateDir },
		);
		expect(existsSync(join(output, "a.md"))).toBe(true);

		await withSession(
			binary,
			async (session) => {
				await session.open(openInput({ output_root: output, items }));
				await session.skip("a.md", "v1", "policy changed");
				await session.finish();
			},
			{ stateDir },
		);
		expect(existsSync(join(output, "a.md"))).toBe(false);

		await withSession(
			binary,
			async (session) => {
				const state = await session.inspect({ stage: "parse", source_id: "corpus", output_root: output });
				expect(state.items).toEqual([]);
				expect(state.skipped).toEqual([
					{ key: "a.md", version: "v1", reason: "policy changed", fingerprint: "parser-1" },
				]);
			},
			{ stateDir },
		);
	});

	it("asks for an omitted item again once its source version changes", async () => {
		const binary = join(root, "fake-minsync.mjs");
		writeFakeMinSync(binary);
		const output = join(root, "out");
		mkdirSync(output, { recursive: true });
		const stateDir = join(root, "state");

		await withSession(
			binary,
			async (session) => {
				await session.open(openInput({ output_root: output, items: [{ key: "a.md", version: "v1" }] }));
				await session.skip("a.md", "v1", "unsupported type");
				await session.finish();
			},
			{ stateDir },
		);

		await withSession(
			binary,
			async (session) => {
				const report = await session.open(
					openInput({ output_root: output, items: [{ key: "a.md", version: "v2" }] }),
				);
				expect(report.decisions).toEqual([{ key: "a.md", action: "process", version: "v2" }]);
				const state = await session.inspect({ stage: "parse", source_id: "corpus", output_root: output });
				expect(state.items).toEqual([]);
			},
			{ stateDir },
		);
	});

	it("reuses a deliberate omission even for a full run", async () => {
		const binary = join(root, "fake-minsync.mjs");
		writeFakeMinSync(binary);
		const output = join(root, "out");
		mkdirSync(output, { recursive: true });
		const stateDir = join(root, "state");
		const items = [{ key: "a.md", version: "v1" }];

		await withSession(
			binary,
			async (session) => {
				await session.open(openInput({ output_root: output, items }));
				await session.skip("a.md", "v1", "unsupported type");
				await session.finish();
			},
			{ stateDir },
		);

		await withSession(
			binary,
			async (session) => {
				const report = await session.open(openInput({ output_root: output, items, full: true }));
				expect(report.decisions).toEqual([]);
				expect(report.counts.skip).toBe(1);
			},
			{ stateDir },
		);
	});

	it("rejects a skip for a version the run does not hold", async () => {
		const binary = join(root, "fake-minsync.mjs");
		writeFakeMinSync(binary);

		await withSession(binary, async (session) => {
			await session.open(openInput());
			await expect(session.skip("docs/a.pdf", "sha256:other", "unsupported type")).rejects.toMatchObject({
				code: "version_mismatch",
				op: "skip",
			});
			await expect(session.finish()).resolves.toBeDefined();
		});
	});
});

describe("MinSyncStageSession deferred failures", () => {
	/** Commit `a.md` as v1, then fail it as v2 with `fail`. */
	async function seed(
		binary: string,
		output: string,
		stateDir: string,
		options: { readonly retryFingerprint?: string; readonly defer?: boolean; readonly discard?: boolean } = {},
	): Promise<void> {
		mkdirSync(output, { recursive: true });
		writeFileSync(join(output, "a.md"), "alpha");
		await withSession(
			binary,
			async (session) => {
				await session.open(openInput({ output_root: output, items: [{ key: "a.md", version: "v1" }] }));
				await session.commit("a.md", "v1", { path: "a.md" });
				await session.finish();
			},
			{ stateDir },
		);
		await withSession(
			binary,
			async (session) => {
				await session.open(
					openInput({
						output_root: output,
						items: [{ key: "a.md", version: "v2" }],
						retry_fingerprint: options.retryFingerprint,
					}),
				);
				await session.fail("a.md", "v2", "parser crashed", {
					defer: options.defer,
					discard_output: options.discard,
				});
				await session.finish();
			},
			{ stateDir },
		);
	}

	it("holds a deferred failure back and reports why", async () => {
		const binary = join(root, "fake-minsync.mjs");
		writeFakeMinSync(binary);
		const output = join(root, "out");
		const stateDir = join(root, "state");
		await seed(binary, output, stateDir, { defer: true });

		await withSession(
			binary,
			async (session) => {
				const report = await session.open(
					openInput({ output_root: output, items: [{ key: "a.md", version: "v2" }] }),
				);
				expect(report.decisions).toEqual([
					{ key: "a.md", action: "skip", version: "v2", deferred: true, error: "parser crashed" },
				]);
				expect(report.counts.skip).toBe(1);
				expect(report.counts.retry).toBe(0);
				const planned = await session.plan(
					openInput({ output_root: output, items: [{ key: "a.md", version: "v2" }] }),
				);
				expect(planned.decisions).toEqual([
					{ key: "a.md", action: "skip", version: "v2", deferred: true, error: "parser crashed" },
				]);
				const state = await session.inspect({ stage: "parse", source_id: "corpus", output_root: output });
				// The failure did not erase the last good output of the older version.
				expect(state.items).toEqual([expect.objectContaining({ key: "a.md", version: "v1" })]);
				expect(state.failed).toEqual([{ key: "a.md", version: "v2", error: "parser crashed", deferred: true }]);
			},
			{ stateDir },
		);
	});

	it("retries a deferred failure once the run, the retry policy or the settings change", async () => {
		const binary = join(root, "fake-minsync.mjs");
		writeFakeMinSync(binary);
		const output = join(root, "out");
		const stateDir = join(root, "state");
		await seed(binary, output, stateDir, { retryFingerprint: "rf-1", defer: true });

		const items = [{ key: "a.md", version: "v2" }];
		const held: MinSyncStageDecision = {
			key: "a.md",
			action: "skip",
			version: "v2",
			deferred: true,
			error: "parser crashed",
		};
		const cases: readonly [string, Partial<MinSyncStageOpenInput>, readonly MinSyncStageDecision[]][] = [
			["the same policy", { retry_fingerprint: "rf-1" }, [held]],
			[
				"a released run",
				{ retry_fingerprint: "rf-1", retry_failed: true },
				[{ key: "a.md", action: "retry", version: "v2" }],
			],
			["a new retry policy", { retry_fingerprint: "rf-2" }, [{ key: "a.md", action: "retry", version: "v2" }]],
			[
				"new settings",
				{ retry_fingerprint: "rf-1", fingerprint: "parser-2" },
				[{ key: "a.md", action: "retry", version: "v2" }],
			],
			["a full run", { retry_fingerprint: "rf-1", full: true }, [{ key: "a.md", action: "retry", version: "v2" }]],
		];
		for (const [label, overrides, expected] of cases) {
			await withSession(
				binary,
				async (session) => {
					const report = await session.open(openInput({ output_root: output, items, ...overrides }));
					expect(report.decisions, label).toEqual(expected);
				},
				{ stateDir },
			);
		}

		await withSession(
			binary,
			async (session) => {
				const report = await session.open(
					openInput({ output_root: output, items: [{ key: "a.md", version: "v3" }] }),
				);
				expect(report.decisions).toEqual([{ key: "a.md", action: "process", version: "v3" }]);
			},
			{ stateDir },
		);
	});

	it("discards the last good output only when the failure asks for it", async () => {
		const binary = join(root, "fake-minsync.mjs");
		writeFakeMinSync(binary);
		const output = join(root, "out");
		const stateDir = join(root, "state");
		await seed(binary, output, stateDir, { defer: true });
		expect(existsSync(join(output, "a.md"))).toBe(true);

		await withSession(
			binary,
			async (session) => {
				await session.open(openInput({ output_root: output, items: [{ key: "a.md", version: "v3" }] }));
				await session.fail("a.md", "v3", "unusable output", { defer: true, discard_output: true });
				await session.finish();
			},
			{ stateDir },
		);
		expect(existsSync(join(output, "a.md"))).toBe(false);

		await withSession(
			binary,
			async (session) => {
				await session.open(openInput({ output_root: output, items: [{ key: "a.md", version: "v3" }] }));
				const state = await session.inspect({ stage: "parse", source_id: "corpus", output_root: output });
				expect(state.items).toEqual([]);
				expect(state.failed).toEqual([{ key: "a.md", version: "v3", error: "unusable output", deferred: true }]);
			},
			{ stateDir },
		);
	});

	it("clears a held failure once its source is confirmed gone", async () => {
		const binary = join(root, "fake-minsync.mjs");
		writeFakeMinSync(binary);
		const output = join(root, "out");
		const stateDir = join(root, "state");
		await seed(binary, output, stateDir, { defer: true });
		expect(existsSync(join(output, "a.md"))).toBe(true);

		await withSession(
			binary,
			async (session) => {
				// The key is neither present nor unobservable: its record is stale.
				const report = await session.open(openInput({ output_root: output, items: [] }));
				expect(report.decisions).toEqual([{ key: "a.md", action: "delete" }]);
				await session.delete("a.md");
				await session.finish();
			},
			{ stateDir },
		);
		expect(existsSync(join(output, "a.md"))).toBe(false);

		await withSession(
			binary,
			async (session) => {
				expect(await session.inspect({ stage: "parse", source_id: "corpus", output_root: output })).toEqual({
					items: [],
					failed: [],
					skipped: [],
				});
			},
			{ stateDir },
		);
	});

	it("retries an ordinary failure without being asked", async () => {
		const binary = join(root, "fake-minsync.mjs");
		writeFakeMinSync(binary);
		const output = join(root, "out");
		const stateDir = join(root, "state");
		await seed(binary, output, stateDir);

		await withSession(
			binary,
			async (session) => {
				const report = await session.open(
					openInput({ output_root: output, items: [{ key: "a.md", version: "v2" }] }),
				);
				expect(report.decisions).toEqual([{ key: "a.md", action: "retry", version: "v2" }]);
				expect(report.counts.retry).toBe(1);
				const state = await session.inspect({ stage: "parse", source_id: "corpus", output_root: output });
				expect(state.items).toEqual([expect.objectContaining({ key: "a.md", version: "v1" })]);
				expect(state.failed).toEqual([{ key: "a.md", version: "v2", error: "parser crashed", deferred: false }]);
			},
			{ stateDir },
		);
	});
});

describe("MinSyncStageSession read-only state", () => {
	it("lists every item's decision without opening a run", async () => {
		const binary = join(root, "fake-minsync.mjs");
		writeFakeMinSync(binary);
		const output = join(root, "out");
		mkdirSync(output, { recursive: true });
		writeFileSync(join(output, "a.md"), "alpha");
		writeFileSync(join(output, "b.md"), "bravo");
		const stateDir = join(root, "state");
		const seedItems = [
			{ key: "a.md", version: "v1" },
			{ key: "b.md", version: "v1" },
		];

		await withSession(
			binary,
			async (session) => {
				await session.open(openInput({ output_root: output, items: seedItems }));
				await session.commit("a.md", "v1", { path: "a.md" });
				await session.commit("b.md", "v1", { path: "b.md" });
				await session.finish();
			},
			{ stateDir },
		);

		await withSession(
			binary,
			async (session) => {
				const planned = await session.plan(
					openInput({
						output_root: output,
						items: [
							{ key: "a.md", version: "v1" },
							{ key: "d.md", version: "v1" },
						],
						unknown: ["c.md"],
					}),
				);
				expect(planned.decisions).toEqual([
					{ key: "a.md", action: "skip" },
					{ key: "b.md", action: "delete" },
					{ key: "c.md", action: "preserve" },
					{ key: "d.md", action: "process", version: "v1" },
				]);
				expect(planned.counts).toEqual({ process: 1, retry: 0, skip: 1, resumed: 0, delete: 1, preserve: 1 });
				expect(existsSync(join(stateDir, "lock"))).toBe(false);
			},
			{ stateDir },
		);
	});

	it("reconstructs a stage's durable state without a lock", async () => {
		const binary = join(root, "fake-minsync.mjs");
		writeFakeMinSync(binary);
		const output = join(root, "out");
		mkdirSync(output, { recursive: true });
		writeFileSync(join(output, "a.md"), "alpha");
		const stateDir = join(root, "state");

		await withSession(
			binary,
			async (session) => {
				await session.open(openInput({ output_root: output, items: [{ key: "a.md", version: "v1" }] }));
				await session.commit("a.md", "v1", { path: "a.md" });
				await session.finish();
			},
			{ stateDir },
		);
		const snapshot = readFileSync(join(stateDir, "snapshot.json"), "utf8");
		const writtenAt = statSync(join(output, "a.md"), { bigint: true }).mtimeNs;

		await withSession(
			binary,
			async (session) => {
				const state = await session.inspect({
					stage: "parse",
					source_id: "corpus",
					output_root: output,
					verify: "digest",
				});
				expect(state.failed).toEqual([]);
				expect(state.skipped).toEqual([]);
				expect(state.items).toHaveLength(1);
				const item = state.items[0]!;
				expect(item).toMatchObject({ key: "a.md", version: "v1", fingerprint: "parser-1" });
				expect(item.output).toMatchObject({ path: "a.md", size: 5, digest: expect.stringMatching(/^sha256:/) });
				expect(item.output.mtime_ns).toMatch(/^\d+$/);
				expect(Math.abs(new Date(item.updated_at).getTime() - Number(writtenAt / 1_000_000n))).toBeLessThan(2_000);
			},
			{ stateDir },
		);
		// Reading state is not a run: nothing was locked and nothing was written.
		expect(existsSync(join(stateDir, "lock"))).toBe(false);
		expect(existsSync(join(stateDir, "txn.json"))).toBe(false);
		expect(readFileSync(join(stateDir, "snapshot.json"), "utf8")).toBe(snapshot);
	});

	it("reports nothing for a state directory that has none", async () => {
		const binary = join(root, "fake-minsync.mjs");
		writeFakeMinSync(binary);
		const output = join(root, "empty-out");
		const stateDir = join(root, "empty-state");

		await withSession(
			binary,
			async (session) => {
				expect(await session.inspect({ stage: "parse", source_id: "corpus", output_root: output })).toEqual({
					items: [],
					failed: [],
					skipped: [],
				});
				expect(
					await session.plan(openInput({ output_root: output, items: [{ key: "a.md", version: "v1" }] })),
				).toEqual({
					counts: { process: 1, retry: 0, skip: 0, resumed: 0, delete: 0, preserve: 0 },
					decisions: [{ key: "a.md", action: "process", version: "v1" }],
				});
			},
			{ stateDir },
		);
		expect(existsSync(stateDir)).toBe(false);
		expect(existsSync(output)).toBe(false);
	});

	it("reads the state of a run that is still open without taking its lock", async () => {
		const binary = join(root, "fake-minsync.mjs");
		writeFakeMinSync(binary);
		const output = join(root, "out");
		mkdirSync(output, { recursive: true });
		writeFileSync(join(output, "a.md"), "alpha");
		const stateDir = join(root, "state");
		const items = [{ key: "a.md", version: "v1" }];

		await withSession(
			binary,
			async (active) => {
				await active.open(openInput({ output_root: output, items }));
				// Committed into the run's journal only: no snapshot exists yet.
				await active.commit("a.md", "v1", { path: "a.md" });
				expect(existsSync(join(stateDir, "snapshot.json"))).toBe(false);

				await withSession(binary, async (observer) => {
					const state = await observer.inspect({ stage: "parse", source_id: "corpus", output_root: output });
					expect(state.items).toEqual([expect.objectContaining({ key: "a.md", version: "v1" })]);
					const planned = await observer.plan(openInput({ output_root: output, items }));
					expect(planned.decisions).toEqual([{ key: "a.md", action: "skip" }]);
					expect(planned.counts.resumed).toBe(1);
					await expect(observer.open(openInput({ output_root: output, items }))).rejects.toMatchObject({
						code: "lock_failed",
						op: "open",
					});
				});

				// The observer never held the lock, so the run still finishes.
				expect(await active.finish()).toEqual({ committed: 1, deleted: 0, failed: 0, pending: 0 });
			},
			{ stateDir },
		);
	});
});

describe("readMinSyncStage", () => {
	it("reads a stage's durable state in one synchronous call", async () => {
		const binary = join(root, "fake-minsync.mjs");
		writeFakeMinSync(binary);
		const output = join(root, "out");
		mkdirSync(output, { recursive: true });
		writeFileSync(join(output, "a.md"), "alpha");
		const stateDir = join(root, "state");

		await withSession(
			binary,
			async (session) => {
				await session.open(openInput({ output_root: output, items: [{ key: "a.md", version: "v1" }] }));
				await session.commit("a.md", "v1", { path: "a.md" });
				await session.finish();
			},
			{ stateDir },
		);

		const state = readMinSyncStage(binary, stateDir, root, {
			stage: "parse",
			source_id: "corpus",
			output_root: output,
		});
		expect(state.failed).toEqual([]);
		expect(state.skipped).toEqual([]);
		expect(state.items).toEqual([
			expect.objectContaining({
				key: "a.md",
				version: "v1",
				fingerprint: "parser-1",
				output: expect.objectContaining({ path: "a.md", size: 5, mtime_ns: expect.stringMatching(/^\d+$/) }),
			}),
		]);
	});

	it("accepts an answer from a helper that exits non-zero on EOF", () => {
		const binary = join(root, "eof.mjs");
		writeStub(binary, { kind: "responses", lines: [INSPECT_OK], exitCode: 1 });
		const state = readMinSyncStage(binary, join(root, "state"), root, {
			stage: "parse",
			source_id: "corpus",
			output_root: join(root, "out"),
		});
		expect(state.items[0]?.key).toBe("docs/a.pdf");
	});

	it("names the protocol a helper without inspect is missing", () => {
		const binary = join(root, "old.mjs");
		writeStub(binary, {
			kind: "responses",
			lines: [
				JSON.stringify({
					ok: false,
					op: "unknown",
					error: { code: "invalid_request", message: "unknown variant `inspect`" },
				}),
			],
		});
		const failure = caughtError(() =>
			readMinSyncStage(binary, join(root, "state"), root, {
				stage: "parse",
				source_id: "corpus",
				output_root: join(root, "out"),
			}),
		);
		expect(failure).toMatchObject({ code: "unsupported_protocol", op: "inspect" });
		expect(failure.message).toContain("`inspect`");
	});

	it("reports a helper it cannot run", () => {
		const failure = caughtError(() =>
			readMinSyncStage(join(root, "missing-minsync"), join(root, "state"), root, {
				stage: "parse",
				source_id: "corpus",
				output_root: join(root, "out"),
			}),
		);
		expect(failure).toMatchObject({ code: "spawn_failed", op: "inspect" });
	});
});

/** The `MinSyncStageError` a call must reject with. */
function caughtError(run: () => unknown): MinSyncStageError {
	try {
		run();
	} catch (error) {
		expect(error).toBeInstanceOf(MinSyncStageError);
		if (!(error instanceof MinSyncStageError)) throw error;
		return error;
	}
	throw new Error("the call did not fail");
}

async function rejection(run: Promise<unknown>): Promise<MinSyncStageError> {
	const failure = await run.catch((cause: unknown) => cause);
	expect(failure).toBeInstanceOf(MinSyncStageError);
	if (!(failure instanceof MinSyncStageError)) throw failure;
	return failure;
}

function exited(child: ChildProcess): Promise<boolean> {
	if (child.exitCode !== null || child.signalCode !== null) return Promise.resolve(true);
	return new Promise<boolean>((resolve) => {
		child.once("close", () => resolve(true));
	});
}

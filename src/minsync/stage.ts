import { type ChildProcess, spawn } from "node:child_process";
import { createInterface, type Interface } from "node:readline";
import { abortReason } from "../process/abort-reason.ts";
import { trackLiveChild, untrackLiveChild } from "../process/lifecycle.ts";
import { portableSpawnCommand } from "../process/portable-spawn.ts";
import { terminateProcessTree } from "../process/terminate-tree.ts";

/**
 * The host side of MinSync's materialization-stage protocol: one JSON request
 * per line on the child's stdin, one JSON response per line on its stdout
 * (`minsync stage --state-dir DIR`).
 *
 * One session is one run. MinSync holds the stage's state lock from `open`
 * until `finish` or exit, and the OS releases it on any exit, so a session that
 * ends without `finish` (EOF, a signal, a crash) leaves the journal behind and
 * the next `open` resumes from it. Requests are strictly sequential: one
 * request is in flight at a time, whatever the caller does.
 *
 * Keys and output paths stay exactly as given: both are relative to the
 * session's source listing and to `output_root`, and MinSync validates them at
 * its own boundary. MinSync is the only authority on what a request means — a
 * refusal rejects with its protocol code and the session stays usable, while a
 * request that MinSync never answers (EOF, a dead host-side pipe) rejects too,
 * so nothing waits on a process that is gone. Each request returns its own
 * promise; a caller that does not await one owns its rejection.
 */

/** What MinSync decided for one item. Only `process`, `retry` and `delete` are work to do. */
export type MinSyncStageAction = "process" | "retry" | "skip" | "delete" | "preserve";

export type MinSyncStageVerify = "fast" | "digest";

/** One current source item, relative to the stage's source listing. */
export interface MinSyncStageItem {
	readonly key: string;
	/**
	 * Any host-chosen string that changes whenever the item's content or its
	 * processing does — the mirror sends a stable JSON identity of the source
	 * path, parser and sizes. Opaque here: the transport passes it through.
	 */
	readonly version: string;
}

export interface MinSyncStageOpenInput {
	/** Stage name: 1-64 characters of `[A-Za-z0-9._-]`, not starting with `.` or `-`. */
	readonly stage: string;
	/** Corpus identity; one state directory belongs to exactly one. */
	readonly source_id: string;
	/** Everything that changes what the stage produces (parser name and version, output format, ...). */
	readonly fingerprint: string;
	/** Directory holding every output; MinSync creates it if missing. */
	readonly output_root: string;
	/** The whole current source. An item absent from `items` and from `unknown` is confirmed gone. */
	readonly items: readonly MinSyncStageItem[];
	/** Keys the source could not be observed for: their previous output is kept, never deleted. */
	readonly unknown?: readonly string[];
	/** Reuse only what this run committed itself, via its own ledger. */
	readonly full?: boolean;
	readonly verify?: MinSyncStageVerify;
	/** Wait for another run's lock instead of failing. */
	readonly wait?: boolean;
}

export interface MinSyncStageCounts {
	readonly process: number;
	readonly retry: number;
	readonly skip: number;
	/** Skips that rest on the interrupted run's ledger (a subset of `skip`). */
	readonly resumed: number;
	readonly delete: number;
	readonly preserve: number;
}

export interface MinSyncStageDecision {
	readonly key: string;
	readonly action: MinSyncStageAction;
	/** The source version to process; absent for `skip`, `delete` and `preserve`. */
	readonly version?: string;
}

export interface MinSyncStageOpenReport {
	readonly run_token: string;
	/** An interrupted run's ledger was resumed. */
	readonly resumed: boolean;
	readonly counts: MinSyncStageCounts;
	/** Work still to do, in key order. */
	readonly decisions: readonly MinSyncStageDecision[];
}

/** What the host wrote, relative to `output_root`. `size` and `digest` are optional cross-checks. */
export interface MinSyncStageOutput {
	readonly path: string;
	readonly size?: number;
	readonly digest?: string;
}

/** What MinSync recorded: measured itself, not what the host claimed. */
export interface MinSyncStageRecordedOutput {
	readonly path: string;
	readonly size: number;
	readonly digest: string;
}

export interface MinSyncStageFinishReport {
	/** Items committed in this run, including ones resumed from its ledger. */
	readonly committed: number;
	readonly deleted: number;
	readonly failed: number;
	/** Items the next run still has to do. */
	readonly pending: number;
}

interface MinSyncStageExit {
	readonly code: number | null;
	readonly signal: NodeJS.Signals | null;
}

/**
 * A stage session failure. `code` is MinSync's protocol code when it refused the
 * request (`invalid_output_path`, `output_missing`, `version_mismatch`,
 * `unknown_item`, `still_present`, `lock_failed`, ...) and a transport code when
 * it never answered: `spawn_failed`, `eof`, `invalid_response`, `process_failed`.
 */
export class MinSyncStageError extends Error {
	readonly code: string;
	readonly op: string | undefined;
	/** Tail of the session's stderr, for a failure MinSync could not report itself. */
	readonly stderr: string;

	constructor(code: string, message: string, details: { readonly op?: string; readonly stderr?: string } = {}) {
		super(message);
		this.name = "MinSyncStageError";
		this.code = code;
		this.op = details.op;
		this.stderr = details.stderr ?? "";
	}
}

/**
 * Grace the session gives the child to exit on its own (stdin EOF) before
 * SIGTERM, and again before SIGKILL. `close` is the only caller that waits, and
 * it must never leave a stage lock holder behind.
 */
const TERMINATE_GRACE_MS = 5_000;

/** Most stderr kept for diagnostics; a chatty child must not grow a buffer without bound. */
const STDERR_LIMIT = 8 * 1024;

/** Longest response fragment echoed into an error message. */
const LINE_ECHO_LIMIT = 200;

interface PendingRequest {
	readonly op: string;
	readonly resolve: (line: string) => void;
	readonly reject: (error: Error) => void;
}

interface MinSyncStageRequest {
	readonly op: string;
	readonly [key: string]: unknown;
}

export class MinSyncStageSession {
	private readonly child: ChildProcess;
	private readonly lines: Interface;
	private readonly signal: AbortSignal | undefined;
	private readonly exitPromise: Promise<MinSyncStageExit>;
	private readonly settleExit: (exit: MinSyncStageExit) => void;
	/** The caller gave up: stop the tree, then reject with the caller's own reason. */
	private readonly abortHandler = (): void => {
		const signal = this.signal;
		if (signal === undefined) return;
		this.terminate();
		this.failSession(abortReason(signal));
	};
	private stderrTail = "";
	private stderrTruncated = false;
	private pending: PendingRequest | null = null;
	private closedReason: Error | null = null;
	private exit: MinSyncStageExit | null = null;
	private tail: Promise<unknown> = Promise.resolve();
	private closing: Promise<void> | null = null;
	private escalation: NodeJS.Timeout | undefined;

	/**
	 * Start `minsync stage --state-dir stateDir` in `cwd`. `stateDir` holds the
	 * run's lock, marker, journal and snapshot, and belongs to one stage and one
	 * source. An aborted `signal` rejects the in-flight request with the signal's
	 * reason and takes the process tree down, which releases the stage lock.
	 */
	constructor(binaryPath: string, stateDir: string, cwd: string, signal?: AbortSignal) {
		this.signal = signal;
		const portable = portableSpawnCommand(binaryPath, ["stage", "--state-dir", stateDir]);
		this.child = spawn(portable.command, [...portable.args], {
			cwd,
			// POSIX: lead a process group so terminateProcessTree reaches every descendant.
			detached: process.platform !== "win32",
			// stdin is a pipe this host owns. There is no process timeout: a stage
			// legitimately runs for hours. A host that dies closes the pipe, the
			// child reads EOF and exits non-zero, and the run stays resumable.
			stdio: ["pipe", "pipe", "pipe"],
		});
		trackLiveChild(this.child);
		// Executor form: the ES2022 lib this package targets has no Promise.withResolvers.
		let settleExit: (exit: MinSyncStageExit) => void = () => {};
		this.exitPromise = new Promise<MinSyncStageExit>((resolve) => {
			settleExit = resolve;
		});
		this.settleExit = settleExit;
		this.child.stderr?.setEncoding("utf8");
		this.child.stderr?.on("data", (chunk: string) => this.appendStderr(chunk));
		this.lines = createInterface({ input: this.child.stdout!, crlfDelay: Number.POSITIVE_INFINITY });
		this.lines.on("line", (line) => this.onLine(line));
		this.child.on("error", (error) => {
			this.failSession(
				new MinSyncStageError("spawn_failed", `Running \`${binaryPath} stage\` failed: ${error.message}`, {
					stderr: this.stderrText(),
				}),
			);
		});
		this.child.stdin?.on("error", (error) => {
			// A write to a session that is already gone (EPIPE) has to settle its
			// request instead of surfacing as an unhandled stream error.
			this.failSession(
				new MinSyncStageError("eof", `MinSync stage stdin failed: ${error.message}`, { stderr: this.stderrText() }),
			);
		});
		this.child.on("close", (code, exitSignal) => this.onExit(code, exitSignal));
		if (signal?.aborted) {
			// The event already fired: take the tree down here, or the child would
			// hold the stage lock for a session that can never issue a request.
			this.abortHandler();
			return;
		}
		signal?.addEventListener("abort", this.abortHandler, { once: true });
	}

	/**
	 * Open a run, or resume the interrupted one, and return what it still has to
	 * do. A refusal (bad key, duplicate item, held lock, ...) rejects with its
	 * protocol code.
	 */
	async open(input: MinSyncStageOpenInput): Promise<MinSyncStageOpenReport> {
		const response = await this.send({
			op: "open",
			stage: input.stage,
			source_id: input.source_id,
			fingerprint: input.fingerprint,
			output_root: input.output_root,
			items: input.items.map((item) => ({ key: item.key, version: item.version })),
			unknown: input.unknown,
			full: input.full,
			verify: input.verify,
			wait: input.wait,
		});
		return readOpenReport(response);
	}

	/**
	 * Commit an item's output: `output.path` is relative to `output_root` and must
	 * already be written. MinSync checks the file and returns what it recorded.
	 */
	async commit(key: string, version: string, output: MinSyncStageOutput): Promise<MinSyncStageRecordedOutput> {
		const response = await this.send({
			op: "commit",
			key,
			version,
			output: { path: output.path, size: output.size, digest: output.digest },
		});
		return readRecordedOutput(response.output);
	}

	/** Record that an item could not be produced; its previous output stays intact. */
	async fail(key: string, version: string, error: string): Promise<void> {
		await this.send({ op: "fail", key, version, error });
	}

	/**
	 * Delete an item's recorded output, which must be absent from `items` and from
	 * `unknown`. Idempotent; returns whether a file was removed.
	 */
	async delete(key: string): Promise<boolean> {
		const response = await this.send({ op: "delete", key });
		return response.output_removed === true;
	}

	/**
	 * Fold this run into the snapshot and end the session with exit code 0, which
	 * is what makes the run complete rather than resumable. Fails if the session
	 * dies without finishing, or finishes with any other exit code.
	 */
	async finish(): Promise<MinSyncStageFinishReport> {
		const response = await this.send({ op: "finish" });
		const report = readFinishReport(response);
		const exit = await this.exitPromise;
		if (exit.code !== 0) {
			const cause = exit.signal === null ? `exit code ${exit.code ?? "unknown"}` : `signal ${exit.signal}`;
			throw new MinSyncStageError("process_failed", `MinSync stage ended with ${cause} instead of exit code 0.`, {
				op: "finish",
				stderr: this.stderrText(),
			});
		}
		return report;
	}

	/**
	 * End the session without `finish`: the run stays in the journal and the next
	 * `open` resumes it. Idempotent, and always reaps the child — stdin closes,
	 * then SIGTERM, then SIGKILL — so no stage lock holder is left behind.
	 */
	async close(): Promise<void> {
		this.closing ??= this.reap();
		return this.closing;
	}

	private async reap(): Promise<void> {
		const stdin = this.child.stdin;
		try {
			if (stdin !== null && !stdin.destroyed) stdin.end();
		} catch {
			// The child exited while its stdin was being ended; the kill below still reaps it.
		}
		if (this.exit !== null) return;
		const terminateTimer = setTimeout(() => terminateProcessTree(this.child, "SIGTERM"), TERMINATE_GRACE_MS);
		const killTimer = setTimeout(() => terminateProcessTree(this.child, "SIGKILL"), TERMINATE_GRACE_MS * 2);
		try {
			await this.exitPromise;
		} finally {
			clearTimeout(terminateTimer);
			clearTimeout(killTimer);
		}
	}

	/** One outstanding line-in/line-out exchange at a time, whatever the caller awaits. */
	private send(request: MinSyncStageRequest): Promise<Record<string, unknown>> {
		const result = this.tail.then(() => this.exchange(request));
		this.tail = result.then(
			() => undefined,
			() => undefined,
		);
		return result;
	}

	private async exchange(request: MinSyncStageRequest): Promise<Record<string, unknown>> {
		const closed = this.closedReason;
		if (closed !== null) throw closed;
		const { signal } = this;
		if (signal?.aborted) throw abortReason(signal);
		const line = await this.write(request);
		const response = parseResponse(line, request.op, this.stderrText());
		if (response.ok === true) return response;
		throw refusal(response, request.op, this.stderrText());
	}

	private write(request: MinSyncStageRequest): Promise<string> {
		return new Promise<string>((resolve, reject) => {
			const stdin = this.child.stdin;
			if (stdin === null || stdin.destroyed || !stdin.writable) {
				reject(
					this.closedReason ?? new MinSyncStageError("eof", "MinSync stage stdin is closed.", { op: request.op }),
				);
				return;
			}
			this.pending = { op: request.op, resolve, reject };
			try {
				stdin.write(`${JSON.stringify(request)}\n`, (error) => {
					if (error === null || error === undefined) return;
					this.failSession(
						new MinSyncStageError("eof", `Writing to MinSync stage failed: ${error.message}`, {
							op: request.op,
							stderr: this.stderrText(),
						}),
					);
				});
			} catch (error) {
				this.failSession(
					new MinSyncStageError(
						"eof",
						`Writing to MinSync stage failed: ${error instanceof Error ? error.message : String(error)}`,
						{ op: request.op, stderr: this.stderrText() },
					),
				);
			}
		});
	}

	private onLine(line: string): void {
		const pending = this.pending;
		// One request is outstanding at a time, so a line without one is protocol noise.
		if (pending === null) return;
		this.pending = null;
		pending.resolve(line);
	}

	private onExit(code: number | null, signal: NodeJS.Signals | null): void {
		clearTimeout(this.escalation);
		this.escalation = undefined;
		this.signal?.removeEventListener("abort", this.abortHandler);
		untrackLiveChild(this.child);
		this.exit = { code, signal };
		const pending = this.pending;
		const cause = signal === null ? `exit code ${code ?? "unknown"}` : `signal ${signal}`;
		this.settleExit(this.exit);
		this.failSession(
			new MinSyncStageError(
				"eof",
				pending === null
					? `MinSync stage session ended (${cause}).`
					: `MinSync stage ended (${cause}) before answering \`${pending.op}\`.`,
				{ op: pending?.op, stderr: this.stderrText() },
			),
		);
	}

	/** SIGTERM the process tree, escalating to SIGKILL when it does not go. */
	private terminate(): void {
		terminateProcessTree(this.child, "SIGTERM");
		this.escalation ??= setTimeout(() => terminateProcessTree(this.child, "SIGKILL"), TERMINATE_GRACE_MS);
	}

	/** The session is unusable from here on: reject whatever is in flight. */
	private failSession(error: Error): void {
		if (this.closedReason === null) this.closedReason = error;
		const pending = this.pending;
		if (pending === null) return;
		this.pending = null;
		pending.reject(error);
	}

	private appendStderr(chunk: string): void {
		const combined = this.stderrTail + chunk;
		if (combined.length <= STDERR_LIMIT) {
			this.stderrTail = combined;
			return;
		}
		this.stderrTail = combined.slice(combined.length - STDERR_LIMIT);
		this.stderrTruncated = true;
	}

	private stderrText(): string {
		const text = this.stderrTail.trim();
		if (text.length === 0) return "";
		return this.stderrTruncated ? `…${text}` : text;
	}
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** A response line can be arbitrarily long; an error message never needs all of it. */
function head(line: string): string {
	return line.length <= LINE_ECHO_LIMIT ? line : `${line.slice(0, LINE_ECHO_LIMIT)}…`;
}

function invalid(message: string, op: string): MinSyncStageError {
	return new MinSyncStageError("invalid_response", message, { op });
}

function parseResponse(line: string, op: string, stderr: string): Record<string, unknown> {
	let value: unknown;
	try {
		value = JSON.parse(line);
	} catch {
		throw new MinSyncStageError(
			"invalid_response",
			`MinSync stage wrote a line that is not JSON while answering \`${op}\`: ${head(line)}`,
			{ op, stderr },
		);
	}
	if (!isRecord(value)) {
		throw new MinSyncStageError("invalid_response", `MinSync stage answered \`${op}\` with ${head(line)}`, {
			op,
			stderr,
		});
	}
	return value;
}

function refusal(response: Record<string, unknown>, op: string, stderr: string): MinSyncStageError {
	const error = isRecord(response.error) ? response.error : {};
	const code = typeof error.code === "string" ? error.code : "internal";
	const detail =
		typeof error.message === "string" && error.message.length > 0
			? error.message
			: `MinSync stage refused \`${op}\`.`;
	return new MinSyncStageError(code, detail, { op, stderr });
}

function isAction(value: unknown): value is MinSyncStageAction {
	switch (value) {
		case "process":
		case "retry":
		case "skip":
		case "delete":
		case "preserve":
			return true;
		default:
			return false;
	}
}

function readCount(source: Record<string, unknown>, name: string, op: string): number {
	const count = source[name];
	if (typeof count !== "number") throw invalid(`The \`${op}\` response carried no \`${name}\`.`, op);
	return count;
}

function readCounts(value: unknown, op: string): MinSyncStageCounts {
	if (!isRecord(value)) throw invalid(`The \`${op}\` response carried no \`counts\` object.`, op);
	return {
		process: readCount(value, "process", op),
		retry: readCount(value, "retry", op),
		skip: readCount(value, "skip", op),
		resumed: readCount(value, "resumed", op),
		delete: readCount(value, "delete", op),
		preserve: readCount(value, "preserve", op),
	};
}

function readDecision(value: unknown, op: string): MinSyncStageDecision {
	if (!isRecord(value)) throw invalid(`The \`${op}\` response carried a decision that is not an object.`, op);
	const { key, action, version } = value;
	if (typeof key !== "string") throw invalid(`The \`${op}\` response carried a decision without a \`key\`.`, op);
	if (!isAction(action)) throw invalid(`The decision for \`${key}\` carried an unknown action.`, op);
	return typeof version === "string" ? { key, action, version } : { key, action };
}

function readOpenReport(response: Record<string, unknown>): MinSyncStageOpenReport {
	const runToken = response.run_token;
	if (typeof runToken !== "string" || runToken.length === 0) {
		throw invalid("The `open` response carried no `run_token`.", "open");
	}
	if (typeof response.resumed !== "boolean") throw invalid("The `open` response carried no `resumed` flag.", "open");
	const { resumed, decisions } = response;
	if (!Array.isArray(decisions)) throw invalid("The `open` response carried no `decisions` list.", "open");
	return {
		run_token: runToken,
		resumed,
		counts: readCounts(response.counts, "open"),
		decisions: decisions.map((decision) => readDecision(decision, "open")),
	};
}

function readRecordedOutput(value: unknown): MinSyncStageRecordedOutput {
	if (!isRecord(value)) throw invalid("The `commit` response carried no `output`.", "commit");
	const { path, size, digest } = value;
	if (typeof path !== "string" || typeof size !== "number" || typeof digest !== "string") {
		throw invalid("The `commit` response carried an incomplete `output`.", "commit");
	}
	return { path, size, digest };
}

function readFinishReport(response: Record<string, unknown>): MinSyncStageFinishReport {
	return {
		committed: readCount(response, "committed", "finish"),
		deleted: readCount(response, "deleted", "finish"),
		failed: readCount(response, "failed", "finish"),
		pending: readCount(response, "pending", "finish"),
	};
}

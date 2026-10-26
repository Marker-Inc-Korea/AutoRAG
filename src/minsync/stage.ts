import { type ChildProcess, spawn, spawnSync } from "node:child_process";
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
 *
 * `plan` and `inspect` are the read-only doors onto the same state: neither
 * opens a run, takes the lock, or writes anything, so either may be used
 * before `open` or while this session's own run is open.
 *
 * `open` is the only mutation, so it performs one `capabilities` handshake
 * first: a build that does not speak this transport's protocol is refused with
 * `unsupported_protocol` before any state is touched.
 */

/** What MinSync decided for one item. Only `process`, `retry` and `delete` are work to do. */
export type MinSyncStageAction = "process" | "retry" | "skip" | "delete" | "preserve";

export type MinSyncStageVerify = "fast" | "digest";

/**
 * What a MinSync build says it can do, from the `capabilities` op: the stage
 * protocol revision and the named capabilities it implements.
 */
export interface MinSyncStageCapabilities {
	/** Stage protocol revision; the durable-outcome protocol this host needs is 2. */
	readonly protocol: number;
	readonly features: readonly string[];
}

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
	/**
	 * Identity of the host's retry policy. A deferred failure is only retried
	 * once this differs from the one it was deferred under, so a stage that
	 * changes how it retries does not inherit the previous policy's deferrals.
	 */
	readonly retry_fingerprint?: string;
	/** Release every deferred failure now, whatever it was deferred against. */
	readonly retry_failed?: boolean;
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
	/**
	 * Set on the `skip` that stands in for a deferred failure: MinSync is
	 * deliberately not retrying until the source, the settings or the retry
	 * fingerprint change, or the host releases it. `open` reports these so the
	 * host can diagnose them; an unchanged successful skip stays a count.
	 */
	readonly deferred?: boolean;
	/** The recorded failure message behind a deferred `skip`. */
	readonly error?: string;
}

export interface MinSyncStageOpenReport {
	readonly run_token: string;
	/** An interrupted run's ledger was resumed. */
	readonly resumed: boolean;
	readonly counts: MinSyncStageCounts;
	/**
	 * Work still to do, in key order, plus a `skip` for every deferred failure
	 * MinSync is holding back. Unchanged successful skips stay in `counts`.
	 */
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

/** How `fail` records an item beyond the failure itself. */
export interface MinSyncStageFailOptions {
	/**
	 * Keep MinSync from retrying this exact failure until the source, the
	 * settings or the retry fingerprint change, or the host releases it with
	 * `retry_failed`. The item's previous output stays either way.
	 */
	readonly defer?: boolean;
	/**
	 * Remove the item's known output through MinSync, so a later run can no
	 * longer reuse it. Without this the last good output is preserved.
	 */
	readonly discard_output?: boolean;
}

/** A stage to reconstruct the durable state of; no open session is needed. */
export interface MinSyncStageInspectInput {
	readonly stage: string;
	readonly source_id: string;
	readonly output_root: string;
	readonly verify?: MinSyncStageVerify;
}

/** One durable output, as the recorded commit measured it. */
export interface MinSyncStageInspectOutput {
	readonly path: string;
	readonly size: number;
	readonly digest: string;
	/** Modification time at commit, in nanoseconds, as a decimal string. */
	readonly mtime_ns: string;
}

/** One source item MinSync holds a verified output for. */
export interface MinSyncStageInspectItem {
	readonly key: string;
	/** The host's own source version for the item; MinSync only stores it. */
	readonly version: string;
	readonly fingerprint: string;
	readonly output: MinSyncStageInspectOutput;
	/** Stable timestamp of this output, derived from the recorded mtime. */
	readonly updated_at: string;
}

/** The latest failure MinSync holds for an item. */
export interface MinSyncStageInspectFailure {
	readonly key: string;
	/** The source version that failed, absent when the host did not name one. */
	readonly version: string | null;
	readonly error: string;
	/** MinSync is holding the failure back instead of retrying it. */
	readonly deferred: boolean;
}

/** A deliberate omission MinSync recorded for an item. */
export interface MinSyncStageInspectSkip {
	readonly key: string;
	readonly version: string;
	readonly reason: string;
	readonly fingerprint: string;
}

/**
 * The durable state of a stage: every output that still verifies, every
 * failure and every deliberate omission. An item can appear in `items` with
 * its last good output and in `failed` for a newer version at the same time.
 */
export interface MinSyncStageInspectReport {
	readonly items: readonly MinSyncStageInspectItem[];
	readonly failed: readonly MinSyncStageInspectFailure[];
	readonly skipped: readonly MinSyncStageInspectSkip[];
}

export interface MinSyncStagePlanReport {
	readonly counts: MinSyncStageCounts;
	/**
	 * Every item's decision, in key order — `skip` and `preserve` included, so
	 * the host can report staleness without opening a run.
	 */
	readonly decisions: readonly MinSyncStageDecision[];
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

/**
 * The stage protocol revision whose semantics this transport requires: an
 * older build has no deliberate skips, no deferred failures, and no read-only
 * `inspect` or `plan`, so a run must not start against it.
 */
const REQUIRED_STAGE_PROTOCOL = 2;

/** What a run needs the build to implement before the host mutates anything. */
const REQUIRED_STAGE_FEATURES = [
	"retry_fingerprint",
	"retry_failed",
	"deferred_failure",
	"discard_output",
	"deliberate_skip",
	"plan",
	"inspect",
] as const;

/**
 * The one sentence naming the protocol an older MinSync is missing, whichever
 * extended call runs into it.
 */
const EXTENDED_PROTOCOL_REQUIREMENT =
	"Reading and writing durable parsing outcomes needs a MinSync whose `stage` command answers `capabilities`, `inspect`, `plan` and `skip`.";

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
	/** The one capability handshake this session performs, before it mutates. */
	private handshake: Promise<void> | null = null;
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
	 *
	 * A run is the first thing this transport mutates, so it first asks the
	 * build what it implements — once per session — and refuses to start
	 * against one that cannot own durable parsing outcomes.
	 */
	async open(input: MinSyncStageOpenInput): Promise<MinSyncStageOpenReport> {
		return this.enqueue(async () => {
			// The handshake and the open are one queued job, so a request the
			// caller never awaited still cannot slip ahead of them.
			await this.capabilities();
			const response = await this.exchange({ op: "open", ...openRequest(input) });
			return readOpenReport(response);
		});
	}
	/** Check this binary's contract without opening or creating a stage. */
	verifyProtocol(): Promise<void> {
		return this.enqueue(() => this.capabilities());
	}

	/**
	 * The one capability handshake this session performs, started on first use
	 * and always inside the request queue: a build from before the extended
	 * protocol answers the capabilities op with a refusal, or claims less than
	 * a run requires, and both become `unsupported_protocol`.
	 */
	private capabilities(): Promise<void> {
		this.handshake ??= this.checkCapabilities();
		return this.handshake;
	}

	private async checkCapabilities(): Promise<void> {
		let answer: Record<string, unknown>;
		try {
			// The probe carries no fields, so a build that refuses it as an invalid
			// request cannot know the op at all.
			answer = await this.exchange({ op: "capabilities" });
		} catch (error) {
			if (error instanceof MinSyncStageError && error.code === "invalid_request") {
				throw unsupportedProtocol("capabilities", error.message, error.stderr);
			}
			throw error;
		}
		const capabilities = readCapabilities(answer);
		const missing = REQUIRED_STAGE_FEATURES.filter((feature) => !capabilities.features.includes(feature));
		if (capabilities.protocol >= REQUIRED_STAGE_PROTOCOL && missing.length === 0) return;
		throw new MinSyncStageError("unsupported_protocol", describeCapabilities(capabilities, missing), {
			op: "capabilities",
		});
	}

	/**
	 * Preview what a run with the same request would do, without opening one:
	 * no lock, no journal, no snapshot write. Unlike `open` it lists every
	 * decision — `skip` and `preserve` included — so the host can report
	 * staleness without a session.
	 */
	async plan(input: MinSyncStageOpenInput): Promise<MinSyncStagePlanReport> {
		const response = await this.send({ op: "plan", ...openRequest(input) });
		return readPlanReport(response);
	}

	/**
	 * Reconstruct a stage's durable state from its snapshot and journal and
	 * verify every recorded output. Non-mutating: it takes no lock, so it is
	 * safe while another run of the same stage is open (that run's own journal
	 * records are merged in), and it needs no `open` of its own.
	 */
	async inspect(input: MinSyncStageInspectInput): Promise<MinSyncStageInspectReport> {
		const response = await this.send({
			op: "inspect",
			stage: input.stage,
			source_id: input.source_id,
			output_root: input.output_root,
			verify: input.verify,
		});
		return readInspectReport(response);
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

	/**
	 * Record that an item could not be produced. Its previous output stays
	 * intact unless `discard_output` asks MinSync to remove it, and `defer`
	 * keeps MinSync from retrying this same failure until the source, the
	 * settings or the retry fingerprint change.
	 */
	async fail(key: string, version: string, error: string, options: MinSyncStageFailOptions = {}): Promise<void> {
		await this.send({
			op: "fail",
			key,
			version,
			error,
			defer: options.defer,
			discard_output: options.discard_output,
		});
	}

	/**
	 * Record a deliberate omission: the item was considered and intentionally
	 * produced nothing, which is not a failure. MinSync removes any previous
	 * output of the item and stops asking for it while the source version and
	 * the settings stay the same.
	 */
	async skip(key: string, version: string, reason: string): Promise<void> {
		await this.send({ op: "skip", key, version, reason });
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
		return this.enqueue(() => this.exchange(request));
	}

	/**
	 * Run `work` after every request already issued and before any later one,
	 * so a job of several exchanges — the handshake and the open — is atomic
	 * with respect to a caller that does not await its requests.
	 */
	private enqueue<T>(work: () => Promise<T>): Promise<T> {
		const result = this.tail.then(work);
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
		const stderr = this.stderrText();
		const refused = refusal(response, request.op, stderr);
		const unsupported = unsupportedOp(response, request.op, refused, stderr);
		throw unsupported ?? refused;
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

/**
 * A build that does not know an op at all answers it with `invalid_request`
 * under a different op name — 0.5.0 answers `"unknown"` — while a build that
 * knows the op and disliked the request echoes that op. Only the first is a
 * missing protocol, so only that one is reported as such.
 */
function unsupportedOp(
	response: Record<string, unknown>,
	op: string,
	error: MinSyncStageError,
	stderr: string,
): MinSyncStageError | null {
	if (error.code !== "invalid_request" || response.op === op) return null;
	return unsupportedProtocol(op, error.message, stderr);
}

/** The refusal a build that does not know `op` produces, as this host reports it. */
function unsupportedProtocol(op: string, detail: string, stderr: string): MinSyncStageError {
	return new MinSyncStageError(
		"unsupported_protocol",
		`MinSync refused \`${op}\` as an invalid request (${detail}). ${EXTENDED_PROTOCOL_REQUIREMENT}`,
		{ op, stderr },
	);
}

function readCapabilities(response: Record<string, unknown>): MinSyncStageCapabilities {
	const { protocol, features } = response;
	if (typeof protocol !== "number" || !Number.isInteger(protocol)) {
		throw invalid("The `capabilities` response carried no `protocol` number.", "capabilities");
	}
	if (!Array.isArray(features)) {
		throw invalid("The `capabilities` response carried no `features` list.", "capabilities");
	}
	const names: string[] = [];
	for (const feature of features) {
		if (typeof feature !== "string") {
			throw invalid("The `capabilities` response carried a non-text feature.", "capabilities");
		}
		names.push(feature);
	}
	return { protocol, features: names };
}

/** Why a build is not good enough to own durable parsing outcomes. */
function describeCapabilities(capabilities: MinSyncStageCapabilities, missing: readonly string[]): string {
	const reported = capabilities.features.length === 0 ? "no features" : capabilities.features.join(", ");
	const shortfall = missing.length === 0 ? "its protocol is too old" : `it is missing ${missing.join(", ")}`;
	return `MinSync reports stage protocol ${capabilities.protocol} with ${reported}: ${shortfall}, while a run needs protocol ${REQUIRED_STAGE_PROTOCOL} with ${REQUIRED_STAGE_FEATURES.join(", ")}. ${EXTENDED_PROTOCOL_REQUIREMENT}`;
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
	const { key, action, version, deferred, error } = value;
	if (typeof key !== "string") throw invalid(`The \`${op}\` response carried a decision without a \`key\`.`, op);
	if (!isAction(action)) throw invalid(`The decision for \`${key}\` carried an unknown action.`, op);
	if (deferred !== undefined && typeof deferred !== "boolean") {
		throw invalid(`The decision for \`${key}\` carried a non-boolean \`deferred\`.`, op);
	}
	if (error !== undefined && typeof error !== "string") {
		throw invalid(`The decision for \`${key}\` carried a non-text \`error\`.`, op);
	}
	const decision: { key: string; action: MinSyncStageAction; version?: string; deferred?: boolean; error?: string } = {
		key,
		action,
	};
	if (typeof version === "string") decision.version = version;
	if (typeof deferred === "boolean") decision.deferred = deferred;
	if (typeof error === "string") decision.error = error;
	return decision;
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

/** The fields `open` and `plan` share; `plan` takes the very same request. */
function openRequest(input: MinSyncStageOpenInput): Record<string, unknown> {
	return {
		stage: input.stage,
		source_id: input.source_id,
		fingerprint: input.fingerprint,
		output_root: input.output_root,
		items: input.items.map((item) => ({ key: item.key, version: item.version })),
		unknown: input.unknown,
		full: input.full,
		verify: input.verify,
		retry_fingerprint: input.retry_fingerprint,
		retry_failed: input.retry_failed,
		wait: input.wait,
	};
}

function readPlanReport(response: Record<string, unknown>): MinSyncStagePlanReport {
	const { decisions } = response;
	if (!Array.isArray(decisions)) throw invalid("The `plan` response carried no `decisions` list.", "plan");
	return {
		counts: readCounts(response.counts, "plan"),
		decisions: decisions.map((decision) => readDecision(decision, "plan")),
	};
}

function readInspectItem(value: unknown): MinSyncStageInspectItem {
	const op = "inspect";
	if (!isRecord(value)) throw invalid("The `inspect` response carried an item that is not an object.", op);
	const { key, version, fingerprint, output, updated_at } = value;
	if (typeof key !== "string") throw invalid("An `inspect` item carried no `key`.", op);
	if (typeof version !== "string") throw invalid(`The \`inspect\` item for \`${key}\` carried no \`version\`.`, op);
	if (typeof fingerprint !== "string")
		throw invalid(`The \`inspect\` item for \`${key}\` carried no \`fingerprint\`.`, op);
	if (typeof updated_at !== "string")
		throw invalid(`The \`inspect\` item for \`${key}\` carried no \`updated_at\`.`, op);
	if (!isRecord(output)) throw invalid(`The \`inspect\` item for \`${key}\` carried no \`output\`.`, op);
	const { path, size, digest, mtime_ns } = output;
	if (typeof path !== "string" || typeof size !== "number" || typeof digest !== "string") {
		throw invalid(`The \`inspect\` item for \`${key}\` carried an incomplete \`output\`.`, op);
	}
	if (typeof mtime_ns !== "string") {
		throw invalid(`The \`inspect\` item for \`${key}\` carried a non-text \`output.mtime_ns\`.`, op);
	}
	return { key, version, fingerprint, updated_at, output: { path, size, digest, mtime_ns } };
}

function readInspectFailure(value: unknown): MinSyncStageInspectFailure {
	const op = "inspect";
	if (!isRecord(value)) throw invalid("The `inspect` response carried a failure that is not an object.", op);
	const { key, version, error, deferred } = value;
	if (typeof key !== "string") throw invalid("An `inspect` failure carried no `key`.", op);
	if (typeof error !== "string") throw invalid(`The \`inspect\` failure for \`${key}\` carried no \`error\`.`, op);
	if (typeof deferred !== "boolean")
		throw invalid(`The \`inspect\` failure for \`${key}\` carried no \`deferred\` flag.`, op);
	if (version !== null && version !== undefined && typeof version !== "string") {
		throw invalid(`The \`inspect\` failure for \`${key}\` carried a non-text \`version\`.`, op);
	}
	return { key, version: typeof version === "string" ? version : null, error, deferred };
}

function readInspectSkip(value: unknown): MinSyncStageInspectSkip {
	const op = "inspect";
	if (!isRecord(value)) throw invalid("The `inspect` response carried a skip that is not an object.", op);
	const { key, version, reason, fingerprint } = value;
	if (typeof key !== "string") throw invalid("An `inspect` skip carried no `key`.", op);
	if (typeof version !== "string") throw invalid(`The \`inspect\` skip for \`${key}\` carried no \`version\`.`, op);
	if (typeof reason !== "string") throw invalid(`The \`inspect\` skip for \`${key}\` carried no \`reason\`.`, op);
	if (typeof fingerprint !== "string")
		throw invalid(`The \`inspect\` skip for \`${key}\` carried no \`fingerprint\`.`, op);
	return { key, version, reason, fingerprint };
}

function readInspectReport(response: Record<string, unknown>): MinSyncStageInspectReport {
	const { items, failed, skipped } = response;
	if (!Array.isArray(items)) throw invalid("The `inspect` response carried no `items` list.", "inspect");
	if (!Array.isArray(failed)) throw invalid("The `inspect` response carried no `failed` list.", "inspect");
	if (!Array.isArray(skipped)) throw invalid("The `inspect` response carried no `skipped` list.", "inspect");
	return {
		items: items.map(readInspectItem),
		failed: failed.map(readInspectFailure),
		skipped: skipped.map(readInspectSkip),
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

/**
 * Node's own limit for one synchronous answer: an `inspect` listing of a whole
 * corpus must not be truncated, and the engine answers from local state.
 */
const INSPECT_MAX_BUFFER = 2 ** 31 - 1;

/**
 * Read a stage's durable state in one synchronous round trip, for callers that
 * reconstruct state while they are already synchronous. Exactly one `inspect`
 * line is written and stdin is closed, so a helper that answers and then exits
 * non-zero on EOF is still fine — code 0 or 1 with a valid answer is success.
 *
 * Anything else throws `MinSyncStageError`: MinSync's own refusal code, a
 * transport code (`spawn_failed`, `process_failed`) or `unsupported_protocol`
 * when the build simply never answered.
 */
export function readMinSyncStage(
	binaryPath: string,
	stateDir: string,
	cwd: string,
	input: MinSyncStageInspectInput,
): MinSyncStageInspectReport {
	const portable = portableSpawnCommand(binaryPath, ["stage", "--state-dir", stateDir]);
	const request = JSON.stringify({
		op: "inspect",
		stage: input.stage,
		source_id: input.source_id,
		output_root: input.output_root,
		verify: input.verify,
	});
	// No timeout: this is a local read of the engine's own state.
	const result = spawnSync(portable.command, [...portable.args], {
		cwd,
		input: `${request}\n`,
		encoding: "utf8",
		maxBuffer: INSPECT_MAX_BUFFER,
		stdio: ["pipe", "pipe", "pipe"],
	});
	const stderr = (result.stderr ?? "").slice(-STDERR_LIMIT).trim();
	// A signal means the answer may be half-written; only a clean exit counts.
	if (result.signal === null && (result.status === 0 || result.status === 1)) {
		const answer = readInspectLine(result.stdout ?? "", stderr);
		if (answer !== null) return answer;
	}
	const failure = describeSpawnFailure(result);
	throw new MinSyncStageError(failure.code, `${failure.message} ${EXTENDED_PROTOCOL_REQUIREMENT}`, {
		op: "inspect",
		stderr,
	});
}

/** The `inspect` answer on stdout, or `null` when the helper never sent one. */
function readInspectLine(stdout: string, stderr: string): MinSyncStageInspectReport | null {
	for (const line of stdout.split("\n")) {
		const text = line.trim();
		if (text.length === 0) continue;
		let value: unknown;
		try {
			value = JSON.parse(text);
		} catch {
			continue;
		}
		if (!isRecord(value)) continue;
		if (value.op === "inspect" && value.ok === true) return readInspectReport(value);
		if (value.ok === false) {
			const refused = refusal(value, "inspect", stderr);
			// A build without the extended protocol rejects the unknown op this way.
			throw unsupportedOp(value, "inspect", refused, stderr) ?? refused;
		}
	}
	return null;
}

/** Why a synchronous `stage` call produced no answer. */
function describeSpawnFailure(result: {
	readonly status: number | null;
	readonly signal: NodeJS.Signals | null;
	readonly error?: Error | undefined;
}): { readonly code: string; readonly message: string } {
	if (result.error !== undefined) {
		return { code: "spawn_failed", message: `Running \`minsync stage\` failed: ${result.error.message}.` };
	}
	if (result.signal !== null) {
		return {
			code: "process_failed",
			message: `MinSync stage was killed by signal ${result.signal} before answering \`inspect\`.`,
		};
	}
	if (result.status !== null && result.status !== 0 && result.status !== 1) {
		return {
			code: "process_failed",
			message: `MinSync stage exited with code ${result.status} before answering \`inspect\`.`,
		};
	}
	return { code: "unsupported_protocol", message: "MinSync stage answered nothing to `inspect`." };
}

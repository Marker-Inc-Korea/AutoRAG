import { type ChildProcess, spawn } from "node:child_process";
import { abortReason } from "../process/abort-reason.ts";
import { portableSpawnCommand } from "../process/portable-spawn.ts";
import { terminateProcessTree } from "../process/terminate-tree.ts";

export interface ProcessResult {
	readonly ok: boolean;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number | null;
	readonly timedOut?: boolean;
}

/**
 * Grace between the polite SIGTERM to the process group and the SIGKILL that
 * follows when something in the tree ignores it. Long enough for MinSync to
 * release its `.minsync/lock` on a graceful stop, short enough that an abort
 * never leaves a caller waiting on a wedged descendant.
 */
const TERMINATE_GRACE_MS = 5_000;

const HOST_SIGNALS = ["SIGINT", "SIGTERM", "SIGHUP"] as const;

/**
 * Children still running. A POSIX child leads its own process group, so it no
 * longer receives the terminal's Ctrl-C or the host's group-wide signals; the
 * host takes the tree down with it instead of orphaning a lock-holding sync.
 */
const liveChildren = new Set<ChildProcess>();
const hostSignalHandlers = new Map<NodeJS.Signals, () => void>();

function killLiveChildren(): void {
	for (const child of liveChildren) terminateProcessTree(child, "SIGKILL");
}

function installHostCleanup(): void {
	if (hostSignalHandlers.size > 0) return;
	process.on("exit", killLiveChildren);
	for (const signal of HOST_SIGNALS) {
		const handler = (): void => {
			killLiveChildren();
			// Listening for a signal replaces its default "terminate" action. When
			// nothing else listens, restore it so the host still dies from the signal.
			if (process.listenerCount(signal) === 1) {
				removeHostCleanup();
				process.kill(process.pid, signal);
			}
		};
		hostSignalHandlers.set(signal, handler);
		process.on(signal, handler);
	}
}

function removeHostCleanup(): void {
	process.off("exit", killLiveChildren);
	for (const [signal, handler] of hostSignalHandlers) process.off(signal, handler);
	hostSignalHandlers.clear();
}

export function spawnProcess(
	command: string,
	args: readonly string[],
	cwd: string,
	options: { readonly timeoutMs?: number; readonly signal?: AbortSignal; readonly env?: NodeJS.ProcessEnv } = {},
): Promise<ProcessResult> {
	// Executor form: the ES2022 lib this package targets has no Promise.withResolvers.
	return new Promise((resolve, reject) => {
		const { signal } = options;
		if (signal?.aborted) {
			reject(abortReason(signal));
			return;
		}
		const portable = portableSpawnCommand(command, args);
		const posix = process.platform !== "win32";
		const child = spawn(portable.command, [...portable.args], {
			cwd,
			...(options.env === undefined ? {} : { env: options.env }),
			// POSIX: lead a process group so terminateProcessTree reaches every
			// descendant with kill(-pid). Windows: keep the parent/child chain
			// that `taskkill /t` walks.
			detached: posix,
			stdio: ["ignore", "pipe", "pipe"],
		});
		if (posix) {
			liveChildren.add(child);
			installHostCleanup();
		}
		let stdout = "";
		let stderr = "";
		let timedOut = false;
		let escalation: ReturnType<typeof setTimeout> | undefined;
		child.stdout.setEncoding("utf8");
		child.stderr.setEncoding("utf8");
		child.stdout.on("data", (chunk: string) => {
			stdout += chunk;
		});
		child.stderr.on("data", (chunk: string) => {
			stderr += chunk;
		});
		const terminate = (): void => {
			terminateProcessTree(child, "SIGTERM");
			escalation ??= setTimeout(() => terminateProcessTree(child, "SIGKILL"), TERMINATE_GRACE_MS);
		};
		const timer =
			options.timeoutMs !== undefined
				? setTimeout(() => {
						timedOut = true;
						terminate();
					}, options.timeoutMs)
				: undefined;
		// The caller gave up on the result: stop the process tree instead of letting it run on.
		signal?.addEventListener("abort", terminate, { once: true });
		const settle = (): void => {
			clearTimeout(timer);
			clearTimeout(escalation);
			signal?.removeEventListener("abort", terminate);
			liveChildren.delete(child);
			if (liveChildren.size === 0) removeHostCleanup();
		};
		child.on("error", (error) => {
			settle();
			if (signal?.aborted) reject(abortReason(signal));
			else resolve({ ok: false, stdout, stderr: error.message, code: null });
		});
		child.on("close", (code) => {
			settle();
			if (signal?.aborted) {
				reject(abortReason(signal));
			} else if (timedOut) {
				resolve({ ok: false, stdout, stderr: "process timed out", code: null, timedOut: true });
			} else {
				resolve({ ok: code === 0, stdout, stderr, code });
			}
		});
	});
}

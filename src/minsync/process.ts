import { spawn } from "node:child_process";
import { abortReason } from "../process/abort-reason.ts";
import { trackLiveChild, untrackLiveChild } from "../process/lifecycle.ts";
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

export function spawnProcess(
	command: string,
	args: readonly string[],
	cwd: string,
	options: {
		readonly timeoutMs?: number;
		readonly signal?: AbortSignal;
		readonly env?: NodeJS.ProcessEnv;
		readonly onStdoutLine?: (line: string) => void;
	} = {},
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
		trackLiveChild(child);
		let stdout = "";
		let stderr = "";
		let stdoutLine = "";
		let timedOut = false;
		let escalation: ReturnType<typeof setTimeout> | undefined;
		child.stdout.setEncoding("utf8");
		child.stderr.setEncoding("utf8");
		child.stdout.on("data", (chunk: string) => {
			stdout += chunk;
			if (options.onStdoutLine === undefined) return;
			stdoutLine += chunk;
			let newline = stdoutLine.indexOf("\n");
			while (newline >= 0) {
				options.onStdoutLine(stdoutLine.slice(0, newline).replace(/\r$/, ""));
				stdoutLine = stdoutLine.slice(newline + 1);
				newline = stdoutLine.indexOf("\n");
			}
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
			untrackLiveChild(child);
		};
		child.on("error", (error) => {
			settle();
			if (signal?.aborted) reject(abortReason(signal));
			else resolve({ ok: false, stdout, stderr: error.message, code: null });
		});
		child.on("close", (code) => {
			if (stdoutLine.length > 0) options.onStdoutLine?.(stdoutLine.replace(/\r$/, ""));
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

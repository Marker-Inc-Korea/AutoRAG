import { spawn } from "node:child_process";
import { portableSpawnCommand } from "../process/portable-spawn.ts";
import { terminateProcessTree } from "../process/terminate-tree.ts";

export interface ProcessResult {
	readonly ok: boolean;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number | null;
	readonly timedOut?: boolean;
}

export function spawnProcess(
	command: string,
	args: readonly string[],
	cwd: string,
	options: { readonly timeoutMs?: number; readonly signal?: AbortSignal } = {},
): Promise<ProcessResult> {
	// Executor form: the ES2022 lib this package targets has no Promise.withResolvers.
	return new Promise((resolve, reject) => {
		const { signal } = options;
		if (signal?.aborted) {
			reject(abortError(signal));
			return;
		}
		const portable = portableSpawnCommand(command, args);
		// A signal-bearing run leads its own process group on POSIX so an abort reaches descendants too;
		// without a signal the child stays in the caller's group exactly as before.
		const child = spawn(portable.command, [...portable.args], {
			cwd,
			stdio: ["ignore", "pipe", "pipe"],
			...(signal !== undefined && process.platform !== "win32" ? { detached: true } : {}),
		});
		let stdout = "";
		let stderr = "";
		let timedOut = false;
		child.stdout.setEncoding("utf8");
		child.stderr.setEncoding("utf8");
		child.stdout.on("data", (chunk: string) => {
			stdout += chunk;
		});
		child.stderr.on("data", (chunk: string) => {
			stderr += chunk;
		});
		const timer =
			options.timeoutMs !== undefined
				? setTimeout(() => {
						timedOut = true;
						child.kill("SIGTERM");
					}, options.timeoutMs)
				: undefined;
		// The caller gave up on the result: stop the process instead of letting it run on.
		const onAbort = (): void => {
			terminateProcessTree(child);
		};
		signal?.addEventListener("abort", onAbort, { once: true });
		const settle = (): void => {
			clearTimeout(timer);
			signal?.removeEventListener("abort", onAbort);
		};
		child.on("error", (error) => {
			settle();
			if (signal?.aborted) reject(abortError(signal));
			else resolve({ ok: false, stdout, stderr: error.message, code: null });
		});
		child.on("close", (code) => {
			settle();
			if (signal?.aborted) {
				reject(abortError(signal));
			} else if (timedOut) {
				resolve({ ok: false, stdout, stderr: "process timed out", code: null, timedOut: true });
			} else {
				resolve({ ok: code === 0, stdout, stderr, code });
			}
		});
	});
}

/** The rejection for a process its caller aborted: the signal's reason when it is an error. */
function abortError(signal: AbortSignal): Error {
	return signal.reason instanceof Error ? signal.reason : new DOMException("The operation was aborted", "AbortError");
}

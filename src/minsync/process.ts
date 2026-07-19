import { spawn } from "node:child_process";

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
	options: { readonly timeoutMs?: number } = {},
): Promise<ProcessResult> {
	return new Promise((resolve) => {
		const child = spawn(command, [...args], { cwd, stdio: ["ignore", "pipe", "pipe"] });
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
		child.on("error", (error) => {
			if (timer) clearTimeout(timer);
			resolve({ ok: false, stdout, stderr: error.message, code: null });
		});
		child.on("close", (code) => {
			if (timer) clearTimeout(timer);
			if (timedOut) {
				resolve({ ok: false, stdout, stderr: "process timed out", code: null, timedOut: true });
			} else {
				resolve({ ok: code === 0, stdout, stderr, code });
			}
		});
	});
}

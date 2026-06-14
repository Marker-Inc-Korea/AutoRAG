import { spawn } from "node:child_process";

export interface ProcessResult {
	readonly ok: boolean;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number | null;
}

export function spawnProcess(command: string, args: readonly string[], cwd: string): Promise<ProcessResult> {
	return new Promise((resolve) => {
		const child = spawn(command, [...args], { cwd, stdio: ["ignore", "pipe", "pipe"] });
		let stdout = "";
		let stderr = "";
		child.stdout.setEncoding("utf8");
		child.stderr.setEncoding("utf8");
		child.stdout.on("data", (chunk: string) => {
			stdout += chunk;
		});
		child.stderr.on("data", (chunk: string) => {
			stderr += chunk;
		});
		child.on("error", (error) => {
			resolve({ ok: false, stdout, stderr: error.message, code: null });
		});
		child.on("close", (code) => {
			resolve({ ok: code === 0, stdout, stderr, code });
		});
	});
}

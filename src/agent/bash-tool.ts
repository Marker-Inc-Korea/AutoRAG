import { spawn, spawnSync } from "node:child_process";
import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import { classifyFilesystemRoot, isDatalessPlaceholder } from "../filesystem/cloud-placeholder.ts";

export const BASH_TOOL_NAME = "bash";

const DEFAULT_TIMEOUT_MS = 120_000;
const DEFAULT_MAX_OUTPUT_BYTES = 131_072;

const bashSchema = Type.Object({
	command: Type.String({
		description: "Shell command to run (bash -c). Use standard tools like ls, find, grep, cat, head, sed, rg.",
	}),
	cwd: Type.Optional(
		Type.String({ description: "Working directory for the command. Defaults to the configured workspace root." }),
	),
	timeoutMs: Type.Optional(Type.Integer({ description: "Optional timeout in milliseconds for this command." })),
});

export interface BashToolOptions {
	/** Default working directory for spawned commands. */
	readonly cwd: string;
	/** Maximum wall-clock time per command before it is killed. */
	readonly timeoutMs?: number;
	/** Maximum captured output bytes before truncation. */
	readonly maxOutputBytes?: number;
}

export interface BashToolDetails {
	readonly method: "bash";
	readonly command: string;
	readonly exitCode: number | undefined;
	readonly timedOut: boolean;
	readonly truncated: boolean;
}

const CLOUD_CONTENT_COMMAND = /\b(?:rg|ripgrep|grep|egrep|fgrep|cat|less|more|head|tail|sed|awk)\b/i;

interface RunResult {
	readonly output: string;
	readonly exitCode: number | undefined;
	readonly timedOut: boolean;
	readonly truncated: boolean;
}

function runCommand(
	command: string,
	cwd: string,
	timeoutMs: number,
	maxOutputBytes: number,
	signal?: AbortSignal,
): Promise<RunResult> {
	return new Promise((resolve) => {
		const shell = process.platform === "win32" ? "bash.exe" : "/bin/bash";
		const child = spawn(shell, ["-c", command], { cwd, detached: true });
		const chunks: Buffer[] = [];
		let total = 0;
		let truncated = false;
		let timedOut = false;
		let settled = false;

		const collect = (data: Buffer) => {
			if (total >= maxOutputBytes) {
				truncated = true;
				return;
			}
			const remaining = maxOutputBytes - total;
			if (data.length > remaining) {
				chunks.push(data.subarray(0, remaining));
				total = maxOutputBytes;
				truncated = true;
			} else {
				chunks.push(data);
				total += data.length;
			}
		};

		child.stdout?.on("data", collect);
		child.stderr?.on("data", collect);

		const finish = (exitCode: number | undefined) => {
			if (settled) return;
			settled = true;
			clearTimeout(timer);
			signal?.removeEventListener("abort", onAbort);
			resolve({ output: Buffer.concat(chunks).toString("utf8"), exitCode, timedOut, truncated });
		};

		const killProcessTree = () => {
			if (child.pid === undefined) return;
			if (process.platform === "win32") {
				spawnSync("taskkill", ["/pid", String(child.pid), "/t", "/f"], { stdio: "ignore" });
				return;
			}
			try {
				process.kill(-child.pid, "SIGKILL");
			} catch (error) {
				if ((error as NodeJS.ErrnoException).code !== "ESRCH") throw error;
			}
		};

		const timer = setTimeout(() => {
			timedOut = true;
			killProcessTree();
			child.stdout?.destroy();
			child.stderr?.destroy();
			setTimeout(() => finish(undefined), 50);
		}, timeoutMs);

		const onAbort = () => {
			killProcessTree();
			child.stdout?.destroy();
			child.stderr?.destroy();
			setTimeout(() => finish(undefined), 50);
		};
		signal?.addEventListener("abort", onAbort, { once: true });

		child.on("error", () => finish(undefined));
		child.on("close", (code) => finish(code === null ? undefined : code));
	});
}

/**
 * Real shell access for the librarian agent. AutoRAG navigates and reads the
 * configured collection directly through this tool (ls, find, grep, cat, …).
 * Output — including real filesystem paths — is returned to the model verbatim.
 */
export function createBashTool(options: BashToolOptions): AgentTool<typeof bashSchema, BashToolDetails> {
	const defaultTimeout = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
	const maxOutputBytes = options.maxOutputBytes ?? DEFAULT_MAX_OUTPUT_BYTES;
	return {
		name: BASH_TOOL_NAME,
		label: "Bash",
		description:
			"Run a shell command to explore and read the document collection (ls, find, grep, cat, head, sed, rg). Returns combined stdout/stderr.",
		parameters: bashSchema,
		async execute(_toolCallId, params, signal): Promise<AgentToolResult<BashToolDetails>> {
			const command = String(params.command ?? "").trim();
			if (command.length === 0) {
				return {
					content: [{ type: "text", text: "Command was empty; nothing was run." }],
					details: { method: "bash", command: "", exitCode: undefined, timedOut: false, truncated: false },
				};
			}
			const cwd = typeof params.cwd === "string" && params.cwd.length > 0 ? params.cwd : options.cwd;
			if (CLOUD_CONTENT_COMMAND.test(command)) {
				const classification = await classifyFilesystemRoot(cwd);
				if (classification.kind === "file-provider") {
					return {
						content: [
							{
								type: "text",
								text:
									"AUTORAG_CLOUD_PLACEHOLDER_BLOCKED: content access under a cloud placeholder root was refused to prevent remote hydration. " +
									"Use the configured cloud-drive datasource or a materialized local file.",
							},
						],
						details: { method: "bash", command, exitCode: undefined, timedOut: false, truncated: false },
					};
				}
				if (await isDatalessPlaceholder(cwd)) {
					return {
						content: [
							{
								type: "text",
								text:
									"AUTORAG_CLOUD_PLACEHOLDER_BLOCKED: the working path is a data-less cloud placeholder. " +
									"Use the configured cloud-drive datasource or materialize the file first.",
							},
						],
						details: { method: "bash", command, exitCode: undefined, timedOut: false, truncated: false },
					};
				}
			}
			const timeoutMs =
				typeof params.timeoutMs === "number" && params.timeoutMs > 0 ? params.timeoutMs : defaultTimeout;
			const run = await runCommand(command, cwd, timeoutMs, maxOutputBytes, signal);

			const parts: string[] = [];
			parts.push(run.output.length > 0 ? run.output : "(no output)");
			if (run.timedOut) parts.push(`\n(command timed out after ${timeoutMs}ms and was killed)`);
			if (run.truncated) parts.push(`\n(output truncated at ${maxOutputBytes} bytes)`);
			if (run.exitCode !== undefined && run.exitCode !== 0 && !run.timedOut) {
				parts.push(`\n(exit code ${run.exitCode})`);
			}

			return {
				content: [{ type: "text", text: parts.join("") }],
				details: {
					method: "bash",
					command,
					exitCode: run.exitCode,
					timedOut: run.timedOut,
					truncated: run.truncated,
				},
			};
		},
	};
}

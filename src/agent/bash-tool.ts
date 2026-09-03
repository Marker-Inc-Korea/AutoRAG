import { spawn } from "node:child_process";
import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import { classifyFilesystemRoot, isDatalessPlaceholder } from "../filesystem/cloud-placeholder.ts";

export const BASH_TOOL_NAME = "bash";

const DEFAULT_TIMEOUT_MS = 120_000;
const DEFAULT_MAX_OUTPUT_BYTES = 131_072;
const KILL_EXIT_GRACE_MS = 5_000;

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
	readonly aborted: boolean;
	readonly exitCode: number | undefined;
	readonly terminationFailed: boolean;
	readonly timedOut: boolean;
	readonly truncated: boolean;
}

const CLOUD_CONTENT_COMMAND = /\b(?:rg|ripgrep|grep|egrep|fgrep|cat|less|more|head|tail|sed|awk)\b/i;

interface RunResult {
	readonly output: string;
	readonly exitCode: number | undefined;
	readonly aborted: boolean;
	readonly terminationFailed: boolean;
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
		if (signal?.aborted) {
			resolve({
				output: "",
				exitCode: undefined,
				aborted: true,
				terminationFailed: false,
				timedOut: false,
				truncated: false,
			});
			return;
		}
		const shell = process.platform === "win32" ? "bash.exe" : "/bin/bash";
		const child = spawn(shell, ["-c", command], {
			cwd,
			// Unix: new process group so kill(-pid) reaps descendants.
			// Windows: keep the Win32 parent/child chain for taskkill /T.
			detached: process.platform !== "win32",
			windowsHide: true,
		});
		const chunks: Buffer[] = [];
		let total = 0;
		let truncated = false;
		let timedOut = false;
		let aborted = false;
		let settled = false;
		let killGraceTimer: ReturnType<typeof setTimeout> | undefined;
		let terminationRequested = false;
		let terminationStarted = false;
		let terminationSettled = false;
		let terminationFailed = false;
		let pendingExitCode: number | undefined;

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
			if (terminationRequested && !terminationSettled) {
				pendingExitCode = exitCode;
				return;
			}
			settled = true;
			clearTimeout(timer);
			if (killGraceTimer !== undefined) clearTimeout(killGraceTimer);
			signal?.removeEventListener("abort", onAbort);
			resolve({
				output: Buffer.concat(chunks).toString("utf8"),
				exitCode,
				aborted,
				terminationFailed,
				timedOut,
				truncated,
			});
		};

		const killDirectChild = (): boolean => {
			try {
				return child.kill("SIGKILL");
			} catch {
				return false;
			}
		};

		const killProcessTree = async (): Promise<boolean> => {
			if (child.pid === undefined) return true;
			if (process.platform === "win32") {
				return await new Promise<boolean>((resolve) => {
					const killer = spawn("taskkill", ["/pid", String(child.pid), "/t", "/f"], { stdio: "ignore" });
					let completed = false;
					const graceTimer = setTimeout(() => {
						if (completed) return;
						completed = true;
						killer.kill();
						resolve(killDirectChild());
					}, KILL_EXIT_GRACE_MS);
					const finishKill = (success: boolean) => {
						if (completed) return;
						completed = true;
						clearTimeout(graceTimer);
						if (!success) killDirectChild();
						resolve(success);
					};
					killer.on("error", () => finishKill(false));
					killer.on("close", (code) => finishKill(code === 0));
				});
			}
			try {
				process.kill(-child.pid, "SIGKILL");
				return true;
			} catch {
				killDirectChild();
				return false;
			}
		};

		const releaseAfterKill = async () => {
			if (terminationStarted) return;
			terminationStarted = true;
			const killed = await killProcessTree();
			terminationSettled = true;
			terminationFailed = !killed;
			child.stdout?.destroy();
			child.stderr?.destroy();
			killGraceTimer = setTimeout(() => finish(undefined), KILL_EXIT_GRACE_MS);
			if (pendingExitCode !== undefined) finish(pendingExitCode);
		};

		const timer = setTimeout(() => {
			timedOut = true;
			terminationRequested = true;
			void releaseAfterKill();
		}, timeoutMs);

		const onAbort = () => {
			aborted = true;
			terminationRequested = true;
			void releaseAfterKill();
		};
		if (signal) {
			if (signal.aborted) onAbort();
			else signal.addEventListener("abort", onAbort, { once: true });
		}

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
					details: {
						method: "bash",
						command: "",
						aborted: false,
						exitCode: undefined,
						terminationFailed: false,
						timedOut: false,
						truncated: false,
					},
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
						details: {
							method: "bash",
							command,
							aborted: false,
							exitCode: undefined,
							terminationFailed: false,
							timedOut: false,
							truncated: false,
						},
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
						details: {
							method: "bash",
							command,
							aborted: false,
							exitCode: undefined,
							terminationFailed: false,
							timedOut: false,
							truncated: false,
						},
					};
				}
			}
			const timeoutMs =
				typeof params.timeoutMs === "number" && params.timeoutMs > 0 ? params.timeoutMs : defaultTimeout;
			const run = await runCommand(command, cwd, timeoutMs, maxOutputBytes, signal);

			const parts: string[] = [];
			parts.push(run.output.length > 0 ? run.output : run.aborted ? "(command aborted)" : "(no output)");
			if (run.timedOut) parts.push(`\n(command timed out after ${timeoutMs}ms and was killed)`);
			if (run.truncated) parts.push(`\n(output truncated at ${maxOutputBytes} bytes)`);
			if (run.exitCode !== undefined && run.exitCode !== 0 && !run.timedOut) {
				parts.push(`\n(exit code ${run.exitCode})`);
			}
			if (run.terminationFailed) parts.push("\n(process termination failed)");

			return {
				content: [{ type: "text", text: parts.join("") }],
				details: {
					method: "bash",
					command,
					aborted: run.aborted,
					exitCode: run.exitCode,
					terminationFailed: run.terminationFailed,
					timedOut: run.timedOut,
					truncated: run.truncated,
				},
			};
		},
	};
}

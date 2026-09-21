import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { BASH_TOOL_NAME, createBashTool } from "../../src/agent/bash-tool.ts";

const PID_WAIT_MS = 10_000;
const TREE_TIMEOUT_MS = 8_000;
const RESOLVE_WAIT_MS = 15_000;
const DEATH_WAIT_MS = 2_000;

let tmpDir: string;

async function waitForFile(path: string, timeoutMs = PID_WAIT_MS): Promise<void> {
	const deadline = Date.now() + timeoutMs;
	while (Date.now() < deadline) {
		if (existsSync(path) && readFileSync(path, "utf8").trim().length > 0) return;
		await new Promise((resolve) => setTimeout(resolve, 25));
	}
	throw new Error(`Timed out waiting for ${basename(path)}`);
}

function isProcessAlive(pid: number): boolean {
	if (!Number.isInteger(pid) || pid <= 0) return false;
	try {
		process.kill(pid, 0);
		return true;
	} catch {
		return false;
	}
}

async function waitForProcessDeath(pid: number, timeoutMs = DEATH_WAIT_MS): Promise<void> {
	const deadline = Date.now() + timeoutMs;
	while (Date.now() < deadline) {
		if (!isProcessAlive(pid)) return;
		await new Promise((resolve) => setTimeout(resolve, 25));
	}
	throw new Error(`Process ${pid} still alive`);
}

function hangingChildCommand(pidFileName: string): string {
	writeFileSync(
		join(tmpDir, "hang.mjs"),
		`import { writeFileSync } from "node:fs";
writeFileSync(${JSON.stringify(pidFileName)}, String(process.pid));
setInterval(() => {}, 1000);
`,
	);
	return "node hang.mjs";
}

function readChildPid(pidPath: string): number {
	const childPid = Number(readFileSync(pidPath, "utf8").trim());
	if (!Number.isInteger(childPid) || childPid <= 0) {
		throw new Error(`Invalid child pid in ${basename(pidPath)}`);
	}
	return childPid;
}

async function awaitToolResult<T>(execution: Promise<T>, label: string): Promise<T> {
	return await Promise.race([
		execution,
		new Promise<never>((_, reject) =>
			setTimeout(() => reject(new Error(`${label} did not resolve promptly`)), RESOLVE_WAIT_MS),
		),
	]);
}

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-bash-tool-"));
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
});

describe("createBashTool", () => {
	it("exposes the bash tool name", () => {
		const tool = createBashTool({ cwd: tmpDir });
		expect(tool.name).toBe(BASH_TOOL_NAME);
		expect(BASH_TOOL_NAME).toBe("bash");
	});

	it("runs a command and returns stdout, including real paths", async () => {
		writeFileSync(join(tmpDir, "notes.txt"), "hello autorag\n");
		const tool = createBashTool({ cwd: tmpDir });

		const result = await tool.execute("call-1", { command: "cat notes.txt" });

		const text = result.content.map((part) => (part.type === "text" ? part.text : "")).join("");
		expect(text).toContain("hello autorag");
		expect(result.details.exitCode).toBe(0);
	});

	it("runs in the configured cwd", async () => {
		writeFileSync(join(tmpDir, "marker.txt"), "here\n");
		const tool = createBashTool({ cwd: tmpDir });

		const result = await tool.execute("call-2", { command: "ls" });
		const text = result.content.map((part) => (part.type === "text" ? part.text : "")).join("");
		expect(text).toContain("marker.txt");
	});

	it("refuses content access under known cloud-provider roots", async () => {
		const tool = createBashTool({ cwd: String.raw`G:\Google Drive\My Drive` });

		const result = await tool.execute("call-cloud-blocked", { command: "rg renewal ." });

		expect(result.details.exitCode).toBeUndefined();
		expect(result.content[0]).toMatchObject({
			type: "text",
			text: expect.stringContaining("AUTORAG_CLOUD_PLACEHOLDER_BLOCKED"),
		});
	});

	it("reports a non-zero exit code without throwing", async () => {
		const tool = createBashTool({ cwd: tmpDir });
		const result = await tool.execute("call-3", { command: "exit 3" });
		expect(result.details.exitCode).toBe(3);
	});

	it("returns a message for an empty command without spawning", async () => {
		const tool = createBashTool({ cwd: tmpDir });
		const result = await tool.execute("call-4", { command: "   " });
		const text = result.content.map((part) => (part.type === "text" ? part.text : "")).join("");
		expect(text.toLowerCase()).toContain("empty");
		expect(result.details.exitCode).toBeUndefined();
	});

	it("truncates output beyond the byte cap", async () => {
		const tool = createBashTool({ cwd: tmpDir, maxOutputBytes: 64 });
		const result = await tool.execute("call-5", { command: "yes autorag | head -n 1000" });
		const text = result.content.map((part) => (part.type === "text" ? part.text : "")).join("");
		expect(result.details.truncated).toBe(true);
		expect(text).toContain("truncated");
	});

	it("times out a long-running command", async () => {
		const tool = createBashTool({ cwd: tmpDir, timeoutMs: 200 });
		const result = await tool.execute("call-6", { command: "sleep 5" });
		expect(result.details.timedOut).toBe(true);
	});

	it("does not run a command when the signal is already aborted", async () => {
		const controller = new AbortController();
		controller.abort();
		const markerPath = join(tmpDir, "already-aborted-ran");
		const tool = createBashTool({ cwd: tmpDir, timeoutMs: 1_000 });

		const result = await tool.execute(
			"call-already-aborted",
			{ command: "printf ran > already-aborted-ran" },
			controller.signal,
		);

		expect(existsSync(markerPath)).toBe(false);
		expect(result.content[0]).toMatchObject({
			type: "text",
			text: expect.stringContaining("(command aborted)"),
		});
		expect(result.details.timedOut).toBe(false);
	});

	it("kills a process tree and resolves on process tree timeout", async () => {
		const pidPath = join(tmpDir, "timeout-child.pid");
		const tool = createBashTool({ cwd: tmpDir, timeoutMs: TREE_TIMEOUT_MS });
		const execution = tool.execute("call-tree-timeout", {
			command: hangingChildCommand("timeout-child.pid"),
		});

		await waitForFile(pidPath);
		const childPid = readChildPid(pidPath);
		const result = await awaitToolResult(execution, "bash tool");

		expect(result.details.timedOut).toBe(true);
		expect(result.details.terminationFailed).toBe(false);
		expect(result.details.aborted).toBe(false);
		expect(result.content[0]).toMatchObject({
			type: "text",
			text: expect.stringContaining(`(command timed out after ${TREE_TIMEOUT_MS}ms and was killed)`),
		});
		await waitForProcessDeath(childPid);
	}, 20_000);

	it("kills a process tree when aborted", async () => {
		const pidPath = join(tmpDir, "abort-child.pid");
		const controller = new AbortController();
		const tool = createBashTool({ cwd: tmpDir, timeoutMs: 30_000 });
		const execution = tool.execute(
			"call-tree-abort",
			{ command: hangingChildCommand("abort-child.pid") },
			controller.signal,
		);

		await waitForFile(pidPath);
		const childPid = readChildPid(pidPath);
		controller.abort();
		const result = await awaitToolResult(execution, "aborted bash tool");

		expect(result.details.timedOut).toBe(false);
		expect(result.details.aborted).toBe(true);
		expect(result.details.terminationFailed).toBe(false);
		await waitForProcessDeath(childPid);
	}, 20_000);

	it("never blocks datasource CLI binaries: the agent may drive katok/discrawl directly", async () => {
		const tool = createBashTool({ cwd: tmpDir });
		const result = await tool.execute("call-direct", { command: "echo katok search keyword x" });
		const text = result.content.map((part) => (part.type === "text" ? part.text : "")).join("");
		expect(text).toContain("katok search keyword x");
		expect(text).not.toContain("AUTORAG_MANAGED_CLI_BLOCKED");
		expect(result.details.exitCode).toBe(0);
	});
});

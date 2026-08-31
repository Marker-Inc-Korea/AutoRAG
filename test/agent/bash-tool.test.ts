import { existsSync, mkdtempSync, readFileSync, rmSync, unwatchFile, watchFile, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { BASH_TOOL_NAME, createBashTool, isManagedCliDirectInvocation } from "../../src/agent/bash-tool.ts";
import { type ManagedCliConfigProvider, ManagedCliRegistry } from "../../src/cli/managed-cli-config.ts";

let tmpDir: string;

async function waitForFile(path: string): Promise<void> {
	if (existsSync(path)) return;
	await new Promise<void>((resolve, reject) => {
		const timeout = setTimeout(() => {
			unwatchFile(path, onChange);
			reject(new Error(`Timed out waiting for ${basename(path)}`));
		}, 2_000);
		const onChange = () => {
			if (!existsSync(path)) return;
			clearTimeout(timeout);
			unwatchFile(path, onChange);
			resolve();
		};
		watchFile(path, { interval: 25 }, onChange);
	});
}

function isProcessAlive(pid: number): boolean {
	try {
		process.kill(pid, 0);
		return true;
	} catch {
		return false;
	}
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

	it("kills a process tree and resolves on process tree timeout", async () => {
		const pidPath = join(tmpDir, "timeout-child.pid");
		const tool = createBashTool({ cwd: tmpDir, timeoutMs: 100 });
		const execution = tool.execute("call-tree-timeout", {
			command: "sleep 10 & echo $! > timeout-child.pid; wait",
		});

		await waitForFile(pidPath);
		const childPid = Number(readFileSync(pidPath, "utf8"));
		const result = await Promise.race([
			execution,
			new Promise<never>((_, reject) =>
				setTimeout(() => reject(new Error("bash tool did not resolve promptly")), 1_000),
			),
		]);

		expect(result.details.timedOut).toBe(true);
		expect(result.content[0]).toMatchObject({
			type: "text",
			text: expect.stringContaining("(command timed out after 100ms and was killed)"),
		});
		expect(isProcessAlive(childPid)).toBe(false);
	});

	it("kills a process tree when aborted", async () => {
		const pidPath = join(tmpDir, "abort-child.pid");
		const controller = new AbortController();
		const tool = createBashTool({ cwd: tmpDir, timeoutMs: 10_000 });
		const execution = tool.execute(
			"call-tree-abort",
			{ command: "sleep 10 & echo $! > abort-child.pid; wait" },
			controller.signal,
		);

		await waitForFile(pidPath);
		const childPid = Number(readFileSync(pidPath, "utf8"));
		controller.abort();
		const result = await Promise.race([
			execution,
			new Promise<never>((_, reject) =>
				setTimeout(() => reject(new Error("aborted bash tool did not resolve promptly")), 1_000),
			),
		]);

		expect(result.details.timedOut).toBe(false);
		expect(isProcessAlive(childPid)).toBe(false);
	});

	it("blocks registered datasource CLIs through direct, absolute, env, chain, pipeline, subshell, and quoted forms", () => {
		const registry = new ManagedCliRegistry();
		const fixture: ManagedCliConfigProvider = {
			tool: "discrawl",
			aliases: ["discord-crawl"],
			binaryPaths: ["/opt/bin/discrawl"],
			materialize: async () => {
				throw new Error("unused");
			},
			inspect: async () => {
				throw new Error("unused");
			},
		};
		registry.register(fixture);
		for (const command of [
			"discrawl search hello",
			"/opt/bin/discrawl search hello",
			"env HOME=/tmp discrawl search hello",
			"printf x; discrawl search hello",
			"cat input | discrawl search hello",
			"(discrawl search hello)",
			"'discrawl' search hello",
			"env 'HOME=/tmp' '/opt/bin/discrawl' search hello",
		]) {
			expect(isManagedCliDirectInvocation(command, registry)).toBe(true);
		}
	});

	it("does not block unrelated commands or false-positive names", () => {
		const registry = new ManagedCliRegistry();
		registry.register({
			tool: "discrawl",
			materialize: async () => {
				throw new Error("unused");
			},
			inspect: async () => {
				throw new Error("unused");
			},
		});
		expect(isManagedCliDirectInvocation("echo discrawlx", registry)).toBe(false);
		expect(isManagedCliDirectInvocation("find . -name discrawl", registry)).toBe(false);
		expect(isManagedCliDirectInvocation("cat notes.txt", registry)).toBe(false);
	});

	it("reports a stable remediation when a managed CLI is blocked before spawning", async () => {
		const registry = new ManagedCliRegistry();
		registry.register({
			tool: "discrawl",
			materialize: async () => {
				throw new Error("unused");
			},
			inspect: async () => {
				throw new Error("unused");
			},
		});
		const tool = createBashTool({ cwd: tmpDir, managedCliRegistry: registry });
		const result = await tool.execute("call-blocked", { command: "discrawl --help" });
		expect(result.details.exitCode).toBeUndefined();
		expect(result.content[0]).toMatchObject({
			type: "text",
			text: expect.stringContaining("AUTORAG_MANAGED_CLI_BLOCKED"),
		});
	});
});

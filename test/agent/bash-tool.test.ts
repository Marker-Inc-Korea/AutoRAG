import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { BASH_TOOL_NAME, createBashTool, isManagedCliDirectInvocation } from "../../src/agent/bash-tool.ts";
import { type ManagedCliConfigProvider, ManagedCliRegistry } from "../../src/cli/managed-cli-config.ts";

let tmpDir: string;

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

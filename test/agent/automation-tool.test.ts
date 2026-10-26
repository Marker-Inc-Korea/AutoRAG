import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createManageAutomationsTool, MANAGE_AUTOMATIONS_TOOL_NAME } from "../../src/agent/automation-tool.ts";

let root: string;
let home: string;
let userHome: string;
let inbox: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-automation-tool-"));
	userHome = join(root, "user");
	home = join(root, "autorag-home");
	inbox = join(root, "inbox");
	mkdirSync(userHome, { recursive: true });
	mkdirSync(home, { recursive: true });
	mkdirSync(inbox, { recursive: true });
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function tool(
	exec?: (cmd: string, args: readonly string[]) => Promise<{ code: number; stdout: string; stderr: string }>,
) {
	return createManageAutomationsTool({
		home,
		env: { HOME: userHome, PATH: process.env.PATH },
		platform: "darwin",
		waitForRunningMs: 0,
		...(exec ? { exec } : {}),
	});
}

async function call(params: Record<string, unknown>, exec?: Parameters<typeof tool>[0]) {
	const result = await tool(exec).execute("call-1", params as never);
	const text = result.content.map((part) => (part.type === "text" ? part.text : "")).join("\n");
	return { text, details: result.details };
}

function savedFile(): { automations: Array<Record<string, unknown>> } {
	return JSON.parse(readFileSync(join(home, "automations.json"), "utf8"));
}

describe("manage_automations tool", () => {
	it("is named manage_automations", () => {
		expect(tool().name).toBe(MANAGE_AUTOMATIONS_TOOL_NAME);
		expect(MANAGE_AUTOMATIONS_TOOL_NAME).toBe("manage_automations");
	});

	it("save creates the rules file, keeps ~ unexpanded on disk, and upserts by id", async () => {
		await call({
			action: "save",
			id: "pdf",
			automation: { paths: ["~/inbox"], include: ["**/*.pdf"], run: ["node", "-e", "0"] },
		});
		expect(savedFile().automations).toEqual([
			{ id: "pdf", paths: ["~/inbox"], include: ["**/*.pdf"], run: ["node", "-e", "0"] },
		]);

		await call({ action: "save", id: "pdf", automation: { paths: [inbox], events: ["delete"], run: ["true"] } });
		await call({ action: "save", id: "other", automation: { paths: [inbox], run: ["true"] } });
		expect(savedFile().automations.map((a) => [a.id, a.events ?? null])).toEqual([
			["pdf", ["delete"]],
			["other", null],
		]);
	});

	it("save rejects an invalid rule and leaves the existing file untouched", async () => {
		await call({ action: "save", id: "ok", automation: { paths: [inbox], run: ["true"] } });
		const before = readFileSync(join(home, "automations.json"), "utf8");
		await expect(
			call({ action: "save", id: "bad", automation: { paths: ["relative/dir"], run: ["true"] } }),
		).rejects.toThrow(/absolute/);
		await expect(call({ action: "save", id: "bad", automation: { paths: [inbox], run: [] } })).rejects.toThrow(/run/);
		expect(readFileSync(join(home, "automations.json"), "utf8")).toBe(before);
	});

	it("delete removes one rule and errors on an unknown id", async () => {
		await call({ action: "save", id: "a", automation: { paths: [inbox], run: ["true"] } });
		await call({ action: "save", id: "b", automation: { paths: [inbox], run: ["true"] } });
		await call({ action: "delete", id: "a" });
		expect(savedFile().automations.map((a) => a.id)).toEqual(["b"]);
		await expect(call({ action: "delete", id: "missing" })).rejects.toThrow(/missing/);
	});

	it("list reports the service state, each rule, and its last run", async () => {
		await call({
			action: "save",
			id: "pdf",
			automation: { paths: [inbox], include: ["**/*.pdf"], run: ["node", "-e", "0"] },
		});
		await call({ action: "run", id: "pdf" });
		const { text } = await call({ action: "list" });
		expect(text).toContain("installed=false");
		expect(text).toContain("running=false");
		expect(text).toContain("pdf");
		expect(text).toContain("**/*.pdf");
		expect(text).toMatch(/last run: .*exit 0/);
	});

	it("run pipes only the matching events to the command and reports its exit and output", async () => {
		const script = join(root, "echo.mjs");
		writeFileSync(
			script,
			"let s='';for await(const c of process.stdin)s+=c;console.log('got',JSON.parse(s).length);process.exit(3);",
		);
		await call({
			action: "save",
			id: "pdf",
			automation: { paths: [inbox], include: ["**/*.pdf"], run: ["node", script] },
		});
		const { text } = await call({
			action: "run",
			id: "pdf",
			events: [
				{ type: "create", path: join(inbox, "a.pdf") },
				{ type: "create", path: join(inbox, "b.txt") },
			],
		});
		expect(text).toContain("exit 3");
		expect(text).toContain("got 1");
	});

	it("install_service and uninstall_service manage the OS login service", async () => {
		const calls: string[] = [];
		const exec = async (cmd: string, args: readonly string[]) => {
			calls.push(`${cmd} ${args[0]}`);
			return { code: 0, stdout: "", stderr: "" };
		};
		const installed = await call({ action: "install_service" }, exec);
		expect(installed.text).toContain("com.autorag.automate");
		const plist = join(userHome, "Library", "LaunchAgents");
		expect(existsSync(plist)).toBe(true);
		await call({ action: "uninstall_service" }, exec);
		expect(calls).toEqual(["launchctl bootout", "launchctl bootstrap", "launchctl bootout"]);
	});

	it("requires id for save, delete, and run", async () => {
		await expect(call({ action: "save", automation: { paths: [inbox], run: ["true"] } })).rejects.toThrow(/id/);
		await expect(call({ action: "delete" })).rejects.toThrow(/id/);
		await expect(call({ action: "run" })).rejects.toThrow(/id/);
	});
});

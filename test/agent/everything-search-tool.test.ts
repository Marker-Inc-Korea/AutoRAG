import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { createEverythingSearchTool, EVERYTHING_SEARCH_TOOL_NAME } from "../../src/agent/everything-search-tool.ts";
import { buildSystemPrompt } from "../../src/agent/system-prompt.ts";
import type { EverythingRunner, EverythingSearchRequest } from "../../src/everything/index.ts";

const FIXTURE_DIR = "test/fixtures/sample-project";
let tmpDir: string;

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-everything-agent-"));
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true, maxRetries: 20, retryDelay: 100 });
});

describe("everything_search tool", () => {
	it("forwards typed filters and returns paths as sources", async () => {
		const requests: EverythingSearchRequest[] = [];
		const tool = createEverythingSearchTool({
			async searchEverything(request) {
				requests.push(request);
				return {
					ok: true,
					results: [
						{ path: "C:\\docs\\하위", type: "folder", size: undefined, dateModified: "2026-10-01T17:09:33" },
						{ path: "C:\\docs\\하위\\환불 정책.txt", type: "file", size: 3, dateModified: "2026-10-01T17:09:33" },
					],
				};
			},
		});
		const result = await tool.execute("call-1", { query: "환불 ext:txt", kind: "files", maxResults: 20 });
		expect(requests).toEqual([{ query: "환불 ext:txt", kind: "files", maxResults: 20 }]);
		expect(result.details).toMatchObject({
			method: EVERYTHING_SEARCH_TOOL_NAME,
			resultCount: 2,
			sources: ["C:\\docs\\하위", "C:\\docs\\하위\\환불 정책.txt"],
		});
		const text = result.content[0]?.type === "text" ? result.content[0].text : "";
		expect(text).toContain("[1] folder C:\\docs\\하위");
		expect(text).toContain("[2] file C:\\docs\\하위\\환불 정책.txt size=3 modified=2026-10-01T17:09:33");
	});

	it("caps maxResults and reports Everything failures verbatim", async () => {
		const requests: EverythingSearchRequest[] = [];
		const tool = createEverythingSearchTool({
			async searchEverything(request) {
				requests.push(request);
				return { ok: false, reason: "search-failed", message: "es.exe exit 7: Error 7: IPC reply invalid." };
			},
		});
		const result = await tool.execute("call-2", { query: "x", maxResults: 100_000 });
		expect(requests[0]?.maxResults).toBe(1000);
		const text = result.content[0]?.type === "text" ? result.content[0].text : "";
		expect(text).toContain("search-failed");
		expect(text).toContain("es.exe exit 7: Error 7: IPC reply invalid.");
		expect(result.details).toMatchObject({ resultCount: 0, sources: [] });
	});

	it("rejects an empty query without calling Everything", async () => {
		let called = false;
		const tool = createEverythingSearchTool({
			async searchEverything() {
				called = true;
				return { ok: true, results: [] };
			},
		});
		const result = await tool.execute("call-3", { query: "   " });
		expect(called).toBe(false);
		expect(result.details).toMatchObject({ resultCount: 0 });
	});
});

describe("system prompt Everything guidance", () => {
	it("documents Everything only when the tool is available", () => {
		const withTool = buildSystemPrompt({
			toolNames: ["bash", EVERYTHING_SEARCH_TOOL_NAME],
			manifests: [],
		});
		expect(withTool).toContain(`- **${EVERYTHING_SEARCH_TOOL_NAME}**:`);
		expect(withTool).toContain("## Windows Everything File-Name Search");
		expect(withTool).toContain("ext:");
		expect(withTool).not.toContain(`- **${EVERYTHING_SEARCH_TOOL_NAME}**: caller-provided tool`);

		const withoutTool = buildSystemPrompt({ toolNames: ["bash"], manifests: [] });
		expect(withoutTool).not.toContain("Everything");
	});

	it("routes file discovery to everything_search instead of a jikji_find tool that is not registered", () => {
		const prompt = buildSystemPrompt({
			toolNames: ["bash", EVERYTHING_SEARCH_TOOL_NAME, "search_all_documents"],
			manifests: [],
		});
		expect(prompt).not.toContain("jikji_find");
		expect(prompt).toContain("use `everything_search` actively to locate relevant files and folders");
	});

	it("names both discovery tools when Jikji and Everything are registered", () => {
		const prompt = buildSystemPrompt({
			toolNames: ["bash", "jikji_find", EVERYTHING_SEARCH_TOOL_NAME],
			manifests: [],
			jikjiIndexingEnabled: true,
		});
		expect(prompt).toMatch(/`jikji_find`[^\n]*`everything_search`|`everything_search`[^\n]*`jikji_find`/);
	});

	it("does not steer a jikji-less agent toward jikji_find in the search prompt", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			workspacePath: tmpDir,
			memoryPath: join(tmpDir, "memory.json"),
			minSync: { autoInstall: false },
			jikji: false,
			everything: false,
			fsearch: false,
		});
		expect(agent.getSystemPrompt()).not.toContain("jikji_find");
		expect(agent.buildSearchPrompt("q", {})).not.toContain("jikji_find");
	});
});

interface AgentInternals {
	innerAgent: { state: { tools: Array<{ name: string; execute: (...args: unknown[]) => Promise<unknown> }> } };
}

function toolNames(agent: AutoRAGAgent): string[] {
	return (agent as unknown as AgentInternals).innerAgent.state.tools.map((tool) => tool.name);
}

describe("AutoRAGAgent Everything wiring", () => {
	const fakeRunner: EverythingRunner = async (_command, args) => {
		if (args.includes("-get-everything-version")) return { code: 0, stdout: "1.4.1.1032\r\n", stderr: "" };
		if (args.includes("-get-result-count")) return { code: 0, stdout: "7\r\n", stderr: "" };
		if (args.includes("-json")) {
			return { code: 0, stdout: '[{"filename":"C:\\\\docs\\\\a.txt","size":1}]', stderr: "" };
		}
		return { code: 0, stdout: "", stderr: "" };
	};

	function windowsAgent() {
		return new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			workspacePath: tmpDir,
			memoryPath: join(tmpDir, "memory.json"),
			minSync: { autoInstall: false },
			jikji: false,
			webSearch: false,
			everything: {
				platform: "win32",
				resolveBinaries: async () => ({
					ok: true,
					everythingPath: "C:\\c\\everything.exe",
					esPath: "C:\\c\\es.exe",
					source: "cached",
				}),
				run: fakeRunner,
				launch: () => {},
			},
		});
	}

	it("registers everything_search and its prompt section on Windows", () => {
		const agent = windowsAgent();
		expect(toolNames(agent)).toContain(EVERYTHING_SEARCH_TOOL_NAME);
		expect(agent.getSystemPrompt()).toContain("## Windows Everything File-Name Search");
	});

	it("omits Everything on non-Windows hosts and when disabled", () => {
		const mac = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			workspacePath: tmpDir,
			memoryPath: join(tmpDir, "memory.json"),
			minSync: { autoInstall: false },
			jikji: false,
			everything: { platform: "darwin" },
			fsearch: false,
		});
		expect(toolNames(mac)).not.toContain(EVERYTHING_SEARCH_TOOL_NAME);
		expect(mac.getSystemPrompt()).not.toContain("Everything");

		const disabled = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			workspacePath: tmpDir,
			memoryPath: join(tmpDir, "memory.json"),
			minSync: { autoInstall: false },
			jikji: false,
			everything: false,
			fsearch: false,
		});
		expect(toolNames(disabled)).not.toContain(EVERYTHING_SEARCH_TOOL_NAME);
	});

	it("indexes Everything during refresh and reports its component status", async () => {
		const agent = windowsAgent();
		const result = await agent.refresh(false, { methods: ["everything"] });
		expect(result.everything).toEqual({ ok: true, indexedItems: 7 });
		expect(agent.refreshComponentStatus().everything).toBe("ready");
	});

	it("exits the workspace Everything instance via stopEverything", async () => {
		const calls: string[][] = [];
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			workspacePath: tmpDir,
			memoryPath: join(tmpDir, "memory.json"),
			minSync: { autoInstall: false },
			jikji: false,
			webSearch: false,
			everything: {
				platform: "win32",
				resolveBinaries: async () => ({
					ok: true,
					everythingPath: "C:\\c\\everything.exe",
					esPath: "C:\\c\\es.exe",
					source: "cached",
				}),
				run: async (_command, args) => {
					calls.push([...args]);
					return { code: 0, stdout: "1.4.1.1032\r\n", stderr: "" };
				},
				launch: () => {},
			},
		});
		await agent.stopEverything();
		expect(calls).toHaveLength(1);
		expect(calls[0]).toEqual(["-instance", expect.stringMatching(/^autorag-[0-9a-f]{12}$/), "-exit"]);
	});

	it("surfaces an Everything indexing failure as a refresh error diagnostic", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			workspacePath: tmpDir,
			memoryPath: join(tmpDir, "memory.json"),
			minSync: { autoInstall: false },
			jikji: false,
			everything: {
				platform: "win32",
				resolveBinaries: async () => ({
					ok: false,
					reason: "install-failed",
					message: "es.exe SHA-256 abc does not match pinned def",
				}),
			},
		});
		const result = await agent.refresh(false, { methods: ["everything"] });
		expect(result.everything).toMatchObject({ ok: false, reason: "es.exe SHA-256 abc does not match pinned def" });
		expect(result.diagnostics).toContainEqual(
			expect.objectContaining({
				code: "everything-index-failed",
				severity: "error",
				message: expect.stringContaining("es.exe SHA-256 abc does not match pinned def"),
			}),
		);
		expect(agent.refreshComponentStatus().everything).toBe("degraded");
	});
});

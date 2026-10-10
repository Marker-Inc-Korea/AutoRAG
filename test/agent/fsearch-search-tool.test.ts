import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { createFSearchSearchTool, FSEARCH_SEARCH_TOOL_NAME } from "../../src/agent/fsearch-search-tool.ts";
import { buildSystemPrompt } from "../../src/agent/system-prompt.ts";
import type { FSearchRunner, FSearchSearchRequest } from "../../src/fsearch/index.ts";

const FIXTURE_DIR = "test/fixtures/sample-project";
let tmpDir: string;

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-fsearch-agent-"));
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true, maxRetries: 20, retryDelay: 100 });
});

describe("fsearch_search tool", () => {
	it("forwards typed filters and returns paths as sources", async () => {
		const requests: FSearchSearchRequest[] = [];
		const tool = createFSearchSearchTool({
			async searchFsearch(request) {
				requests.push(request);
				return {
					ok: true,
					backend: "fsearch-cli",
					total: 2,
					results: [
						{ path: "/docs/sub", name: "sub", type: "folder", size: undefined, dateModified: undefined },
						{
							path: "/docs/환불 정책.txt",
							name: "환불 정책.txt",
							type: "file",
							size: 3,
							dateModified: "2026-10-01T17:09:33.000Z",
						},
					],
				};
			},
		});
		const result = await tool.execute("call-1", { query: "환불 ext:txt", kind: "files", maxResults: 20 });
		expect(requests).toEqual([{ query: "환불 ext:txt", kind: "files", maxResults: 20 }]);
		expect(result.details).toMatchObject({
			method: FSEARCH_SEARCH_TOOL_NAME,
			resultCount: 2,
			sources: ["/docs/sub", "/docs/환불 정책.txt"],
		});
		const text = result.content[0]?.type === "text" ? result.content[0].text : "";
		expect(text).toContain("[1] folder /docs/sub");
		expect(text).toContain("[2] file /docs/환불 정책.txt size=3 modified=2026-10-01T17:09:33.000Z");
	});

	it("caps maxResults and reports fsearch failures verbatim", async () => {
		const requests: FSearchSearchRequest[] = [];
		const tool = createFSearchSearchTool({
			async searchFsearch(request) {
				requests.push(request);
				return {
					ok: false,
					reason: "search-failed",
					message: "fsearch-cli exit 2: fsearch-cli: database is corrupt",
				};
			},
		});
		const result = await tool.execute("call-2", { query: "x", maxResults: 100_000 });
		expect(requests[0]?.maxResults).toBe(1000);
		const text = result.content[0]?.type === "text" ? result.content[0].text : "";
		expect(text).toContain("search-failed");
		expect(text).toContain("fsearch-cli: database is corrupt");
		expect(result.details).toMatchObject({ resultCount: 0, sources: [] });
	});

	it("marks slow-walk fallback results so the model knows why it was slow", async () => {
		const tool = createFSearchSearchTool({
			async searchFsearch() {
				return {
					ok: true,
					backend: "walk",
					note: "fsearch-cli is not installed; used a slow filesystem walk",
					results: [{ path: "/docs/a.txt", name: "a.txt", type: "file", size: 1, dateModified: undefined }],
				};
			},
		});
		const result = await tool.execute("call-3", { query: "a" });
		const text = result.content[0]?.type === "text" ? result.content[0].text : "";
		expect(text).toContain("slow filesystem walk");
		expect(text).toContain("[1] file /docs/a.txt");
	});

	it("rejects an empty query without calling fsearch", async () => {
		let called = false;
		const tool = createFSearchSearchTool({
			async searchFsearch() {
				called = true;
				return { ok: true, backend: "fsearch-cli", results: [] };
			},
		});
		const result = await tool.execute("call-4", { query: "   " });
		expect(called).toBe(false);
		expect(result.details).toMatchObject({ resultCount: 0 });
	});
});

describe("system prompt FSearch guidance", () => {
	it("documents FSearch only when the tool is available", () => {
		const withTool = buildSystemPrompt({
			toolNames: ["bash", FSEARCH_SEARCH_TOOL_NAME, "emit_autorag_results"],
			manifests: [],
		});
		expect(withTool).toContain(`- **${FSEARCH_SEARCH_TOOL_NAME}**:`);
		expect(withTool).toContain("## FSearch File-Name Search");
		expect(withTool).toContain("ext:");
		expect(withTool).not.toContain(`- **${FSEARCH_SEARCH_TOOL_NAME}**: caller-provided tool`);

		const withoutTool = buildSystemPrompt({ toolNames: ["bash", "emit_autorag_results"], manifests: [] });
		expect(withoutTool).not.toContain("FSearch");
	});

	it("routes file discovery to fsearch_search when it is the only discovery tool", () => {
		const prompt = buildSystemPrompt({
			toolNames: ["bash", FSEARCH_SEARCH_TOOL_NAME, "search_all_documents", "emit_autorag_results"],
			manifests: [],
		});
		expect(prompt).toContain("use `fsearch_search` actively to locate relevant files and folders");
	});

	it("names fsearch_search alongside the other registered discovery tools", () => {
		const prompt = buildSystemPrompt({
			toolNames: ["bash", "jikji_find", FSEARCH_SEARCH_TOOL_NAME, "emit_autorag_results"],
			manifests: [],
			jikjiIndexingEnabled: true,
		});
		expect(prompt).toMatch(/`jikji_find`[^\n]*`fsearch_search`|`fsearch_search`[^\n]*`jikji_find`/);
	});
});

interface AgentInternals {
	innerAgent: { state: { tools: Array<{ name: string; execute: (...args: unknown[]) => Promise<unknown> }> } };
}

function toolNames(agent: AutoRAGAgent): string[] {
	return (agent as unknown as AgentInternals).innerAgent.state.tools.map((tool) => tool.name);
}

describe("AutoRAGAgent FSearch wiring", () => {
	function macAgent(overrides: Record<string, unknown> = {}) {
		return new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			workspacePath: tmpDir,
			memoryPath: join(tmpDir, "memory.json"),
			minSync: { autoInstall: false },
			jikji: false,
			everything: false,
			webSearch: false,
			fsearch: {
				platform: "darwin",
				run: (async (_command, args) => {
					if (args[0] === "--version") return { code: 0, stdout: "fsearch-cli 0.3\n", stderr: "" };
					if (args[0] === "stats") return { code: 0, stdout: '{"live":true,"files":5,"folders":2}', stderr: "" };
					return { code: 0, stdout: "", stderr: "" };
				}) satisfies FSearchRunner,
				launch: () => 5150,
				isProcessAlive: () => false,
				startupTimeoutMs: 50,
				pollIntervalMs: 1,
				...overrides,
			},
		});
	}

	it("registers fsearch_search and its prompt section on macOS/Linux", () => {
		const agent = macAgent();
		expect(toolNames(agent)).toContain(FSEARCH_SEARCH_TOOL_NAME);
		expect(agent.getSystemPrompt()).toContain("## FSearch File-Name Search");
	});

	it("omits fsearch_search on Windows, when disabled, and for remote peers", () => {
		const windows = macAgent({ platform: "win32" });
		expect(toolNames(windows)).not.toContain(FSEARCH_SEARCH_TOOL_NAME);

		const disabled = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			workspacePath: tmpDir,
			memoryPath: join(tmpDir, "memory.json"),
			minSync: { autoInstall: false },
			jikji: false,
			fsearch: false,
		});
		expect(toolNames(disabled)).not.toContain(FSEARCH_SEARCH_TOOL_NAME);

		const remote = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			workspacePath: tmpDir,
			memoryPath: join(tmpDir, "memory.json"),
			minSync: { autoInstall: false },
			jikji: false,
			webSearch: false,
			remoteSession: true,
			fsearch: { platform: "darwin" },
		});
		expect(toolNames(remote)).not.toContain(FSEARCH_SEARCH_TOOL_NAME);
	});

	it("indexes FSearch during refresh and reports its component status", async () => {
		const agent = macAgent();
		const result = await agent.refresh(false, { methods: ["fsearch"] });
		expect(result.fsearch).toEqual({ ok: true, indexedItems: 7 });
		expect(agent.refreshComponentStatus().fsearch).toBe("ready");
	});

	it("marks the component unavailable when fsearch-cli is not installed, with a warning diagnostic", async () => {
		const agent = macAgent({
			run: async () => ({ code: null, stdout: "", stderr: "fsearch-cli: spawn fsearch-cli ENOENT" }),
		});
		const result = await agent.refresh(false, { methods: ["fsearch"] });
		expect(result.fsearch).toMatchObject({ ok: false, reason: "binary-missing" });
		expect(result.diagnostics).toContainEqual(
			expect.objectContaining({
				code: "fsearch-binary-missing",
				severity: "warning",
				message: expect.stringContaining("spawn fsearch-cli ENOENT"),
			}),
		);
		expect(agent.refreshComponentStatus().fsearch).toBe("unavailable");
	});

	it("surfaces an FSearch indexing failure as a refresh error diagnostic", async () => {
		const agent = macAgent({
			run: (async (_command, args) => {
				if (args[0] === "--version") return { code: 0, stdout: "fsearch-cli 0.3\n", stderr: "" };
				if (args[0] === "index") return { code: 1, stdout: "", stderr: "fsearch-cli: permission denied" };
				return { code: 0, stdout: "", stderr: "" };
			}) satisfies FSearchRunner,
		});
		const result = await agent.refresh(false, { methods: ["fsearch"] });
		expect(result.fsearch).toMatchObject({ ok: false, reason: "index-failed" });
		expect(result.diagnostics).toContainEqual(
			expect.objectContaining({
				code: "fsearch-index-failed",
				severity: "error",
				message: expect.stringContaining("permission denied"),
			}),
		);
		expect(agent.refreshComponentStatus().fsearch).toBe("degraded");
	});

	it("serves searchFsearch through the client and stops the watch daemon via stopFsearch", async () => {
		const killed: number[] = [];
		let watchUp = false;
		const agent = macAgent({
			run: (async (_command, args) => {
				if (args[0] === "--version") return { code: 0, stdout: "fsearch-cli 0.3\n", stderr: "" };
				if (args[0] === "stats") {
					return {
						code: 0,
						stdout: watchUp ? '{"live":true,"files":0,"folders":0}' : '{"live":false,"files":0,"folders":0}',
						stderr: "",
					};
				}
				if (args[0] === "search") {
					return {
						code: 0,
						stdout:
							'{"path":"/docs/a.txt","name":"a.txt","type":"file","size":1,"mtime":1790956800}\n{"done":true,"num_results":1,"num_returned":1}\n',
						stderr: "",
					};
				}
				// Like the real CLI, `index --db <path>` writes the database file.
				if (args[0] === "index") writeFileSync(args[args.indexOf("--db") + 1]!, "db");
				return { code: 0, stdout: "", stderr: "" };
			}) satisfies FSearchRunner,
			launch: () => {
				watchUp = true;
				return 5150;
			},
			isProcessAlive: () => true,
			killProcess: (pid: number) => {
				killed.push(pid);
				watchUp = false;
			},
		});
		// Refresh builds the database; a search only reads it.
		await agent.refresh(false, { methods: ["fsearch"] });
		const result = await agent.searchFsearch({ query: "a" });
		expect(result).toMatchObject({ ok: true, backend: "fsearch-cli" });
		await agent.stopFsearch();
		expect(killed).toContain(5150);
	});
});

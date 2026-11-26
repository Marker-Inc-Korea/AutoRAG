import { mkdirSync, mkdtempSync, realpathSync, rmSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent, type AutoRAGAgentOptions } from "../../src/agent/agent.ts";
import { DupeyCliError } from "../../src/dupey/index.ts";
import { loadMirrorIndex } from "../../src/mirror/index.ts";

let workspace: string;
let rootA: string;
let rootB: string;

beforeEach(() => {
	workspace = realpathSync(mkdtempSync(join(tmpdir(), "autorag-dupey-diag-")));
	rootA = join(workspace, "root-a");
	rootB = join(workspace, "root-b");
	mkdirSync(rootA, { recursive: true });
	mkdirSync(rootB, { recursive: true });
});

afterEach(() => {
	rmSync(workspace, { recursive: true, force: true });
});

function writeDuplicatePair(dir: string, content: string): { readonly older: string; readonly newer: string } {
	const older = join(dir, "older.txt");
	const newer = join(dir, "newer.txt");
	writeFileSync(older, content);
	writeFileSync(newer, content);
	utimesSync(older, 1, 1);
	utimesSync(newer, 2, 2);
	return { older, newer };
}

function scanResultForPair(dir: string, older: string, newer: string, hash: string): string {
	return JSON.stringify({
		dir,
		files: [
			{ path: older, content_hash: hash },
			{ path: newer, content_hash: hash },
		],
		families: [
			{
				id: 1,
				relation: "exact",
				files: [older, newer],
				members: [],
				edges: [{ relation: "exact", a: older, b: newer }],
			},
		],
		errors: [],
	});
}

function makeAgent(extra: Omit<AutoRAGAgentOptions, "searchPaths"> = {}): AutoRAGAgent {
	return new AutoRAGAgent({
		searchPaths: [rootA, rootB],
		memoryPath: join(workspace, "memory.json"),
		workspacePath: workspace,
		jikji: false,
		everything: false,
		fsearch: false,
		minSync: { autoInstall: false },
		...extra,
	});
}

describe("duplicate-scan-failed diagnostics", () => {
	it("warns on the failing root with sanitized cause while still excluding duplicates on healthy roots", async () => {
		const pairA = writeDuplicatePair(rootA, "identical-a\n");
		const pairB = writeDuplicatePair(rootB, "identical-b\n");
		const failingCalls: string[][] = [];

		const agent = makeAgent({
			dupey: {
				run: async (args) => {
					failingCalls.push([...args]);
					if (args[0] === "scan" && args[1] === resolve(rootA)) {
						throw new DupeyCliError(`dupey failed with exit code 23: EIO reading ${pairA.older}`);
					}
					if (args[0] === "scan" && args[1] === resolve(rootB)) {
						return scanResultForPair(rootB, pairB.older, pairB.newer, "hash-b");
					}
					throw new Error(`unexpected dupey command: ${args.join(" ")}`);
				},
			},
		});

		const refreshed = await agent.refresh(true, { methods: ["parsed"] });
		const index = loadMirrorIndex(workspace);

		// Failing root indexed both files; healthy root excluded its older duplicate.
		expect(index.entries["/root-a/older.txt"]).toBeDefined();
		expect(index.entries["/root-a/newer.txt"]).toBeDefined();
		expect(index.entries["/root-b/newer.txt"]).toBeDefined();
		expect(index.entries["/root-b/older.txt"]).toBeUndefined();

		const warnings = refreshed.diagnostics.filter((d) => d.code === "duplicate-scan-failed");
		expect(warnings).toHaveLength(1);
		expect(warnings[0]).toMatchObject({
			code: "duplicate-scan-failed",
			severity: "warning",
			source: "/root-a",
			reason: "scan-failed",
		});
		// Real filesystem paths must not leak into the diagnostic message.
		expect(warnings[0]?.message).not.toContain(workspace);
		expect(warnings[0]?.message).toContain("<path>");
		expect(warnings[0]?.message).toContain("exit code 23");

		// A fresh process (new AutoRAGAgent) surfaces the persisted warning in status without calling dupey.
		const statusCalls: string[][] = [];
		const statusAgent = makeAgent({
			dupey: {
				run: async (args) => {
					statusCalls.push([...args]);
					return JSON.stringify({ dir: args[1], files: [], families: [], errors: [] });
				},
			},
		});
		const status = await statusAgent.getRefreshStatus();
		expect(statusCalls).toEqual([]);
		expect(status.diagnostics.filter((d) => d.code === "duplicate-scan-failed")).toEqual([
			{
				code: "duplicate-scan-failed",
				severity: "warning",
				source: "/root-a",
				reason: "scan-failed",
				message: warnings[0]?.message,
			},
		]);
	});

	it("reports reason='timeout' when dupey times out, preserves across datasource-only refresh, and clears on recovery", async () => {
		const pairA = writeDuplicatePair(rootA, "identical-a\n");
		const pairB = writeDuplicatePair(rootB, "identical-b\n");
		let rootATimesOut = true;

		const dupeyRun = async (args: readonly string[]): Promise<string> => {
			if (args[0] === "scan" && args[1] === resolve(rootA)) {
				if (rootATimesOut) throw new DupeyCliError("dupey timed out.", { timedOut: true });
				return scanResultForPair(rootA, pairA.older, pairA.newer, "hash-a");
			}
			if (args[0] === "scan" && args[1] === resolve(rootB)) {
				return scanResultForPair(rootB, pairB.older, pairB.newer, "hash-b");
			}
			throw new Error(`unexpected dupey command: ${args.join(" ")}`);
		};

		const agent = makeAgent({ dupey: { run: dupeyRun } });
		const first = await agent.refresh(true, { methods: ["parsed"] });
		const timeoutWarnings = first.diagnostics.filter((d) => d.code === "duplicate-scan-failed");
		expect(timeoutWarnings).toHaveLength(1);
		expect(timeoutWarnings[0]).toMatchObject({
			code: "duplicate-scan-failed",
			severity: "warning",
			source: "/root-a",
			reason: "timeout",
		});

		// A datasource-only refresh does not run parsed mirrors and keeps the last completed warning.
		await agent.refresh(false, { methods: ["datasources"] });
		const statusAfterDatasource = await makeAgent({ dupey: { run: dupeyRun } }).getRefreshStatus();
		expect(statusAfterDatasource.diagnostics.filter((d) => d.code === "duplicate-scan-failed")).toHaveLength(1);

		// Once root-a recovers, the next completed parsed refresh replaces the marker and clears the warning.
		rootATimesOut = false;
		const recovered = await agent.refresh(false, { methods: ["parsed"] });
		expect(recovered.diagnostics.filter((d) => d.code === "duplicate-scan-failed")).toEqual([]);
		const statusAfterRecovery = await makeAgent({ dupey: { run: dupeyRun } }).getRefreshStatus();
		expect(statusAfterRecovery.diagnostics.filter((d) => d.code === "duplicate-scan-failed")).toEqual([]);
	});
});

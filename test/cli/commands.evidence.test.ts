import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { runEvidence } from "../../src/cli/commands/evidence.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";

let root: string;
let configPath: string;
let memoryPath: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-evidence-"));
	configPath = join(root, "config.json");
	memoryPath = join(root, "memory.json");
	writeFileSync(
		configPath,
		JSON.stringify({ searchPaths: [root], workspacePath: root, memoryPath, minSync: false, jikji: false }),
	);
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function makeCtx(positionals: string[], flags: Record<string, string | boolean> = {}) {
	const stdout: string[] = [];
	const stderr: string[] = [];
	const ctx: CommandContext = {
		positionals,
		flags: { config: configPath, ...flags },
		json: true,
		debug: false,
		cwd: root,
		stdout: (line) => stdout.push(line),
		stderr: (line) => stderr.push(line),
	};
	return { ctx, stdout, stderr };
}

describe("autorag evidence", () => {
	it("returns persisted raw chunks for a session and result", async () => {
		writeFileSync(
			memoryPath,
			JSON.stringify({
				version: 4,
				curatedResults: [
					{
						resultId: "session-1:1",
						sessionId: "session-1",
						number: 1,
						query: "alpha",
						title: "Alpha",
						summary: "summary",
						resultHash: "hash",
						evidenceIds: ["minsync:chunk-1"],
						createdAt: 1,
					},
				],
				evidenceChunks: [
					{
						stableEvidenceId: "minsync:chunk-1",
						method: "minsync",
						source: "/docs/alpha.md",
						content: "raw chunk body",
						excerptHash: "hash",
						firstSeenAt: 1,
						lastSeenAt: 1,
						chunkIndex: 3,
					},
				],
				feedbackSignals: [],
				signalDefaults: { explicitWeight: 1, followupWeight: 0.25, retryWeight: -0.25, implicitCap: 0.5 },
				warnings: [],
				insights: [],
				pendingInsightSignals: [],
			}),
		);
		const { ctx, stdout } = makeCtx(["session-1"], { result: "1" });
		expect(await runEvidence(ctx)).toBe(0);
		const output = JSON.parse(stdout[0]);
		expect(output.results[0].chunks[0]).toMatchObject({
			source: "/docs/alpha.md",
			content: "raw chunk body",
			chunkIndex: 3,
		});
	});

	it("returns usage error for an unknown session", async () => {
		const { ctx, stderr } = makeCtx(["missing"]);
		expect(await runEvidence(ctx)).toBe(2);
		expect(stderr.join("\n")).toContain("No evidence found");
	});
});

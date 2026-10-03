import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { AutoRAGResultsDetails } from "../../src/agent/emit-results-tool.ts";
import { createAutoRAGLite } from "../../src/core.ts";
import { RetrievalMemory } from "../../src/memory/memory.ts";

let tmpDir: string;

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-lite-persisted-feedback-"));
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
});

function writeConfig(): string {
	const configPath = join(tmpDir, "config.json");
	writeFileSync(
		configPath,
		JSON.stringify({
			searchPaths: [tmpDir],
			workspacePath: tmpDir,
			memoryPath: join(tmpDir, "memory.json"),
			minSync: false,
			jikji: false,
			everything: false,
		}),
	);
	return configPath;
}

const report: AutoRAGResultsDetails = {
	answer: "[1] Director approval is required before payout.",
	results: [
		{
			number: 1,
			title: "Refund exception approval",
			summary: "Director approval is required before payout.",
			evidence: [{ excerpt: "Refund exceptions require director approval before payout." }],
			confidence: 0.9,
		},
	],
	mapping: [
		{
			number: 1,
			source: "opaque:src",
			method: "manual",
			content: "Refund exceptions require director approval before payout.",
			evidenceRefs: [],
		},
	],
	warnings: [],
};

describe("AutoRAGLite persisted feedback", () => {
	it("applies numbered feedback against the persisted report registry after a restart", () => {
		const configPath = writeConfig();
		const first = createAutoRAGLite({ flags: { config: configPath } });
		const persisted = first.recordReport("refund policy", report);
		expect(persisted.sessionId).not.toBe("");

		// A fresh facade models a process restart: its in-memory session registry is empty.
		const restarted = createAutoRAGLite({ flags: { config: configPath } });
		expect(restarted.getResultRegistry(persisted.sessionId).size).toBe(0);
		const applied = restarted.recordPersistedFeedbackByNumbers(persisted.sessionId, [1]);
		expect(applied).toBe(true);

		const memory = new RetrievalMemory({ storagePath: join(tmpDir, "memory.json") });
		memory.load();
		expect(memory.getSignalCount()).toBeGreaterThan(0);
	});

	it("returns false for an unknown session", () => {
		const lite = createAutoRAGLite({ flags: { config: writeConfig() } });
		expect(lite.recordPersistedFeedbackByNumbers("never-recorded", [1])).toBe(false);
	});
});

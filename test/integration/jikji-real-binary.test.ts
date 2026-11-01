import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";

/**
 * Exercises the find → discover protocol against the real jikji 0.2.1 binary,
 * the version whose envelopes the retry step was written against. Skipped when
 * that exact version is not on PATH.
 */
const version = spawnSync("jikji", ["--version"], { encoding: "utf8" });
const hasJikji021 = version.status === 0 && version.stdout.trim() === "jikji 0.2.1";

let tmp: string;
let docs: string;

beforeEach(() => {
	tmp = mkdtempSync(join(tmpdir(), "autorag-jikji-real-"));
	docs = join(tmp, "docs");
	mkdirSync(docs, { recursive: true });
	writeFileSync(join(docs, "refund.txt"), "refund policy director approval\n");
	writeFileSync(join(docs, "notes.txt"), "meeting notes\n");
	writeFileSync(join(docs, "AGENTS.md"), "user-owned agent rules\n");
	writeFileSync(join(docs, "CLAUDE.md"), "user-owned claude rules\n");
});

afterEach(() => {
	rmSync(tmp, { recursive: true, force: true, maxRetries: 20, retryDelay: 100 });
});

function agent(): AutoRAGAgent {
	return new AutoRAGAgent({
		searchPaths: [docs],
		memoryPath: join(tmp, "memory.json"),
		workspacePath: tmp,
		minSync: { autoInstall: false },
		jikji: { env: { JIKJI_DATA_DIR: join(tmp, "jikji-data") } },
		everything: false,
		fsearch: false,
	});
}

describe.skipIf(!hasJikji021)("real jikji 0.2.1 retry protocol", () => {
	it("reaches raw_fallback_after_retry from a miss and leaves user rule files untouched", {
		timeout: 120_000,
	}, async () => {
		const subject = agent();
		await subject.prepareJikji();

		const result = await subject.findJikji("zzzqqq unrelatedterm");

		expect(result.diagnostics).toEqual([]);
		expect(result.policy).toMatchObject({
			handoffAction: "raw_fallback_after_retry",
			rawFallbackAllowed: true,
			maxRawFallbackCommands: 2,
		});
		expect(readFileSync(join(docs, "AGENTS.md"), "utf8")).toBe("user-owned agent rules\n");
		expect(readFileSync(join(docs, "CLAUDE.md"), "utf8")).toBe("user-owned claude rules\n");
	});

	it("answers a hit directly without a retry", { timeout: 120_000 }, async () => {
		const subject = agent();
		await subject.prepareJikji();

		const result = await subject.findJikji("refund policy");

		expect(result.policy).toMatchObject({ handoffAction: "direct_use", rawFallbackAllowed: false });
		expect(result.answerPack?.answerPaths).toContain(realpathSync(join(docs, "refund.txt")));
	});
});

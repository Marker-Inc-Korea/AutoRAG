import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { JikjiMethod } from "../../src/retrieval/methods/jikji.ts";

let root: string;
let docs: string;
let notes: string;
let binaryPath: string;
let logPath: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-jikji-method-test-"));
	docs = join(root, "docs");
	notes = join(root, "notes");
	binaryPath = join(root, "fake-jikji.mjs");
	logPath = join(root, "calls.jsonl");
	mkdirSync(docs, { recursive: true });
	mkdirSync(notes, { recursive: true });
	writeFileSync(join(docs, "q3-report.txt"), "Q3 revenue grew from enterprise contracts.\n");
	writeFileSync(join(notes, "memo.txt"), "Enterprise memo.\n");
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function writeFakeJikji(payloads: readonly unknown[], exitCode = 0): void {
	writeFileSync(
		binaryPath,
		`#!/usr/bin/env node
import { appendFileSync } from "node:fs";
const args = process.argv.slice(2);
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args }) + "\\n");
const root = args[1];
const payloads = ${JSON.stringify(payloads)};
const payload = root.includes("notes") && payloads.length > 1 ? payloads[1] : payloads[0];
if (payload === "MALFORMED") process.stdout.write("{not-json");
else console.log(JSON.stringify(payload));
process.exit(${exitCode});
`,
	);
	chmodSync(binaryPath, 0o755);
}

function highPayload(): unknown {
	return {
		query_type: "single_file",
		confidence: "high",
		handoff_action: "direct_use",
		index_status: "ready",
		paths: ["q3-report.txt"],
		answer_paths: ["q3-report.txt"],
		evidence_pack: [
			{
				path: "q3-report.txt",
				why: ["body-coverage"],
				matched_terms: ["enterprise", "revenue"],
				evidence: ["Q3 revenue grew from enterprise contracts"],
				next_read: { kind: "original", path: "q3-report.txt" },
			},
		],
		judge_candidate_slate: [
			{
				path: "q3-report.txt",
				score: 42,
				evidence: ["Judge confirms Q3 revenue report"],
				next_read: { kind: "original", path: "q3-report.txt" },
			},
		],
	};
}

function compactPayload(): unknown {
	return {
		confidence: "low",
		handoff_action: "jikji_retry",
		paths: [],
		answer_paths: [],
		evidence_pack: [],
		judge_candidate_slate: [],
		candidates: [
			{
				p: "memo.txt",
				s: 0.25,
				why: ["filename"],
				terms: ["enterprise"],
				ev: "Enterprise memo fallback evidence",
			},
		],
	};
}

function outsidePayload(): unknown {
	return {
		confidence: "high",
		paths: [
			"../secret.txt",
			"/etc/passwd",
			"C:/Users/secret.txt",
			"file:///tmp/secret.txt",
			"//server/share/secret.txt",
		],
		answer_paths: [
			"../secret.txt",
			"/etc/passwd",
			"C:/Users/secret.txt",
			"file:///tmp/secret.txt",
			"//server/share/secret.txt",
		],
		evidence_pack: [],
		judge_candidate_slate: [],
		candidates: [{ p: "../secret.txt", why: [], terms: [], ev: "secret" }],
	};
}

function readLoggedArgs(): readonly string[][] {
	return readFileSync(logPath, "utf8")
		.trim()
		.split("\n")
		.filter(Boolean)
		.map((line) => (JSON.parse(line) as { args: string[] }).args);
}

describe("JikjiMethod", () => {
	it("describes Jikji as an active hybrid local discovery method", () => {
		const method = new JikjiMethod({ root, searchPaths: [docs], binaryPath });

		expect(method.describe()).toEqual({
			name: "jikji",
			type: "hybrid",
			description: "optional Jikji CLI local file-discovery retrieval over configured source directories",
			status: "active",
			capabilities: [
				"local-file-discovery",
				"cli-json",
				"fielded-search",
				"agent-handoff",
				"opaque-root-relative-paths",
			],
		});
	});

	it("maps Jikji answer paths to opaque retrieval results", async () => {
		writeFakeJikji([highPayload()]);
		const method = new JikjiMethod({ root, searchPaths: [docs], binaryPath });

		const results = await method.retrieve("enterprise revenue", { topK: 5 });

		expect(results).toHaveLength(1);
		expect(results[0]).toMatchObject({
			source: "/docs/q3-report.txt",
			content: "Q3 revenue grew from enterprise contracts",
			score: 42,
			metadata: {
				method: "jikji",
				confidence: "high",
				handoffAction: "direct_use",
				indexStatus: "ready",
				queryType: "single_file",
				why: ["body-coverage"],
				matchedTerms: ["enterprise", "revenue"],
			},
		});
		expect(JSON.stringify(results)).not.toContain(root);
		expect(readLoggedArgs()[0]).toEqual(["find", docs, "enterprise revenue", "--json", "--top-k", "5"]);
	});

	it("maps compact candidates when answer paths are absent", async () => {
		writeFakeJikji([compactPayload()]);
		const method = new JikjiMethod({ root, searchPaths: [notes], binaryPath });

		const results = await method.retrieve("enterprise", {});

		expect(results).toHaveLength(1);
		expect(results[0]).toMatchObject({
			source: "/notes/memo.txt",
			content: "Enterprise memo fallback evidence",
			score: 0.25,
			metadata: { method: "jikji", confidence: "low", handoffAction: "jikji_retry" },
		});
	});

	it("deduplicates duplicate source results within one method result", async () => {
		const payload = highPayload() as Record<string, unknown>;
		payload.paths = ["q3-report.txt"];
		payload.answer_paths = ["q3-report.txt", "q3-report.txt"];
		writeFakeJikji([payload]);
		const method = new JikjiMethod({ root, searchPaths: [docs], binaryPath });

		const results = await method.retrieve("enterprise revenue", {});

		expect(results.map((result) => result.source)).toEqual(["/docs/q3-report.txt"]);
	});

	it("drops paths outside the configured source root", async () => {
		writeFakeJikji([outsidePayload()]);
		const method = new JikjiMethod({ root, searchPaths: [docs], binaryPath });

		const results = await method.retrieve("secret", {});

		expect(results).toEqual([]);
	});

	it("does not expose unsafe next-read metadata paths", async () => {
		writeFakeJikji([
			{
				confidence: "medium",
				paths: [],
				answer_paths: [],
				evidence_pack: [],
				judge_candidate_slate: [],
				candidates: [
					{
						p: "q3-report.txt",
						s: 0.5,
						why: [],
						terms: [],
						ev: "safe evidence",
						next_read: { kind: "original", path: "C:/Users/secret" },
					},
				],
			},
		]);
		const method = new JikjiMethod({ root, searchPaths: [docs], binaryPath });

		const results = await method.retrieve("semantic", {});

		expect(results[0]?.metadata.nextRead).toEqual({ kind: "original" });
		expect(JSON.stringify(results)).not.toContain("C:/Users/secret");
	});

	it("runs configured source roots sequentially", async () => {
		writeFakeJikji([highPayload(), compactPayload()]);
		const method = new JikjiMethod({ root, searchPaths: [docs, notes], binaryPath, topK: 7 });

		const results = await method.retrieve("enterprise", {});

		expect(results.map((result) => result.source).sort()).toEqual(["/docs/q3-report.txt", "/notes/memo.txt"]);
		expect(readLoggedArgs()).toEqual([
			["find", docs, "enterprise", "--json", "--top-k", "7"],
			["find", notes, "enterprise", "--json", "--top-k", "7"],
		]);
	});

	it("returns empty results for expected Jikji failures", async () => {
		writeFakeJikji(["MALFORMED"]);
		const method = new JikjiMethod({ root, searchPaths: [docs], binaryPath });

		await expect(method.retrieve("enterprise", {})).resolves.toEqual([]);
	});
});

import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { AutoRAGResultsDetails } from "../../src/agent/emit-results-tool.ts";
import { runReport } from "../../src/cli/commands/report.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import { normalizeSessionEvidenceRef } from "../../src/memory/memory.ts";

let root: string;
let configPath: string;
let memoryPath: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-report-"));
	configPath = join(root, "config.json");
	memoryPath = join(root, "memory.json");
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function makeCtx(
	positionals: string[],
	flags: Record<string, string | boolean | undefined> = {},
): {
	ctx: CommandContext;
	stdout: string[];
	stderr: string[];
} {
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

function validReport(overrides?: Partial<AutoRAGResultsDetails>): AutoRAGResultsDetails {
	return {
		answer: "[1] test answer",
		results: [
			{
				number: 1,
				title: "Result one",
				summary: "Summary one",
				confidence: 0.8,
				evidence: [{ excerpt: "evidence excerpt" }],
			},
		],
		mapping: [
			{
				number: 1,
				source: "file:///do-not-read",
				method: "grep",
				content: "source content",
				evidenceRefs: [
					normalizeSessionEvidenceRef({
						method: "grep",
						source: "file:///do-not-read",
						content: "source content",
					}),
				],
			},
		],
		warnings: [],
		...overrides,
	};
}

function writeConfig(): void {
	writeFileSync(
		configPath,
		JSON.stringify({
			searchPaths: [root],
			workspacePath: root,
			memoryPath,
			minSync: false,
			jikji: false,
		}),
	);
}

function getMemoryByteLength(): number {
	if (!existsSync(memoryPath)) return 0;
	return readFileSync(memoryPath).byteLength;
}

describe("runReport", () => {
	it("accepts a valid report and creates a session", async () => {
		writeConfig();
		const reportPath = join(root, "report.json");
		writeFileSync(reportPath, JSON.stringify(validReport()));

		const { ctx, stdout } = makeCtx(["test query"], { input: reportPath });
		const code = await runReport(ctx);
		expect(code).toBe(0);
		const parsed = JSON.parse(stdout[0]);
		expect(parsed.ok).toBe(true);
		expect(typeof parsed.sessionId).toBe("string");
		expect(existsSync(memoryPath)).toBe(true);
	});

	it("rejects result confidence of 1.1 with exit 2 and no memory written", async () => {
		writeConfig();
		const reportPath = join(root, "report-invalid-confidence.json");
		writeFileSync(
			reportPath,
			JSON.stringify(
				validReport({
					results: [
						{
							number: 1,
							title: "X",
							summary: "Y",
							evidence: [{ excerpt: "E" }],
							confidence: 1.1,
						},
					],
				}),
			),
		);

		const { ctx, stderr } = makeCtx(["test query"], { input: reportPath });
		const code = await runReport(ctx);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toContain("ok");
		expect(existsSync(memoryPath)).toBe(false);
	});

	it("rejects result confidence of -0.1 with exit 2 and no memory written", async () => {
		writeConfig();
		const reportPath = join(root, "report-negative-confidence.json");
		writeFileSync(
			reportPath,
			JSON.stringify(
				validReport({
					results: [
						{
							number: 1,
							title: "X",
							summary: "Y",
							evidence: [{ excerpt: "E" }],
							confidence: -0.1,
						},
					],
				}),
			),
		);

		const { ctx, stderr } = makeCtx(["test query"], { input: reportPath });
		const code = await runReport(ctx);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toContain("ok");
		expect(existsSync(memoryPath)).toBe(false);
	});

	it("rejects non-finite result confidence (Infinity) with exit 2", async () => {
		writeConfig();
		const reportPath = join(root, "report-inf-confidence.json");
		writeFileSync(
			reportPath,
			JSON.stringify(
				validReport({
					results: [
						{
							number: 1,
							title: "X",
							summary: "Y",
							evidence: [{ excerpt: "E" }],
							confidence: Infinity,
						},
					],
				}),
			),
		);

		const { ctx, stderr } = makeCtx(["test query"], { input: reportPath });
		const code = await runReport(ctx);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toContain("ok");
		expect(existsSync(memoryPath)).toBe(false);
	});

	it("rejects NaN result confidence with exit 2", async () => {
		writeConfig();
		const reportPath = join(root, "report-nan-confidence.json");
		writeFileSync(
			reportPath,
			JSON.stringify(
				validReport({
					results: [
						{
							number: 1,
							title: "X",
							summary: "Y",
							evidence: [{ excerpt: "E" }],
							confidence: NaN,
						},
					],
				}),
			),
		);

		const { ctx, stderr } = makeCtx(["test query"], { input: reportPath });
		const code = await runReport(ctx);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toContain("ok");
		expect(existsSync(memoryPath)).toBe(false);
	});

	it("rejects evidence reference confidence of 1.05 with exit 2 and no memory written", async () => {
		writeConfig();
		const reportPath = join(root, "report-evidence-confidence.json");
		writeFileSync(
			reportPath,
			JSON.stringify(
				validReport({
					mapping: [
						{
							number: 1,
							source: "file:///do-not-read",
							method: "grep",
							content: "content",
							evidenceRefs: [
								{
									method: "grep",
									source: "file:///do-not-read",
									content: "content",
									confidence: 1.05,
								},
							],
						},
					],
				}),
			),
		);

		const { ctx, stderr } = makeCtx(["test query"], { input: reportPath });
		const code = await runReport(ctx);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toContain("ok");
		expect(existsSync(memoryPath)).toBe(false);
	});

	it("rejects non-finite evidence reference confidence with exit 2", async () => {
		writeConfig();
		const reportPath = join(root, "report-evidence-inf.json");
		writeFileSync(
			reportPath,
			JSON.stringify(
				validReport({
					mapping: [
						{
							number: 1,
							source: "file:///do-not-read",
							method: "grep",
							content: "content",
							evidenceRefs: [
								{
									method: "grep",
									source: "file:///do-not-read",
									content: "content",
									confidence: Number.NEGATIVE_INFINITY,
								},
							],
						},
					],
				}),
			),
		);

		const { ctx, stderr } = makeCtx(["test query"], { input: reportPath });
		const code = await runReport(ctx);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toContain("ok");
		expect(existsSync(memoryPath)).toBe(false);
	});

	it("preserves memory byte-identical when rejecting invalid confidence (no side effects)", async () => {
		writeConfig();
		// Make a valid report first, to create memory
		const reportPath = join(root, "report-good.json");
		writeFileSync(reportPath, JSON.stringify(validReport()));
		const { ctx: ctxGood } = makeCtx(["query"], { input: reportPath });
		await runReport(ctxGood);
		expect(existsSync(memoryPath)).toBe(true);
		const memoryBytesBefore = getMemoryByteLength();

		// Now submit invalid confidence — must preserve existing memory byte-identical
		const reportBadPath = join(root, "report-bad.json");
		writeFileSync(
			reportBadPath,
			JSON.stringify(
				validReport({
					results: [
						{
							number: 1,
							title: "X",
							summary: "Y",
							evidence: [{ excerpt: "E" }],
							confidence: 1.1,
						},
					],
				}),
			),
		);
		const { ctx: ctxBad, stderr } = makeCtx(["other query"], { input: reportBadPath });
		const code = await runReport(ctxBad);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toContain("ok");
		expect(getMemoryByteLength()).toBe(memoryBytesBefore);
	});

	it("rejects malformed JSON (non-report) with exit 2", async () => {
		writeConfig();
		const reportPath = join(root, "bad-json.json");
		writeFileSync(reportPath, "{not-json");
		const { ctx, stderr } = makeCtx(["query"], { input: reportPath });
		const code = await runReport(ctx);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toContain("ok");
		expect(existsSync(memoryPath)).toBe(false);
	});

	it("rejects mapping without matching result number", async () => {
		writeConfig();
		const reportPath = join(root, "mismatch.json");
		writeFileSync(
			reportPath,
			JSON.stringify(
				validReport({
					results: [{ number: 1, title: "X", summary: "Y", evidence: [{ excerpt: "E" }], confidence: 0.5 }],
					mapping: [
						{
							number: 2,
							source: "s",
							method: "grep",
							content: "c",
							evidenceRefs: [
								normalizeSessionEvidenceRef({
									method: "grep",
									source: "s",
									content: "c",
								}),
							],
						},
					],
				}),
			),
		);
		const { ctx, stderr } = makeCtx(["query"], { input: reportPath });
		const code = await runReport(ctx);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toContain("one-to-one");
		expect(existsSync(memoryPath)).toBe(false);
	});

	it("rejects duplicate result numbers", async () => {
		writeConfig();
		const reportPath = join(root, "duplicate.json");
		writeFileSync(
			reportPath,
			JSON.stringify({
				answer: "[1] [2] test",
				results: [
					{ number: 1, title: "A", summary: "A", evidence: [{ excerpt: "e" }], confidence: 0.5 },
					{ number: 1, title: "B", summary: "B", evidence: [{ excerpt: "e" }], confidence: 0.5 },
				],
				mapping: [
					{
						number: 1,
						source: "s",
						method: "grep",
						content: "c",
						evidenceRefs: [normalizeSessionEvidenceRef({ method: "grep", source: "s", content: "c" })],
					},
					{
						number: 1,
						source: "s2",
						method: "grep",
						content: "c2",
						evidenceRefs: [normalizeSessionEvidenceRef({ method: "grep", source: "s2", content: "c2" })],
					},
				],
			}),
		);
		const { ctx } = makeCtx(["query"], { input: reportPath });
		const code = await runReport(ctx);
		expect(code).toBe(2);
		expect(existsSync(memoryPath)).toBe(false);
	});

	describe("runReport config error exit codes", () => {
		it("returns exit 2 when --config points to a missing file", async () => {
			const { ctx, stderr } = makeCtx(["test query"], {
				input: join(root, "report.json"),
				config: join(root, "nonexistent-config.json"),
			});
			writeFileSync(join(root, "report.json"), JSON.stringify(validReport()));
			const code = await runReport(ctx);
			expect(code).toBe(2);
			expect(stderr.join("\n")).toContain("Config file not found");
		});

		it("returns exit 2 when --config points to a malformed JSON file", async () => {
			const malformedPath = join(root, "bad-config.json");
			writeFileSync(malformedPath, "{not-json");
			const { ctx, stderr } = makeCtx(["test query"], {
				input: join(root, "report.json"),
				config: malformedPath,
			});
			writeFileSync(join(root, "report.json"), JSON.stringify(validReport()));
			const code = await runReport(ctx);
			expect(code).toBe(2);
			expect(stderr.join("\n")).toContain("Failed to parse config file");
		});
	});
});

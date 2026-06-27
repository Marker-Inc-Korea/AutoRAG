import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, watch, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { DEFAULT_JIKJI_OPTIONS, JikjiClient, parseJikjiFindPayload } from "../../src/jikji/index.ts";

type LoggedCall = {
	readonly args: readonly string[];
	readonly envMedia: string | null;
};

let root: string;
let corpusRoot: string;
let binDir: string;
let binaryPath: string;
let logPath: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-jikji-client-test-"));
	corpusRoot = join(root, "corpus");
	binDir = join(root, "bin");
	binaryPath = join(binDir, "jikji");
	logPath = join(root, "jikji-calls.jsonl");
	mkdirSync(binDir, { recursive: true });
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function writeFakeJikji(body: string, exitCode = 0): void {
	writeFileSync(
		binaryPath,
		`#!/usr/bin/env node
import { appendFileSync } from "node:fs";

const args = process.argv.slice(2);
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({
  args,
  envMedia: process.env.JIKJI_ENABLE_MEDIA_INDEX ?? null
}) + "\\n");
${body}
process.exit(${exitCode});
`,
	);
	chmodSync(binaryPath, 0o755);
}

function payloadJson(): string {
	return JSON.stringify({
		mode: "find",
		answer_pack_version: 1,
		root: corpusRoot,
		query: "enterprise revenue report",
		query_type: "single_file",
		confidence: "high",
		confidence_score: 1,
		recommended_action: "return_top1_after_light_verification",
		handoff_action: "direct_use",
		paths: ["docs/q3-report.txt"],
		answer_paths: ["docs/q3-report.txt"],
		index_status: "ready",
		command: "jikji find",
		evidence_pack: [
			{
				path: "docs/q3-report.txt",
				why: ["body-coverage"],
				matched_terms: ["enterprise", "revenue"],
				evidence: ["Q3 revenue grew from enterprise contracts"],
				next_read: { kind: "original", path: "docs/q3-report.txt" },
			},
		],
		candidates: [
			{
				p: "docs/q3-report.txt",
				s: 1327.049,
				rank: 1,
				why: ["body-coverage"],
				terms: ["enterprise", "revenue"],
				ev: "Q3 revenue grew from enterprise contracts",
				next_read: { kind: "original", path: "docs/q3-report.txt" },
			},
		],
	});
}

function loggedCalls(): readonly LoggedCall[] {
	return readFileSync(logPath, "utf8")
		.trim()
		.split("\n")
		.filter((line) => line.length > 0)
		.map(parseLoggedCall);
}

function waitForLogFile(): Promise<void> {
	return new Promise((resolve, reject) => {
		if (existsSync(logPath)) {
			resolve();
			return;
		}
		const watcher = watch(root, (eventType, filename) => {
			if (eventType === "rename" && filename === "jikji-calls.jsonl") {
				clearTimeout(timeout);
				watcher.close();
				resolve();
			}
		});
		const timeout = setTimeout(() => {
			watcher.close();
			reject(new Error("timed out waiting for fake Jikji log"));
		}, 1000);
	});
}

function parseLoggedCall(line: string): LoggedCall {
	const parsed: unknown = JSON.parse(line);
	if (!isLoggedCall(parsed)) throw new Error(`unexpected fake Jikji log: ${line}`);
	return parsed;
}

function isLoggedCall(value: unknown): value is LoggedCall {
	if (!isRecord(value)) return false;
	return (
		Array.isArray(value.args) &&
		value.args.every((arg) => typeof arg === "string") &&
		isNullableString(value.envMedia)
	);
}

function isNullableString(value: unknown): value is string | null {
	return value === null || typeof value === "string";
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}

function clientWithPath(): JikjiClient {
	return new JikjiClient({
		env: { PATH: `${binDir}:${process.env.PATH ?? ""}`, JIKJI_ENABLE_MEDIA_INDEX: "1" },
	});
}

describe("JikjiClient", () => {
	it("publishes the bounded default options", () => {
		// Given
		const expectedDefaults = {
			binaryPath: "jikji",
			timeoutMs: 10_000,
			maxBufferBytes: 1_048_576,
			includeHidden: false,
			includeSensitive: false,
			parseTimeout: 5,
			maxFiles: 0,
			staleAfterSeconds: 86_400,
			exclude: [],
			topK: 20,
		};

		// When
		const defaults = DEFAULT_JIKJI_OPTIONS;

		// Then
		expect(defaults).toEqual(expectedDefaults);
	});

	it("runs jikji find with bounded json output", async () => {
		// Given
		writeFakeJikji(`console.log(${JSON.stringify(payloadJson())});`);
		const client = clientWithPath();

		// When
		const result = await client.find(corpusRoot, "enterprise revenue report");

		// Then
		expect(result).toMatchObject({ ok: true });
		expect(result.ok ? result.payload.answerPaths : []).toEqual(["docs/q3-report.txt"]);
		expect(loggedCalls()).toEqual([
			{
				args: ["find", corpusRoot, "enterprise revenue report", "--json", "--top-k", "20"],
				envMedia: null,
			},
		]);
	});

	it("uses configured binaryPath for jikji find", async () => {
		// Given
		writeFakeJikji(`console.log(${JSON.stringify(payloadJson())});`);
		const client = new JikjiClient({ binaryPath, env: { JIKJI_ENABLE_MEDIA_INDEX: "1" } });

		// When
		const result = await client.find(corpusRoot, "enterprise revenue report", { topK: 7 });

		// Then
		expect(result).toMatchObject({ ok: true });
		expect(loggedCalls()).toEqual([
			{
				args: ["find", corpusRoot, "enterprise revenue report", "--json", "--top-k", "7"],
				envMedia: null,
			},
		]);
	});

	it("does not pass hidden sensitive or media flags by default", async () => {
		// Given
		writeFakeJikji(`console.log(${JSON.stringify(payloadJson())});`);
		const client = clientWithPath();

		// When
		await client.find(corpusRoot, "enterprise revenue report");

		// Then
		const call = loggedCalls()[0];
		expect(call?.args).not.toContain("--include-hidden");
		expect(call?.args).not.toContain("--include-sensitive");
		expect(call?.args).not.toContain("--enable-media-index");
	});

	it("passes explicit hidden and sensitive flags", async () => {
		// Given
		writeFakeJikji(`console.log(${JSON.stringify(payloadJson())});`);
		const client = new JikjiClient({
			env: { PATH: `${binDir}:${process.env.PATH ?? ""}` },
			includeHidden: true,
			includeSensitive: true,
		});

		// When
		await client.find(corpusRoot, "enterprise revenue report");

		// Then
		const call = loggedCalls()[0];
		expect(call?.args).toContain("--include-hidden");
		expect(call?.args).toContain("--include-sensitive");
		expect(call?.args).not.toContain("--enable-media-index");
	});

	it("returns failure for timeout without throwing", async () => {
		// Given
		writeFakeJikji("setInterval(() => undefined, 1000);\nawait new Promise(() => undefined);");
		const client = new JikjiClient({
			env: { PATH: `${binDir}:${process.env.PATH ?? ""}` },
			timeoutMs: 50,
		});

		// When
		const result = await client.find(corpusRoot, "slow query");

		// Then
		expect(result).toMatchObject({ ok: false, reason: "timeout" });
	});

	it("returns failure for malformed json without throwing", async () => {
		// Given
		writeFakeJikji('process.stdout.write("{not json");');
		const client = clientWithPath();

		// When
		const result = await client.find(corpusRoot, "enterprise revenue report");

		// Then
		expect(result).toMatchObject({ ok: false, reason: "malformed-json" });
	});

	it("returns failure for invalid payload shape without throwing", () => {
		// Given
		const payload = JSON.stringify({ mode: "find", paths: [123] });

		// When
		const result = parseJikjiFindPayload(payload);

		// Then
		expect(result).toMatchObject({ ok: false, reason: "invalid-payload" });
	});

	it("terminates the child when AbortController aborts", async () => {
		// Given
		writeFakeJikji("setInterval(() => undefined, 1000);\nawait new Promise(() => undefined);");
		const controller = new AbortController();
		const client = new JikjiClient({
			env: { PATH: `${binDir}:${process.env.PATH ?? ""}` },
			timeoutMs: 5000,
		});

		// When
		const pending = client.find(corpusRoot, "cancel query", { signal: controller.signal });
		await waitForLogFile();
		controller.abort();
		const result = await pending;

		// Then
		expect(result).toMatchObject({ ok: false, reason: "aborted" });
		expect(loggedCalls()).toHaveLength(1);
	});

	it("returns failure for oversized stdout without throwing", async () => {
		// Given
		writeFakeJikji('process.stdout.write("x".repeat(64));');
		const client = new JikjiClient({
			env: { PATH: `${binDir}:${process.env.PATH ?? ""}` },
			maxBufferBytes: 8,
		});

		// When
		const result = await client.find(corpusRoot, "oversized query");

		// Then
		expect(result).toMatchObject({ ok: false, reason: "stdout-too-large" });
	});
});

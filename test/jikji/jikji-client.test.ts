import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, watch, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { DEFAULT_JIKJI_OPTIONS, JikjiClient } from "../../src/jikji/index.ts";

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
	it("publishes bounded prepare defaults without retrieval options", () => {
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
		};

		const defaults = DEFAULT_JIKJI_OPTIONS;

		expect(defaults).toEqual(expectedDefaults);
		expect(JSON.stringify(defaults)).not.toContain("topK");
	});

	it("runs jikji prepare with bounded json output", async () => {
		writeFakeJikji("console.log(JSON.stringify({ prepared: true }));");
		const client = clientWithPath();

		const result = await client.prepare(corpusRoot);

		expect(result).toMatchObject({ ok: true });
		expect(loggedCalls()).toEqual([
			{
				args: ["prepare", corpusRoot, "--json"],
				envMedia: null,
			},
		]);
	});

	it("uses configured binaryPath for jikji prepare", async () => {
		writeFakeJikji("console.log(JSON.stringify({ prepared: true }));");
		const client = new JikjiClient({ binaryPath, env: { JIKJI_ENABLE_MEDIA_INDEX: "1" } });

		const result = await client.prepare(corpusRoot);

		expect(result).toMatchObject({ ok: true });
		expect(loggedCalls()).toEqual([
			{
				args: ["prepare", corpusRoot, "--json"],
				envMedia: null,
			},
		]);
	});

	it("does not pass hidden sensitive or media flags by default", async () => {
		writeFakeJikji("console.log(JSON.stringify({ prepared: true }));");
		const client = clientWithPath();

		await client.prepare(corpusRoot);

		const call = loggedCalls()[0];
		expect(call?.args).not.toContain("--include-hidden");
		expect(call?.args).not.toContain("--include-sensitive");
		expect(call?.args).not.toContain("--enable-media-index");
	});

	it("passes explicit prepare flags", async () => {
		writeFakeJikji("console.log(JSON.stringify({ prepared: true }));");
		const client = new JikjiClient({
			env: { PATH: `${binDir}:${process.env.PATH ?? ""}` },
			includeHidden: true,
			includeSensitive: true,
			parseTimeout: 5,
			maxFiles: 10,
			staleAfterSeconds: 60,
			exclude: ["private/**"],
		});

		await client.prepare(corpusRoot);

		expect(loggedCalls()[0]?.args).toEqual([
			"prepare",
			corpusRoot,
			"--json",
			"--include-hidden",
			"--include-sensitive",
			"--parse-timeout",
			"5",
			"--max-files",
			"10",
			"--stale-after-seconds",
			"60",
			"--exclude",
			"private/**",
		]);
		expect(loggedCalls()[0]?.envMedia).toBeNull();
	});

	it("returns failure for timeout without throwing", async () => {
		writeFakeJikji("setInterval(() => undefined, 1000);\nawait new Promise(() => undefined);");
		const client = new JikjiClient({
			env: { PATH: `${binDir}:${process.env.PATH ?? ""}` },
			timeoutMs: 50,
		});

		const result = await client.prepare(corpusRoot);

		expect(result).toMatchObject({ ok: false, reason: "timeout" });
	});

	it("terminates the child when AbortController aborts", async () => {
		writeFakeJikji("setInterval(() => undefined, 1000);\nawait new Promise(() => undefined);");
		const controller = new AbortController();
		const client = new JikjiClient({
			env: { PATH: `${binDir}:${process.env.PATH ?? ""}` },
			timeoutMs: 5000,
		});

		const pending = client.prepare(corpusRoot, { signal: controller.signal });
		await waitForLogFile();
		controller.abort();
		const result = await pending;

		expect(result).toMatchObject({ ok: false, reason: "aborted" });
		expect(loggedCalls()).toHaveLength(1);
	});

	it("returns failure for oversized stdout without throwing", async () => {
		writeFakeJikji('process.stdout.write("x".repeat(64));');
		const client = new JikjiClient({
			env: { PATH: `${binDir}:${process.env.PATH ?? ""}` },
			maxBufferBytes: 8,
		});

		const result = await client.prepare(corpusRoot);

		expect(result).toMatchObject({ ok: false, reason: "stdout-too-large" });
	});
});

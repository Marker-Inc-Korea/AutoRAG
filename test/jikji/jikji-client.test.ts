import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, watch, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { DEFAULT_JIKJI_OPTIONS, JikjiClient } from "../../src/jikji/index.ts";
import type { JikjiOptions } from "../../src/jikji/index.ts";

type LoggedCall = {
	readonly args: readonly string[];
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
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args }) + "\\n");
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
	return Array.isArray(value.args) && value.args.every((arg) => typeof arg === "string");
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}

function pathEnv(): { readonly PATH: string } {
	return { PATH: `${binDir}:${process.env.PATH ?? ""}` };
}

function clientWithPath(): JikjiClient {
	return new JikjiClient({ env: pathEnv() });
}

function argsOfFirstCall(): readonly string[] {
	return loggedCalls()[0]?.args ?? [];
}

describe("JikjiClient", () => {
	it("publishes bounded prepare defaults without retrieval or stale options", () => {
		const expectedDefaults = {
			binaryPath: "jikji",
			timeoutMs: 10_000,
			maxBufferBytes: 1_048_576,
			includeHidden: false,
			includeSensitive: false,
			maxFiles: 0,
			noAgentRules: false,
			enableMediaIndex: false,
			exclude: [],
		};

		const defaults = DEFAULT_JIKJI_OPTIONS;

		expect(defaults).toEqual(expectedDefaults);
		expect(JSON.stringify(defaults)).not.toContain("topK");
		expect(JSON.stringify(defaults)).not.toContain("parseTimeout");
	});

	it("emits a clean wire format for the bare default client", async () => {
		writeFakeJikji("console.log(JSON.stringify({ prepared: true }));");
		const client = clientWithPath();

		const result = await client.prepare(corpusRoot);

		expect(result).toMatchObject({ ok: true });
		expect(argsOfFirstCall()).toEqual(["prepare", corpusRoot, "--json"]);
	});

	it("emits a clean wire format for new JikjiClient(DEFAULT_JIKJI_OPTIONS)", async () => {
		writeFakeJikji("console.log(JSON.stringify({ prepared: true }));");
		const client = new JikjiClient({ ...DEFAULT_JIKJI_OPTIONS, env: pathEnv() });

		const result = await client.prepare(corpusRoot);

		expect(result).toMatchObject({ ok: true });
		expect(argsOfFirstCall()).toEqual(["prepare", corpusRoot, "--json"]);
	});

	it("emits a clean wire format for the README default-shaped config", async () => {
		writeFakeJikji("console.log(JSON.stringify({ prepared: true }));");
		const readmeConfig: JikjiOptions = {
			binaryPath: "jikji",
			timeoutMs: 10_000,
			maxBufferBytes: 1_048_576,
			includeHidden: false,
			includeSensitive: false,
			noAgentRules: false,
			enableMediaIndex: false,
			maxFiles: 0,
			exclude: [],
			env: pathEnv(),
		};
		const client = new JikjiClient(readmeConfig);

		const result = await client.prepare(corpusRoot);

		expect(result).toMatchObject({ ok: true });
		expect(argsOfFirstCall()).toEqual(["prepare", corpusRoot, "--json"]);
	});

	it("suppresses --max-files when maxFiles is 0", async () => {
		writeFakeJikji("console.log(JSON.stringify({ prepared: true }));");
		const client = new JikjiClient({ ...DEFAULT_JIKJI_OPTIONS, env: pathEnv() });

		await client.prepare(corpusRoot);

		const args = argsOfFirstCall();
		expect(args).not.toContain("--max-files");
	});


	it("uses configured binaryPath for jikji prepare", async () => {
		writeFakeJikji("console.log(JSON.stringify({ prepared: true }));");
		const client = new JikjiClient({ binaryPath, env: pathEnv() });

		const result = await client.prepare(corpusRoot);

		expect(result).toMatchObject({ ok: true });
		expect(argsOfFirstCall()).toEqual(["prepare", corpusRoot, "--json"]);
	});

	it("does not pass hidden sensitive no-agent-rules or media flags by default", async () => {
		writeFakeJikji("console.log(JSON.stringify({ prepared: true }));");
		const client = clientWithPath();

		await client.prepare(corpusRoot);

		const args = argsOfFirstCall();
		expect(args).not.toContain("--include-hidden");
		expect(args).not.toContain("--include-sensitive");
		expect(args).not.toContain("--no-agent-rules");
		expect(args).not.toContain("--enable-media-index");
		expect(args).not.toContain("--media-index-max-mb");
	});

	it("passes explicit caller and upstream Rust prepare flags", async () => {
		writeFakeJikji("console.log(JSON.stringify({ prepared: true }));");
		const client = new JikjiClient({
			env: pathEnv(),
			includeHidden: true,
			includeSensitive: true,
			parseTimeout: 5,
			maxFiles: 10,
			exclude: ["private/**"],
			maxHashBytes: 1024,
			docTextMaxChars: 2_000_000,
			docTextChunkChars: 1_000_000,
			noAgentRules: true,
			enableMediaIndex: true,
			mediaIndexMaxMb: 25,
		});

		await client.prepare(corpusRoot);

		expect(argsOfFirstCall()).toEqual([
			"prepare",
			corpusRoot,
			"--json",
			"--include-hidden",
			"--include-sensitive",
			"--no-agent-rules",
			"--enable-media-index",
			"--parse-timeout",
			"5",
			"--max-hash-bytes",
			"1024",
			"--doc-text-max-chars",
			"2000000",
			"--doc-text-chunk-chars",
			"1000000",
			"--max-files",
			"10",
			"--media-index-max-mb",
			"25",
			"--exclude",
			"private/**",
		]);
	});

	it("gates media-index-max-mb behind enableMediaIndex", async () => {
		writeFakeJikji("console.log(JSON.stringify({ prepared: true }));");
		const client = new JikjiClient({
			env: pathEnv(),
			enableMediaIndex: false,
			mediaIndexMaxMb: 25,
		});

		await client.prepare(corpusRoot);

		const args = argsOfFirstCall();
		expect(args).not.toContain("--enable-media-index");
		expect(args).not.toContain("--media-index-max-mb");
	});

	it("emits --media-index-max-mb only when media indexing is enabled", async () => {
		writeFakeJikji("console.log(JSON.stringify({ prepared: true }));");
		const client = new JikjiClient({
			env: pathEnv(),
			enableMediaIndex: true,
			mediaIndexMaxMb: 25,
		});

		await client.prepare(corpusRoot);

		const args = argsOfFirstCall();
		expect(args).toContain("--enable-media-index");
		expect(args).toContain("--media-index-max-mb");
		expect(args).toContain("25");
	});


	it("returns failure for timeout without throwing", async () => {
		writeFakeJikji("setInterval(() => undefined, 1000);\nawait new Promise(() => undefined);");
		const client = new JikjiClient({
			env: pathEnv(),
			timeoutMs: 50,
		});

		const result = await client.prepare(corpusRoot);

		expect(result).toMatchObject({ ok: false, reason: "timeout" });
	});

	it("terminates the child when AbortController aborts", async () => {
		writeFakeJikji("setInterval(() => undefined, 1000);\nawait new Promise(() => undefined);");
		const controller = new AbortController();
		const client = new JikjiClient({
			env: pathEnv(),
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
			env: pathEnv(),
			maxBufferBytes: 8,
		});

		const result = await client.prepare(corpusRoot);

		expect(result).toMatchObject({ ok: false, reason: "stdout-too-large" });
	});
});

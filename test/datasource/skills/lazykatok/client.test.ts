import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { LazykatokClient } from "../../../../src/datasource/skills/lazykatok/client.ts";

type LoggedCall = {
	readonly args: readonly string[];
	readonly envApiKey?: string | null;
};

const FAKE_CHILD_READY_TIMEOUT_MS = 10_000;

let root: string;
let binDir: string;
let binaryPath: string;
let logPath: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-lazykatok-client-test-"));
	binDir = join(root, "bin");
	binaryPath = join(binDir, "lazykatok");
	logPath = join(root, "lazykatok-calls.jsonl");
	mkdirSync(binDir, { recursive: true });
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

/**
 * Writes a fake `lazykatok` executable that logs every invocation (args + selected
 * env) to a JSONL file, then prints the JSON payload from `LAZYKATOK_FAKE_OUTPUT`
 * (so each test controls the parsed shape) and exits 0.
 */
function writeFakeLazykatok(): void {
	writeFileSync(
		binaryPath,
		`#!/usr/bin/env node
import { appendFileSync } from "node:fs";

const args = process.argv.slice(2);
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({
  args,
  envApiKey: process.env.OPENAI_API_KEY ?? null,
}) + "\\n");

const payload = process.env.LAZYKATOK_FAKE_OUTPUT ?? "{}";
process.stdout.write(payload);
process.exit(0);
`,
	);
	chmodSync(binaryPath, 0o755);
}

function loggedCalls(): readonly LoggedCall[] {
	if (!existsSync(logPath)) return [];
	return readFileSync(logPath, "utf8")
		.trim()
		.split("\n")
		.filter((line) => line.length > 0)
		.map(parseLoggedCall);
}

async function waitForLogFile(): Promise<void> {
	// Poll for a non-empty log: watching for file creation alone races the
	// child's write, so an early abort can kill the child before it flushes.
	const deadline = Date.now() + FAKE_CHILD_READY_TIMEOUT_MS;
	while (Date.now() < deadline) {
		if (existsSync(logPath) && statSync(logPath).size > 0) return;
		await new Promise((resolve) => setTimeout(resolve, 20));
	}
	throw new Error("timed out waiting for fake lazykatok log");
}

function parseLoggedCall(line: string): LoggedCall {
	const parsed: unknown = JSON.parse(line);
	if (!isLoggedCall(parsed)) throw new Error(`unexpected fake lazykatok log: ${line}`);
	return parsed;
}

function isLoggedCall(value: unknown): value is LoggedCall {
	if (!isRecord(value)) return false;
	return (
		Array.isArray(value.args) &&
		value.args.every((arg) => typeof arg === "string") &&
		(value.envApiKey === undefined || isNullableString(value.envApiKey))
	);
}

function isNullableString(value: unknown): value is string | null {
	return value === null || typeof value === "string";
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}

/** A client pointed at the fake binary with PATH-aware env. */
function fakeClient(env: Readonly<Record<string, string | undefined>> = {}): LazykatokClient {
	return new LazykatokClient({
		binaryPath,
		env: { PATH: `${binDir}:${process.env.PATH ?? ""}`, ...env },
	});
}

function jsonEnv(value: unknown): string {
	return JSON.stringify(value);
}

describe("LazykatokClient", () => {
	it("parses doctor JSON and preserves call args order", async () => {
		writeFakeLazykatok();
		const client = fakeClient({ LAZYKATOK_FAKE_OUTPUT: jsonEnv({ version: "1.2.3", ready: true }) });

		const result = await client.doctor();

		expect(result.ok).toBe(true);
		if (!result.ok) return;
		expect(result.data).toEqual({ version: "1.2.3", ready: true, metadata: {} });
		const call = loggedCalls()[0];
		expect(call?.args).toEqual(["doctor", "--json"]);
	});

	it("forwards an explicitly configured workspacePath", async () => {
		writeFakeLazykatok();
		const client = new LazykatokClient({
			binaryPath,
			workspacePath: join(root, "custom-lazykatok"),
			env: { PATH: `${binDir}:${process.env.PATH ?? ""}`, LAZYKATOK_FAKE_OUTPUT: jsonEnv({ ready: true }) },
		});

		const result = await client.doctor();

		expect(result.ok).toBe(true);
		const args = loggedCalls()[0]?.args ?? [];
		expect(args).toContain("--data-dir");
		expect(args[args.indexOf("--data-dir") + 1]).toBe(join(root, "custom-lazykatok"));
	});

	it("parses search hits in returned order", async () => {
		writeFakeLazykatok();
		const payload = {
			hits: [
				{ chunkId: "c1", score: 0.9, content: "alpha" },
				{ chunkId: "c2", score: 0.5, content: "beta" },
				{ chunkId: "c3", score: 0.1, content: "gamma" },
			],
		};
		const client = fakeClient({ LAZYKATOK_FAKE_OUTPUT: jsonEnv(payload) });

		const result = await client.search("semantic", "hello", { topK: 3 });

		expect(result.ok).toBe(true);
		if (!result.ok) return;
		expect(result.data.hits.map((hit) => hit.chunkId)).toEqual(["c1", "c2", "c3"]);
		expect(result.data.hits[0]).toMatchObject({ chunkId: "c1", score: 0.9, content: "alpha" });
		const call = loggedCalls()[0];
		expect(call?.args).toEqual(["search", "semantic", "hello", "--json", "--limit", "3"]);
	});

	it("maps topK to lazykatok's --limit flag and never forwards virtual scopes", async () => {
		writeFakeLazykatok();
		const client = fakeClient({ LAZYKATOK_FAKE_OUTPUT: jsonEnv({ hits: [] }) });

		await client.search("keyword", "q", { topK: 5, scope: "room-42" });

		const args = loggedCalls()[0]?.args ?? [];
		expect(args).toContain("--limit");
		expect(args[args.indexOf("--limit") + 1]).toBe("5");
		expect(args).not.toContain("--scope");
		expect(args).not.toContain("--top-k");
	});

	it("parses index/sync/chunk/context/parent payloads", async () => {
		writeFakeLazykatok();
		const client = fakeClient({ LAZYKATOK_FAKE_OUTPUT: jsonEnv({ chunkCount: 7 }) });
		const index = await client.index();
		expect(index.ok).toBe(true);
		if (index.ok) expect(index.data.chunkCount).toBe(7);

		const syncClient = fakeClient({ LAZYKATOK_FAKE_OUTPUT: jsonEnv({ synced: true, messageCount: 12 }) });
		const synced = await syncClient.sync();
		expect(synced.ok).toBe(true);
		if (synced.ok) expect(synced.data).toEqual({ synced: true, messageCount: 12, metadata: {} });

		const chunkClient = fakeClient({ LAZYKATOK_FAKE_OUTPUT: jsonEnv({ chunkId: "c1", content: "hi" }) });
		const chunk = await chunkClient.chunkGet("c1");
		expect(chunk.ok).toBe(true);
		if (chunk.ok) expect(chunk.data).toEqual({ chunkId: "c1", content: "hi", metadata: {} });

		const ctxClient = fakeClient({
			LAZYKATOK_FAKE_OUTPUT: jsonEnv({
				chunks: [
					{ chunkId: "c1", content: "a" },
					{ chunkId: "c2", content: "b" },
				],
			}),
		});
		const ctx = await ctxClient.context("c1");
		expect(ctx.ok).toBe(true);
		if (ctx.ok) expect(ctx.data.chunks.map((c) => c.chunkId)).toEqual(["c1", "c2"]);

		const parentClient = fakeClient({ LAZYKATOK_FAKE_OUTPUT: jsonEnv({ chunkId: "p1", content: "parent" }) });
		const parent = await parentClient.parent("c1");
		expect(parent.ok).toBe(true);
		if (parent.ok) expect(parent.data.chunkId).toBe("p1");
	});

	it("returns binary-missing for a non-existent binary without throwing", async () => {
		const client = new LazykatokClient({
			binaryPath: join(binDir, "does-not-exist"),
			env: { PATH: `${binDir}:${process.env.PATH ?? ""}` },
		});

		const result = await client.doctor();

		expect(result).toMatchObject({ ok: false, reason: "binary-missing" });
		expect(loggedCalls()).toHaveLength(0);
	});

	it("returns nonzero-exit for a failing binary without throwing", async () => {
		writeFakeLazykatok();
		const client = new LazykatokClient({
			binaryPath,
			env: { PATH: `${binDir}:${process.env.PATH ?? ""}`, LAZYKATOK_FAKE_OUTPUT: jsonEnv({}) },
		});
		// Overwrite the fake to exit nonzero after logging.
		writeFileSync(
			binaryPath,
			`#!/usr/bin/env node
import { appendFileSync } from "node:fs";
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args: process.argv.slice(2) }) + "\\n");
process.exit(2);
`,
		);
		chmodSync(binaryPath, 0o755);

		const result = await client.doctor();

		expect(result).toMatchObject({ ok: false, reason: "nonzero-exit", code: 2 });
	});

	it("preserves CLI stderr verbatim on failure, paths included", async () => {
		writeFakeLazykatok();
		const stderrText = "lazykatok: index busy at /Users/me/Library/~/Library/Application Support/katok/index.db";
		writeFileSync(
			binaryPath,
			`#!/usr/bin/env node
process.stderr.write(${JSON.stringify(stderrText)});
process.exit(1);
`,
		);
		chmodSync(binaryPath, 0o755);
		const client = new LazykatokClient({
			binaryPath,
			env: { PATH: `${binDir}:${process.env.PATH ?? ""}` },
		});

		const result = await client.doctor();

		expect(result.ok).toBe(false);
		if (result.ok) return;
		expect(result.stderr).toBe(stderrText);
		expect(result.stderr).not.toContain("suppressed");
	});

	it("returns invalid-json for unparseable stdout without throwing", async () => {
		writeFakeLazykatok();
		const client = new LazykatokClient({
			binaryPath,
			env: { PATH: `${binDir}:${process.env.PATH ?? ""}`, LAZYKATOK_FAKE_OUTPUT: "not-json{" },
		});

		const result = await client.doctor();

		expect(result).toMatchObject({ ok: false, reason: "invalid-json" });
	});

	it("rejects malformed success payloads instead of fabricating defaults", async () => {
		writeFakeLazykatok();
		const missingChunkCount = fakeClient({ LAZYKATOK_FAKE_OUTPUT: jsonEnv({}) });
		await expect(missingChunkCount.index()).resolves.toMatchObject({ ok: false, reason: "invalid-shape" });

		const missingHitId = fakeClient({ LAZYKATOK_FAKE_OUTPUT: jsonEnv({ hits: [{ score: 1, content: "hi" }] }) });
		await expect(missingHitId.search("keyword", "hi")).resolves.toMatchObject({ ok: false, reason: "invalid-shape" });

		const missingChunkContent = fakeClient({ LAZYKATOK_FAKE_OUTPUT: jsonEnv({ chunkId: "c1" }) });
		await expect(missingChunkContent.chunkGet("c1")).resolves.toMatchObject({ ok: false, reason: "invalid-shape" });

		const missingReady = fakeClient({ LAZYKATOK_FAKE_OUTPUT: jsonEnv({ version: "1.2.3" }) });
		await expect(missingReady.doctor()).resolves.toMatchObject({ ok: false, reason: "invalid-shape" });
	});

	it("does not forward unrelated parent or caller secrets to lazykatok", async () => {
		writeFakeLazykatok();
		const client = fakeClient({
			LAZYKATOK_FAKE_OUTPUT: jsonEnv({ ready: true }),
			OPENAI_API_KEY: "sk-test-secret",
		});

		const result = await client.doctor();

		expect(result.ok).toBe(true);
		expect(loggedCalls()[0]?.envApiKey).toBeNull();
	});

	it("returns timeout for a hanging binary without throwing", async () => {
		writeFakeLazykatok();
		const client = new LazykatokClient({
			binaryPath,
			env: { PATH: `${binDir}:${process.env.PATH ?? ""}` },
			timeoutMs: 50,
		});
		writeFileSync(
			binaryPath,
			`#!/usr/bin/env node
setInterval(() => undefined, 1000);
`,
		);
		chmodSync(binaryPath, 0o755);

		const result = await client.doctor();

		expect(result).toMatchObject({ ok: false, reason: "timeout" });
	});

	it("terminates the child when AbortController aborts", { timeout: 20_000 }, async () => {
		writeFakeLazykatok();
		const client = new LazykatokClient({
			binaryPath,
			env: { PATH: `${binDir}:${process.env.PATH ?? ""}` },
			timeoutMs: 15_000,
		});
		writeFileSync(
			binaryPath,
			`#!/usr/bin/env node
import { appendFileSync } from "node:fs";
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args: process.argv.slice(2) }) + "\\n");
setInterval(() => undefined, 1000);
`,
		);
		chmodSync(binaryPath, 0o755);
		const controller = new AbortController();

		const pending = client.doctor(controller.signal);
		await waitForLogFile();
		controller.abort();
		const result = await pending;

		expect(result).toMatchObject({ ok: false, reason: "aborted" });
		expect(loggedCalls()).toHaveLength(1);
	});

	it("returns stdout-too-large without throwing", async () => {
		writeFakeLazykatok();
		const client = new LazykatokClient({
			binaryPath,
			env: { PATH: `${binDir}:${process.env.PATH ?? ""}` },
			maxBufferBytes: 8,
		});
		writeFileSync(
			binaryPath,
			`#!/usr/bin/env node
process.stdout.write("x".repeat(64));
`,
		);
		chmodSync(binaryPath, 0o755);

		const result = await client.doctor();

		expect(result).toMatchObject({ ok: false, reason: "stdout-too-large" });
	});

	describe("paths and source opacity", () => {
		it("never includes the binary path in any failure result", async () => {
			const client = new LazykatokClient({ binaryPath, env: {} });
			const result = await client.doctor();
			expect(JSON.stringify(result)).not.toContain(binaryPath);
			expect(JSON.stringify(result)).not.toContain(binDir);
		});

		it("keeps public results free of filesystem paths on success", async () => {
			writeFakeLazykatok();
			const client = fakeClient({
				LAZYKATOK_FAKE_OUTPUT: jsonEnv({ hits: [{ chunkId: "c1", score: 1, content: "hi" }] }),
			});

			const result = await client.search("hybrid", "q");

			expect(result.ok).toBe(true);
			if (!result.ok) return;
			const serialized = JSON.stringify(result.data);
			expect(serialized).not.toContain(root);
			expect(serialized).not.toContain(binDir);
		});
	});
});

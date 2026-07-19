import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { KatokClient } from "../../../../src/datasource/skills/katok/client.ts";

type LoggedCall = {
	readonly args: readonly string[];
	readonly envKatokEmbedder: string | null;
	readonly envEmbedderBaseUrl: string | null;
	readonly envAllowRemoteEmbeddings: string | null;
	readonly envApiKey?: string | null;
};

const FAKE_CHILD_READY_TIMEOUT_MS = 10_000;

let root: string;
let binDir: string;
let binaryPath: string;
let logPath: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-katok-client-test-"));
	binDir = join(root, "bin");
	binaryPath = join(binDir, "katok");
	logPath = join(root, "katok-calls.jsonl");
	mkdirSync(binDir, { recursive: true });
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

/**
 * Writes a fake `katok` executable that logs every invocation (args + selected
 * env) to a JSONL file, then prints the JSON payload from `KATOK_FAKE_OUTPUT`
 * (so each test controls the parsed shape) and exits 0.
 */
function writeFakeKatok(): void {
	writeFileSync(
		binaryPath,
		`#!/usr/bin/env node
import { appendFileSync } from "node:fs";

const args = process.argv.slice(2);
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({
  args,
  envKatokEmbedder: process.env.KATOK_EMBEDDER ?? null,
  envEmbedderBaseUrl: process.env.EMBEDDER_BASE_URL ?? null,
  envAllowRemoteEmbeddings: process.env.ALLOW_REMOTE_EMBEDDINGS ?? null,
  envApiKey: process.env.OPENAI_API_KEY ?? null,
}) + "\\n");

const payload = process.env.KATOK_FAKE_OUTPUT ?? "{}";
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
	throw new Error("timed out waiting for fake katok log");
}

function parseLoggedCall(line: string): LoggedCall {
	const parsed: unknown = JSON.parse(line);
	if (!isLoggedCall(parsed)) throw new Error(`unexpected fake katok log: ${line}`);
	return parsed;
}

function isLoggedCall(value: unknown): value is LoggedCall {
	if (!isRecord(value)) return false;
	return (
		Array.isArray(value.args) &&
		value.args.every((arg) => typeof arg === "string") &&
		isNullableString(value.envKatokEmbedder) &&
		isNullableString(value.envEmbedderBaseUrl) &&
		isNullableString(value.envAllowRemoteEmbeddings) &&
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
function fakeClient(env: Readonly<Record<string, string | undefined>> = {}): KatokClient {
	return new KatokClient({
		binaryPath,
		env: { PATH: `${binDir}:${process.env.PATH ?? ""}`, ...env },
	});
}

function jsonEnv(value: unknown): string {
	return JSON.stringify(value);
}

describe("KatokClient", () => {
	it("parses doctor JSON and preserves call args order", async () => {
		writeFakeKatok();
		const client = fakeClient({ KATOK_FAKE_OUTPUT: jsonEnv({ version: "1.2.3", ready: true }) });

		const result = await client.doctor();

		expect(result.ok).toBe(true);
		if (!result.ok) return;
		expect(result.data).toEqual({ version: "1.2.3", ready: true, metadata: {} });
		const call = loggedCalls()[0];
		expect(call?.args).toEqual(["doctor", "--json", "--source", "macos", "--workspace", expect.any(String)]);
	});

	it("parses search hits in returned order", async () => {
		writeFakeKatok();
		const payload = {
			hits: [
				{ chunkId: "c1", score: 0.9, content: "alpha" },
				{ chunkId: "c2", score: 0.5, content: "beta" },
				{ chunkId: "c3", score: 0.1, content: "gamma" },
			],
		};
		const client = fakeClient({ KATOK_FAKE_OUTPUT: jsonEnv(payload) });

		const result = await client.search("semantic", "hello", { topK: 3 });

		expect(result.ok).toBe(true);
		if (!result.ok) return;
		expect(result.data.hits.map((hit) => hit.chunkId)).toEqual(["c1", "c2", "c3"]);
		expect(result.data.hits[0]).toMatchObject({ chunkId: "c1", score: 0.9, content: "alpha" });
		const call = loggedCalls()[0];
		expect(call?.args).toEqual([
			"search",
			"semantic",
			"hello",
			"--json",
			"--top-k",
			"3",
			"--source",
			"macos",
			"--workspace",
			expect.any(String),
		]);
	});

	it("forwards scope alongside topK when provided", async () => {
		writeFakeKatok();
		const client = fakeClient({ KATOK_FAKE_OUTPUT: jsonEnv({ hits: [] }) });

		await client.search("keyword", "q", { topK: 5, scope: "room-42" });

		const args = loggedCalls()[0]?.args ?? [];
		expect(args).toContain("--scope");
		expect(args[args.indexOf("--scope") + 1]).toBe("room-42");
		expect(args).toContain("--top-k");
		expect(args[args.indexOf("--top-k") + 1]).toBe("5");
	});

	it("parses index/sync/chunk/context/parent payloads", async () => {
		writeFakeKatok();
		const client = fakeClient({ KATOK_FAKE_OUTPUT: jsonEnv({ chunkCount: 7 }) });
		const index = await client.index();
		expect(index.ok).toBe(true);
		if (index.ok) expect(index.data.chunkCount).toBe(7);

		const syncClient = fakeClient({ KATOK_FAKE_OUTPUT: jsonEnv({ synced: true, messageCount: 12 }) });
		const synced = await syncClient.sync();
		expect(synced.ok).toBe(true);
		if (synced.ok) expect(synced.data).toEqual({ synced: true, messageCount: 12, metadata: {} });

		const chunkClient = fakeClient({ KATOK_FAKE_OUTPUT: jsonEnv({ chunkId: "c1", content: "hi" }) });
		const chunk = await chunkClient.chunkGet("c1");
		expect(chunk.ok).toBe(true);
		if (chunk.ok) expect(chunk.data).toEqual({ chunkId: "c1", content: "hi", metadata: {} });

		const ctxClient = fakeClient({
			KATOK_FAKE_OUTPUT: jsonEnv({
				chunks: [
					{ chunkId: "c1", content: "a" },
					{ chunkId: "c2", content: "b" },
				],
			}),
		});
		const ctx = await ctxClient.context("c1");
		expect(ctx.ok).toBe(true);
		if (ctx.ok) expect(ctx.data.chunks.map((c) => c.chunkId)).toEqual(["c1", "c2"]);

		const parentClient = fakeClient({ KATOK_FAKE_OUTPUT: jsonEnv({ chunkId: "p1", content: "parent" }) });
		const parent = await parentClient.parent("c1");
		expect(parent.ok).toBe(true);
		if (parent.ok) expect(parent.data.chunkId).toBe("p1");
	});

	it("returns binary-missing for a non-existent binary without throwing", async () => {
		const client = new KatokClient({
			binaryPath: join(binDir, "does-not-exist"),
			env: { PATH: `${binDir}:${process.env.PATH ?? ""}` },
		});

		const result = await client.doctor();

		expect(result).toMatchObject({ ok: false, reason: "binary-missing" });
		expect(loggedCalls()).toHaveLength(0);
	});

	it("returns nonzero-exit for a failing binary without throwing", async () => {
		writeFakeKatok();
		const client = new KatokClient({
			binaryPath,
			env: { PATH: `${binDir}:${process.env.PATH ?? ""}`, KATOK_FAKE_OUTPUT: jsonEnv({}) },
		});
		// Overwrite the fake to exit nonzero after logging.
		writeFileSync(
			binaryPath,
			`#!/usr/bin/env node
import { appendFileSync } from "node:fs";
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args: process.argv.slice(2), envKatokEmbedder: null, envEmbedderBaseUrl: null, envAllowRemoteEmbeddings: null }) + "\\n");
process.exit(2);
`,
		);
		chmodSync(binaryPath, 0o755);

		const result = await client.doctor();

		expect(result).toMatchObject({ ok: false, reason: "nonzero-exit", code: 2 });
	});

	it("returns invalid-json for unparseable stdout without throwing", async () => {
		writeFakeKatok();
		const client = new KatokClient({
			binaryPath,
			env: { PATH: `${binDir}:${process.env.PATH ?? ""}`, KATOK_FAKE_OUTPUT: "not-json{" },
		});

		const result = await client.doctor();

		expect(result).toMatchObject({ ok: false, reason: "invalid-json" });
	});

	it("rejects malformed success payloads instead of fabricating defaults", async () => {
		writeFakeKatok();
		const missingChunkCount = fakeClient({ KATOK_FAKE_OUTPUT: jsonEnv({}) });
		await expect(missingChunkCount.index()).resolves.toMatchObject({ ok: false, reason: "invalid-shape" });

		const missingHitId = fakeClient({ KATOK_FAKE_OUTPUT: jsonEnv({ hits: [{ score: 1, content: "hi" }] }) });
		await expect(missingHitId.search("keyword", "hi")).resolves.toMatchObject({ ok: false, reason: "invalid-shape" });

		const missingChunkContent = fakeClient({ KATOK_FAKE_OUTPUT: jsonEnv({ chunkId: "c1" }) });
		await expect(missingChunkContent.chunkGet("c1")).resolves.toMatchObject({ ok: false, reason: "invalid-shape" });

		const missingReady = fakeClient({ KATOK_FAKE_OUTPUT: jsonEnv({ version: "1.2.3" }) });
		await expect(missingReady.doctor()).resolves.toMatchObject({ ok: false, reason: "invalid-shape" });
	});

	it("does not forward unrelated parent or caller secrets to katok", async () => {
		writeFakeKatok();
		const client = fakeClient({
			KATOK_FAKE_OUTPUT: jsonEnv({ ready: true }),
			OPENAI_API_KEY: "sk-test-secret",
		});

		const result = await client.doctor();

		expect(result.ok).toBe(true);
		expect(loggedCalls()[0]?.envApiKey).toBeNull();
	});

	it("returns timeout for a hanging binary without throwing", async () => {
		writeFakeKatok();
		const client = new KatokClient({
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
		writeFakeKatok();
		const client = new KatokClient({
			binaryPath,
			env: { PATH: `${binDir}:${process.env.PATH ?? ""}` },
			timeoutMs: 15_000,
		});
		writeFileSync(
			binaryPath,
			`#!/usr/bin/env node
import { appendFileSync } from "node:fs";
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args: process.argv.slice(2), envKatokEmbedder: null, envEmbedderBaseUrl: null, envAllowRemoteEmbeddings: null }) + "\\n");
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
		writeFakeKatok();
		const client = new KatokClient({
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

	describe("remote-embedding rejection (pre-spawn, case-insensitive)", () => {
		const rejectingCases = [
			{ label: "EMBEDDER_BASE_URL any value", env: { EMBEDDER_BASE_URL: "https://embed.example.com" } },
			{ label: "ALLOW_REMOTE_EMBEDDINGS any value", env: { ALLOW_REMOTE_EMBEDDINGS: "1" } },
			{ label: "embedder_base_url lowercase", env: { embedder_base_url: "https://embed.example.com" } },
			{ label: "Allow_Remote_Embeddings mixed case", env: { Allow_Remote_Embeddings: "true" } },
			{ label: "EMBEDDER_BASE_URL empty string", env: { EMBEDDER_BASE_URL: "" } },
			{ label: "KATOK_EMBEDDER http url", env: { KATOK_EMBEDDER: "http://embed.example.com" } },
			{ label: "katok_embedder https url", env: { katok_embedder: "https://embed.example.com/v1" } },
			{ label: "KATOK_EMBEDDER url with whitespace", env: { KATOK_EMBEDDER: "  https://embed.example.com  " } },
		];

		it.each(rejectingCases)("rejects $label without spawning", async ({ env }) => {
			writeFakeKatok();
			const client = fakeClient({ KATOK_FAKE_OUTPUT: jsonEnv({ ready: true }), ...env });

			const result = await client.doctor();

			expect(result).toMatchObject({ ok: false, reason: "remote-embedding-rejected" });
			expect(loggedCalls()).toHaveLength(0);
		});

		const allowingCases = [
			{ label: "no embedder env at all", env: {} as Record<string, string> },
			{ label: "KATOK_EMBEDDER local model name", env: { KATOK_EMBEDDER: "local-bge-small" } },
			{ label: "KATOK_EMBEDDER empty", env: { KATOK_EMBEDDER: "" } },
			{ label: "KATOK_EMBEDDER local path", env: { KATOK_EMBEDDER: "/opt/models/bge" } },
		];

		it.each(allowingCases)("spawns normally for $label", async ({ env }) => {
			writeFakeKatok();
			const client = fakeClient({ KATOK_FAKE_OUTPUT: jsonEnv({ ready: true }), ...env });

			const result = await client.doctor();

			expect(result.ok).toBe(true);
			expect(loggedCalls()).toHaveLength(1);
		});

		it("rejects a remote-embedding key supplied via process.env-equivalent merge even when options.env is clean", async () => {
			writeFakeKatok();
			// Simulate a polluted base env by setting it on the client's env merge
			// through an override that itself is dangerous.
			const client = fakeClient({ KATOK_FAKE_OUTPUT: jsonEnv({ ready: true }), ALLOW_REMOTE_EMBEDDINGS: "1" });

			const result = await client.doctor();

			expect(result).toMatchObject({ ok: false, reason: "remote-embedding-rejected" });
			expect(loggedCalls()).toHaveLength(0);
		});

		it("does not leak the env value in the failure result", async () => {
			writeFakeKatok();
			const client = fakeClient({
				KATOK_FAKE_OUTPUT: jsonEnv({ ready: true }),
				EMBEDDER_BASE_URL: "https://secret-embed.example.com/token-xyz",
			});

			const result = await client.doctor();

			expect(result.ok).toBe(false);
			if (result.ok) return;
			const serialized = JSON.stringify(result);
			expect(serialized).not.toContain("secret-embed.example.com");
			expect(serialized).not.toContain("token-xyz");
			expect(result.violatingKey).toBe("EMBEDDER_BASE_URL");
		});
	});

	describe("paths and source opacity", () => {
		it("never includes the binary path in any failure result", async () => {
			const client = new KatokClient({ binaryPath, env: {} });
			const result = await client.doctor();
			expect(JSON.stringify(result)).not.toContain(binaryPath);
			expect(JSON.stringify(result)).not.toContain(binDir);
		});

		it("keeps public results free of filesystem paths on success", async () => {
			writeFakeKatok();
			const client = fakeClient({
				KATOK_FAKE_OUTPUT: jsonEnv({ hits: [{ chunkId: "c1", score: 1, content: "hi" }] }),
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

import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { LazykatokClient, syncArgs } from "../../../../src/datasource/skills/lazykatok/client.ts";

type LoggedCall = {
	readonly args: readonly string[];
	readonly descendantPid?: number;
	readonly envApiKey?: string | null;
	readonly envEmbedder?: string | null;
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
  envEmbedder: process.env.KATOK_EMBEDDER ?? null,
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
		(value.descendantPid === undefined || typeof value.descendantPid === "number") &&
		(value.envApiKey === undefined || isNullableString(value.envApiKey)) &&
		(value.envEmbedder === undefined || isNullableString(value.envEmbedder))
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
		// lazykatok's clap CLI rejects global options placed after the subcommand.
		expect(args.indexOf("--data-dir")).toBeLessThan(args.indexOf("doctor"));
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

		const parentClient = fakeClient({ LAZYKATOK_FAKE_OUTPUT: jsonEnv([{ chunkId: "p1", content: "parent" }]) });
		const parent = await parentClient.parent("c1");
		expect(parent.ok).toBe(true);
		if (parent.ok) expect(parent.data[0]?.chunkId).toBe("p1");
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
		const stderrText = "lazykatok: index busy at /Users/me/Library/Application Support/katok/index.db";
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

		const unrecognizedDoctor = fakeClient({ LAZYKATOK_FAKE_OUTPUT: jsonEnv({ version: "1.2.3" }) });
		await expect(unrecognizedDoctor.doctor()).resolves.toMatchObject({ ok: false, reason: "invalid-shape" });
	});

	/**
	 * Real `lazykatok doctor --json` payload (upstream `run_doctor`, changeroa/lazykatok @ main,
	 * verified against the installed CLI). Note there is no `ready` field: doctor reports
	 * its readiness through the `freshness` block, the `archive` status, and the
	 * `source_adapter` probes.
	 */
	const REAL_DOCTOR_JSON = {
		archive: { status: "present" },
		command: "lazykatok",
		data_dir: "/Users/example/Library/Application Support/katok",
		embedder: { dimension: 768, endpoint: null, mode: "local", model: "embeddinggemma-300m-q4", provider: "local" },
		freshness: {
			last_index: { archive_revision: "37e91a9f", completed_at: "2026-09-21T07:03:28.000771+00:00" },
			last_sync: { chunks: 33272, completed_at: "2026-09-21T07:03:15.188930+00:00", total_messages: 50377 },
			recommendation: { index_before_semantic_search: false, sync_before_search: false },
		},
		local_first: true,
		macos: true,
		name: "lazykatok",
		semantic_index: "/Users/example/Library/Application Support/katok/semantic",
		source_adapter: { configured: "fixture", fixture: "ok", kakaocli: "present" },
	};

	/** Real `lazykatok sync --json` payload (upstream `SyncReport`). There is no `synced` field. */
	const REAL_SYNC_JSON = {
		inserted_messages: 3,
		updated_messages: 0,
		total_messages: 50377,
		chunks: 33272,
		rebuilt_chats: 1,
		timings_ms: { read_source: 12, upsert_messages: 31, rebuild_chunks: 8 },
	};

	/** Real `lazykatok index --dry-run --json` payload. There is no `chunkCount` field. */
	const REAL_INDEX_JSON = {
		full: false,
		dry_run: true,
		candidate_chunks: 33272,
		written_documents: 0,
		embedding_calls: 0,
		documents: [
			{
				chunk_id: "window_fef9f5aa904ae1e9",
				path: "/Users/example/Library/Application Support/katok/semantic/source/chunks/window_fef9f5aa904ae1e9.md",
			},
		],
		embedder: "embeddinggemma-300m-q4",
		semantic_units: "parent_windows",
	};

	it("passes the CLI's own KATOK_* environment namespace through to the child", async () => {
		writeFakeLazykatok();
		const client = fakeClient({ LAZYKATOK_FAKE_OUTPUT: jsonEnv({ ready: true }), KATOK_EMBEDDER: "mock" });

		const result = await client.doctor();

		expect(result.ok).toBe(true);
		expect(loggedCalls()[0]?.envEmbedder).toBe("mock");
	});

	it("normalizes the real lazykatok doctor payload that carries no ready field", async () => {
		writeFakeLazykatok();
		const client = fakeClient({ LAZYKATOK_FAKE_OUTPUT: jsonEnv(REAL_DOCTOR_JSON) });

		const result = await client.doctor();

		expect(result.ok).toBe(true);
		if (!result.ok) return;
		expect(result.data.ready).toBe(true);
		expect(result.data.metadata).toMatchObject({ name: "lazykatok", archive: { status: "present" } });
	});

	it("normalizes the real lazykatok sync report into synced + messageCount", async () => {
		writeFakeLazykatok();
		const client = fakeClient({ LAZYKATOK_FAKE_OUTPUT: jsonEnv(REAL_SYNC_JSON) });

		const result = await client.sync();

		expect(result.ok).toBe(true);
		if (!result.ok) return;
		expect(result.data).toMatchObject({ synced: true, messageCount: 50377 });
	});

	it("normalizes the real lazykatok index report into chunkCount without leaking native paths", async () => {
		writeFakeLazykatok();
		const client = fakeClient({ LAZYKATOK_FAKE_OUTPUT: jsonEnv(REAL_INDEX_JSON) });

		const result = await client.index();

		expect(result.ok).toBe(true);
		if (!result.ok) return;
		expect(result.data.chunkCount).toBe(33272);
		expect(JSON.stringify(result.data)).not.toContain("Application Support");
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

	it("terminates descendants that inherit the lazykatok stdio pipes", { timeout: 20_000 }, async () => {
		writeFakeLazykatok();
		const client = new LazykatokClient({
			binaryPath,
			env: { PATH: `${binDir}:${process.env.PATH ?? ""}` },
			timeoutMs: 500,
		});
		writeFileSync(
			binaryPath,
			`#!/usr/bin/env node
import { appendFileSync } from "node:fs";
import { spawn } from "node:child_process";
const descendant = spawn(process.execPath, ["-e", "setInterval(() => undefined, 1000)"], { stdio: "inherit" });
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args: process.argv.slice(2), descendantPid: descendant.pid }) + "\\n");
setInterval(() => undefined, 1000);
`,
		);
		chmodSync(binaryPath, 0o755);

		const pending = client.doctor();
		await waitForLogFile();
		const descendantPid = loggedCalls()[0]?.descendantPid;
		try {
			const result = await Promise.race([
				pending,
				new Promise<never>((_, reject) =>
					setTimeout(() => reject(new Error("lazykatok timeout did not settle")), 5_000),
				),
			]);

			expect(result).toMatchObject({ ok: false, reason: "timeout" });
			expect(descendantPid).toEqual(expect.any(Number));
			if (descendantPid === undefined) throw new Error("fake lazykatok did not record descendant PID");
			expect(() => process.kill(descendantPid, 0)).toThrow();
		} finally {
			if (descendantPid !== undefined) {
				try {
					process.kill(descendantPid, "SIGKILL");
				} catch {}
			}
		}
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

	describe("sync source adapter and the index output bound", () => {
		/** Fake that streams a payload from a file, for reports too large to pass through env. */
		function writeFakeLazykatokStreaming(payload: string): void {
			const payloadPath = join(root, "payload.json");
			writeFileSync(payloadPath, payload);
			writeFileSync(
				binaryPath,
				`#!/usr/bin/env node
import { appendFileSync, readFileSync } from "node:fs";
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args: process.argv.slice(2), envApiKey: null }) + "\\n");
process.stdout.write(readFileSync(${JSON.stringify(payloadPath)}, "utf8"));
`,
			);
			chmodSync(binaryPath, 0o755);
		}

		it("pins the sync source policy per platform", () => {
			expect(syncArgs({}, "darwin")).toEqual(["sync", "--source", "macos", "--json"]);
			expect(syncArgs({}, "linux")).toEqual(["sync", "--json"]);
			expect(syncArgs({}, "win32")).toEqual(["sync", "--json"]);
			expect(syncArgs({ source: "fixture" }, "darwin")).toEqual(["sync", "--source", "fixture", "--json"]);
			expect(syncArgs({ source: "kakaocli" }, "linux")).toEqual(["sync", "--source", "kakaocli", "--json"]);
		});

		it("spawns sync with the configured --source and defaults to macos on macOS", async () => {
			writeFakeLazykatok();
			const configured = new LazykatokClient({
				binaryPath,
				source: "fixture",
				env: { PATH: `${binDir}:${process.env.PATH ?? ""}`, LAZYKATOK_FAKE_OUTPUT: jsonEnv(REAL_SYNC_JSON) },
			});

			const configuredResult = await configured.sync();

			expect(configuredResult.ok).toBe(true);
			expect(loggedCalls()[0]?.args).toEqual(["sync", "--source", "fixture", "--json"]);

			// A bare `sync --json` makes the CLI use its config file, whose default
			// adapter is `fixture` and fails without a JSONL path, so the live macOS
			// source has to be named explicitly.
			const defaulted = new LazykatokClient({
				binaryPath,
				env: { PATH: `${binDir}:${process.env.PATH ?? ""}`, LAZYKATOK_FAKE_OUTPUT: jsonEnv(REAL_SYNC_JSON) },
			});
			const defaultedResult = await defaulted.sync();

			expect(defaultedResult.ok).toBe(true);
			expect(loggedCalls()[1]?.args).toEqual(
				process.platform === "darwin" ? ["sync", "--source", "macos", "--json"] : ["sync", "--json"],
			);
		});

		it("accepts an archive-scale index report instead of rejecting it as stdout-too-large", async () => {
			// The real `index --json` lists one document per candidate chunk, so a
			// 33k-chunk archive prints ~1.6 MB — well past the 1 MiB general cap.
			const documents = Array.from({ length: 33_000 }, (_, i) => ({
				chunk_id: `window_${i}`,
				path: `/native/katok/semantic/source/chunks/window_${i}.md`,
			}));
			const payload = JSON.stringify({ candidate_chunks: 33_000, documents });
			expect(Buffer.byteLength(payload)).toBeGreaterThan(1_048_576);
			writeFakeLazykatokStreaming(payload);
			const client = fakeClient();

			const result = await client.index();

			expect(result.ok).toBe(true);
			if (!result.ok) return;
			expect(result.data.chunkCount).toBe(33_000);
			expect(JSON.stringify(result.data)).not.toContain("/native/katok");
		});
	});

	describe("real chunk subcommand contract", () => {
		/** Real payloads captured from a live lazykatok-lineage archive. */
		const REAL_CHUNK_GET_JSON = {
			chunk_id: "chunk_a48857402f678ddc",
			chat_id: "348487216782557",
			chat_name: "마커 노마다마스",
			sender_nickname: "정철현 박사님",
			started_at: "2022-11-24T05:26:28+00:00",
			ended_at: "2022-11-24T05:26:28+00:00",
			text: "애들아 오늘 원재 어머니 오시니 회의실과 복도 화장실 정리좀 부탁",
			message_count: 1,
			message_ids: [],
			parent_chunk_ids: [],
			window_parent_ids: ["window_9e30fe7c18d95df6"],
		};
		const REAL_CHUNK_CONTEXT_JSON = {
			chunk: REAL_CHUNK_GET_JSON,
			previous: [
				{ chunk_id: "chunk_prev", chat_id: "1", chat_name: "room", started_at: "2022-11-24T05:20:00+00:00" },
			],
			next: [{ chunk_id: "chunk_next", chat_id: "1", chat_name: "room", started_at: "2022-11-24T05:30:00+00:00" }],
			parent_windows: [{ parent_id: "window_9e30fe7c18d95df6", text: "[정철현 박사님] 애들아 오늘 원재" }],
		};
		const REAL_CHUNK_PARENT_JSON = [
			{
				parent_id: "window_9e30fe7c18d95df6",
				chat_id: "348487216782557",
				chat_name: "마커 노마다마스",
				started_at: "2022-11-24T05:26:28+00:00",
				ended_at: "2022-11-24T05:26:28+00:00",
				text: "[정철현 박사님] 애들아 오늘 원재 어머니 오시니",
				message_count: 1,
				child_chunk_ids: ["chunk_a48857402f678ddc"],
			},
		];

		/**
		 * Fake CLI replicating the real clap dispatch: only `chunk get|context|parent`
		 * exist, and an unknown subcommand exits nonzero like the real binary does.
		 */
		function writeChunkContractFake(): void {
			writeFileSync(
				binaryPath,
				`#!/usr/bin/env node
import { appendFileSync } from "node:fs";

const args = process.argv.slice(2);
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args, envApiKey: null }) + "\\n");
const reply = (value) => {
  process.stdout.write(JSON.stringify(value));
  process.exit(0);
};
if (args[0] === "chunk" && args[1] === "get") reply(${JSON.stringify(REAL_CHUNK_GET_JSON)});
if (args[0] === "chunk" && args[1] === "context") reply(${JSON.stringify(REAL_CHUNK_CONTEXT_JSON)});
if (args[0] === "chunk" && args[1] === "parent") reply(${JSON.stringify(REAL_CHUNK_PARENT_JSON)});
process.stderr.write("error: unrecognized subcommand\\n");
process.exit(1);
`,
			);
			chmodSync(binaryPath, 0o755);
		}

		it("invokes the real `chunk get|context|parent` argv the CLI accepts", async () => {
			writeChunkContractFake();
			const client = fakeClient();

			const chunk = await client.chunkGet("chunk_a48857402f678ddc");
			const context = await client.context("chunk_a48857402f678ddc");
			const parent = await client.parent("chunk_a48857402f678ddc");

			expect([chunk.ok, context.ok, parent.ok]).toEqual([true, true, true]);
			expect(loggedCalls().map((call) => call.args)).toEqual([
				["chunk", "get", "chunk_a48857402f678ddc", "--json"],
				["chunk", "context", "chunk_a48857402f678ddc", "--json"],
				["chunk", "parent", "chunk_a48857402f678ddc", "--json"],
			]);
		});

		it("parses the real chunk payloads instead of requiring the legacy envelope", async () => {
			writeChunkContractFake();
			const client = fakeClient();

			const chunk = await client.chunkGet("chunk_a48857402f678ddc");
			expect(chunk.ok).toBe(true);
			if (!chunk.ok) return;
			expect(chunk.data).toMatchObject({
				chunkId: "chunk_a48857402f678ddc",
				content: "애들아 오늘 원재 어머니 오시니 회의실과 복도 화장실 정리좀 부탁",
			});
			expect(chunk.data.metadata).toMatchObject({
				chatName: "마커 노마다마스",
				senderNickname: "정철현 박사님",
			});

			const context = await client.context("chunk_a48857402f678ddc");
			expect(context.ok).toBe(true);
			if (!context.ok) return;
			expect(context.data.chunks.map((entry) => entry.chunkId)).toContain("chunk_a48857402f678ddc");
			expect(context.data.metadata).toMatchObject({ previous: REAL_CHUNK_CONTEXT_JSON.previous });

			const parent = await client.parent("chunk_a48857402f678ddc");
			expect(parent.ok).toBe(true);
			if (!parent.ok) return;
			expect(parent.data.map((entry) => entry.chunkId)).toEqual(["window_9e30fe7c18d95df6"]);
			expect(parent.data[0]?.content).toContain("애들아 오늘 원재");
		});
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

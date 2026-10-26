import { execFileSync } from "node:child_process";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MailcrawlClient, MailcrawlSkill } from "../../../src/datasource/skills/mailcrawl/index.ts";

let root: string;
let binaryPath: string;
let logPath: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-mailcrawl-"));
	mkdirSync(join(root, "bin"), { recursive: true });
	binaryPath = join(root, "bin", "mailcrawl");
	logPath = join(root, "calls.jsonl");
});
afterEach(() => rmSync(root, { recursive: true, force: true }));

function writeFake(output: string, delayMs = 0): void {
	writeFileSync(
		binaryPath,
		`#!/usr/bin/env node
import { appendFileSync } from "node:fs";
if (${delayMs} > 0) await new Promise((resolve) => setTimeout(resolve, ${delayMs}));
const embeddingEnv = Object.fromEntries(
  Object.entries(process.env).filter(([key]) => key.startsWith("MAILCRAWL_")),
);
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({
  args: process.argv.slice(2),
  dataDir: process.env.MAILCRAWL_DATA_DIR ?? null,
  openai: process.env.OPENAI_API_KEY ?? null,
  embeddingEnv
}) + "\\n");
process.stdout.write(${JSON.stringify(output)});
`,
	);
	chmodSync(binaryPath, 0o755);
}

function calls(): readonly {
	readonly args: readonly string[];
	readonly dataDir: string | null;
	readonly openai: string | null;
	readonly embeddingEnv: Record<string, string>;
}[] {
	if (!existsSync(logPath)) return [];
	return readFileSync(logPath, "utf8")
		.trim()
		.split("\n")
		.map((line) => JSON.parse(line));
}

describe("MailcrawlClient", () => {
	it("runs sync and search through mailcrawl's native store by default", async () => {
		writeFake(JSON.stringify({ added: 2, updated: 1, unchanged: 4, chunksAdded: 3, archiveRevision: "r1" }));
		const client = new MailcrawlClient({
			binaryPath,
			account: "personal",
			mailbox: "INBOX",
			env: { OPENAI_API_KEY: "must-not-leak" },
		});

		expect(await client.sync()).toMatchObject({ ok: true, data: { messages: 2, chunksAdded: 3 } });

		writeFake(
			JSON.stringify([
				{
					chunkId: "msg-1:latest:0",
					messageId: "msg-1",
					threadId: "thread-1",
					accountId: "personal",
					mailbox: "INBOX",
					subject: "Contract renewal",
					from: "legal@example.com",
					to: ["me@example.com"],
					date: "2026-08-30T10:00:00Z",
					snippet: "Renewal is approved.",
					score: 0.9,
				},
			]),
		);
		expect(await client.search("hybrid", "renewal", { topK: 5 })).toMatchObject({
			ok: true,
			hits: [{ chunkId: "msg-1:latest:0", messageId: "msg-1" }],
		});

		const recorded = calls();
		expect(recorded[0]?.args).toEqual(["sync", "--json", "--account", "personal", "--mailbox", "INBOX"]);
		expect(recorded[1]?.args).toEqual([
			"search",
			"--mode",
			"hybrid",
			"--account",
			"personal",
			"--mailbox",
			"INBOX",
			"--limit",
			"5",
			"--json",
			"renewal",
		]);
		expect(recorded.every((call) => call.dataDir === null)).toBe(true);
		expect(recorded.every((call) => call.openai === null)).toBe(true);
	});
	it("returns bounded failure results without throwing", async () => {
		const client = new MailcrawlClient({ binaryPath: join(root, "missing") });
		expect(await client.search("bm25", "query")).toMatchObject({ ok: false, reason: "binary-missing" });
	});

	it("preserves CLI stderr verbatim on failure, paths included", async () => {
		const stderrText = "mailcrawl: archive locked at /Users/me/Library/Mail/mailcrawl/archive.db";
		writeFileSync(
			binaryPath,
			`#!/usr/bin/env node
process.stderr.write(${JSON.stringify(stderrText)});
process.exit(3);
`,
		);
		chmodSync(binaryPath, 0o755);
		const client = new MailcrawlClient({ binaryPath });

		const result = await client.search("bm25", "renewal");

		expect(result.ok).toBe(false);
		if (result.ok) return;
		expect(result.stderr).toBe(stderrText);
		expect(result.stderr).not.toContain("suppressed");
	});

	it("applies dataDir when used without a workspace", async () => {
		writeFake(JSON.stringify([]));
		const dataDir = join(root, "custom-data");
		const client = new MailcrawlClient({ binaryPath, dataDir });

		await client.search("bm25", "query");

		expect(calls()[0]?.dataDir).toBe(dataDir);
		if (process.platform !== "win32") expect(statSync(dataDir).mode & 0o777).toBe(0o700);
	});

	it("rejects malformed successful JSON responses", async () => {
		writeFake("{}");
		const client = new MailcrawlClient({ binaryPath });

		expect(await client.sync()).toMatchObject({ ok: false, reason: "invalid-output" });
		expect(await client.index()).toMatchObject({ ok: false, reason: "invalid-output" });

		writeFake(JSON.stringify([{}]));
		expect(await client.search("bm25", "query")).toMatchObject({ ok: false, reason: "invalid-output" });
	});

	it("forwards native fixture sync flags", async () => {
		const fixture = join(root, "messages.json");
		writeFake(JSON.stringify({ added: 1, unchanged: 0, chunksAdded: 1 }));
		const client = new MailcrawlClient({
			binaryPath,
			dataDir: join(root, "data"),
			source: "fixture",
			fixture,
		});

		expect(await client.sync()).toMatchObject({ ok: true, data: { messages: 1, chunksAdded: 1 } });
		expect(calls()[0]?.args).toEqual(["sync", "--json", "--source", "fixture", "--fixture", fixture]);
	});

	it("accepts 0.1.4 reused-only semantic index JSON", async () => {
		writeFake(JSON.stringify({ embedded: 0, reused: 3, generation: "gen-1" }));
		const client = new MailcrawlClient({ binaryPath, dataDir: join(root, "data") });

		expect(await client.index()).toMatchObject({
			ok: true,
			data: { embedded: 0, reused: 3, generation: "gen-1" },
		});
	});

	it("accepts the 0.2.0 LanceDB index report and keeps the native embedder identity", async () => {
		writeFake(
			JSON.stringify({
				embedded: 2,
				reused: 3,
				archiveRevision: "rev-2",
				rebuilt: false,
				embedder: "native:Qwen/Qwen3-Embedding-0.6B:1024",
			}),
		);
		const client = new MailcrawlClient({ binaryPath, dataDir: join(root, "data") });

		expect(await client.index()).toMatchObject({
			ok: true,
			data: {
				embedded: 2,
				reused: 3,
				archiveRevision: "rev-2",
				rebuilt: false,
				embedder: "native:Qwen/Qwen3-Embedding-0.6B:1024",
			},
		});
		expect(calls()[0]?.args).toEqual(["index", "--json"]);
	});

	it("lets a long 0.2.0 native index run outlive the interactive search timeout", async () => {
		writeFake(JSON.stringify({ embedded: 1, reused: 0, embedder: "native:Qwen/Qwen3-Embedding-0.6B:1024" }), 900);
		const client = new MailcrawlClient({ binaryPath, timeoutMs: 200, indexTimeoutMs: 5_000 });

		expect(await client.index()).toMatchObject({ ok: true });
		expect(await client.search("bm25", "refund")).toMatchObject({ ok: false, reason: "timeout" });
	});

	it("rejects a non-loopback 0.2.0 loopback-http embedding endpoint before spawning", async () => {
		writeFake(JSON.stringify({ embedded: 1, reused: 0, embedder: "loopback-http:qwen:1024" }));
		const client = new MailcrawlClient({
			binaryPath,
			env: {
				MAILCRAWL_EMBEDDER_PROVIDER: "loopback-http",
				MAILCRAWL_EMBED_URL: "https://embeddings.example.com",
				MAILCRAWL_EMBED_MODEL: "qwen3-embedding-0.6b",
				MAILCRAWL_EMBED_DIM: "1024",
			},
		});

		expect(await client.index()).toMatchObject({ ok: false, reason: "remote-embedding-rejected" });
		expect(calls()).toEqual([]);
	});

	it("forwards a loopback 0.2.0 embedding endpoint so the shared runtime stays local", async () => {
		writeFake(JSON.stringify({ embedded: 1, reused: 0, embedder: "loopback-http:qwen3-embedding-0.6b:1024" }));
		const client = new MailcrawlClient({
			binaryPath,
			env: {
				MAILCRAWL_EMBEDDER_PROVIDER: "loopback-http",
				MAILCRAWL_EMBED_URL: "http://127.0.0.1:18080",
				MAILCRAWL_EMBED_MODEL: "qwen3-embedding-0.6b",
				MAILCRAWL_EMBED_DIM: "1024",
				MAILCRAWL_EMBED_BATCH_SIZE: "4",
			},
		});

		expect(await client.index()).toMatchObject({ ok: true });
		expect(calls()[0]?.embeddingEnv).toMatchObject({
			MAILCRAWL_EMBEDDER_PROVIDER: "loopback-http",
			MAILCRAWL_EMBED_URL: "http://127.0.0.1:18080",
			MAILCRAWL_EMBED_MODEL: "qwen3-embedding-0.6b",
			MAILCRAWL_EMBED_DIM: "1024",
			MAILCRAWL_EMBED_BATCH_SIZE: "4",
		});
	});

	it("rejects remote embedding configuration before spawning", async () => {
		const client = new MailcrawlClient({
			binaryPath,
			env: { MAILCRAWL_EMBEDDER: "https://embeddings.example.com" },
		});

		expect(await client.index()).toMatchObject({ ok: false, reason: "remote-embedding-rejected" });
		expect(calls()).toEqual([]);
	});

	it("bounds UTF-8 process output by bytes", async () => {
		writeFake(JSON.stringify("😀".repeat(32)));
		const client = new MailcrawlClient({ binaryPath, maxBufferBytes: 8 });

		expect(await client.search("bm25", "query")).toMatchObject({ ok: false, reason: "stdout-too-large" });
	});
});

describe("MailcrawlClient process lifecycle", () => {
	/** kill(pid, 0) succeeds for zombies until init reaps them; those are not running. */
	function isRunning(pid: number): boolean {
		try {
			process.kill(pid, 0);
		} catch {
			return false;
		}
		try {
			return !/^Z/.test(execFileSync("ps", ["-o", "stat=", "-p", String(pid)], { encoding: "utf8" }).trim());
		} catch {
			return false;
		}
	}

	function writeScript(body: string): void {
		writeFileSync(binaryPath, `#!/usr/bin/env node\n${body}`);
		chmodSync(binaryPath, 0o755);
	}

	const spawnedPids: number[] = [];
	afterEach(() => {
		for (const pid of spawnedPids.splice(0)) {
			try {
				process.kill(pid, "SIGKILL");
			} catch {}
		}
	});

	// Real wall-clock bound on purpose: the failure under test is a promise that never settles,
	// which fake timers cannot observe. The tsconfig lib predates Promise.withResolvers.
	async function settleWithin<T>(promise: Promise<T>, ms: number, what: string): Promise<T> {
		let timer: ReturnType<typeof setTimeout> | undefined;
		const bound = new Promise<never>((_, reject) => {
			timer = setTimeout(() => reject(new Error(`${what} stayed pending for ${ms}ms`)), ms);
		});
		try {
			return await Promise.race([promise, bound]);
		} finally {
			clearTimeout(timer);
		}
	}

	it.skipIf(process.platform === "win32")(
		"settles and reaps a descendant that keeps the inherited stdio open after the direct child exits",
		{ timeout: 15_000 },
		async () => {
			writeScript(`import { spawn } from "node:child_process";
const descendant = spawn(process.execPath, ["-e", "setInterval(() => undefined, 1000)"], { stdio: "inherit" });
process.stdout.write(JSON.stringify({ added: 1, grandchildPid: descendant.pid }));
process.exit(0);
`);
			const client = new MailcrawlClient({ binaryPath, timeoutMs: 30_000, indexTimeoutMs: 30_000 });

			const result = await settleWithin(client.sync(), 4_000, "mailcrawl sync");

			expect(result.ok).toBe(true);
			const pid = Number(/"grandchildPid":(\d+)/.exec(result.stdout)?.[1]);
			expect(pid).toBeGreaterThan(0);
			spawnedPids.push(pid);
			await vi.waitFor(() => expect(isRunning(pid)).toBe(false), { timeout: 3_000 });
		},
	);

	it.skipIf(process.platform === "win32")(
		"escalates past a SIGTERM-ignoring child and reaps its descendant on timeout",
		{ timeout: 15_000 },
		async () => {
			const pidFile = join(root, "descendant.pid");
			writeScript(`import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";
process.on("SIGTERM", () => undefined);
const descendant = spawn(process.execPath, ["-e", "process.on('SIGTERM', () => undefined); setInterval(() => undefined, 1000)"], { stdio: "inherit" });
writeFileSync(${JSON.stringify(pidFile)}, String(descendant.pid));
setInterval(() => undefined, 1000);
`);
			// The wall must outlive Node startup under load, or the fixture is killed before it records the
			// descendant pid. A fixed 500ms raced that and left the pid file missing intermittently.
			const client = new MailcrawlClient({ binaryPath, timeoutMs: 3_000 });

			const pending = client.search("keyword", "anything");
			await vi.waitFor(() => expect(existsSync(pidFile)).toBe(true), { timeout: 10_000 });
			const pid = Number(readFileSync(pidFile, "utf8"));
			spawnedPids.push(pid);

			expect(await settleWithin(pending, 8_000, "mailcrawl search")).toMatchObject({ ok: false, reason: "timeout" });
			await vi.waitFor(() => expect(isRunning(pid)).toBe(false), { timeout: 3_000 });
		},
	);

	it.skipIf(process.platform === "win32")(
		"still honors abort after an output-cap termination was already requested",
		{ timeout: 15_000 },
		async () => {
			const pidFile = join(root, "descendant.pid");
			writeScript(`import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";
process.on("SIGTERM", () => undefined);
const descendant = spawn(process.execPath, ["-e", "process.on('SIGTERM', () => undefined); setInterval(() => undefined, 1000)"], { stdio: "inherit" });
writeFileSync(${JSON.stringify(pidFile)}, String(descendant.pid));
process.stdout.write("x".repeat(64));
setInterval(() => undefined, 1000);
`);
			const controller = new AbortController();
			const client = new MailcrawlClient({ binaryPath, maxBufferBytes: 8, timeoutMs: 30_000 });

			const pending = client.search("keyword", "anything", { signal: controller.signal });
			await vi.waitFor(() => expect(existsSync(pidFile)).toBe(true), { timeout: 5_000 });
			controller.abort();

			expect(await settleWithin(pending, 5_000, "mailcrawl search")).toMatchObject({ ok: false });
			const pid = Number(readFileSync(pidFile, "utf8"));
			spawnedPids.push(pid);
			await vi.waitFor(() => expect(isRunning(pid)).toBe(false), { timeout: 3_000 });
		},
	);
});

describe("MailcrawlClient default non-interactive budget", () => {
	it("gives sync and index one hour, and search the short interactive default", async () => {
		writeFake(JSON.stringify({ added: 0, updated: 0, unchanged: 1, chunksAdded: 1, archiveRevision: "r1" }));
		const spy = vi.spyOn(globalThis, "setTimeout");
		try {
			const client = new MailcrawlClient({ binaryPath });
			await client.sync();
			const syncDelays = spy.mock.calls
				.map((call) => call[1])
				.filter((ms) => typeof ms === "number" && ms >= 60_000);
			expect(syncDelays).toContain(3_600_000);
			spy.mockClear();
			await client.search("bm25", "refund");
			const searchDelays = spy.mock.calls
				.map((call) => call[1])
				.filter((ms) => typeof ms === "number" && ms >= 60_000);
			expect(searchDelays).toContain(60_000);
			expect(searchDelays).not.toContain(3_600_000);
		} finally {
			spy.mockRestore();
		}
	});
});

describe("MailcrawlSkill", () => {
	it("indexes and exposes BM25, semantic, and hybrid opaque retrieval methods", async () => {
		const client = {
			async sync() {
				return { ok: true as const, data: { messages: 2, chunksAdded: 2 }, stdout: "", stderr: "", code: 0 };
			},
			async index() {
				return { ok: true as const, data: { embedded: 2 }, stdout: "", stderr: "", code: 0 };
			},
			async search() {
				const hits = [
					{
						chunkId: "m1:latest:0",
						messageId: "m1",
						threadId: "t1",
						accountId: "acct",
						mailbox: "INBOX",
						subject: "Policy",
						from: "a@example.com",
						to: [],
						date: "2026-08-30",
						snippet: "Director approval required.",
						score: 1,
						mode: "bm25" as const,
					},
				];
				return { ok: true as const, hits, stdout: "", stderr: "", code: 0 };
			},
		};
		const skill = new MailcrawlSkill({ client, instanceId: "personal" });
		expect(await skill.index()).toMatchObject({ ok: true, chunkCount: 2 });
		expect(skill.retrievalMethods().map((method) => method.describe().name)).toEqual(["mailcrawl-hybrid"]);
		const result = await skill
			.retrievalMethods()[0]
			?.retrieve("approval", { topK: 5, scope: "/mailcrawl/personal/**" });
		expect(result?.[0]?.source).toBe("/mailcrawl/personal/chunks/m1:latest:0");
		expect(skill.skillManifest().content).toContain("mailcrawl");
	});

	it("filters configured account and mailbox at retrieval time", async () => {
		const client = {
			async sync() {
				return { ok: true as const, data: { messages: 1, chunksAdded: 1 }, stdout: "", stderr: "", code: 0 };
			},
			async index() {
				return { ok: true as const, data: { embedded: 1 }, stdout: "", stderr: "", code: 0 };
			},
			async search() {
				return {
					ok: true as const,
					hits: [
						{
							chunkId: "other",
							messageId: "other",
							threadId: "thread",
							accountId: "other",
							mailbox: "INBOX",
							subject: "Other",
							from: "other@example.com",
							to: [],
							date: "2026-08-30",
							snippet: "Other account",
							score: 1,
							mode: "bm25" as const,
						},
						{
							chunkId: "personal",
							messageId: "personal",
							threadId: "thread",
							accountId: "personal",
							mailbox: "INBOX",
							subject: "Personal",
							from: "me@example.com",
							to: [],
							date: "2026-08-30",
							snippet: "Personal account",
							score: 1,
							mode: "bm25" as const,
						},
					],
					stdout: "",
					stderr: "",
					code: 0,
				};
			},
		};
		const skill = new MailcrawlSkill({
			client,
			instanceId: "personal",
			account: "personal",
			mailbox: "INBOX",
		});

		const result = await skill.retrievalMethods()[0]?.retrieve("account", {
			topK: 5,
			scope: "/mailcrawl/personal/**",
		});

		expect(result).toHaveLength(1);
		expect(result?.[0]?.metadata.accountId).toBe("personal");
	});

	it("falls back to BM25 when hybrid search cannot run, and searches hybrid first", async () => {
		const modes: string[] = [];
		const hit = {
			chunkId: "m1:latest:0",
			messageId: "m1",
			threadId: "t1",
			accountId: "acct",
			mailbox: "INBOX",
			subject: "Policy",
			from: "a@example.com",
			to: [],
			date: "2026-08-31",
			snippet: "Director approval required.",
			score: 1,
			mode: "bm25" as const,
		};
		const client = {
			async sync() {
				return { ok: true as const, data: { messages: 1, chunksAdded: 1 }, stdout: "", stderr: "", code: 0 };
			},
			async index() {
				return { ok: true as const, data: { embedded: 0 }, stdout: "", stderr: "", code: 0 };
			},
			async search(mode: string) {
				modes.push(mode);
				if (mode === "hybrid") {
					return {
						ok: false as const,
						reason: "nonzero-exit" as const,
						stdout: "",
						stderr: "semantic index missing",
						code: 1,
					};
				}
				return { ok: true as const, hits: [hit], stdout: "", stderr: "", code: 0 };
			},
		};
		const skill = new MailcrawlSkill({ client, instanceId: "personal" });

		const methods = skill.retrievalMethods();
		const results = await Promise.all(methods.map((method) => method.retrieve("approval", { topK: 5 })));

		expect(modes).toEqual(["hybrid", "bm25"]);
		expect(results[0]?.[0]).toMatchObject({
			source: "/mailcrawl/personal/chunks/m1:latest:0",
			metadata: { fallbackFrom: "mailcrawl-hybrid" },
		});
	});

	it("does not touch BM25 when hybrid search succeeds", async () => {
		const modes: string[] = [];
		const client = {
			async sync() {
				return { ok: true as const, data: { messages: 0, chunksAdded: 0 }, stdout: "", stderr: "", code: 0 };
			},
			async index() {
				return { ok: true as const, data: { embedded: 0 }, stdout: "", stderr: "", code: 0 };
			},
			async search(mode: string) {
				modes.push(mode);
				return { ok: true as const, hits: [], stdout: "", stderr: "", code: 0 };
			},
		};
		const skill = new MailcrawlSkill({ client, instanceId: "personal" });

		await Promise.all(skill.retrievalMethods().map((method) => method.retrieve("approval", { topK: 5 })));

		expect(modes).toEqual(["hybrid"]);
	});

	it("keeps BM25 available and maps sources when the 0.2.0 semantic index is unavailable", async () => {
		const client = {
			async sync() {
				return { ok: true as const, data: { messages: 1, chunksAdded: 1 }, stdout: "", stderr: "", code: 0 };
			},
			async index() {
				return {
					ok: false as const,
					reason: "nonzero-exit" as const,
					stdout: "",
					stderr: "semantic index embedder mismatch: indexed with legacy-onnx, active embedder is native",
					code: 1,
				};
			},
			async search() {
				return {
					ok: true as const,
					hits: [
						{
							chunkId: "m1:latest:0",
							messageId: "m1",
							threadId: "t1",
							accountId: "acct",
							mailbox: "INBOX",
							subject: "Policy",
							from: "a@example.com",
							to: [],
							date: "2026-08-31",
							snippet: "Director approval required.",
							score: 1,
							mode: "bm25" as const,
						},
					],
					stdout: "",
					stderr: "",
					code: 0,
				};
			},
		};
		const skill = new MailcrawlSkill({ client, instanceId: "personal" });

		const indexed = await skill.index();
		expect(indexed.ok).toBe(true);
		expect(indexed.diagnostics?.[0]?.code).toBe("datasource-index-failed");
		expect(indexed.diagnostics?.[0]?.message).toContain("BM25 remains available");

		const retrieved = await skill
			.retrievalMethods()[0]
			?.retrieve("approval", { topK: 5, scope: "/mailcrawl/personal/**" });
		expect(retrieved?.[0]?.source).toBe("/mailcrawl/personal/chunks/m1:latest:0");
	});

	it("rejects unsafe instance identifiers", () => {
		expect(() => new MailcrawlSkill({ instanceId: "../outside" })).toThrow(/safe single path segment/);
	});
});

import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { JikjiOptions } from "../../src/jikji/index.ts";
import { DEFAULT_JIKJI_OPTIONS, JikjiClient } from "../../src/jikji/index.ts";

type LoggedCall = {
	readonly args: readonly string[];
};

const FAKE_CHILD_READY_TIMEOUT_MS = 10_000;

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
	rmSync(root, { recursive: true, force: true, maxRetries: 20, retryDelay: 100 });
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

async function waitForLogFile(): Promise<void> {
	// Poll for a non-empty log: watching for file creation alone races the
	// child's write, so an early abort can kill the child before it flushes.
	const deadline = Date.now() + FAKE_CHILD_READY_TIMEOUT_MS;
	while (Date.now() < deadline) {
		if (existsSync(logPath) && statSync(logPath).size > 0) return;
		await new Promise((resolve) => setTimeout(resolve, 20));
	}
	throw new Error("timed out waiting for fake Jikji log");
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
	return { PATH: `${binDir}${delimiter}${process.env.PATH ?? ""}` };
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
			writeAgentRules: false,
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
		expect(argsOfFirstCall()).toEqual(["prepare", corpusRoot, "--json", "--no-agent-rules"]);
	});

	it("emits a clean wire format for new JikjiClient(DEFAULT_JIKJI_OPTIONS)", async () => {
		writeFakeJikji("console.log(JSON.stringify({ prepared: true }));");
		const client = new JikjiClient({ ...DEFAULT_JIKJI_OPTIONS, env: pathEnv() });

		const result = await client.prepare(corpusRoot);

		expect(result).toMatchObject({ ok: true });
		expect(argsOfFirstCall()).toEqual(["prepare", corpusRoot, "--json", "--no-agent-rules"]);
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
		expect(argsOfFirstCall()).toEqual(["prepare", corpusRoot, "--json", "--no-agent-rules"]);
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
		expect(argsOfFirstCall()).toEqual(["prepare", corpusRoot, "--json", "--no-agent-rules"]);
	});

	it("runs native commands from the managed workspace boundary", async () => {
		writeFakeJikji("console.log(JSON.stringify({ prepared: true }));");
		const client = new JikjiClient({ binaryPath, root });

		const result = await client.prepare(corpusRoot);

		expect(result.ok).toBe(true);
		expect(argsOfFirstCall()).toEqual(["prepare", corpusRoot, "--json", "--no-agent-rules"]);
	});

	it("emits --no-agent-rules by default and suppresses hidden/sensitive/media flags", async () => {
		writeFakeJikji("console.log(JSON.stringify({ prepared: true }));");
		const client = clientWithPath();

		await client.prepare(corpusRoot);

		const args = argsOfFirstCall();
		expect(args).toContain("--no-agent-rules");
		expect(args).not.toContain("--include-hidden");
		expect(args).not.toContain("--include-sensitive");
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

	it("terminates the child when AbortController aborts", { timeout: 20_000 }, async () => {
		writeFakeJikji("setInterval(() => undefined, 1000);\nawait new Promise(() => undefined);");
		const controller = new AbortController();
		const client = new JikjiClient({
			env: pathEnv(),
			timeoutMs: 15_000,
		});

		const pending = client.prepare(corpusRoot, { signal: controller.signal });
		await waitForLogFile();
		controller.abort();
		const result = await pending;

		expect(result).toMatchObject({ ok: false, reason: "aborted" });
		expect(loggedCalls()).toHaveLength(1);
	});

	describe("process-tree lifecycle", () => {
		const SPAWN_STUCK_DESCENDANT = `
const { spawn } = await import("node:child_process");
const { writeFileSync } = await import("node:fs");
const descendant = spawn(process.execPath, ["-e", "setInterval(() => undefined, 1000)"], { stdio: "inherit" });
writeFileSync(PIDFILE, String(descendant.pid));
`;

		function pidFile(): string {
			return join(root, "descendant.pid");
		}

		function stuckDescendantBody(parentBehavior: string): string {
			return `const PIDFILE = ${JSON.stringify(pidFile())};\n${SPAWN_STUCK_DESCENDANT}\n${parentBehavior}`;
		}

		async function descendantPid(): Promise<number> {
			const deadline = Date.now() + FAKE_CHILD_READY_TIMEOUT_MS;
			while (Date.now() < deadline) {
				if (existsSync(pidFile()) && statSync(pidFile()).size > 0) return Number(readFileSync(pidFile(), "utf8"));
				await new Promise((resolve) => setTimeout(resolve, 20));
			}
			throw new Error("timed out waiting for the descendant pid");
		}

		function isAlive(pid: number): boolean {
			try {
				process.kill(pid, 0);
				return true;
			} catch {
				return false;
			}
		}

		async function expectReaped(pid: number): Promise<void> {
			const deadline = Date.now() + 5_000;
			while (isAlive(pid) && Date.now() < deadline) await new Promise((resolve) => setTimeout(resolve, 20));
			expect(isAlive(pid)).toBe(false);
		}

		it.skipIf(process.platform === "win32")(
			"reaps a descendant holding the pipes when the wall expires",
			{ timeout: 20_000 },
			async () => {
				writeFakeJikji(stuckDescendantBody("await new Promise(() => undefined);"));
				const client = new JikjiClient({ env: pathEnv(), timeoutMs: 2_000 });

				const result = await client.prepare(corpusRoot);

				expect(result).toMatchObject({ ok: false, reason: "timeout" });
				await expectReaped(await descendantPid());
			},
		);

		it.skipIf(process.platform === "win32")(
			"does not hang when the direct child exits but a descendant keeps the pipes open",
			{ timeout: 20_000 },
			async () => {
				writeFakeJikji(stuckDescendantBody("process.exit(0);"));
				const client = new JikjiClient({ env: pathEnv(), timeoutMs: 15_000 });

				const result = await client.prepare(corpusRoot);

				expect(result).toMatchObject({ ok: true });
				await expectReaped(await descendantPid());
			},
		);

		it.skipIf(process.platform === "win32")("reaps a descendant on abort", { timeout: 20_000 }, async () => {
			writeFakeJikji(stuckDescendantBody("await new Promise(() => undefined);"));
			const controller = new AbortController();
			const client = new JikjiClient({ env: pathEnv(), timeoutMs: 15_000 });

			const pending = client.prepare(corpusRoot, { signal: controller.signal });
			const pid = await descendantPid();
			controller.abort();

			expect(await pending).toMatchObject({ ok: false, reason: "aborted" });
			await expectReaped(pid);
		});
	});

	describe("prepare budget", () => {
		function populate(files: number): void {
			mkdirSync(corpusRoot, { recursive: true });
			for (let index = 0; index < files; index += 1) writeFileSync(join(corpusRoot, `f${index}.txt`), "x");
		}

		const SLEEP_THEN_OK = "await new Promise((resolve) => setTimeout(resolve, 600));";

		it("keeps a flat wall when prepareTimeoutMsPerFile is 0", async () => {
			populate(50);
			writeFakeJikji(SLEEP_THEN_OK);
			const client = new JikjiClient({ env: pathEnv(), timeoutMs: 200, prepareTimeoutMsPerFile: 0 });

			expect(await client.prepare(corpusRoot)).toMatchObject({ ok: false, reason: "timeout" });
		});

		it("lets a large root finish under a budget that a small root would exceed", async () => {
			writeFakeJikji(SLEEP_THEN_OK);
			const client = new JikjiClient({ env: pathEnv(), timeoutMs: 200, prepareTimeoutMsPerFile: 50 });

			populate(2);
			expect(await client.prepare(corpusRoot)).toMatchObject({ ok: false, reason: "timeout" });

			rmSync(corpusRoot, { recursive: true, force: true });
			populate(40);
			expect(await client.prepare(corpusRoot)).toMatchObject({ ok: true });
		});

		it("defaults the per-file slice to the parse timeout, so no fixed total wall applies", async () => {
			populate(40);
			writeFakeJikji(SLEEP_THEN_OK);
			// 200 ms base + 40 files * 1 s (parseTimeout 1) far exceeds the 600 ms run.
			const client = new JikjiClient({ env: pathEnv(), timeoutMs: 200, parseTimeout: 1 });

			expect(await client.prepare(corpusRoot)).toMatchObject({ ok: true });
		});

		it("lets a per-call timeoutMs override the client budget", async () => {
			writeFakeJikji(SLEEP_THEN_OK);
			const client = new JikjiClient({ env: pathEnv(), timeoutMs: 15_000 });

			expect(await client.prepare(corpusRoot, { timeoutMs: 200 })).toMatchObject({ ok: false, reason: "timeout" });
		});

		it("passes a per-call parseTimeout to --parse-timeout, overriding the client value", async () => {
			writeFakeJikji("console.log('{}');");
			const client = new JikjiClient({ env: pathEnv(), parseTimeout: 5 });

			await client.prepare(corpusRoot, { parseTimeout: 120 });
			await client.prepare(corpusRoot);

			const [first, second] = loggedCalls();
			expect(first?.args).toEqual(expect.arrayContaining(["--parse-timeout", "120"]));
			expect(first?.args).not.toContain("5");
			expect(second?.args).toEqual(expect.arrayContaining(["--parse-timeout", "5"]));
		});
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

	it("writeAgentRules:true omits --no-agent-rules", async () => {
		writeFakeJikji("console.log(JSON.stringify({ prepared: true }));");
		const client = new JikjiClient({ env: pathEnv(), writeAgentRules: true });

		await client.prepare(corpusRoot);

		const args = argsOfFirstCall();
		expect(args).not.toContain("--no-agent-rules");
		expect(args).toEqual(["prepare", corpusRoot, "--json"]);
	});

	it("writeAgentRules:false still emits --no-agent-rules", async () => {
		writeFakeJikji("console.log(JSON.stringify({ prepared: true }));");
		const client = new JikjiClient({ env: pathEnv(), writeAgentRules: false });

		await client.prepare(corpusRoot);

		const args = argsOfFirstCall();
		expect(args).toContain("--no-agent-rules");
	});
});

const ANSWER_PACK = {
	answer_paths: ["/repo/src/a.ts"],
	paths: ["/repo/src/a.ts", "/repo/src/b.ts"],
	candidates: [
		{ path: "/repo/src/a.ts", next_read: "cache", label: "A", score: 0.9 },
		{ path: "/repo/src/b.ts", next_read: "wiki" },
	],
	evidence_pack: [{ path: "/repo/src/a.ts", next_read: "cache" }],
	handoff_action: "direct_use",
	tool_call_policy: {
		stop_after_find: true,
		forbidden_tools: ["bash"],
		allowed_followups: ["jikji_find"],
	},
	agent_should_not_rerank: true,
};

describe("JikjiClient.find", () => {
	it("emits only ['find', root, query, '--json', '--no-background-refresh'] by default", async () => {
		writeFakeJikji(`console.log(${JSON.stringify(JSON.stringify(ANSWER_PACK))});`);
		const client = clientWithPath();

		await client.find(corpusRoot, "how does X work");

		expect(argsOfFirstCall()).toEqual(["find", corpusRoot, "how does X work", "--json", "--no-background-refresh"]);
	});

	// `jikji find` spawns a detached `jikji prepare <root>` (without
	// --no-agent-rules) when the index is changed or stale. That indexes during
	// a query and rewrites AGENTS.md/CLAUDE.md/.cursorrules in the user's root.
	it("always opts out of upstream's detached background refresh, even with every find option set", async () => {
		writeFakeJikji(`console.log(${JSON.stringify(JSON.stringify(ANSWER_PACK))});`);
		const client = clientWithPath();

		await client.find(corpusRoot, "q", {
			topK: 3,
			first: true,
			fresh: true,
			autoPrepare: true,
			staleAfterSeconds: 30,
		});

		expect(argsOfFirstCall()).toContain("--no-background-refresh");
	});

	it("emits --top-k only when set", async () => {
		writeFakeJikji(`console.log(${JSON.stringify(JSON.stringify(ANSWER_PACK))});`);
		const client = clientWithPath();

		await client.find(corpusRoot, "q", { topK: 5 });

		const args = argsOfFirstCall();
		expect(args).toContain("--top-k");
		expect(args).toContain("5");
	});

	it("does not emit --top-k when unset", async () => {
		writeFakeJikji(`console.log(${JSON.stringify(JSON.stringify(ANSWER_PACK))});`);
		const client = clientWithPath();

		await client.find(corpusRoot, "q");

		expect(argsOfFirstCall()).not.toContain("--top-k");
	});

	it("emits --first only when set", async () => {
		writeFakeJikji(`console.log(${JSON.stringify(JSON.stringify(ANSWER_PACK))});`);
		const client = clientWithPath();

		await client.find(corpusRoot, "q", { first: true });

		expect(argsOfFirstCall()).toContain("--first");
	});

	it("emits --fresh only when set", async () => {
		writeFakeJikji(`console.log(${JSON.stringify(JSON.stringify(ANSWER_PACK))});`);
		const client = clientWithPath();

		await client.find(corpusRoot, "q", { fresh: true });

		expect(argsOfFirstCall()).toContain("--fresh");
	});

	it("emits --auto-prepare only when set", async () => {
		writeFakeJikji(`console.log(${JSON.stringify(JSON.stringify(ANSWER_PACK))});`);
		const client = clientWithPath();

		await client.find(corpusRoot, "q", { autoPrepare: true });

		expect(argsOfFirstCall()).toContain("--auto-prepare");
	});

	it("does not emit --auto-prepare by default", async () => {
		writeFakeJikji(`console.log(${JSON.stringify(JSON.stringify(ANSWER_PACK))});`);
		const client = clientWithPath();

		await client.find(corpusRoot, "q");

		expect(argsOfFirstCall()).not.toContain("--auto-prepare");
	});

	it("emits --stale-after-seconds only when set", async () => {
		writeFakeJikji(`console.log(${JSON.stringify(JSON.stringify(ANSWER_PACK))});`);
		const client = clientWithPath();

		await client.find(corpusRoot, "q", { staleAfterSeconds: 60 });

		const args = argsOfFirstCall();
		expect(args).toContain("--stale-after-seconds");
		expect(args).toContain("60");
	});

	it("emits all find flags together in the right order", async () => {
		writeFakeJikji(`console.log(${JSON.stringify(JSON.stringify(ANSWER_PACK))});`);
		const client = clientWithPath();

		await client.find(corpusRoot, "query", {
			topK: 3,
			first: true,
			fresh: true,
			autoPrepare: true,
			staleAfterSeconds: 30,
		});

		expect(argsOfFirstCall()).toEqual([
			"find",
			corpusRoot,
			"query",
			"--json",
			"--no-background-refresh",
			"--top-k",
			"3",
			"--first",
			"--fresh",
			"--auto-prepare",
			"--stale-after-seconds",
			"30",
		]);
	});

	it("parses a valid answer-pack into ok:true with right answerPaths", async () => {
		writeFakeJikji(`console.log(${JSON.stringify(JSON.stringify(ANSWER_PACK))});`);
		const client = clientWithPath();

		const result = await client.find(corpusRoot, "q");

		expect(result).toMatchObject({ ok: true });
		if (result.ok) {
			expect(result.answerPack.answerPaths).toEqual(["/repo/src/a.ts"]);
			expect(result.answerPack.candidates).toHaveLength(2);
			expect(result.answerPack.candidates[0]?.nextRead).toBe("cache");
			expect(result.answerPack.handoffAction).toBe("direct_use");
			expect(result.answerPack.toolCallPolicy.stopAfterFind).toBe(true);
			expect(result.answerPack.toolCallPolicy.forbiddenTools).toEqual(["bash"]);
			expect(result.answerPack.agentShouldNotRerank).toBe(true);
			expect(result.code).toBe(0);
		}
	});

	it("returns ok:false reason 'bad-answer-pack' on malformed stdout", async () => {
		writeFakeJikji('process.stdout.write("this is not json");');
		const client = clientWithPath();

		const result = await client.find(corpusRoot, "q");

		expect(result).toMatchObject({ ok: false, reason: "bad-answer-pack" });
	});

	it("returns ok:false reason 'bad-answer-pack' on JSON missing required fields", async () => {
		writeFakeJikji('console.log(JSON.stringify({ hello: "world" }));');
		const client = clientWithPath();

		const result = await client.find(corpusRoot, "q");

		expect(result).toMatchObject({ ok: false, reason: "bad-answer-pack" });
	});

	it("returns ok:false reason 'nonzero-exit' on nonzero exit", async () => {
		writeFakeJikji("console.error('index not prepared');", 2);
		const client = clientWithPath();

		const result = await client.find(corpusRoot, "q");

		expect(result).toMatchObject({ ok: false, reason: "nonzero-exit" });
		if (!result.ok) {
			expect(result.code).toBe(2);
		}
	});

	it("returns ok:false reason 'spawn-error' when binary is missing", async () => {
		const client = new JikjiClient({ binaryPath: join(root, "does-not-exist"), env: pathEnv() });

		const result = await client.find(corpusRoot, "q");

		expect(result).toMatchObject({ ok: false, reason: "spawn-error" });
	});

	it("returns ok:false reason 'timeout' without throwing", async () => {
		writeFakeJikji("setInterval(() => undefined, 1000);\nawait new Promise(() => undefined);");
		const client = new JikjiClient({ env: pathEnv(), timeoutMs: 50 });

		const result = await client.find(corpusRoot, "q");

		expect(result).toMatchObject({ ok: false, reason: "timeout" });
	});

	it("terminates the child when AbortController aborts", { timeout: 20_000 }, async () => {
		writeFakeJikji("setInterval(() => undefined, 1000);\nawait new Promise(() => undefined);");
		const controller = new AbortController();
		const client = new JikjiClient({ env: pathEnv(), timeoutMs: 15_000 });

		const pending = client.find(corpusRoot, "q", { signal: controller.signal });
		await waitForLogFile();
		controller.abort();
		const result = await pending;

		expect(result).toMatchObject({ ok: false, reason: "aborted" });
		expect(loggedCalls()).toHaveLength(1);
	});

	it("canonicalizes a symlinked root before prepare and find so Jikji's SQLite root lookup matches", async () => {
		const realCorpus = join(root, "real-corpus");
		const symlinkCorpus = join(root, "link-corpus");
		mkdirSync(realCorpus, { recursive: true });
		const { realpathSync, symlinkSync } = await import("node:fs");
		symlinkSync(realCorpus, symlinkCorpus, "dir");
		const canonical = realpathSync(realCorpus);
		writeFakeJikji(
			`if (args[0] === "prepare") { console.log(JSON.stringify({ prepared: true })); } else { console.log(JSON.stringify({ answer_paths: [], paths: [], candidates: [], evidence_pack: [], handoff_action: "direct_use", tool_call_policy: { stop_after_find: false, forbidden_tools: [], allowed_followups: [] }, agent_should_not_rerank: false })); }`,
		);
		const client = clientWithPath();

		await client.prepare(symlinkCorpus);
		const found = await client.find(symlinkCorpus, "query");

		expect(found.ok).toBe(true);
		const recorded = loggedCalls();
		expect(recorded[0]?.args[1]).toBe(canonical);
		expect(recorded[1]?.args[1]).toBe(canonical);
	});
});

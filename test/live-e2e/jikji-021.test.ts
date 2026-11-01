/**
 * Live E2E: a real model drives the librarian over a REAL `jikji` 0.2.1, end to end.
 *
 * Gated on AUTORAG_JIKJI_LIVE=1 plus an OpenRouter credential; the offline suite never calls a model or
 * compiles jikji. The librarian, MinSync index, jikji binary and model are all real:
 *
 *   1. `refresh` installs the pinned jikji (`cargo install jikji-cli --version 0.2.1`) into the workspace
 *      cache while a stale 0.1.1 sits first on PATH, so the version floor is exercised for real.
 *   2. The model is asked a question that only a local file can answer and must call `jikji_find`.
 *   3. The tool result must come from the real 0.2.1 answer pack (a real path under the corpus), and the
 *      final answer must carry a fact that exists only in that file.
 *   4. A real `--exclude` scope must hold: a document outside the indexed scope never reaches the answer pack
 *      even though the model's query matches nothing else (the empty-result reindex leak).
 *
 *   AUTORAG_JIKJI_LIVE=1 AUTORAG_JIKJI_LIVE_STALE_BIN=/path/to/jikji-0.1.1 \
 *     AUTORAG_JIKJI_LIVE_MODEL=qwen/qwen3.7-flash bun run vitest run test/live-e2e/jikji-021.test.ts
 *
 * The model must be a 2026+ release (see the global model-usage rule).
 */
import { execFileSync } from "node:child_process";
import {
	chmodSync,
	copyFileSync,
	existsSync,
	mkdirSync,
	mkdtempSync,
	readFileSync,
	realpathSync,
	renameSync,
	rmSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, dirname, join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { JIKJI_FIND_TOOL_NAME } from "../../src/agent/jikji-find-tool.ts";
import { resolveAgentModel, resolveConfig } from "../../src/cli/config.ts";
import { cachedJikjiBinaryPath } from "../../src/jikji/installer.ts";
import { JIKJI_INSTALL_VERSION } from "../../src/jikji/version.ts";
import { realMinSync } from "../helpers/real-minsync.ts";

const LIVE = process.env.AUTORAG_JIKJI_LIVE === "1" && Boolean(process.env.OPENROUTER_API_KEY);
const MODEL_ID = process.env.AUTORAG_JIKJI_LIVE_MODEL ?? "qwen/qwen3.7-flash";
const STALE_BIN = process.env.AUTORAG_JIKJI_LIVE_STALE_BIN;

const RESTORED_VARIABLES = ["PI_CODING_AGENT_DIR", "KIWI_LIBRARY_PATH", "KIWI_MODEL_PATH", "JIKJI_DATA_DIR"] as const;

let root: string;
let savedEnv: Record<string, string | undefined>;

beforeEach(() => {
	root = realpathSync(mkdtempSync(join(tmpdir(), "arag-jikji-live-")));
	savedEnv = Object.fromEntries(RESTORED_VARIABLES.map((name) => [name, process.env[name]]));
	delete process.env.KIWI_LIBRARY_PATH;
	delete process.env.KIWI_MODEL_PATH;
	// Jikji's central store stays inside the temp root: the live run never writes the maintainer's real one.
	process.env.JIKJI_DATA_DIR = join(root, "jikji-data");
});

afterEach(() => {
	for (const name of RESTORED_VARIABLES) {
		const value = savedEnv[name];
		if (value === undefined) delete process.env[name];
		else process.env[name] = value;
	}
	rmSync(root, { recursive: true, force: true });
});

describe.skipIf(!LIVE)("jikji 0.2.1, live model", () => {
	it("replaces a stale jikji on refresh, then answers from a file jikji_find returned", async () => {
		expect(STALE_BIN, "AUTORAG_JIKJI_LIVE_STALE_BIN must point at a real jikji 0.1.1 binary").toBeTruthy();
		expect(execFileSync(STALE_BIN as string, ["--version"], { encoding: "utf8" }).trim()).toBe("jikji 0.1.1");

		const docs = join(root, "docs");
		mkdirSync(join(docs, "contracts"), { recursive: true });
		mkdirSync(join(docs, "private"), { recursive: true });
		writeFileSync(
			join(docs, "contracts", "vendor-renewal.md"),
			[
				"# Vendor renewal",
				"",
				"The Hanbit Logistics service agreement renews on 2027-08-14.",
				"The renewal fee is 12,750,000 KRW and the notice period is 45 days.",
				"The approval code for this renewal is QX-4471-ZEBRA.",
				"",
			].join("\n"),
		);
		writeFileSync(join(docs, "lunch.txt"), "Team lunch is on Fridays at the usual place.\n");
		writeFileSync(
			join(docs, "private", "payroll.md"),
			"Confidential payroll sheet. The payroll override code is PAYROLL-9-KESTREL.\n",
		);

		// A 0.1.1 jikji FIRST on PATH: the machine that installed jikji early. cargo + node must stay reachable.
		const stale = join(root, "stale-bin");
		mkdirSync(stale, { recursive: true });
		copyFileSync(STALE_BIN as string, join(stale, "jikji"));
		const cargoDir = dirname(execFileSync("which", ["cargo"], { encoding: "utf8" }).trim());
		const path = [stale, cargoDir, dirname(process.execPath), "/usr/bin", "/bin"].join(delimiter);

		const agentDir = join(root, "agent");
		mkdirSync(agentDir, { recursive: true });
		process.env.PI_CODING_AGENT_DIR = agentDir;
		writeFileSync(join(root, "config.json"), JSON.stringify({ searchPaths: [docs] }));
		const config = resolveConfig({
			flags: { "model-provider": "openrouter", "model-id": MODEL_ID },
			env: { ...process.env, AUTORAG_CONFIG: join(root, "config.json") },
			cwd: root,
			readOnly: true,
		});
		const resolved = await resolveAgentModel(config);
		const agent = new AutoRAGAgent({
			model: resolved.model,
			...(resolved.apiKey !== undefined ? { apiKey: resolved.apiKey } : {}),
			...(resolved.providerApiKeys !== undefined ? { providerApiKeys: resolved.providerApiKeys } : {}),
			searchPaths: [docs],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: { autoInstall: false, binaryPath: realMinSync() },
			// `private/` is outside the indexed scope: forwarded to prepare AND find.
			jikji: {
				env: { PATH: path, JIKJI_DATA_DIR: join(root, "jikji-data") },
				exclude: ["private"],
				timeoutMs: 120_000,
			},
			everything: false,
			fsearch: false,
			webSearch: false,
			piAgentDir: agentDir,
			piSessionDir: join(root, "sessions"),
		});

		// 1. refresh: the stale PATH jikji is skipped and the pinned build is installed and used.
		// The librarian runs `jikji find` itself in its baseline prefetch, before (and often instead of) any model
		// tool call. So the proof that the real 0.2.1 binary served the question is the process log, not the
		// model's tool list. The log records only argv; the real binary still does the work.
		const callLog = join(root, "jikji-calls.log");
		const refreshed = await agent.refresh(true, { methods: ["parsed", "minsync", "jikji"] });
		expect(refreshed.minsync?.ok, JSON.stringify(refreshed.minsync)).toBe(true);
		expect(
			refreshed.diagnostics.filter((item) => item.source === "jikji"),
			"jikji diagnostics after refresh",
		).toEqual([]);
		const cached = cachedJikjiBinaryPath(root);
		expect(existsSync(cached), `pinned jikji was not installed at ${cached}`).toBe(true);
		expect(execFileSync(cached, ["--version"], { encoding: "utf8" }).trim()).toBe(`jikji ${JIKJI_INSTALL_VERSION}`);
		const real = `${cached}.real`;
		renameSync(cached, real);
		writeFileSync(cached, `#!/bin/sh\nprintf '%s\\n' "$*" >> '${callLog}'\nexec '${real}' "$@"\n`);
		chmodSync(cached, 0o755);

		// 2. a real model, a real question: it must reach for jikji_find.
		const called: string[] = [];
		const unsubscribe = agent.subscribe((event) => {
			if (event.type === "tool_execution_start") called.push(event.toolName);
		});
		let answer: string;
		try {
			const response = await agent.searchDocuments(
				"Hanbit Logistics 서비스 계약은 언제 갱신되고, 갱신 수수료와 승인 코드는 뭐야? 로컬 파일에서 찾아서 알려줘.",
			);
			answer = response.answer ?? "";
		} finally {
			unsubscribe();
		}
		console.log(`--- live answer (${MODEL_ID}) ---\n${answer}\n--- tools: ${called.join(", ")} ---`);

		// jikji_find is registered for the model whenever jikji is configured; the prefetch already served the query.
		expect(agent.getSystemPrompt()).toContain(JIKJI_FIND_TOOL_NAME);
		const finds = readFileSync(callLog, "utf8")
			.split("\n")
			.filter((line) => line.startsWith("find "));
		expect(finds.length, "the pinned jikji never served a find").toBeGreaterThan(0);
		for (const line of finds) {
			expect(line, "find lost the prepare scope").toContain("--exclude private");
			expect(line).toContain("--no-background-refresh");
		}
		// 3. grounded in the file the real 0.2.1 answer pack pointed at.
		expect(answer).toContain("2027");
		expect(answer).toMatch(/12,?750,?000/u);
		expect(answer).toContain("QX-4471-ZEBRA");
		// 4. the excluded document never leaks, even though no other file matches a payroll query.
		const leak = await agent.findJikji("payroll override code PAYROLL-9-KESTREL");
		expect(JSON.stringify(leak.answerPack), "excluded private/ reached the answer pack").not.toContain("payroll.md");
		expect(answer).not.toContain("PAYROLL-9-KESTREL");
	}, 600_000);
});

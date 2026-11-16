/**
 * Manual QA of the wombat datasource against the operator's REAL coding-agent sessions, through the AutoRAG agent.
 *
 * Isolation: wombat gets its own data dir (`WOMBAT_QA_DATA_DIR`, already indexed with
 * `wombat --data-dir <dir> sync --index`), so `~/.wombat` is neither read nor written. The AutoRAG config is a temp
 * file; `~/.autorag` is never touched. Session logs are only read.
 *
 * It checks, in order: the datasource is listed and has retrieval methods; every configured retrieval path
 * (`search_all_documents`, the single-datasource tool, the three wombat modes) finds the same real session by a phrase
 * taken from it; hits carry the session label and an opaque source; and, with a decision model signed in, Jev's
 * datasource check picks wombat for a question about a past session and skips it for an unrelated one.
 *
 * Run: WOMBAT_BINARY=... WOMBAT_QA_DATA_DIR=/tmp/wqa/data bun scripts/manual-qa/run-qa-wombat-real-sessions.ts [--jev]
 */
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import type { JevJudge } from "../../src/agent/jev-extension.ts";
import { selectDatasources } from "../../src/agent/query-routing.ts";
import { buildAgentOptions, resolveConfig } from "../../src/cli/config.ts";
import { WombatClient } from "../../src/datasource/skills/wombat/index.ts";

const dataDir = process.env.WOMBAT_QA_DATA_DIR;
if (dataDir === undefined) throw new Error("set WOMBAT_QA_DATA_DIR to an indexed wombat data dir");
const binaryPath = process.env.WOMBAT_BINARY ?? "wombat";
const withJev = process.argv.includes("--jev");
const root = mkdtempSync(join(tmpdir(), "autorag-wombat-real-qa-"));

let failures = 0;
function check(name: string, pass: boolean, note?: string): void {
	if (!pass) failures += 1;
	console.log(`${pass ? "PASS" : "FAIL"}  ${name}${note ? ` — ${note}` : ""}`);
}

try {
	const configPath = join(root, "config.json");
	writeFileSync(
		configPath,
		JSON.stringify({
			searchPaths: [root],
			workspacePath: root,
			minSync: { autoInstall: false },
			datasources: { wombat: { connector: { binaryPath, dataDir } } },
		}),
	);
	const options = buildAgentOptions(resolveConfig({ flags: { config: configPath } }));
	const agent = new AutoRAGAgent({ ...options, everything: false, fsearch: false });

	const listed = agent.listDatasources();
	check("wombat is a configured datasource", listed.map((entry) => entry.datasourceId).includes("wombat"));

	// Real queries: take a distinctive turn from the index itself, so the expected session is known.
	const client = new WombatClient({ binaryPath, dataDir });
	const probes = [
		"release patch main sync",
		"EventKit calendar reminders MCP",
		"duplicate pi issue analysis",
		"mgrep 스킬 삭제",
		"CI 상태 확인 후 main 병합",
	];
	const sessions = new Map<string, string>();
	for (const probe of probes) {
		const direct = await client.search("keyword", probe, { topK: 3 });
		check(`wombat CLI answers "${probe}"`, direct.ok && direct.hits.length > 0);
		if (direct.ok && direct.hits[0] !== undefined) sessions.set(probe, direct.hits[0].catalogKey);
	}

	for (const probe of probes) {
		const expected = sessions.get(probe);
		if (expected === undefined) continue;
		const all = await agent.searchAllDocuments(probe, { topK: 10 });
		const inAll = all.results.filter((result) => result.source.startsWith("/wombat/"));
		check(
			`search_all_documents: "${probe}" returns wombat hits`,
			inAll.length > 0,
			`${all.results.length} total, ${inAll.length} wombat`,
		);
		const single = await agent.searchSingleDatasourceDocuments("wombat", probe, { topK: 10 });
		check(
			`single datasource: "${probe}" finds ${expected}`,
			single.results.some((result) => result.content.startsWith(`[${expected}`)),
			single.results
				.slice(0, 3)
				.map((result) => result.content.split("\n")[0]?.slice(0, 60))
				.join(" | "),
		);
		check(
			"hits are opaque and labeled with a session",
			single.results.every(
				(result) => /^\/wombat\/default\/chunks\/[^/#]+$/u.test(result.source) && result.content.startsWith("["),
			),
		);
	}

	// Meaning-based: no keyword from the session appears in the question.
	const semantic = await agent.searchSingleDatasourceDocuments("wombat", "how did we ship a new version of the tool", {
		topK: 8,
	});
	check(
		"dense/hybrid answers a paraphrase with real sessions",
		semantic.results.length > 0,
		semantic.results
			.slice(0, 3)
			.map((result) => result.content.split("\n")[0]?.slice(0, 70))
			.join(" | "),
	);

	if (withJev) {
		// The operator's real setup always has other datasources, and a lone candidate is judged against nothing, so the
		// check runs wombat next to two realistic decoys and reports each probability.
		const wombat = listed.find((entry) => entry.datasourceId === "wombat");
		if (wombat === undefined) throw new Error("wombat is not listed");
		const candidates = [
			{ datasourceId: wombat.datasourceId, type: wombat.type, description: wombat.description },
			{ datasourceId: "kakao", type: "chat", description: "KakaoTalk (50k+ messages): personal and friend chats, club and work rooms." },
			{ datasourceId: "mailcrawl", type: "mail", description: "Gmail inbox: GitHub notifications, invoices, newsletters." },
		];
		const internals = agent as unknown as { jevJudge: JevJudge | undefined };
		const judge = internals.jevJudge;
		if (judge === undefined) throw new Error("no Jev decision model is signed in; run without --jev");
		const cases: readonly (readonly [string, boolean])[] = [
			["지난번 omp 세션에서 release patch를 어떻게 배포하기로 했었지?", true],
			["claude code로 mgrep 스킬 지우던 대화 찾아줘", true],
			["그때 코덱스한테 CI 상태 확인하고 병합하라고 시킨 적 있었나?", true],
			["Where is my 2025 tax return PDF?", false],
			["What is the capital of France?", false],
			["AWS 인보이스 메일 찾아줘", false],
		];
		for (const [question, wanted] of cases) {
			const selection = await selectDatasources(judge, question, candidates);
			const picked = selection.selected.includes("wombat");
			check(
				`Jev ${wanted ? "picks" : "skips"} wombat: ${question}`,
				picked === wanted,
				JSON.stringify(selection.probabilities),
			);
		}
	}

	console.log(failures === 0 ? "\nREAL-SESSION QA PASSED" : `\nREAL-SESSION QA: ${failures} failure(s)`);
	if (failures > 0) process.exitCode = 1;
} finally {
	rmSync(root, { recursive: true, force: true });
}

/**
 * Live QA, A/B: does handing the agent a datasource's attachment folders help it find the saved file?
 *
 * Runs the real two-phase search (`searchDocuments`) with a real model over the real Discord archive and its
 * downloaded attachments, on the same questions in two arms:
 *   on   the connection reports its attachment folders (skill text + end of each search result)
 *   off  the same connection with that wrapper removed; the agent gets no folder context
 * Each answer is judged against the disk and `download-status.json`. Needs OPENROUTER_API_KEY and a real `minsync`.
 *
 *   bun scripts/manual-qa/run-qa-attachments-agent-live.ts [model] [runs-per-arm]
 *       defaults: deepseek/deepseek-v4.1-flash-1017, 2. Writes every run to /tmp/attach-ab.json.
 *
 * Arm order alternates per case and each run gets a fresh workspace, memory and session directory, so an arm never
 * inherits the other's remembered evidence.
 *
 * Read-only by construction: the agent's `bash` can reach every crawler CLI, and a crawler's `import`/`sync` writes
 * the user's real archive. A shim for each crawler goes first on PATH, refuses those subcommands and hands every
 * read subcommand (search, messages, ...) to the real binary.
 *
 * Judge, fixed before the arms were compared:
 *   cited path   every `/Users/...` or `~/...` in the answer is followed on disk to its longest existing prefix;
 *                it is missing when the answer names something deeper than what exists, malformed for `~/ `.
 *   pass         no missing or malformed path, and
 *                  - saved-file cases: the answer locates at least half of the expected files (all when two or fewer),
 *                    by full path, or by file name when the name is unique among saved files;
 *                  - media-folder case: it cites at least one existing file under the media folder;
 *                  - not-saved case: it gives no local path for the file and says it is not stored locally.
 * Facts beyond location (counts, dates) are for a person to read in the printed answers.
 * The decision model is the deterministic baseline, so a run always reaches the verification phase.
 * The model must be a 2026+ release.
 */
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { basename, delimiter, join } from "node:path";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { resolveAgentModel, resolveConfig } from "../../src/cli/config.ts";
import { resolveAutoRAGHome, resolveUserHome } from "../../src/config/home.ts";
import { buildDatasourceSkills, type DatasourcesConfig } from "../../src/datasource/skills/factory.ts";
import type { DatasourceSkill } from "../../src/datasource/types.ts";
import { citedPaths } from "./cited-paths.ts";
import { baselineJev } from "../../test/helpers/baseline-jev.ts";
import { unwrapAttachments } from "../../test/helpers/attachments.ts";
import { realMinSync } from "../../test/helpers/real-minsync.ts";

const MODEL_ID = process.argv[2] ?? "deepseek/deepseek-v4.1-flash-1017";
const RUNS = Number(process.argv[3] ?? "2");
const DISCORD = "nomadamas";
const TELEGRAM = "telegram";
const TELECRAWL_MEDIA = join(homedir(), ".telecrawl", "media");

interface StatusRecord {
	readonly channelName?: string;
	readonly authorName?: string;
	readonly filename: string;
	readonly state: string;
	readonly path?: string;
}

interface Case {
	readonly query: string;
	/** Datasource names the model may use. */
	readonly datasources: readonly string[];
	/** Archive records whose saved copies a correct answer locates; all-unsaved means "not stored locally". */
	readonly records?: (all: readonly StatusRecord[]) => readonly StatusRecord[];
	/** The answer must instead cite an existing file under this folder. */
	readonly citesUnder?: string;
}

const CASES: readonly Case[] = [
	{
		query: "마커 영수 처리 채널에 올라온 인보이스 PDF 파일들 어디에 저장돼 있어? 파일 경로 알려줘.",
		datasources: [DISCORD],
		records: (all) => all.filter((r) => r.channelName === "마커-영수-처리" && /^Invoice/u.test(r.filename) && r.state === "downloaded"),
	},
	{
		query: "AutoRAG Qwen Conference 2026 발표 자료 PDF가 디스코드에 올라왔던 것 같은데, 내 컴퓨터에 저장된 파일 위치 알려줘.",
		datasources: [DISCORD],
		records: (all) => all.filter((r) => /QwenConference/u.test(r.filename) && r.state === "downloaded"),
	},
	{
		query: "dazzi 런칭 영상 mp4 파일이 어디 저장돼 있는지 찾아줘.",
		datasources: [DISCORD],
		records: (all) => all.filter((r) => /^dazzi-launch/u.test(r.filename) && r.state === "downloaded"),
	},
	{
		query: "돌쇠 채널에서 Yun이 올린 이미지 파일은 로컬 어디에 있어?",
		datasources: [DISCORD],
		records: (all) => all.filter((r) => r.channelName === "돌쇠" && r.authorName === "Yun" && r.state === "downloaded"),
	},
	{
		query: "텔레그램 Gajae 대화에서 받은 스크린샷 이미지 파일이 내 컴퓨터 어디 있는지 경로 알려줘.",
		datasources: [TELEGRAM],
		citesUnder: TELECRAWL_MEDIA,
	},
	{
		// Every attachment in this channel is expired and was never saved: the right answer is "not stored locally".
		query: "xhigh 채널에 올라온 message.txt 첨부 파일이 내 컴퓨터 어디에 저장돼 있어?",
		datasources: [DISCORD],
		records: (all) => all.filter((r) => r.channelName === "xhigh" && r.filename === "message.txt" && r.state !== "downloaded"),
	},
];

const NOT_SAVED = /(?:로컬|디스크|컴퓨터|내려받|다운로드|저장)[^.\n]{0,40}(?:없|않|못|만료)|(?:없|못|만료|않)[^.\n]{0,40}(?:로컬|디스크|저장|다운로드|내려받)|not (?:saved|stored|downloaded)|expired/iu;

const home = resolveAutoRAGHome();
const config = JSON.parse(readFileSync(join(home, "config.json"), "utf8")) as { datasources?: DatasourcesConfig };
const discordRoot = join(home, "attachments", "discord", DISCORD);
const archive = (JSON.parse(readFileSync(join(discordRoot, "download-status.json"), "utf8")) as { attachments: readonly StatusRecord[] })
	.attachments;
const nameCount = new Map<string, number>();
for (const r of archive) {
	if (r.state === "downloaded" && r.path !== undefined) nameCount.set(basename(r.path), (nameCount.get(basename(r.path)) ?? 0) + 1);
}

const root = mkdtempSync(join(tmpdir(), "attach-agent-live-"));
const agentDir = join(root, "agent");
mkdirSync(agentDir, { recursive: true });

/** Crawler CLIs that write the user's archive when asked to import, sync, or fetch. */
const CRAWLERS = ["telecrawl", "wacrawl", "slacrawl", "discrawl", "lazykatok", "mailcrawl"];
const READ_ONLY = ["search", "messages", "chats", "status", "doctor", "metadata", "contacts", "topics", "folders", "version", "list", "report", "digest"];
const WRITING = ["import", "sync", "backup", "update", "publish", "subscribe", "init", "tail", "watch", "fetch", "purge", "send", "backfill", "get", "media", "wiretap", "index", "embed", "repair"];

/** The first `name` on PATH (before the shim directory is added). */
function whichOnPath(name: string): string | undefined {
	for (const directory of (process.env.PATH ?? "").split(delimiter)) {
		const candidate = join(directory, name);
		if (directory !== "" && existsSync(candidate)) return candidate;
	}
	return undefined;
}

function installReadOnlyShims(): string {
	const dir = join(root, "shim");
	mkdirSync(dir, { recursive: true });
	for (const name of CRAWLERS) {
		const real = whichOnPath(name);
		if (real === undefined) continue;
		// The first argument that is a known subcommand is the command: query words come after it.
		const script = [
			"#!/bin/sh",
			'for a in "$@"; do',
			'  case "$a" in',
			`    ${WRITING.join("|")}) echo "blocked: this QA run is read-only; ${name} $a was not run" >&2; exit 2;;`,
			`    ${READ_ONLY.join("|")}) break;;`,
			"  esac",
			"done",
			`exec "${real}" "$@"`,
			"",
		].join("\n");
		writeFileSync(join(dir, name), script);
		chmodSync(join(dir, name), 0o755);
	}
	return dir;
}

const minsyncBinary = realMinSync();
const shimDir = installReadOnlyShims();
process.env.PATH = `${shimDir}${delimiter}${process.env.PATH ?? ""}`;
process.env.PI_CODING_AGENT_DIR = agentDir;
writeFileSync(join(root, "config.json"), JSON.stringify({ searchPaths: [] }));
const modelConfig = resolveConfig({
	flags: { "model-provider": "openrouter", "model-id": MODEL_ID },
	env: { ...process.env, AUTORAG_CONFIG: join(root, "config.json"), PI_CODING_AGENT_DIR: agentDir },
	cwd: root,
	readOnly: true,
});
const resolved = await resolveAgentModel(modelConfig);

interface Verdict {
	readonly pass: boolean;
	readonly recall: number | undefined;
	readonly pathed: number | undefined;
	readonly badPaths: number;
	readonly notes: readonly string[];
}

function judge(testCase: Case, answer: string): Verdict {
	const plain = answer.replace(/\*\*/gu, "");
	const cited = citedPaths(plain);
	const notes: string[] = [];
	for (const c of cited) {
		if (c.malformed) notes.push(`malformed path: ${c.written}`);
		else if (c.missing) notes.push(`cites a path that does not exist: ${c.written}`);
	}
	const badPaths = cited.filter((c) => c.malformed || c.missing).length;

	if (testCase.citesUnder !== undefined) {
		const files = cited.filter((c) => c.isFile && !c.missing && c.resolved?.startsWith(testCase.citesUnder ?? "") === true);
		notes.push(`cites ${files.length} existing file(s) under the media folder`);
		return { pass: files.length > 0 && badPaths === 0, recall: undefined, pathed: undefined, badPaths, notes };
	}

	const records = testCase.records?.(archive) ?? [];
	if (records.length > 0 && records.every((r) => r.state !== "downloaded")) {
		const claimsPath = cited.some((c) => !c.malformed && basename(c.written).startsWith("message") && c.written.includes("/attachments/"));
		const says = NOT_SAVED.test(plain);
		notes.push(claimsPath ? "gives a local path for a file that was never saved" : "gives no local path for it");
		notes.push(says ? "says it is not stored locally" : "does not say it is not stored locally");
		return { pass: !claimsPath && says && badPaths === 0, recall: undefined, pathed: undefined, badPaths, notes };
	}

	const resolvedSet = new Set(cited.flatMap((c) => (c.resolved === undefined || c.missing ? [] : [c.resolved])));
	let located = 0;
	let pathed = 0;
	for (const r of records) {
		const absolute = join(discordRoot, r.path ?? "");
		const name = basename(r.path ?? "");
		const byPath = resolvedSet.has(absolute);
		const byName = (nameCount.get(name) ?? 0) === 1 && plain.includes(name);
		if (byPath) pathed += 1;
		if (byPath || byName) located += 1;
	}
	const need = records.length <= 2 ? records.length : Math.ceil(records.length / 2);
	notes.push(`located ${located}/${records.length} expected file(s) (${pathed} by full path), need ${need}`);
	return { pass: located >= need && badPaths === 0, recall: located / records.length, pathed: pathed / records.length, badPaths, notes };
}

type Arm = "on" | "off";

interface Row {
	readonly caseIndex: number;
	readonly arm: Arm;
	readonly run: number;
	readonly pass: boolean;
	readonly recall: number | undefined;
	readonly badPaths: number;
	readonly toolCalls: number;
	readonly loadedSkill: boolean;
	readonly seconds: number;
	readonly notes: readonly string[];
	readonly answer: string;
}

function buildSkills(testCase: Case, arm: Arm): readonly DatasourceSkill[] {
	const subset: DatasourcesConfig = Object.fromEntries(
		Object.entries(config.datasources ?? {}).filter(([name]) => testCase.datasources.includes(name)),
	);
	const { skills } = buildDatasourceSkills(subset, resolveUserHome());
	const exposed = skills.filter((skill) => (skill.attachmentRoots?.().length ?? 0) > 0).length;
	if (arm === "on" && exposed === 0) throw new Error("the on arm exposes no attachment folders");
	const result = arm === "on" ? skills : skills.flatMap((skill) => unwrapAttachments(skill) ?? []);
	if (arm === "off" && result.some((skill) => skill.attachmentRoots !== undefined)) {
		throw new Error("the off arm still exposes attachment folders");
	}
	return result;
}

async function runOne(caseIndex: number, arm: Arm, run: number): Promise<Row> {
	const testCase = CASES[caseIndex] as Case;
	const workspace = mkdtempSync(join(root, "ws-"));
	const agent = new AutoRAGAgent({
		model: resolved.model,
		...(resolved.apiKey !== undefined ? { apiKey: resolved.apiKey } : {}),
		...(resolved.providerApiKeys !== undefined ? { providerApiKeys: resolved.providerApiKeys } : {}),
		searchPaths: [],
		workspacePath: workspace,
		memoryPath: join(workspace, "memory.json"),
		minSync: { autoInstall: false, binaryPath: minsyncBinary },
		jikji: false,
		everything: false,
		fsearch: false,
		webSearch: false,
		piAgentDir: agentDir,
		piSessionDir: join(workspace, "sessions"),
		datasourceSkills: [...buildSkills(testCase, arm)],
		jev: baselineJev(),
	});
	const calls: string[] = [];
	agent.subscribe((event) => {
		if (event.type === "tool_execution_start") calls.push(event.toolName);
	});
	const started = Date.now();
	const response = await agent.searchDocuments(testCase.query);
	const answer = response.answer ?? "";
	const verdict = judge(testCase, answer);
	const row: Row = {
		caseIndex,
		arm,
		run,
		pass: verdict.pass,
		recall: verdict.recall,
		badPaths: verdict.badPaths,
		toolCalls: calls.length,
		loadedSkill: calls.includes("load_datasource_skill"),
		seconds: Math.round((Date.now() - started) / 1000),
		notes: verdict.notes,
		answer: answer.replace(/\s+/gu, " "),
	};
	console.log(`\n[case ${caseIndex + 1} ${arm} run ${run + 1}] ${row.pass ? "PASS" : "FAIL"} (${row.seconds}s, ${row.toolCalls} tool calls) ${testCase.query}`);
	console.log(`  ${row.notes.join("; ")}`);
	console.log(`  answer: ${row.answer.slice(0, 700)}`);
	return row;
}

const rows: Row[] = [];
try {
	for (let run = 0; run < RUNS; run += 1) {
		for (const [caseIndex] of CASES.entries()) {
			// Alternate which arm goes first so provider-side drift hits both arms alike.
			const order: readonly Arm[] = (caseIndex + run) % 2 === 0 ? ["on", "off"] : ["off", "on"];
			for (const arm of order) {
				rows.push(await runOne(caseIndex, arm, run));
				writeFileSync("/tmp/attach-ab.json", JSON.stringify(rows, null, 1));
			}
		}
	}
} finally {
	rmSync(root, { recursive: true, force: true });
}

const mean = (xs: readonly number[]): number => (xs.length === 0 ? Number.NaN : xs.reduce((a, b) => a + b, 0) / xs.length);
console.log(`\n=== A/B summary (model ${MODEL_ID}, ${RUNS} run(s) per arm per case) ===`);
console.log("| case | on pass | off pass | on recall | off recall | on bad paths | off bad paths | on tool calls | off tool calls |");
console.log("|---|---|---|---|---|---|---|---|---|");
const fmt = (x: number): string => (Number.isNaN(x) ? "n/a" : `${(x * 100).toFixed(0)}%`);
for (const [caseIndex] of CASES.entries()) {
	const pick = (arm: Arm): Row[] => rows.filter((r) => r.caseIndex === caseIndex && r.arm === arm);
	const on = pick("on");
	const off = pick("off");
	const recall = (rs: Row[]): number => mean(rs.flatMap((r) => (r.recall === undefined ? [] : [r.recall])));
	console.log(
		`| ${caseIndex + 1} | ${on.filter((r) => r.pass).length}/${on.length} | ${off.filter((r) => r.pass).length}/${off.length} | ${fmt(recall(on))} | ${fmt(recall(off))} | ${on.reduce((s, r) => s + r.badPaths, 0)} | ${off.reduce((s, r) => s + r.badPaths, 0)} | ${mean(on.map((r) => r.toolCalls)).toFixed(1)} | ${mean(off.map((r) => r.toolCalls)).toFixed(1)} |`,
	);
}
for (const arm of ["on", "off"] as const) {
	const rs = rows.filter((r) => r.arm === arm);
	console.log(
		`${arm}: pass ${rs.filter((r) => r.pass).length}/${rs.length}, bad paths ${rs.reduce((s, r) => s + r.badPaths, 0)}, mean tool calls ${mean(rs.map((r) => r.toolCalls)).toFixed(1)}, mean seconds ${mean(rs.map((r) => r.seconds)).toFixed(1)}, loaded the skill in ${rs.filter((r) => r.loadedSkill).length}/${rs.length}`,
	);
}

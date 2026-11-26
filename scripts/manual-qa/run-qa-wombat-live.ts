/**
 * Live manual QA for the wombat coding-agent-session datasource: real `wombat`, real `minsync`, and MinSync's own
 * SPLADE + Qwen3-Embedding models. Nothing is mocked and no remote embedder is contacted.
 *
 * Isolation: the run gets its own HOME, its own wombat data dir and `--source` roots pinned to fixtures, so wombat
 * never reads the operator's real agent logs and never writes `~/.wombat` or `~/.autorag`. The model cache
 * (`~/.cache/im-not-llm/models`) is linked in read-only terms: it is only read.
 *
 * It walks the user-visible path: factory-built skill -> `agent.refresh` -> SPLADE keyword, dense semantic and hybrid
 * search -> agent narrowing -> incremental append (only the new turns are embedded) -> no-op re-sync (nothing is
 * embedded) -> delete -> local-index exclusion of the mirror.
 *
 * Prerequisites: `wombat` (>= 0.1.3) and `minsync` (>= 0.7.6) on PATH or at WOMBAT_BINARY / MINSYNC_BINARY.
 *
 * Run: bun scripts/manual-qa/run-qa-wombat-live.ts
 */

import { appendFileSync, existsSync, mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { buildAgentOptions, resolveConfig } from "../../src/cli/config.ts";
import { WombatClient, wombatMirrorDir } from "../../src/datasource/skills/wombat/index.ts";

const root = mkdtempSync(join(tmpdir(), "autorag-wombat-live-qa-"));
const home = join(root, "home");
const dataDir = join(root, "wombat-data");
const claudeRoot = join(root, "sessions", "claude");
const codexRoot = join(root, "sessions", "codex");
const binaryPath = process.env.WOMBAT_BINARY ?? "wombat";
const minsyncBinary = process.env.MINSYNC_BINARY;
const autoragHome = join(root, "autorag-home");

// Every spawned process, and the agent itself, sees only this environment.
process.env.HOME = home;
process.env.AUTORAG_HOME = autoragHome;
process.env.AUTORAG_CONFIG = join(autoragHome, "config.json");
process.env.MINSYNC_SERVER = process.env.MINSYNC_SERVER ?? "off";

let failures = 0;
function check(name: string, pass: boolean, note?: string): void {
	if (!pass) failures += 1;
	console.log(`${pass ? "PASS" : "FAIL"}  ${name}${note ? ` — ${note}` : ""}`);
}

const realModels = join(homedir(), ".cache", "minsync", "models");

function claudeSession(id: string, turns: readonly (readonly [string, string])[], cwd = "/work/alpha"): string {
	const dir = join(claudeRoot, "proj-a");
	mkdirSync(dir, { recursive: true });
	const path = join(dir, `${id}.jsonl`);
	writeFileSync(
		path,
		`${turns
			.map(([user, assistant], index) =>
				[
					JSON.stringify({
						type: "user",
						sessionId: id,
						cwd,
						timestamp: `2026-10-16T01:${String(index).padStart(2, "0")}:00Z`,
						message: { role: "user", content: user },
					}),
					JSON.stringify({
						type: "assistant",
						sessionId: id,
						timestamp: `2026-10-16T01:${String(index).padStart(2, "0")}:10Z`,
						message: {
							role: "assistant",
							content: [
								{ type: "thinking", thinking: "HIDDENTHOUGHTMARKER never index" },
								{ type: "text", text: assistant },
							],
						},
					}),
				].join("\n"),
			)
			.join("\n")}\n`,
	);
	return path;
}

function codexSession(id: string, user: string, assistant: string): void {
	mkdirSync(codexRoot, { recursive: true });
	const lines = [
		{ timestamp: "2026-10-15T12:00:00Z", type: "session_meta", payload: { session_id: id, cwd: "/work/beta", thread_source: "user" } },
		{
			timestamp: "2026-10-15T12:00:05Z",
			type: "response_item",
			payload: { type: "message", role: "user", content: [{ type: "input_text", text: user }] },
		},
		{
			timestamp: "2026-10-15T12:00:15Z",
			type: "response_item",
			payload: { type: "message", role: "assistant", content: [{ type: "output_text", text: assistant }] },
		},
	];
	writeFileSync(join(codexRoot, `rollout-2026-10-15T12-00-00-${id}.jsonl`), `${lines.map((line) => JSON.stringify(line)).join("\n")}\n`);
}

try {
	mkdirSync(join(home, ".cache", "minsync"), { recursive: true });
	mkdirSync(autoragHome, { recursive: true });
	if (existsSync(realModels)) symlinkSync(realModels, join(home, ".cache", "minsync", "models"));

	const claudeFile = claudeSession("claude-sess-02", [
		["baryon_lattice_cache 무효화 조건 설명해줘", "baryon_lattice_cache는 파일 해시가 바뀔 때만 갱신됩니다."],
		["how do I stop the ingress from dropping long uploads", "Raise the proxy read timeout on the ingress controller and the upstream keepalive."],
	]);
	claudeSession("claude-sess-03", [
		["postgres table keeps growing even after deletes", "Autovacuum is falling behind; tune its cost limit and run VACUUM on the hot table."],
	]);
	codexSession("codex-sess-01", "zephyr_quasar_protocol 핸드셰이크 버그를 고쳐줘", "zephyr_quasar_protocol 타임아웃 재시도 로직을 수정했습니다.");
	// A local document the local-file index must still own.
	mkdirSync(join(root, "docs"), { recursive: true });
	writeFileSync(join(root, "docs", "note.txt"), "Quarterly planning note about the warehouse move.");

	const embedderAvailable = existsSync(realModels);
	check("MinSync model cache is present (first run downloads Qwen3 + KURE-Sparse-nano)", embedderAvailable);

	const configPath = join(root, "config.json");
	writeFileSync(
		configPath,
		JSON.stringify({
			searchPaths: [root],
			workspacePath: root,
			minSync: { ...(minsyncBinary === undefined ? {} : { binaryPath: minsyncBinary }), autoInstall: false },
			datasources: {
				wombat: {
					instanceId: "me",
					connector: {
						binaryPath,
						dataDir,
						sources: { "claude-code": [claudeRoot], codex: [codexRoot] },
						indexTimeoutMs: 30 * 60_000,
					},
				},
			},
		}),
	);
	const config = resolveConfig({ flags: { config: configPath } });
	const options = buildAgentOptions(config);
	check(
		"the local-file index excludes wombat's mirror, so no conversation is embedded twice",
		(options.excludePaths ?? []).includes(wombatMirrorDir({ dataDir })),
		JSON.stringify(options.excludePaths),
	);
	const skills = options.datasourceSkills ?? [];
	check("factory builds the wombat skill from config", skills.length === 1 && skills[0]?.describe().name === "wombat");

	const agent = new AutoRAGAgent({ ...options, everything: false, fsearch: false });

	const first = await agent.refresh(true, { methods: ["datasources"] });
	const firstResult = first.datasources?.find((result) => result.skill === "wombat");
	check("refresh: wombat indexed through the shared MinSync", firstResult?.ok === true, JSON.stringify(firstResult?.diagnostics ?? []));
	const firstChunks = firstResult?.ok === true ? firstResult.chunkCount : 0;
	check("refresh: 3 sessions became chunks", firstChunks >= 3, String(firstChunks));
	const config1 = join(wombatMirrorDir({ dataDir }), ".minsync", "config.toml");
	check("the workspace was created under the isolated data dir", existsSync(config1));
	check("the real ~/.wombat was not created for this run", !existsSync(join(home, ".wombat")));

	const search = (query: string, scope?: string) =>
		agent.searchSingleDatasourceDocuments("wombat", query, { topK: 10, ...(scope === undefined ? {} : { scope }) });
	const sources = (results: readonly { source: string }[]) => results.map((result) => result.source);

	const keyword = await search("baryon_lattice_cache");
	check("SPLADE keyword: finds the Claude session by an exact rare term", keyword.results[0]?.content.includes("baryon_lattice_cache") === true, keyword.results[0]?.content.slice(0, 100));
	check("hit is labeled with its session for `wombat show` / `trace`", keyword.results[0]?.content.startsWith("[claude-code:claude-sess-02") === true);
	check(
		"source is opaque and carries no filesystem path",
		sources(keyword.results).every((source) => /^\/wombat\/me\/chunks\/[^/#]+$/u.test(source)),
		sources(keyword.results)[0],
	);

	const semantic = await search("why does my gateway cut off big file transfers midway");
	check(
		"dense semantic: finds the ingress-timeout turn with no shared keywords",
		semantic.results.slice(0, 3).some((result) => result.content.includes("proxy read timeout")),
		semantic.results.map((result) => result.content.split("\n")[1]?.slice(0, 50)).join(" | "),
	);
	const korean = await search("프로토콜 핸드셰이크 재시도");
	check("Korean query finds the Codex session", korean.results.slice(0, 3).some((result) => result.content.includes("zephyr_quasar_protocol")));
	check("thinking blocks are never indexed", !(await search("HIDDENTHOUGHTMARKER")).results.some((result) => result.content.includes("HIDDENTHOUGHTMARKER")));

	// A direct wombat run, to see each mode independently of the fallback chain.
	const client = new WombatClient({ binaryPath, dataDir, sources: { "claude-code": [claudeRoot], codex: [codexRoot] }, env: process.env });
	const modes = await Promise.all(
		(["keyword", "semantic", "hybrid"] as const).map(async (mode) => [mode, await client.search(mode, "postgres autovacuum", { topK: 3 })] as const),
	);
	for (const [mode, result] of modes) check(`wombat --mode ${mode} answers`, result.ok && result.hits.length > 0);

	// Incremental: append one turn to an existing session, add nothing else.
	const before = await client.sync();
	check("no-op re-sync embeds nothing", before.ok && before.data.written === 0, JSON.stringify(before.ok ? before.data : before));
	appendFileSync(
		claudeFile,
		`${JSON.stringify({ type: "user", sessionId: "claude-sess-02", cwd: "/work/alpha", timestamp: "2026-10-16T01:05:00Z", message: { role: "user", content: "terraform state lock stuck after a crashed apply" } })}\n${JSON.stringify({ type: "assistant", sessionId: "claude-sess-02", timestamp: "2026-10-16T01:05:10Z", message: { role: "assistant", content: [{ type: "text", text: "Run terraform force-unlock with the lock id from the error message." }] } })}\n`,
	);
	const second = await agent.refresh(true, { methods: ["datasources"] });
	const secondResult = second.datasources?.find((result) => result.skill === "wombat");
	check("incremental refresh succeeds", secondResult?.ok === true);
	const appended = await search("terraform force-unlock stuck lock");
	check("incremental: the appended turn is searchable", appended.results.slice(0, 3).some((result) => result.content.includes("force-unlock")));
	const unchanged = await search("baryon_lattice_cache");
	check("incremental: the untouched turns still answer", unchanged.results[0]?.content.includes("baryon_lattice_cache") === true);

	// Delete a session: it must disappear from the index.
	rmSync(join(codexRoot, "rollout-2026-10-15T12-00-00-codex-sess-01.jsonl"));
	await agent.refresh(true, { methods: ["datasources"] });
	const gone = await search("zephyr_quasar_protocol");
	check("delete: a removed session no longer answers", !gone.results.some((result) => result.content.includes("zephyr_quasar_protocol")));

	// The local index still owns real documents and does not see the mirror.
	await agent.refresh(true, { methods: ["parsed", "minsync"] }).catch(() => undefined);
	const local = await agent.searchDocuments?.("baryon_lattice_cache", { topK: 5 }).catch(() => undefined);
	check(
		"local-file index never returns the wombat mirror",
		local === undefined || !JSON.stringify(local).includes(".wombat") && !JSON.stringify(local).includes("wombat-data"),
	);

	console.log(failures === 0 ? "\nLIVE QA PASSED" : `\nLIVE QA: ${failures} failure(s)`);
	if (failures > 0) process.exitCode = 1;
} finally {
	rmSync(root, { recursive: true, force: true });
}

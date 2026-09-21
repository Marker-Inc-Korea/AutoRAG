/**
 * Live manual QA for the github-gist datasource (issue #1588): the real
 * GitHub REST API with the gh CLI's authenticated token, incremental
 * re-index, lexical search, and semantic search through the real loopback
 * autorag-gateway (qwen3-embedding-0.6b). The token is never printed or
 * persisted by this script; embeddings stay on the local machine.
 *
 * Prerequisites: `gh auth login` with the `gist` scope. The embedding model
 * is prefetched automatically on first run (pinned cache download).
 *
 * Run: bun scripts/manual-qa/run-qa-github-gist-live.ts
 */

import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { releaseRuntimeHandles, stopRuntime } from "../../src/embedding-runtime/index.ts";
import { buildDatasourceSkills } from "../../src/datasource/skills/factory.ts";
import { GitHubGistSkill } from "../../src/datasource/skills/github-gist/index.ts";

const tmpRoot = mkdtempSync(join(tmpdir(), "autorag-gist-live-qa-"));
const docsDir = join(tmpRoot, "docs");
mkdirSync(docsDir, { recursive: true });
writeFileSync(join(docsDir, "readme.txt"), "placeholder");

let failures = 0;
function check(name: string, pass: boolean, note?: string): void {
	if (!pass) failures += 1;
	console.log(`${pass ? "PASS" : "FAIL"}  ${name}${note ? ` — ${note}` : ""}`);
}

try {
	// Auth comes from the gh CLI; never echo the token itself.
	let ghAuthed = true;
	try {
		execFileSync("gh", ["auth", "token"], { stdio: ["ignore", "pipe", "pipe"], timeout: 10_000 });
	} catch {
		ghAuthed = false;
	}
	check("gh CLI authenticated (token resolves)", ghAuthed);
	if (!ghAuthed) throw new Error("gh auth token failed; run gh auth login with the gist scope");

	const { skills, unknown } = buildDatasourceSkills({ "github-gist": { connector: {} } }, tmpRoot);
	check("factory builds github-gist skill", skills.length === 1 && unknown.length === 0);
	const skill = skills[0] as GitHubGistSkill;

	const agent = new AutoRAGAgent({
		searchPaths: [docsDir],
		workspacePath: tmpRoot,
		minSync: false,
		datasourceSkills: skills,
		datasourceAccess: { allowedTags: ["github", "gists"], allowedScopes: ["/github-gist/**"] },
	});

	const refresh1 = await agent.refresh(true, { methods: ["datasources"] });
	const first = refresh1.datasources?.find((result) => result.skill === "github-gist");
	check("live index: github-gist ok", first?.ok === true);
	const firstChunks = first?.ok === true ? first.chunkCount : 0;
	check("live index: chunks indexed", firstChunks > 0, `${firstChunks} chunk(s)`);
	for (const diagnostic of first?.diagnostics ?? []) {
		console.log(`  diagnostic[${diagnostic.severity}] ${diagnostic.code}: ${diagnostic.message}`);
	}
	check(
		"semantic sidecar synced (no semantic-unavailable diagnostic)",
		(first?.diagnostics ?? []).every((diagnostic) => !diagnostic.message.includes("semantic-unavailable")),
	);

	const refresh2 = await agent.refresh(true, { methods: ["datasources"] });
	const second = refresh2.datasources?.find((result) => result.skill === "github-gist");
	const secondChunks = second?.ok === true ? second.chunkCount : -1;
	check(
		"incremental re-index: no-op keeps chunk count",
		second?.ok === true && secondChunks === firstChunks,
		`${firstChunks} -> ${secondChunks}`,
	);

	// The retired fan-out `search_datasource_documents` tool was replaced by the
	// per-connection `search_datasource_<id>` tools; QA exercises the same lexical
	// retrieval method those tools route to.
	const lexicalMethod = skill.retrievalMethods().find((method) => method.describe().name === "github-gist-lexical");
	check("lexical method registered", lexicalMethod !== undefined);
	if (lexicalMethod !== undefined) {
		const hits = await lexicalMethod.retrieve("Virtual File System", { topK: 5, scope: "/github-gist/**" });
		check(
			"lexical: exact-term gist hit via github-gist-lexical",
			hits.length > 0 && hits.every((hit) => hit.source.startsWith("/github-gist/")),
			hits[0]?.source,
		);
	}

	// Semantic path through the real loopback gateway (lazy ensure, cachedOnly).
	const semantic = skill.retrievalMethods().find((method) => method.describe().name === "github-gist-semantic");
	check("semantic method registered", semantic !== undefined);
	if (semantic !== undefined) {
		const hits = await semantic.retrieve("파일 시스템을 추상화해서 구현하는 방법", { topK: 5 });
		check(
			"semantic: Korean paraphrase query returns gist hits via local gateway",
			hits.length > 0 && hits.every((hit) => hit.source.startsWith("/github-gist/")),
			hits[0] ? `${hits[0].source} score=${hits[0].score.toFixed(3)}` : "no hits",
		);
		const vfs = hits.find((hit) => hit.content.includes("Virtual File System"));
		check("semantic: VFS gist ranked for paraphrased query", vfs !== undefined, vfs?.source);
	}

	console.log(failures === 0 ? "\nLIVE QA PASSED" : `\nLIVE QA: ${failures} failure(s)`);
	if (failures > 0) process.exitCode = 1;
} finally {
	await releaseRuntimeHandles().catch(() => { });
	await stopRuntime().catch(() => { });
	rmSync(tmpRoot, { recursive: true, force: true });
}

/**
 * Live manual QA for the Obsidian vault datasource: a real vault fixture on disk, the real `minsync`
 * binary, and MinSync's own embedding engine (a running `minsync serve` daemon, started by the shared
 * embedder when none runs). Nothing is mocked and no remote embedder is contacted.
 *
 * It walks the user-visible path: factory-built skill -> `agent.refresh` -> BM25 and semantic
 * search -> folder scope -> incremental re-index (edit, add, delete) -> hidden/attachment skipping.
 * The vault is built in a temp directory, so it needs no existing Obsidian install.
 *
 * Prerequisites: the `minsync` binary on PATH (0.4.11 or newer).
 *
 * Run: bun scripts/manual-qa/run-qa-obsidian-live.ts
 */

import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { buildDatasourceSkills } from "../../src/datasource/skills/factory.ts";
import { createEmbedderFromConfig, releaseEmbedders } from "../../src/embedding/index.ts";

const tmpRoot = mkdtempSync(join(tmpdir(), "autorag-obsidian-live-qa-"));
const vault = join(tmpRoot, "vault");
const decomposed = "계획.md".normalize("NFD");

let failures = 0;
function check(name: string, pass: boolean, note?: string): void {
	if (!pass) failures += 1;
	console.log(`${pass ? "PASS" : "FAIL"}  ${name}${note ? ` — ${note}` : ""}`);
}

function note(relativePath: string, body: string): void {
	const path = join(vault, relativePath);
	mkdirSync(join(path, ".."), { recursive: true });
	writeFileSync(path, body);
}

try {
	note("Projects/Roadmap.md", "# Roadmap\nThe mobile app beta launches in October after the design freeze.");
	note("Projects/Budget.md", "# Budget\nHiring is frozen until the FY25 budget is approved by finance.");
	note("Daily/2026-10-01.md", "# Daily\nRead about PostgreSQL vacuum tuning and index bloat.");
	note(`프로젝트/${decomposed}`, "# 계획\n모바일 앱 베타 출시 일정과 디자인 동결 계획.");
	note("Ideas/C# tips 100%.md", "# Tips\nspecialcharsmarker lives in this note.");
	note(".obsidian/workspace.json", '{"hiddenmarker":"obsidian settings must not be indexed"}');
	note(".trash/Deleted.md", "# Deleted\nhiddenmarker in the trash must not be indexed.");
	note("Attachments/diagram.png", "hiddenmarker binary attachment");
	writeFileSync(join(tmpRoot, "readme.txt"), "placeholder corpus for the local-file index");

	const embedder = createEmbedderFromConfig({});
	const health = await embedder.health();
	console.log(`embedding backend: ${health.backend} alive=${health.alive}${health.detail ? ` (${health.detail})` : ""}`);
	check("an embedding backend is alive (run `minsync serve --idle-timeout 0` first)", health.alive);

	const { skills, unknown } = buildDatasourceSkills(
		{ obsidian: { instanceId: "vault", connector: { vaultPath: vault } } },
		tmpRoot,
		{ embedder },
	);
	check("factory builds the obsidian skill with no external CLI", skills.length === 1 && unknown.length === 0);
	check("descriptor does not claim an external CLI", skills[0]?.describe().requiresExternalCli === undefined);

	const agent = new AutoRAGAgent({
		searchPaths: [tmpRoot],
		workspacePath: tmpRoot,
		minSync: { autoInstall: false },
		everything: false,
		fsearch: false,
		datasourceSkills: skills,
	});

	const first = await agent.refresh(true, { methods: ["datasources"] });
	const firstResult = first.datasources?.find((result) => result.skill === "obsidian");
	check("refresh: obsidian indexed through MinSync", firstResult?.ok === true, JSON.stringify(firstResult?.diagnostics ?? []));
	check("refresh: only the 5 notes were indexed", firstResult?.ok === true && firstResult.chunkCount === 5);

	const search = (query: string, scope?: string) =>
		agent.searchSingleDatasourceDocuments("obsidian", query, { topK: 10, ...(scope === undefined ? {} : { scope }) });
	const sources = (results: readonly { source: string }[]) => results.map((result) => result.source);

	const lexical = await search("hiring frozen budget");
	check(
		"BM25: finds the note by exact terms",
		sources(lexical.results)[0] === "/obsidian/vault/Projects/Budget.md",
		sources(lexical.results).join(", "),
	);
	check("BM25: result carries the real file path and title", lexical.results[0]?.metadata.title === "Budget");

	const semantic = await search("when does the phone application get released to testers");
	check(
		"semantic: finds the roadmap note with no shared keywords",
		sources(semantic.results).includes("/obsidian/vault/Projects/Roadmap.md"),
		sources(semantic.results).join(", "),
	);

	const korean = await search("모바일 앱 출시 계획");
	check(
		"Korean query matches a note whose file name is written decomposed (NFD)",
		sources(korean.results).includes("/obsidian/vault/프로젝트/계획.md".normalize("NFC")),
		sources(korean.results).join(", "),
	);

	const folder = await search("notes", "/obsidian/vault/Projects");
	check(
		"folder scope: only notes under Projects",
		folder.results.length > 0 && sources(folder.results).every((source) => source.startsWith("/obsidian/vault/Projects/")),
		sources(folder.results).join(", "),
	);
	const sibling = await search("notes", "/obsidian/vault/Project");
	check("folder scope: a name prefix is not a folder", sibling.results.length === 0);

	const special = await search("specialcharsmarker");
	check(
		"special characters in a note name are escaped in its source",
		sources(special.results)[0] === "/obsidian/vault/Ideas/C%23 tips 100%25.md",
		sources(special.results).join(", "),
	);

	// A vector search always returns its nearest notes, so "nothing hidden" means no hit comes from a hidden
	// path and none carries the hidden text, not an empty result list.
	const hidden = await search("hiddenmarker obsidian settings trash attachment");
	check(
		"hidden folders and attachments are never indexed",
		hidden.results.every(
			(result) => !/\/(?:\.obsidian|\.trash|Attachments)\//u.test(result.source) && !result.content.includes("hiddenmarker"),
		),
		sources(hidden.results).join(", "),
	);

	// Incremental: edit one note, add one, delete one.
	note("Projects/Roadmap.md", "# Roadmap\nThe mobile app beta slipped to November because of the payments review.");
	note("Projects/New.md", "# New\nKubernetes migration runbook for the staging cluster.");
	rmSync(join(vault, "Projects", "Budget.md"));
	const second = await agent.refresh(true, { methods: ["datasources"] });
	const secondResult = second.datasources?.find((result) => result.skill === "obsidian");
	check("incremental refresh succeeds", secondResult?.ok === true);
	check("incremental refresh: note count is 5 (-1 deleted, +1 added)", secondResult?.ok === true && secondResult.chunkCount === 5);
	check("incremental: an edited note answers with its new text", sources((await search("payments review november")).results)[0] === "/obsidian/vault/Projects/Roadmap.md");
	check("incremental: the old text is gone", !(await search("design freeze")).results.some((result) => result.content.includes("design freeze")));
	check("incremental: a deleted note no longer answers", !sources((await search("hiring frozen budget")).results).includes("/obsidian/vault/Projects/Budget.md"));
	check("incremental: a new note is searchable", sources((await search("kubernetes staging cluster")).results)[0] === "/obsidian/vault/Projects/New.md");

	console.log(failures === 0 ? "\nLIVE QA PASSED" : `\nLIVE QA: ${failures} failure(s)`);
	if (failures > 0) process.exitCode = 1;
} finally {
	await releaseEmbedders();
	rmSync(tmpRoot, { recursive: true, force: true });
}

/**
 * Live manual QA for the macOS Spotlight datasource skill (issue #1350).
 *
 * Runs the full agent path on a real Mac: fixture documents are written to a
 * temp directory, we wait for Spotlight to actually index them (polling
 * `mdfind` as the readiness signal), then index + search through
 * AutoRAGAgent and the datasource tool. Also asserts the platform gate
 * reports `unavailable` off-darwin and that sources stay path-opaque.
 *
 * Prerequisites: macOS with Spotlight indexing enabled (`mdutil -s /`).
 * Full Disk Access is only needed when configured queries hit protected
 * locations; this harness uses a temp dir that needs no special permission.
 *
 * Run: bun scripts/manual-qa/run-qa-spotlight-live.ts
 */

import { execFile } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { singleDatasourceToolName } from "../../src/agent/search-single-datasource-tool.ts";
import { buildDatasourceSkills } from "../../src/datasource/skills/factory.ts";
import { SpotlightConnector } from "../../src/datasource/skills/spotlight/connector.ts";

const execFileAsync = promisify(execFile);

let failures = 0;
function check(name: string, pass: boolean, note?: string): void {
	if (!pass) failures += 1;
	console.log(`${pass ? "PASS" : "FAIL"}  ${name}${note ? ` — ${note}` : ""}`);
}

const token = `autoragspotlightqa${Date.now()}`;
const tmpRoot = mkdtempSync(join(tmpdir(), "autorag-spotlight-live-"));
// Fixture dir: Spotlight skips tmpdirs and dot-directories, so use a
// non-hidden directory under $HOME. The workspace stays in tmp.
const docsDir = join(homedir(), `autorag-spotlight-qa-${Date.now()}`);
const fixtureName = `meeting-notes-${token}.txt`;
const spotlightQuery = `kMDItemFSName == "${fixtureName}"cd`;
mkdirSync(docsDir, { recursive: true });
writeFileSync(join(docsDir, fixtureName), `Quarterly planning notes mentioning ${token} budgets.`);
writeFileSync(join(docsDir, "readme.txt"), "placeholder");

/**
 * Force synchronous Spotlight import of the fixture dir, then confirm via
 * mdfind. `mdimport` is the deterministic signal — no timing luck. Note the
 * fixture lives in a non-hidden directory under $HOME: macOS excludes
 * tmpdirs and dot-directories from Spotlight indexing.
 */
async function importAndConfirm(deadlineMs: number): Promise<boolean> {
	await execFileAsync("mdimport", [docsDir]);
	// mdimport returns before the index commits; poll mdfind (the real state
	// signal) until the fixture appears or the deadline passes.
	const started = Date.now();
	while (Date.now() - started < deadlineMs) {
		const { stdout } = await execFileAsync("mdfind", ["-onlyin", docsDir, spotlightQuery]);
		if (stdout.split("\n").includes(join(docsDir, fixtureName))) return true;
	}
	return false;
}

try {
	check("platform is macOS", process.platform === "darwin", process.platform);

	const gated = await new SpotlightConnector({ queries: [token], platform: "linux" }).fetch();
	check("non-macOS platform reports unavailable", !gated.ok && gated.reason === "unavailable");

	// Generous deadline: mds_stores stalls under system load; content
	// indexing can lag minutes behind mdimport on a busy machine.
	const indexed = await importAndConfirm(300_000);
	check("spotlight indexed the fixture documents", indexed, `mdfind -onlyin ${docsDir} '${spotlightQuery}'`);

	const { skills, unknown } = buildDatasourceSkills(
		{
			spotlight: { connector: { queries: [spotlightQuery], onlyIn: docsDir } },
		},
		tmpRoot,
	);
	check("factory builds the spotlight skill", skills.length === 1 && unknown.length === 0);

	const agent = new AutoRAGAgent({
		searchPaths: [docsDir],
		workspacePath: tmpRoot,
		minSync: false,
		datasourceSkills: skills,
		datasourceAccess: { allowedTags: ["spotlight"], allowedScopes: ["/spotlight/**"] },
	});

	const refresh = await agent.refresh(true, { methods: ["datasources"] });
	const spotlight = (refresh.datasources ?? []).find((result) => result.skill === "spotlight");
	check(
		"live index: spotlight",
		spotlight?.ok === true && spotlight.chunkCount > 0,
		spotlight?.ok ? `${spotlight.chunkCount} chunk(s)` : `${spotlight?.code}: ${spotlight?.message}`,
	);

	const agentTools = (agent as unknown as { tools: readonly AgentTool[] }).tools;
	const spotlightTool = agentTools.find((entry) => entry.name === singleDatasourceToolName("spotlight"));
	const hits = await spotlightTool?.execute("live-spotlight", { query: token, topK: 5, scope: "/spotlight/**" });
	check(
		"live search: spotlight returns scoped opaque hits",
		(hits?.details.sources.length ?? 0) > 0 &&
		(hits?.details.sources ?? []).every(
			(source: string) => source.startsWith("/spotlight/") && !source.includes(tmpRoot),
		),
		hits?.details.sources[0],
	);
	const body = (hits?.content ?? []).map((block) => (block.type === "text" ? block.text : "")).join("\n");
	check("live search: hit content contains the fixture token", body.includes(token));

	const manifest = skills[0]?.skillManifest();
	check(
		"manifest documents macOS-only, FDA, and mdfind/mdls/mdutil",
		manifest?.content.includes("macOS") === true &&
		manifest.content.includes("Full Disk Access") &&
		manifest.content.includes("mdfind") &&
		manifest.content.includes("mdls") &&
		manifest.content.includes("mdutil"),
	);

	console.log(failures === 0 ? "\nSPOTLIGHT LIVE QA PASSED" : `\nSPOTLIGHT LIVE QA: ${failures} failure(s)`);
	if (failures > 0) process.exitCode = 1;
} finally {
	rmSync(tmpRoot, { recursive: true, force: true });
	rmSync(docsDir, { recursive: true, force: true });
}

/**
 * Live manual QA for the Nogon browser-history datasource.
 *
 * Runs the full agent path against a REAL Nogon server that a REAL browser extension has been feeding:
 * refresh (probe), retrieval through `AutoRAGAgent.searchSingleDatasourceDocuments` (the method behind the
 * generated `search_datasource_<id>` tool), scope push-down, the web-only guarantee, and error transparency.
 * Nothing is faked: the server, its index, and the captured pages come from `nogon listen` plus the Nogon
 * Chrome/Firefox extension.
 *
 * Prerequisites (see docs/datasources/nogon.md):
 *   1. `nogon listen` is running (default http://127.0.0.1:4433).
 *   2. The Nogon extension captured the pages named in NOGON_QA_EXPECT (comma-separated `url|token`).
 *
 * Run: NOGON_QA_EXPECT='http://host/a.html|tokenA,http://host/b.html|tokenB' \
 *      NOGON_QA_FORBIDDEN_TOKEN=tokenOfAnExcludedPage bun scripts/manual-qa/run-qa-nogon-live.ts
 *
 * State isolation: the workspace lives in a temp dir; the real ~/.autorag is never read or written.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { buildDatasourceSkills } from "../../src/datasource/skills/factory.ts";

let failures = 0;
function check(name: string, pass: boolean, note?: string): void {
	if (!pass) failures += 1;
	console.log(`${pass ? "PASS" : "FAIL"}  ${name}${note ? ` — ${note}` : ""}`);
}

const baseUrl = process.env.NOGON_QA_URL ?? "http://127.0.0.1:4433";
const expected = (process.env.NOGON_QA_EXPECT ?? "")
	.split(",")
	.map((entry) => entry.trim())
	.filter((entry) => entry.length > 0)
	.map((entry) => {
		const [url = "", token = ""] = entry.split("|");
		return { url, token };
	});
if (expected.length === 0) {
	console.error("NOGON_QA_EXPECT is required: comma-separated `url|token` pairs the browser extension captured.");
	process.exit(2);
}
const forbidden = process.env.NOGON_QA_FORBIDDEN_TOKEN;
const OPAQUE_SOURCE = /^\/browser\/browser\/pages\/[^/]+\/[0-9a-f]{16}$/;

const tmpRoot = mkdtempSync(join(tmpdir(), "autorag-nogon-live-"));
try {
	const { skills, unknown } = buildDatasourceSkills({ browser: { type: "nogon", connector: { baseUrl } } }, tmpRoot);
	check("factory builds the nogon skill", skills.length === 1 && unknown.length === 0);

	const agent = new AutoRAGAgent({
		searchPaths: [tmpRoot],
		workspacePath: tmpRoot,
		minSync: { autoInstall: false },
		datasourceSkills: skills,
	});

	const refresh = await agent.refresh(true, { methods: ["datasources"] });
	const probe = (refresh.datasources ?? []).find((result) => result.skill === "browser");
	check(
		"live refresh: nogon probe reports the captured web pages",
		probe?.ok === true && probe.chunkCount >= expected.length,
		probe?.ok ? `${probe.chunkCount} web page(s)` : `${probe?.code}: ${probe?.message}`,
	);

	for (const { url, token } of expected) {
		const { results, diagnostics } = await agent.searchSingleDatasourceDocuments("browser", token, { topK: 5 });
		const top = results[0];
		check(`live search finds ${url}`, top?.metadata.url === url, top?.source);
		check(
			`  source is an opaque /browser/... identity, not a URL or OS path (${token})`,
			results.length > 0 && results.every((hit) => OPAQUE_SOURCE.test(hit.source)),
		);
		check(
			`  content carries the page text captured by the extension (${token})`,
			results.some((hit) => hit.content.toLowerCase().includes(token.toLowerCase())),
		);
		check(
			`  no retrieval diagnostics (${token})`,
			diagnostics.length === 0,
			diagnostics.map((d) => d.message).join("; "),
		);
	}

	// A config written before the Hister -> Nogon rename (`type: "hister"`, connection id `hister`) must keep answering.
	const legacy = buildDatasourceSkills({ hister: { type: "hister", connector: { baseUrl } } }, tmpRoot);
	check("legacy `type: hister` still builds a skill", legacy.skills.length === 1 && legacy.unknown.length === 0);
	if (legacy.skills[0] !== undefined && expected[0] !== undefined) {
		const legacyAgent = new AutoRAGAgent({
			searchPaths: [tmpRoot],
			workspacePath: tmpRoot,
			minSync: { autoInstall: false },
			datasourceSkills: legacy.skills,
		});
		const old = await legacyAgent.searchSingleDatasourceDocuments("hister", expected[0].token, { topK: 5 });
		check(
			"legacy `type: hister` finds the same live page under its own `/hister/...` identity",
			old.results[0]?.metadata.url === expected[0].url && old.results.every((hit) => hit.source.startsWith("/hister/")),
			old.results[0]?.source,
		);
	}

	const first = expected[0];
	if (first !== undefined) {
		const host = new URL(first.url).hostname;
		const scoped = await agent.searchSingleDatasourceDocuments("browser", first.token, {
			topK: 5,
			scope: `/browser/browser/pages/${host}/**`,
		});
		check(
			"scope /browser/browser/pages/<host>/** keeps the page",
			scoped.results.some((hit) => hit.metadata.url === first.url),
		);
		const other = await agent.searchSingleDatasourceDocuments("browser", first.token, {
			topK: 5,
			scope: "/browser/browser/pages/not-this-host.invalid/**",
		});
		check("scope for a different domain returns nothing", other.results.length === 0);
	}

	if (forbidden !== undefined && forbidden.length > 0) {
		const leak = await agent.searchSingleDatasourceDocuments("browser", forbidden, { topK: 5 });
		check("a page the Nogon allow/skip rules excluded is not searchable", leak.results.length === 0);
	}

	const miss = await agent.searchSingleDatasourceDocuments("browser", "zzqqxxnomatchtoken", { topK: 5 });
	check("a query with no match returns nothing", miss.results.length === 0);

	// Regression: a natural-language question must not match every page that merely contains the word "or"/"the".
	const question = `where did I read about ${expected[0]?.token ?? ""} and also zzqqxxnomatchtoken`;
	const asked = await agent.searchSingleDatasourceDocuments("browser", question, { topK: 5 });
	const lexicalTop = asked.results.find((hit) => String(hit.metadata.method).endsWith("-lexical"));
	check(
		"a natural-language question ranks the page that really contains the rare term first, not a stopword match",
		lexicalTop?.metadata.url === expected[0]?.url,
		`${lexicalTop?.metadata.url}`,
	);

	const down = buildDatasourceSkills({ browser: { type: "nogon", connector: { baseUrl: "http://127.0.0.1:1" } } }, tmpRoot)
		.skills[0];
	let message = "";
	try {
		await down?.retrievalMethods()[0]?.retrieve("anything", {});
	} catch (error) {
		message = error instanceof Error ? error.message : String(error);
	}
	check(
		"unreachable Nogon fails loudly with the real connection error",
		/nogon/i.test(message) && /nogon listen/.test(message),
		message.slice(0, 160),
	);

	const manifest = skills[0]?.skillManifest().content ?? "";
	check(
		"manifest names Nogon, its AGPL license and source, and the separate-program relationship",
		manifest.includes("AGPL-3.0") &&
			manifest.includes("asciimoo/nogon") &&
			manifest.includes("separate program"),
	);

	console.log(failures === 0 ? "\nNOGON LIVE QA PASSED" : `\nNOGON LIVE QA: ${failures} failure(s)`);
	if (failures > 0) process.exitCode = 1;
} finally {
	rmSync(tmpRoot, { recursive: true, force: true });
}

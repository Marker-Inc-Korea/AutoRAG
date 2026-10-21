/**
 * Live manual QA for the Hister browser-history datasource.
 *
 * Runs the full agent path against a REAL Hister server that a REAL browser extension has been feeding:
 * refresh (probe), retrieval through `AutoRAGAgent.searchSingleDatasourceDocuments` (the method behind the
 * generated `search_datasource_<id>` tool), scope push-down, the web-only guarantee, and error transparency.
 * Nothing is faked: the server, its index, and the captured pages come from `hister listen` plus the Hister
 * Chrome/Firefox extension.
 *
 * Prerequisites (see docs/datasource-skills.md, "Browser history via Hister"):
 *   1. `hister listen` is running (default http://127.0.0.1:4433).
 *   2. The Hister extension captured the pages named in HISTER_QA_EXPECT (comma-separated `url|token`).
 *
 * Run: HISTER_QA_EXPECT='http://host/a.html|tokenA,http://host/b.html|tokenB' \
 *      HISTER_QA_FORBIDDEN_TOKEN=tokenOfAnExcludedPage bun scripts/manual-qa/run-qa-hister-live.ts
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

const baseUrl = process.env.HISTER_QA_URL ?? "http://127.0.0.1:4433";
const expected = (process.env.HISTER_QA_EXPECT ?? "")
	.split(",")
	.map((entry) => entry.trim())
	.filter((entry) => entry.length > 0)
	.map((entry) => {
		const [url = "", token = ""] = entry.split("|");
		return { url, token };
	});
if (expected.length === 0) {
	console.error("HISTER_QA_EXPECT is required: comma-separated `url|token` pairs the browser extension captured.");
	process.exit(2);
}
const forbidden = process.env.HISTER_QA_FORBIDDEN_TOKEN;
const OPAQUE_SOURCE = /^\/browser\/browser\/pages\/[^/]+\/[0-9a-f]{16}$/;

const tmpRoot = mkdtempSync(join(tmpdir(), "autorag-hister-live-"));
try {
	const { skills, unknown } = buildDatasourceSkills({ browser: { type: "hister", connector: { baseUrl } } }, tmpRoot);
	check("factory builds the hister skill", skills.length === 1 && unknown.length === 0);

	const agent = new AutoRAGAgent({
		searchPaths: [tmpRoot],
		workspacePath: tmpRoot,
		minSync: { autoInstall: false },
		datasourceSkills: skills,
	});

	const refresh = await agent.refresh(true, { methods: ["datasources"] });
	const probe = (refresh.datasources ?? []).find((result) => result.skill === "browser");
	check(
		"live refresh: hister probe reports the captured web pages",
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
		check("a page the Hister allow/skip rules excluded is not searchable", leak.results.length === 0);
	}

	const miss = await agent.searchSingleDatasourceDocuments("browser", "zzqqxxnomatchtoken", { topK: 5 });
	check("a query with no match returns nothing", miss.results.length === 0);

	const down = buildDatasourceSkills({ browser: { type: "hister", connector: { baseUrl: "http://127.0.0.1:1" } } }, tmpRoot)
		.skills[0];
	let message = "";
	try {
		await down?.retrievalMethods()[0]?.retrieve("anything", {});
	} catch (error) {
		message = error instanceof Error ? error.message : String(error);
	}
	check(
		"unreachable Hister fails loudly with the real connection error",
		/hister/i.test(message) && /hister listen/.test(message),
		message.slice(0, 160),
	);

	const manifest = skills[0]?.skillManifest().content ?? "";
	check(
		"manifest names Hister, its AGPL license and source, and the separate-program relationship",
		manifest.includes("AGPL-3.0") &&
			manifest.includes("https://github.com/asciimoo/hister") &&
			manifest.includes("separate program"),
	);

	console.log(failures === 0 ? "\nHISTER LIVE QA PASSED" : `\nHISTER LIVE QA: ${failures} failure(s)`);
	if (failures > 0) process.exitCode = 1;
} finally {
	rmSync(tmpRoot, { recursive: true, force: true });
}

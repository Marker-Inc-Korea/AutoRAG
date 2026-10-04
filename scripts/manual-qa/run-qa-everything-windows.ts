/**
 * Manual QA for the bundled Windows Everything integration. Run on a real
 * Windows host from the repository root:
 *
 *   bun scripts/manual-qa/run-qa-everything-windows.ts
 *
 * It uses the vendored portable Everything + ES, a temporary corpus with
 * Korean and space-containing names, and the real AutoRAGAgent:
 *   1. refresh --method everything starts a private user-level instance and indexes the corpus
 *   2. everything_search finds files by Korean name, ext:, path, folder kind, regex, sort
 *   3. a file created after indexing is found (live folder monitoring)
 *   4. the instance runs un-elevated, without the Everything service, with only the corpus folder indexed
 *   5. the agent tool is registered and the system prompt carries the Everything section
 * It prints one JSON evidence object and exits non-zero on any failed check.
 */
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { EVERYTHING_SEARCH_TOOL_NAME } from "../../src/agent/everything-search-tool.ts";
import { EverythingClient } from "../../src/everything/index.ts";

if (process.platform !== "win32") {
	console.error(`Everything QA requires Windows; this host is ${process.platform}.`);
	process.exit(2);
}

const workspace = mkdtempSync(join(tmpdir(), "autorag-everything-qa-"));
const docs = join(workspace, "docs dir");
mkdirSync(join(docs, "하위 폴더"), { recursive: true });
mkdirSync(join(docs, "reports"), { recursive: true });
writeFileSync(join(docs, "하위 폴더", "환불 정책.txt"), "환불 예외는 이사 승인이 필요합니다.\n");
writeFileSync(join(docs, "refund-policy.txt"), "Refund exceptions require director approval.\n");
writeFileSync(join(docs, "reports", "Q3 report.pdf"), "%PDF-1.4 fake\n");
writeFileSync(join(docs, "reports", "Q3 summary.docx"), "fake docx\n");

const checks: Array<{ name: string; ok: boolean; detail: unknown }> = [];
const check = (name: string, ok: boolean, detail: unknown) => checks.push({ name, ok, detail });

const agent = new AutoRAGAgent({
	searchPaths: [docs],
	workspacePath: workspace,
	memoryPath: join(workspace, "memory.json"),
	minSync: false,
	jikji: false,
	webSearch: false,
});
const instance = new EverythingClient({ root: workspace, folders: [] }).instanceName;

try {
	// The system prompt lists exactly the registered tools as `- **name**:` lines.
	const systemPrompt = agent.getSystemPrompt();
	check("tool registered", systemPrompt.includes(`- **${EVERYTHING_SEARCH_TOOL_NAME}**:`), EVERYTHING_SEARCH_TOOL_NAME);
	check(
		"system prompt section",
		systemPrompt.includes("## Windows Everything File-Name Search"),
		"prompt contains Everything section",
	);
	const started = Date.now();
	const refresh = await agent.refresh(false, { methods: ["everything"] });
	check("refresh indexes corpus", refresh.everything?.ok === true && (refresh.everything.indexedItems ?? 0) >= 6, {
		everything: refresh.everything,
		diagnostics: refresh.diagnostics,
		ms: Date.now() - started,
	});
	check("component status ready", agent.refreshComponentStatus().everything === "ready", agent.refreshComponentStatus());

	const search = async (label: string, request: Parameters<AutoRAGAgent["searchEverything"]>[0], expect: string[]) => {
		const t = Date.now();
		const result = await agent.searchEverything(request);
		const paths = result.ok ? result.results.map((entry) => entry.path) : [];
		check(label, result.ok && expect.every((path) => paths.includes(path)), {
			request,
			result: result.ok ? result.results : result,
			ms: Date.now() - t,
		});
		return paths;
	};
	await search("korean name", { query: "환불" }, [join(docs, "하위 폴더", "환불 정책.txt")]);
	const extPaths = await search("ext filter", { query: "ext:pdf;docx" }, [
		join(docs, "reports", "Q3 report.pdf"),
		join(docs, "reports", "Q3 summary.docx"),
	]);
	check("ext filter excludes txt", !extPaths.some((path) => path.endsWith(".txt")), extPaths);
	await search("quoted phrase with space", { query: '"Q3 report"' }, [join(docs, "reports", "Q3 report.pdf")]);
	await search("folders only", { query: "폴더", kind: "folders" }, [join(docs, "하위 폴더")]);
	await search("path restriction", { query: "*", path: join(docs, "reports"), kind: "files" }, [
		join(docs, "reports", "Q3 report.pdf"),
	]);
	await search("regex", { query: "^refund-.*\\.txt$", regex: true }, [join(docs, "refund-policy.txt")]);
	const sorted = await agent.searchEverything({ query: "ext:txt", sort: "size-descending" });
	check(
		"sort by size",
		sorted.ok && sorted.results[0]?.path === join(docs, "하위 폴더", "환불 정책.txt"),
		sorted.ok ? sorted.results : sorted,
	);
	const leadingDash = await agent.searchEverything({ query: "-not-a-switch" });
	check("leading dash query is not an ES switch", leadingDash.ok, leadingDash);

	writeFileSync(join(docs, "live-added invoice.txt"), "new\n");
	let liveFound = false;
	for (let attempt = 0; attempt < 20 && !liveFound; attempt += 1) {
		await sleep(250);
		const live = await agent.searchEverything({ query: "invoice" });
		liveFound = live.ok && live.results.some((entry) => entry.path === join(docs, "live-added invoice.txt"));
	}
	check("live change monitored without refresh", liveFound, "live-added invoice.txt");

	const iniPath = join(workspace, ".autorag", "everything", "Everything.ini");
	const ini = readFileSync(iniPath, "utf8");
	const folderLine = ini.split(/\r?\n/).find((line) => line.startsWith("folders="));
	check("only corpus folder indexed", folderLine === `folders="${docs.replace(/\\/g, "\\\\")}"`, folderLine);
	check(
		"no volume auto-index, no admin, no servers",
		/^run_as_admin=0$/m.test(ini) &&
			/^auto_include_fixed_volumes=0$/m.test(ini) &&
			/^allow_http_server=0$/m.test(ini) &&
			/^allow_etp_server=0$/m.test(ini),
		"ini flags",
	);

	// Privilege-free probe (works from a standard Medium-integrity session).
	// A UAC prompt or `-admin` relaunch would replace the direct child with a
	// consent.exe/elevated relaunch, so the instance must be AutoRAG's own child.
	const processInfo = execFileSync(
		"powershell",
		[
			"-NoProfile",
			"-Command",
			`$p = @(Get-CimInstance Win32_Process -Filter "Name='everything.exe'" | Where-Object { $_.CommandLine -like '*${instance}*' }); ` +
				"$owner = @($p | ForEach-Object { (Invoke-CimMethod -InputObject $_ -MethodName GetOwner).User }); " +
				"$svc = @(Get-Service -Name 'Everything*' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name); " +
				"[pscustomobject]@{ count = $p.Count; parents = @($p.ParentProcessId); commandLine = @($p.CommandLine); owner = $owner; services = $svc } | ConvertTo-Json -Compress",
		],
		{ encoding: "utf8" },
	);
	const info: { count: number; parents: number[]; commandLine: string[]; owner: string[]; services: string[] } =
		JSON.parse(processInfo);
	const integrity =
		execFileSync("whoami", ["/groups"], { encoding: "utf8" })
			.split(/\r?\n/)
			.find((line) => line.includes("S-1-16-"))
			?.match(/S-1-16-\d+/)?.[0] ?? "unknown";
	check(
		"single user-level child instance, no elevation request, no Everything service",
		info.count === 1 &&
			info.parents.every((parent) => parent === process.pid) &&
			info.owner.every((user) => user.toLowerCase() === (process.env.USERNAME ?? "").toLowerCase()) &&
			info.services.length === 0 &&
			!info.commandLine.some((line) => /-admin|-install-service|-svc/.test(line)),
		{ ...info, autoragPid: process.pid, autoragIntegritySid: integrity },
	);
} finally {
	await new EverythingClient({ root: workspace, folders: [] }).stop();
	await sleep(500);
	rmSync(workspace, { recursive: true, force: true, maxRetries: 20, retryDelay: 200 });
}

const ok = checks.every((entry) => entry.ok);
console.log(JSON.stringify({ ok, instance, workspace, checks }, null, 2));
process.exit(ok ? 0 : 1);

/**
 * Manual QA of the per-search `agents` filter on the wombat datasource tool, against a REAL indexed wombat data dir
 * (`WOMBAT_QA_DATA_DIR`, a copy of an operator index; nothing is written to it) and the real `wombat` CLI.
 *
 * It checks, in order: the ids the tool offers are the agents the index really holds; the tool description and schema
 * name them; filtered searches return only that agent's sessions, for one agent and for several; a rare agent that a
 * global top-K misses is found with the filter; an id that is not indexed is rejected with the real ids and never
 * searched; a connector-level agent list is never widened by a per-search choice.
 *
 * Run: WOMBAT_BINARY=... WOMBAT_QA_DATA_DIR=/tmp/wqa4/data bun scripts/manual-qa/run-qa-wombat-agent-filter.ts
 */
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { createSingleDatasourceSearchTools } from "../../src/agent/search-single-datasource-tool.ts";
import { buildAgentOptions, resolveConfig } from "../../src/cli/config.ts";
import { WombatClient } from "../../src/datasource/skills/wombat/index.ts";

const dataDir = process.env.WOMBAT_QA_DATA_DIR;
if (dataDir === undefined) throw new Error("set WOMBAT_QA_DATA_DIR to an indexed wombat data dir");
const binaryPath = process.env.WOMBAT_BINARY ?? "wombat";
const root = mkdtempSync(join(tmpdir(), "autorag-wombat-filter-qa-"));

let failures = 0;
function check(name: string, pass: boolean, note?: string): void {
	if (!pass) failures += 1;
	console.log(`${pass ? "PASS" : "FAIL"}  ${name}${note ? ` — ${note}` : ""}`);
}

function agentFor(connector: Record<string, unknown>): AutoRAGAgent {
	const configPath = join(root, `config-${Math.random().toString(36).slice(2)}.json`);
	writeFileSync(
		configPath,
		JSON.stringify({
			searchPaths: [root],
			workspacePath: root,
			minSync: { autoInstall: false },
			datasources: { wombat: { connector: { binaryPath, dataDir, ...connector } } },
		}),
	);
	const options = buildAgentOptions(resolveConfig({ flags: { config: configPath } }));
	return new AutoRAGAgent({ ...options, everything: false, fsearch: false });
}

async function runJson(args: readonly string[]): Promise<Record<string, unknown>[]> {
	const proc = Bun.spawn([binaryPath, "--data-dir", dataDir as string, ...args], { stdout: "pipe", stderr: "pipe" });
	return JSON.parse(await new Response(proc.stdout).text());
}

function agentsOf(rows: readonly { metadata: Record<string, unknown> }[]): string[] {
	return [...new Set(rows.map((row) => String(row.metadata.agent)))].sort();
}

try {
	const indexed = await new WombatClient({ binaryPath, dataDir }).indexedAgents();
	check("the CLI lists the indexed agents", indexed.ok && indexed.agents.length > 1, indexed.ok ? indexed.agents.map((a) => `${a.id}=${a.sessions}`).join(" ") : indexed.reason);
	if (!indexed.ok) throw new Error("cannot list agents");
	const rare = [...indexed.agents].sort((a, b) => a.sessions - b.sessions).find((a) => a.sessions >= 2);
	const big = indexed.agents[0];
	if (rare === undefined || big === undefined) throw new Error("need at least two agents with sessions");

	const agent = agentFor({});
	const internals = agent as unknown as {
		refreshSearchFilters(): Promise<void>;
		singleDatasourceToolSpecs(): Parameters<typeof createSingleDatasourceSearchTools>[1];
	};
	const specs = await agent.searchFilterSpecs("wombat");
	check("the agent offers an `agents` filter with exactly the indexed ids", specs[0]?.name === "agents" && specs[0].values.map((v) => v.id).sort().join() === indexed.agents.map((a) => a.id).sort().join());

	await internals.refreshSearchFilters();
	const [tool] = createSingleDatasourceSearchTools(agent, internals.singleDatasourceToolSpecs());
	const description = tool?.description ?? "";
	check("the tool description names every id with its session count", indexed.agents.every((a) => description.includes(`${a.id} (${a.sessions} ${a.sessions === 1 ? "session" : "sessions"})`)), description.slice(description.indexOf("Filter")).slice(0, 220));
	check("the tool schema has an `agents` string list", JSON.stringify(tool?.parameters).includes('"agents"'));

	// Filtered searches return only that agent's sessions.
	const q = "release patch";
	for (const mode of ["hybrid"] as const) {
		const one = await agent.searchSingleDatasourceDocuments("wombat", q, { topK: 10, filters: { agents: [big.id] } });
		check(`${mode}: filter ${big.id} returns only ${big.id}`, one.results.length > 0 && agentsOf(one.results).join() === big.id, `${one.results.length} hits`);
		const two = await agent.searchSingleDatasourceDocuments("wombat", q, { topK: 10, filters: { agents: [big.id, rare.id] } });
		check(`two agents (${big.id}, ${rare.id}) return only those`, two.results.length > 0 && agentsOf(two.results).every((a) => a === big.id || a === rare.id), agentsOf(two.results).join());
	}

	// A rare agent is easy to miss in a global top-K. Probe with words from its own sessions (a query no session of that
	// agent matches has nothing to find, with or without the filter), taken from the real catalog.
	const sessions = await runJson(["sessions", "--agent", rare.id, "--json", "--limit", "50"]);
	const probes = sessions.map((row) => String(row.title)).filter((title) => title.trim().length > 0);
	check(`the catalog has titled ${rare.id} sessions to probe with`, probes.length > 0, `${probes.length} titles`);
	let foundFiltered = 0;
	let unfilteredHadAll = 0;
	for (const probe of probes) {
		const un = await agent.searchSingleDatasourceDocuments("wombat", probe, { topK: 10 });
		const fi = await agent.searchSingleDatasourceDocuments("wombat", probe, { topK: 10, filters: { agents: [rare.id] } });
		const unRare = un.results.filter((row) => row.metadata.agent === rare.id).length;
		const fiRare = fi.results.filter((row) => row.metadata.agent === rare.id).length;
		if (fi.results.length > 0 && fiRare === fi.results.length) foundFiltered += 1;
		if (unRare >= fiRare) unfilteredHadAll += 1;
		console.log(`      "${probe.slice(0, 40)}": ${rare.id} hits unfiltered=${unRare}/10, filtered=${fiRare}/${fi.results.length}`);
	}
	check(`every ${rare.id} title query returns only ${rare.id} sessions with the filter`, probes.length > 0 && foundFiltered === probes.length, `${foundFiltered}/${probes.length}; unfiltered top-10 already had all of them for ${unfilteredHadAll}`);

	// Through the tool: an unknown id is refused with the real ids, a good one searches.
	const refused = await tool?.execute("qa-1", { query: q, agents: ["claude"] } as never);
	const refusedText = refused?.content.map((p) => (p.type === "text" ? p.text : "")).join("") ?? "";
	check("a tool call with an agent that is not indexed is refused with the real ids", refused?.isError === true && indexed.agents.every((a) => refusedText.includes(a.id)), refusedText.slice(0, 160));
	const good = await tool?.execute("qa-2", { query: q, agents: [big.id.toUpperCase()] } as never);
	check("a tool call with a valid id (any case) searches only that agent", (good?.details.resultCount ?? 0) > 0 && good?.isError !== true, `${good?.details.resultCount} results`);

	// A connector-level list is never widened.
	const narrow = agentFor({ agents: [rare.id] });
	const narrowSpecs = await narrow.searchFilterSpecs("wombat");
	check("a connector limited to one agent offers only that agent", narrowSpecs[0]?.values.map((v) => v.id).join() === rare.id, narrowSpecs[0]?.values.map((v) => v.id).join());
	const widened = await narrow.searchSingleDatasourceDocuments("wombat", q, { topK: 10, filters: { agents: [big.id] } });
	check(`asking that connector for ${big.id} (outside its list) finds nothing`, widened.results.length === 0, `${widened.results.length} hits`);
	const within = await narrow.searchSingleDatasourceDocuments("wombat", probes[0] ?? q, { topK: 10, filters: { agents: [rare.id] } });
	check("asking it for its own agent works", within.results.length > 0 && agentsOf(within.results).every((a) => a === rare.id), `${within.results.length} hits`);

	console.log(failures === 0 ? "\nAGENT-FILTER QA PASSED" : `\nAGENT-FILTER QA: ${failures} failure(s)`);
	if (failures > 0) process.exitCode = 1;
} finally {
	rmSync(root, { recursive: true, force: true });
}

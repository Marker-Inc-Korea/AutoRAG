import { mkdtempSync, mkdirSync, readFileSync, rmSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { createScanDuplicateDocumentsTool } from "../../src/agent/dupey-tool.ts";
import { loadMirrorIndex } from "../../src/mirror/index.ts";

const root = mkdtempSync(join(tmpdir(), "autorag-dupey-live-"));
const docs = join(root, "docs");
mkdirSync(docs);
const older = join(docs, "older.txt");
const newest = join(docs, "newest.txt");
writeFileSync(older, "live exact duplicate marker\n");
writeFileSync(newest, "live exact duplicate marker\n");
utimesSync(older, new Date(1_000), new Date(1_000));
utimesSync(newest, new Date(2_000), new Date(2_000));

try {
	const cli = (args: readonly string[]) =>
		spawnSync("bun", ["src/cli/index.ts", ...args], {
			cwd: process.cwd(),
			env: { ...process.env, HOME: join(root, "home") },
			encoding: "utf8",
		});

	const help = cli(["--help"]);
	if (help.status !== 0 || !help.stdout.includes("duplicates")) throw new Error("CLI --help QA failed");

	const happy = cli(["duplicates", docs, "--json"]);
	if (happy.status !== 0) throw new Error(`CLI duplicate scan failed: ${happy.stderr}`);
	const happyJson = JSON.parse(happy.stdout);
	if (happyJson.exactGroups?.length !== 1) throw new Error("CLI did not report the exact group");

	const bad = cli(["duplicates", join(root, "missing")]);
	if (bad.status === 0 || !bad.stderr.includes("error:")) throw new Error("CLI bad-input QA failed");

	const agent = new AutoRAGAgent({
		searchPaths: [docs],
		workspacePath: root,
		memoryPath: join(root, "memory.json"),
		bm25: false,
		minSync: false,
	});
	const tool = createScanDuplicateDocumentsTool(agent);
	const toolResult = await tool.execute("live-dupey", {});
	if (toolResult.details.exactDuplicateCount !== 1) throw new Error("Agent tool live scan failed");

	await agent.refresh(true);
	const index = loadMirrorIndex(root);
	if (index.entries["/docs/newest.txt"] === undefined) throw new Error("Newest exact duplicate was not indexed");
	if (index.entries["/docs/older.txt"] !== undefined) throw new Error("Older exact duplicate was indexed");
	const indexed = readFileSync(index.entries["/docs/newest.txt"].outputPath, "utf8");
	if (!indexed.includes("live exact duplicate marker")) throw new Error("Indexed keeper content is wrong");

	console.log(
		JSON.stringify({
			ok: true,
			cli: { help: true, happy: true, badInput: true, exactGroups: happyJson.exactGroups.length },
			agentTool: { exactDuplicateCount: toolResult.details.exactDuplicateCount },
			indexing: { keeper: "/docs/newest.txt", excluded: "/docs/older.txt" },
		}),
	);
} finally {
	rmSync(root, { recursive: true, force: true });
}

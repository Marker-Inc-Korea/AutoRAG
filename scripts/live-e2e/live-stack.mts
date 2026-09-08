import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { AutoRAGAgent } from "../../src/agent/agent.ts";

const args = process.argv.slice(2);
const value = (name: string): string => {
	const index = args.indexOf(name);
	const result = index >= 0 ? args[index + 1] : undefined;
	if (!result) throw new Error(`missing ${name}`);
	return resolve(result);
};
const root = value("--root");
const workspace = value("--workspace");
const mode = args.includes("--mode") ? args[args.indexOf("--mode") + 1] : "warm";
if (mode !== "cold" && mode !== "warm") throw new Error("invalid mode");

const source = join(root, "corpus", "sample.txt");
const cursorPath = join(workspace, ".minsync", "cursor.json");
const cursorExistedBefore = existsSync(cursorPath);
const agent = new AutoRAGAgent({
	searchPaths: [join(root, "corpus")],
	workspacePath: root,
	memoryPath: join(workspace, "memory.json"),
	jikji: false,
	minSync: {
		workspacePath: workspace,
		autoInstall: false,
		embedder: {
			id: "tei:embeddinggemma:latest",
			baseUrl: "http://127.0.0.1:18080",
			dimension: 768,
			timeoutMs: 120_000,
		},
	},
});
const refresh = await agent.refresh(mode === "cold", { methods: ["parsed", "minsync"] });
if (refresh.minsync?.ok !== true) throw new Error(`minsync-${refresh.minsync?.reason ?? "unready"}`);
if (!existsSync(cursorPath)) throw new Error("minsync-cursor-missing");
const hits = await agent.retrieve("semantic question about the sample corpus", { topK: 5 });
const hit = hits.find((result) => result.source === resolve(source));
if (!hit) throw new Error("semantic-hit-source-mismatch");
if (!existsSync(hit.source)) throw new Error("source-missing");
const content = readFileSync(hit.source, "utf8");
if (content.length === 0) throw new Error("source-empty");
const status = await agent.getRefreshStatus();
const result = {
	mode,
	refresh: { parsed: { scanned: refresh.scanned, written: refresh.written, skipped: refresh.skipped }, minsync: refresh.minsync },
	cursorPath: resolve(cursorPath),
	cursorExists: true,
	hit: { source: hit.source, sourceAbsolute: true, sourceExists: true, sourceReadable: true, score: hit.score },
	status: { state: status.state, stale: status.stale, components: status.components },
	embedding: { endpoint: "http://127.0.0.1:18080", model: "embeddinggemma:latest", dimension: 768, loopbackOnly: true },
	openaiKeyPresent: Boolean(process.env.OPENAI_API_KEY || process.env.AUTORAG_OPENAI_API_KEY),
	incremental: mode === "warm" && cursorExistedBefore && refresh.written === 0,
	fullSync: !cursorExistedBefore,
};
writeFileSync(join(workspace, "live-stack-result.json"), `${JSON.stringify(result, null, 2)}\n`);

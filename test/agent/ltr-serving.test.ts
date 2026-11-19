import { randomUUID } from "node:crypto";
import { once } from "node:events";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import {
	type AssistantMessage,
	type Context,
	type FauxProviderRegistration,
	type FauxResponseStep,
	fauxAssistantMessage,
} from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent, type AutoRAGAgentOptions } from "../../src/agent/agent.ts";
import { MANAGE_LTR_TOOL_NAME } from "../../src/agent/ltr-tool.ts";
import { type LtrModelMeta, listLtrModels, ltrDirectory, saveLtrModel } from "../../src/ltr/model-store.ts";
import { MIN_TRAINING_RUNS } from "../../src/ltr/trainer.ts";
import { LTR_SERVING_FEATURE_NAMES } from "../../src/memory/ltr-export.ts";
import { RankingLogStore, rankingLogPath } from "../../src/memory/ranking-log.ts";
import type { RetrievalOptions, RetrievalResult } from "../../src/retrieval/types.ts";
import { baselineJev } from "../helpers/baseline-jev.ts";
import { recordedRun, stumpModelJson } from "../ltr/helpers.ts";

let root: string;
let docs: string;
let registrations: FauxProviderRegistration[];
beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-ltr-serving-"));
	docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	registrations = [];
});
afterEach(() => {
	for (const registration of registrations) registration.unregister();
	rmSync(root, { recursive: true, force: true });
});

function fauxModel(...responses: FauxResponseStep[]) {
	const registration = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "faux-model" }] });
	registration.setResponses(responses);
	registrations.push(registration);
	return registration.getModel();
}

function lastUserText(context: Context): string {
	const lastUser = [...context.messages].reverse().find((message) => message.role === "user");
	if (lastUser === undefined) return "";
	return typeof lastUser.content === "string"
		? lastUser.content
		: lastUser.content.map((part) => (part.type === "text" ? part.text : "")).join("");
}

/** Rerank endpoint that scores the document list in reverse: the last candidate wins, like a reranker that disagrees with dense retrieval. */
async function startRerankServer() {
	const server = createServer((request, response) => {
		let body = "";
		request.on("data", (chunk) => {
			body += chunk;
		});
		request.on("end", () => {
			const parsed = JSON.parse(body) as { documents: string[] };
			const results = parsed.documents
				.map((document, index) => ({
					document: { text: document },
					index,
					relevance_score: (index + 1) / parsed.documents.length,
				}))
				.reverse();
			response.writeHead(200, { "content-type": "application/json" });
			response.end(JSON.stringify({ model: "test-rerank", results }));
		});
	});
	server.listen(0, "127.0.0.1");
	await once(server, "listening");
	const address = server.address();
	const port = typeof address === "object" && address !== null ? address.port : 0;
	return {
		rerank: { provider: "openrouter", model: "test-rerank", apiKey: "k", baseUrl: `http://127.0.0.1:${port}/` },
		close: () => {
			server.closeAllConnections();
			server.close();
		},
	};
}

const CHUNKS = ["alpha budget", "beta budget", "gamma budget", "delta budget"];
const minSync = {
	isReady: () => true,
	isBinaryMissing: () => false,
	retrieve: async (query: string, _options: RetrievalOptions): Promise<RetrievalResult[]> =>
		CHUNKS.map((name, index) => ({
			id: `hit-${index}`,
			source: join(docs, `${name.split(" ")[0]}.txt`),
			content: `${name} for ${query}`,
			score: 1 - index * 0.1,
			metadata: { method: "minsync" },
		})),
};

function meta(id: string): LtrModelMeta {
	return {
		id,
		createdAt: 1_000,
		featureNames: [...LTR_SERVING_FEATURE_NAMES],
		objective: "rank:pairwise",
		trainedOnRuns: 100,
		loggedRunsAtTraining: 100,
		holdout: {
			queries: 20,
			baseline: { ndcg10: 0.5, mrr: 0.5 },
			model: { ndcg10: 0.6, mrr: 0.6 },
			deltaMrrCi95: [0, 0.2],
		},
		passedGate: true,
	};
}

/** Prefers the candidate dense retrieval ranked first, which the reranker stub puts last. */
const DENSE_FIRST = stumpModelJson([...LTR_SERVING_FEATURE_NAMES], "dense_best_rank", 1.5, 5, 0);

function buildAgent(options: Partial<AutoRAGAgentOptions> & { rerankUrl: AutoRAGAgentOptions["rerank"] }) {
	const prompts: string[] = [];
	const model = fauxModel(
		(context) => {
			prompts.push(lastUserText(context));
			return fauxAssistantMessage("Fast answer.", { stopReason: "stop" }) as AssistantMessage;
		},
		fauxAssistantMessage("Final answer.", { stopReason: "stop" }),
	);
	const { rerankUrl, ...rest } = options;
	const agent = new AutoRAGAgent({
		searchPaths: [docs],
		memoryPath: join(root, "memory.json"),
		workspacePath: root,
		minSync: { autoInstall: false },
		jikji: false,
		memoryEmbedder: false,
		jev: baselineJev(),
		model,
		rerank: rerankUrl,
		...rest,
	});
	(agent as unknown as { minSyncMethod: unknown }).minSyncMethod = minSync;
	return { agent, prompts };
}

async function runSearch(options: Partial<AutoRAGAgentOptions> & { rerankUrl: AutoRAGAgentOptions["rerank"] }) {
	const { agent, prompts } = buildAgent(options);
	const response = await agent.searchDocuments("what is the budget?");
	return { prompts, response };
}

/** Same private-field seam test/agent/agent.test.ts uses to look at the registered tools. */
function registeredTool(agent: AutoRAGAgent, name: string): AgentTool | undefined {
	return (agent as unknown as { tools: AgentTool[] }).tools.find((entry) => entry.name === name);
}

function baselineOrder(prompt: string): string[] {
	const order: string[] = [];
	for (const name of CHUNKS) {
		const at = prompt.indexOf(`${name} for`);
		if (at >= 0) order.push(`${at}:${name.split(" ")[0]}`);
	}
	return order
		.sort((a, b) => Number.parseInt(a, 10) - Number.parseInt(b, 10))
		.map((entry) => entry.split(":")[1] as string);
}

describe("learning-to-rank serving in the search pipeline", () => {
	it("leaves the reranker's order alone by default, even with a trained model on disk", async () => {
		saveLtrModel(ltrDirectory(join(root, "memory.json")), meta("m-1"), DENSE_FIRST);
		const server = await startRerankServer();
		try {
			const { prompts } = await runSearch({ rerankUrl: server.rerank });
			expect(baselineOrder(prompts[0] ?? "")).toEqual(["delta", "gamma", "beta", "alpha"]);
		} finally {
			server.close();
		}
	});

	it("reorders the baseline with the selected model when enabled, and logs which model served the run", async () => {
		saveLtrModel(ltrDirectory(join(root, "memory.json")), meta("m-1"), DENSE_FIRST);
		const server = await startRerankServer();
		try {
			const { prompts } = await runSearch({ rerankUrl: server.rerank, ltr: { enabled: true } });
			expect(baselineOrder(prompts[0] ?? "")[0]).toBe("alpha");
			const store = new RankingLogStore(rankingLogPath(join(root, "memory.json")));
			const [run] = store.list().runs;
			store.close();
			expect(run?.ltr).toEqual({ modelId: "m-1" });
			// Logged rerank is still the reranker's own order, so future training keeps its baseline.
			const alpha = run?.candidates.find((candidate) => candidate.source.endsWith("alpha.txt"));
			expect(alpha?.rerank?.rank).toBe(4);
			expect(alpha?.exposures[0]?.position).toBe(1);
		} finally {
			server.close();
		}
	});

	it("keeps the reranker's order and says why when enabled with no model that passed the gate", async () => {
		const server = await startRerankServer();
		try {
			const { prompts, response } = await runSearch({ rerankUrl: server.rerank, ltr: { enabled: true } });
			expect(baselineOrder(prompts[0] ?? "")).toEqual(["delta", "gamma", "beta", "alpha"]);
			expect(response.diagnostics?.find((diagnostic) => diagnostic.code === "ltr-unavailable")?.message).toContain(
				"autorag ltr train",
			);
		} finally {
			server.close();
		}
	});

	it("serves an explicitly selected model", async () => {
		saveLtrModel(ltrDirectory(join(root, "memory.json")), meta("m-pick"), DENSE_FIRST);
		const server = await startRerankServer();
		try {
			const { prompts } = await runSearch({ rerankUrl: server.rerank, ltr: { enabled: true, model: "m-pick" } });
			expect(baselineOrder(prompts[0] ?? "")[0]).toBe("alpha");
		} finally {
			server.close();
		}
	});

	it("never creates model files just because a search ran", async () => {
		const server = await startRerankServer();
		try {
			await runSearch({ rerankUrl: server.rerank, ltr: { enabled: true } });
			expect(existsSync(join(ltrDirectory(join(root, "memory.json")), "models"))).toBe(false);
		} finally {
			server.close();
		}
	});
});

describe("learning-to-rank management from the agent", () => {
	it("offers manage_ltr, and a change made through it applies to the next search and is saved through the host", async () => {
		saveLtrModel(ltrDirectory(join(root, "memory.json")), meta("m-1"), DENSE_FIRST);
		const server = await startRerankServer();
		const saved: unknown[] = [];
		try {
			const { agent, prompts } = buildAgent({
				rerankUrl: server.rerank,
				ltr: { enabled: false, persist: (settings) => saved.push(settings) },
			});
			const tool = registeredTool(agent, MANAGE_LTR_TOOL_NAME);
			expect(tool).toBeDefined();
			const reply = await tool?.execute("c1", { action: "set", enabled: true } as never);
			expect(JSON.stringify(reply?.content)).toContain("enabled");
			expect(saved).toEqual([{ enabled: true }]);
			await agent.searchDocuments("what is the budget?");
			expect(baselineOrder(prompts[0] ?? "")[0]).toBe("alpha");
		} finally {
			server.close();
		}
	});

	it("says settings cannot be saved when the host gave no way to save them", async () => {
		const server = await startRerankServer();
		try {
			const { agent } = buildAgent({ rerankUrl: server.rerank });
			const tool = registeredTool(agent, MANAGE_LTR_TOOL_NAME);
			await expect(tool?.execute("c1", { action: "set", enabled: true } as never)).rejects.toThrow(
				/no config file/u,
			);
		} finally {
			server.close();
		}
	});

	it("retrains in the background after a logged search when auto-train is due, without delaying the answer", async () => {
		const store = new RankingLogStore(rankingLogPath(join(root, "memory.json")));
		for (let i = 0; i < MIN_TRAINING_RUNS + 20; i += 1) {
			store.append(recordedRun(`seed-${String(i).padStart(3, "0")}`, 1_000 + i, { gold: 0 }).finish(true));
		}
		store.close();
		const python = join(root, "fake-python.sh");
		writeFileSync(python, `#!/bin/sh\ncat > "$4" <<'JSON'\n${DENSE_FIRST}\nJSON\necho '{"xgboost":"fake"}'\n`);
		chmodSync(python, 0o755);
		const server = await startRerankServer();
		try {
			const { agent } = buildAgent({
				rerankUrl: server.rerank,
				ltr: { enabled: false, autoTrain: { enabled: true, minNewRuns: 1 }, python: [python] },
			});
			await agent.searchDocuments("what is the budget?");
			const directory = ltrDirectory(join(root, "memory.json"));
			const deadline = Date.now() + 20_000;
			while (listLtrModels(directory).length === 0 && Date.now() < deadline)
				await new Promise((r) => setTimeout(r, 50));
			expect(listLtrModels(directory)).toHaveLength(1);
		} finally {
			server.close();
		}
	});

	it("does not train in the background unless auto-train is on", async () => {
		const store = new RankingLogStore(rankingLogPath(join(root, "memory.json")));
		for (let i = 0; i < MIN_TRAINING_RUNS + 20; i += 1) {
			store.append(recordedRun(`seed-${String(i).padStart(3, "0")}`, 1_000 + i, { gold: 0 }).finish(true));
		}
		store.close();
		const server = await startRerankServer();
		try {
			await runSearch({ rerankUrl: server.rerank, ltr: { enabled: false } });
			await new Promise((r) => setTimeout(r, 300));
			expect(existsSync(ltrDirectory(join(root, "memory.json")))).toBe(false);
		} finally {
			server.close();
		}
	});
});

import { homedir } from "node:os";
import { join } from "node:path";
import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Agent } from "@earendil-works/pi-agent-core";
import type { Api, Model } from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { loadManifests } from "../manifest/loader.ts";
import type { StoreManifest } from "../manifest/types.ts";
import { RetrievalMemory } from "../memory/memory.ts";
import { PosixRetrieval } from "../retrieval/posix.ts";
import { RetrievalMethodRegistry } from "../retrieval/registry.ts";
import { BM25Retrieval } from "../retrieval/stubs/bm25.ts";
import { HybridRetrieval } from "../retrieval/stubs/hybrid.ts";
import { VectorSearchRetrieval } from "../retrieval/stubs/vector.ts";
import { VisualRetrieval } from "../retrieval/stubs/visual.ts";
import type { RetrievalMethod, RetrievalOptions } from "../retrieval/types.ts";

export interface AutoRAGAgentOptions {
	model?: Model<Api>;
	searchPaths: string[];
	manifestDir?: string;
	memoryPath?: string;
}

const searchToolSchema = Type.Object({
	query: Type.String({ description: "Search query — text pattern, regex, or glob (e.g. **/*.ts)" }),
	topK: Type.Optional(Type.Number({ description: "Maximum number of results to return (default: 20)" })),
	scope: Type.Optional(Type.String({ description: "Directory to restrict search to" })),
});

function createSearchTool(method: RetrievalMethod): AgentTool<typeof searchToolSchema> {
	const desc = method.describe();
	return {
		name: `search_${desc.name}`,
		label: `Search (${desc.name})`,
		description: desc.description,
		parameters: searchToolSchema,
		async execute(
			_toolCallId: string,
			params: { query: string; topK?: number; scope?: string },
			signal?: AbortSignal,
		): Promise<AgentToolResult<{ resultCount: number; method: string }>> {
			const options: RetrievalOptions = {
				topK: params.topK ?? 20,
				scope: params.scope,
				signal,
			};
			const results = await method.retrieve(params.query, options);
			const text =
				results.length === 0 ? "No results found." : results.map((r) => `[${r.source}] ${r.content}`).join("\n");
			return {
				content: [{ type: "text", text }],
				details: { resultCount: results.length, method: desc.name },
			};
		},
	};
}

function buildSystemPrompt(registry: RetrievalMethodRegistry, manifests: StoreManifest[]): string {
	const methods = registry.list();
	const methodList = methods
		.map((m) => {
			const d = m.describe();
			return `- ${d.name} (${d.status}): ${d.description}`;
		})
		.join("\n");

	const manifestSection =
		manifests.length > 0
			? `\n\nAvailable indexed stores:\n${manifests.map((m) => `- ${m.name} (${m.type}): ${m.dataDescription || m.description}`).join("\n")}`
			: "";

	return `You are AutoRAG, a librarian agent that finds relevant resources using multiple retrieval methods.

Available retrieval methods:
${methodList}${manifestSection}

Use the search tools to find information. For content search, use grep-style patterns. For file discovery, use glob patterns like **/*.ts.
When a search returns no results, try different patterns or methods.`;
}

export class AutoRAGAgent {
	private readonly innerAgent: Agent;
	private readonly memory: RetrievalMemory;
	private readonly registry: RetrievalMethodRegistry;
	private lastQuery: string | undefined;

	constructor(options: AutoRAGAgentOptions) {
		const { searchPaths, manifestDir, memoryPath } = options;

		this.registry = new RetrievalMethodRegistry();
		this.registry.register(new PosixRetrieval({ defaultScope: searchPaths[0] ?? process.cwd() }));
		this.registry.register(new VectorSearchRetrieval());
		this.registry.register(new BM25Retrieval());
		this.registry.register(new HybridRetrieval());
		this.registry.register(new VisualRetrieval());

		const manifests = manifestDir ? loadManifests(manifestDir) : [];

		const memPath = memoryPath ?? join(homedir(), ".autorag", "memory.json");
		this.memory = new RetrievalMemory({ storagePath: memPath });
		this.memory.load();

		const tools = this.registry.list().map(createSearchTool);
		const systemPrompt = buildSystemPrompt(this.registry, manifests);

		this.innerAgent = new Agent({
			initialState: {
				systemPrompt,
				model: options.model as Model<Api>,
				tools,
			},
			convertToLlm: (messages) =>
				messages.filter((m) => m.role === "user" || m.role === "assistant" || m.role === "toolResult"),
		});
	}

	async prompt(text: string): Promise<void> {
		this.lastQuery = text;
		await this.innerAgent.prompt(text);
	}

	subscribe(listener: Parameters<Agent["subscribe"]>[0]): () => void {
		return this.innerAgent.subscribe(listener);
	}

	abort(): void {
		this.innerAgent.abort();
	}

	submitFeedback(satisfied: boolean): void {
		if (this.lastQuery) {
			this.memory.recordFeedback(this.lastQuery, "posix", satisfied);
			this.memory.save();
		}
	}

	getRegistry(): RetrievalMethodRegistry {
		return this.registry;
	}

	getSystemPrompt(): string {
		return this.innerAgent.state.systemPrompt;
	}
}

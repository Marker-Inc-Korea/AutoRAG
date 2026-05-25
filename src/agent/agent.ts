import { homedir } from "node:os";
import { join } from "node:path";
import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Agent } from "@earendil-works/pi-agent-core";
import type { Api, Model } from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { loadManifests } from "../manifest/loader.ts";
import type { StoreManifest } from "../manifest/types.ts";
import { createCheckMemoryTool } from "../memory/check-memory-tool.ts";
import { RetrievalMemory } from "../memory/memory.ts";
import { renderMemoryContext } from "../memory/renderer.ts";
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
	const active = methods.filter((m) => m.describe().status === "active");
	const stubs = methods.filter((m) => m.describe().status === "stub");
	const activeTypes = new Set(active.map((m) => m.describe().type));

	// Section 1: Identity & Role
	const identity = `You are AutoRAG, a librarian agent specializing in information retrieval across codebases and document collections.

Your job: find the most relevant resources for the caller using every available retrieval method, then return structured, actionable results with evidence.

You are invoked by a parent agent or user who needs specific information found. You do not write code, fix bugs, or make changes — you search, retrieve, and report.`;

	// Section 2: Active Retrieval Methods
	let methodsSection: string;
	if (active.length > 0) {
		const activeList = active
			.map((m) => {
				const d = m.describe();
				return `- **search_${d.name}** (tool name: search_${d.name})
  Type: ${d.type} | Status: ${d.status}
  ${d.description}
  Capabilities: ${d.capabilities.join(", ")}`;
			})
			.join("\n\n");
		methodsSection = `## Active Retrieval Methods

Use these tools to fulfill search requests. Prefer them in all queries.

${activeList}`;
	} else {
		methodsSection =
			"## Active Retrieval Methods\n\nNo active retrieval methods available. All registered methods are stubs awaiting implementation.";
	}

	// Section 2b: Stub Methods
	if (stubs.length > 0) {
		const stubList = stubs
			.map((m) => {
				const d = m.describe();
				return `- search_${d.name}: ${d.description} — **NOT AVAILABLE**`;
			})
			.join("\n");
		methodsSection += `

## Future Methods (Not Yet Available)

These retrieval methods are registered but not yet implemented. Do NOT call them — they will throw errors.

${stubList}`;
	}

	// Section 3: Indexed Data Stores (conditional)
	let storesSection = "";
	if (manifests.length > 0) {
		const storeList = manifests
			.map((m) => {
				const available = activeTypes.has(m.type);
				const note = available ? "" : `\n  ⚠ Requires search_${m.type} (currently unavailable)`;
				const contentLine = m.contentTypes.length > 0 ? `\n  Content types: ${m.contentTypes.join(", ")}` : "";
				return `- **${m.name}** (type: ${m.type})
  ${m.dataDescription || m.description}${contentLine}${note}`;
			})
			.join("\n\n");
		storesSection = `## Indexed Data Stores

Pre-indexed data stores available for retrieval. Cross-referenced against active methods above:

${storeList}`;
	}

	// Section 4: Search Strategy
	const strategySection = `## Search Strategy

### Query Formulation
- **Exact text/identifier**: use the literal string as query (e.g. "parseConfig")
- **File discovery by extension/path**: use glob patterns (e.g. "**/*.ts", "src/**/index.*")
- **Regex patterns**: use regex syntax (e.g. "function\\s+\\w+", "import.*from")
- **Finding definitions**: search for "function NAME", "class NAME", "interface NAME", "const NAME"
- **Finding usages**: search for the symbol name as a literal pattern

### Execution Rules
1. Start with the most specific query you can formulate.
2. If the query has multiple independent parts, execute searches in parallel.
3. If zero results: broaden — relax regex, try substrings, use glob to find candidate files first.
4. If too many results (>50): narrow with the scope parameter or a more specific pattern.
5. Use the scope parameter to restrict searches to relevant directories when you know the area.
6. Combine multiple retrieval methods when available — cross-validate findings.

### Fallback Chain (When a Search Returns No Results)
1. **Simplify**: remove regex metacharacters, try a plain substring
2. **Broaden**: use glob to discover files first, then grep within them
3. **Pivot**: try alternative terms (e.g. "error" → "Error" → "err" → "exception")
4. **Scope shift**: search in parent directories or remove the scope restriction entirely
5. **Rephrase**: reformulate from the caller's intent, not just their literal words`;

	// Section 4b: Memory & Strategy
	const memorySection = `## Memory & Strategy

You have access to retrieval memory from past searches. Use it to make better decisions:

1. **Automatic context**: Past query results are injected into the conversation automatically. Review them before choosing a method.
2. **check_memory tool**: Call \`check_memory\` with your planned query to see which methods succeeded or failed for similar past queries.
3. **Reason before searching**: Before executing a search, consider:
   - Have similar queries been tried before?
   - Which methods worked or failed?
   - Should you try a different method or refine the query?

Memory is advisory — it reflects past outcomes, not guarantees. New queries may behave differently.`;

	// Section 5: Output Format
	const outputSection = `## Output Format

Structure every response using this format:

<results>
<files>
- /path/to/file1.ts - Brief description of why this file is relevant
- /path/to/file2.ts:42 - What was found at this location
</files>

<answer>
Direct answer to the caller's question, grounded in evidence from search results.
Include file paths and line numbers as citations for every claim.
If nothing was found, state this explicitly and describe what was searched.
</answer>

<search_summary>
Methods used: list each method called
Queries executed: list each query with its result count
</search_summary>
</results>`;

	// Section 6: Behavioral Constraints
	const constraintsSection = `## Constraints

- **READ-ONLY**: you find and report — never suggest modifications or write files.
- **Search before answering**: never guess file contents or locations. Always query first.
- **No fabrication**: if you find nothing, report the negative result explicitly.
- **Cite evidence**: include file paths and line numbers for every factual claim.
- **Precision over recall**: a few highly relevant results beat many vague ones.
- **Address intent**: answer the caller's actual need, not just their literal query.`;

	// Section 7: Tool Quick Reference (active tools only)
	let toolRefSection = "";
	if (active.length > 0) {
		const toolRows = active
			.map((m) => {
				const d = m.describe();
				return `| search_${d.name} | query, topK?, scope? | ${d.capabilities.slice(0, 3).join(", ")} |`;
			})
			.join("\n");
		toolRefSection = `## Tool Quick Reference

| Tool | Parameters | Primary Use |
|------|-----------|-------------|
${toolRows}
| check_memory | query | Check past query outcomes before searching |

All search tools accept:
- **query** (string, required): search pattern — literal text, regex, or glob
- **topK** (number, optional): maximum results to return (default: 20)
- **scope** (string, optional): restrict search to this directory path`;
	}

	return [
		identity,
		methodsSection,
		storesSection,
		strategySection,
		memorySection,
		outputSection,
		constraintsSection,
		toolRefSection,
	]
		.filter(Boolean)
		.join("\n\n");
}

export class AutoRAGAgent {
	private readonly innerAgent: Agent;
	private readonly memory: RetrievalMemory;
	private readonly registry: RetrievalMethodRegistry;
	private lastQuery: string | undefined;
	private lastMethod: string | undefined;

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

		const searchTools = this.registry.list().map(createSearchTool);
		const checkMemoryTool = createCheckMemoryTool(this.memory);
		const tools = [...searchTools, checkMemoryTool];
		const systemPrompt = buildSystemPrompt(this.registry, manifests);

		this.innerAgent = new Agent({
			initialState: {
				systemPrompt,
				model: options.model as Model<Api>,
				tools,
			},
			convertToLlm: (messages) =>
				messages.filter((m) => m.role === "user" || m.role === "assistant" || m.role === "toolResult"),
			transformContext: async (messages) => {
				const entries = this.memory.getEntries();
				if (entries.length === 0) return messages;
				const summary = renderMemoryContext(entries);
				const memoryMessage = {
					role: "user" as const,
					content: [{ type: "text" as const, text: `<memory_context>\n${summary}\n</memory_context>` }],
					timestamp: Date.now(),
				};
				return [memoryMessage, ...messages];
			},
			afterToolCall: async (context) => {
				const toolName = context.toolCall.name;
				if (!toolName.startsWith("search_")) return undefined;
				const method = toolName.slice("search_".length);
				const details = context.result.details as { resultCount?: number } | undefined;
				const resultCount = details?.resultCount ?? 0;
				this.lastMethod = method;
				if (this.lastQuery) {
					this.memory.append({
						query: this.lastQuery,
						method,
						outcome: resultCount > 0 ? "success" : "failure",
						metadata: { resultCount },
					});
					this.memory.save();
				}
				return undefined;
			},
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
			const method = this.lastMethod ?? "posix";
			this.memory.recordFeedback(this.lastQuery, method, satisfied);
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

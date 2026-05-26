import { randomUUID } from "node:crypto";
import { homedir } from "node:os";
import { join } from "node:path";
import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Agent } from "@earendil-works/pi-agent-core";
import type { Api, Model } from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { loadManifests } from "../manifest/loader.ts";
import type { StoreManifest } from "../manifest/types.ts";
import { createCheckMemoryTool } from "../memory/check-memory-tool.ts";
import type { ResultFeedback } from "../memory/memory.ts";
import { RetrievalMemory } from "../memory/memory.ts";
import { renderMemoryContext } from "../memory/renderer.ts";
import { PosixRetrieval } from "../retrieval/posix.ts";
import { RetrievalMethodRegistry } from "../retrieval/registry.ts";
import { BM25Retrieval } from "../retrieval/stubs/bm25.ts";
import { HybridRetrieval } from "../retrieval/stubs/hybrid.ts";
import { VectorSearchRetrieval } from "../retrieval/stubs/vector.ts";
import { VisualRetrieval } from "../retrieval/stubs/visual.ts";
import type { CuratedResult, RetrievalMethod, RetrievalOptions } from "../retrieval/types.ts";
import { createReadFileTool } from "../tool/read-file.ts";

export interface PromptSession {
	sessionId: string;
	query: string;
	timestamp: number;
}

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
		): Promise<AgentToolResult<{ resultCount: number; method: string; sources: string[] }>> {
			const options: RetrievalOptions = {
				topK: params.topK ?? 20,
				scope: params.scope,
				signal,
			};
			const results = await method.retrieve(params.query, options);
			const sources = results.map((r) => r.source);
			const text =
				results.length === 0
					? "No results found."
					: `Found ${results.length} results:\n` +
						results.map((r, i) => `${i + 1}. ${r.source} — ${r.content}`).join("\n");
			return {
				content: [{ type: "text", text }],
				details: { resultCount: results.length, method: desc.name, sources },
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
	const identity = `You are AutoRAG, a librarian agent that searches, reads, curates, and reports information from codebases and document collections.

Your job: find relevant files, read their contents, extract the key insights, and deliver curated knowledge units to the caller. You do NOT output raw file paths or grep results — you curate information.

You are invoked by a parent agent or user who needs specific information found. You do not write code, fix bugs, or make changes.`;

	// Section 1b: Workflow
	const workflowSection = `## Workflow

1. **SEARCH** — Use search tools to find candidate files matching the query
2. **READ** — Use \`read_file\` to examine promising files from search results
3. **CURATE** — Extract key insights: function names, types, logic, purposes, line ranges
4. **OUTPUT** — Deliver numbered curated knowledge units (NO file paths exposed to caller)
5. **MAP** — Tag each knowledge unit with internal source for feedback tracking`;

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
[1] authenticate() function — Middleware that extracts and verifies JWT from Request.
    Parses Bearer header, calls jwt.verify, sets req.user. (lines 42-67)

[2] AuthConfig interface — JWT configuration type with secret, expiry, and refresh settings.
    Fields: tokenExpiry, refreshEnabled, secretKey. (lines 5-12)

<answer>
Direct answer to the caller's question. Reference results by number (e.g. [1], [2]).
If nothing was found, state this explicitly and describe what was searched.
</answer>
</results>

<internal_mapping>
1:src/middleware/auth.ts:posix
2:src/config/auth.ts:posix
</internal_mapping>

## Output Rules

- **NEVER** include file paths in <results> or <answer> blocks — the caller must not see them.
- Each [N] is a curated knowledge unit: name, purpose, key details, and line range.
- <internal_mapping> MUST appear AFTER </results> with format \`N:filepath:method\` per line.
- Every numbered unit MUST have a corresponding mapping entry.
- The caller can reference results by number for feedback (e.g. "1,3 useful").`;

	// Section 6: Behavioral Constraints
	const constraintsSection = `## Constraints

- **No raw paths**: never expose file paths in <results> or <answer>. Paths go only in <internal_mapping>.
- **READ-ONLY**: you find and report — never suggest modifications or write files.
- **Search then read**: search for candidates first, then read_file to examine content before curating.
- **No fabrication**: if you find nothing, report the negative result explicitly.
- **Curate, don't dump**: extract key insights — function names, types, purposes, line ranges. Not raw lines.
- **Precision over recall**: a few highly relevant curated units beat many vague ones.
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
| read_file | path, startLine?, endLine? | Read file content for curation |
| check_memory | query | Check past query outcomes before searching |

All search tools accept:
- **query** (string, required): search pattern — literal text, regex, or glob
- **topK** (number, optional): maximum results to return (default: 20)
- **scope** (string, optional): restrict search to this directory path`;
	}

	return [
		identity,
		workflowSection,
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

export function parseInternalMapping(text: string): CuratedResult[] {
	const match = text.match(/<internal_mapping>([\s\S]*?)<\/internal_mapping>/);
	if (!match) return [];
	const lines = match[1].trim().split("\n");
	const results: CuratedResult[] = [];
	for (const line of lines) {
		const trimmed = line.trim();
		if (!trimmed) continue;
		const firstColon = trimmed.indexOf(":");
		const lastColon = trimmed.lastIndexOf(":");
		if (firstColon === -1 || firstColon === lastColon) continue;
		const index = Number.parseInt(trimmed.slice(0, firstColon), 10);
		if (Number.isNaN(index)) continue;
		const source = trimmed.slice(firstColon + 1, lastColon);
		const method = trimmed.slice(lastColon + 1);
		results.push({ index, content: "", source, method });
	}
	return results;
}

export class AutoRAGAgent {
	private readonly innerAgent: Agent;
	private readonly memory: RetrievalMemory;
	private readonly registry: RetrievalMethodRegistry;
	private lastQuery: string | undefined;
	private lastSessionId: string | undefined;
	private lastMethod: string | undefined;
	private readonly sessions = new Map<string, { query: string; registry: Map<number, CuratedResult> }>();

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
		const readFileTool = createReadFileTool({ searchPaths });
		const checkMemoryTool = createCheckMemoryTool(this.memory);
		const tools = [...searchTools, readFileTool, checkMemoryTool];
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
				const details = context.result.details as { resultCount?: number; sources?: string[] } | undefined;
				const resultCount = details?.resultCount ?? 0;
				const sources = details?.sources ?? [];
				this.lastMethod = method;
				if (this.lastQuery) {
					const entry = this.memory.append({
						query: this.lastQuery,
						method,
						outcome: "pending",
						metadata: { resultCount },
					});
					this.memory.registerAttempt({
						id: entry.id,
						query: this.lastQuery,
						method,
						sources,
						timestamp: entry.timestamp,
					});
					this.memory.save();
				}
				return undefined;
			},
		});
	}

	async prompt(text: string): Promise<PromptSession> {
		const sessionId = randomUUID();
		this.lastQuery = text;
		this.lastSessionId = sessionId;
		const registry = new Map<number, CuratedResult>();
		this.sessions.set(sessionId, { query: text, registry });

		let lastAssistantText = "";
		const unsub = this.innerAgent.subscribe((event) => {
			if (event.type === "message_end" && "message" in event) {
				const msg = event.message as { role?: string; content?: Array<{ type: string; text?: string }> };
				if (msg.role === "assistant" && Array.isArray(msg.content)) {
					lastAssistantText = msg.content
						.filter((c) => c.type === "text" && c.text)
						.map((c) => c.text!)
						.join("\n");
				}
			}
		});

		try {
			await this.innerAgent.prompt(text);
		} finally {
			unsub();
		}

		const mapped = parseInternalMapping(lastAssistantText);
		for (const entry of mapped) {
			registry.set(entry.index, entry);
		}

		return { sessionId, query: text, timestamp: Date.now() };
	}

	subscribe(listener: Parameters<Agent["subscribe"]>[0]): () => void {
		return this.innerAgent.subscribe(listener);
	}

	abort(): void {
		this.innerAgent.abort();
	}

	submitFeedback(sessionId: string | undefined, satisfied: boolean): void {
		const sid = sessionId ?? this.lastSessionId;
		const session = sid ? this.sessions.get(sid) : undefined;
		const query = session?.query ?? this.lastQuery;
		if (query) {
			this.memory.resolvePendingEntries(query, null, satisfied ? "useful" : "not_useful");
			this.memory.save();
		}
	}

	recordResultFeedback(feedback: ResultFeedback[]): void {
		this.memory.recordResultFeedback(feedback);
		this.memory.save();
	}

	recordFeedbackByNumbers(sessionId: string, usefulNumbers: number[], notUsefulNumbers: number[] = []): void {
		const session = this.sessions.get(sessionId);
		if (!session) return;
		const feedback: ResultFeedback[] = [];
		for (const n of usefulNumbers) {
			const entry = session.registry.get(n);
			if (entry) feedback.push({ source: entry.source, useful: true });
		}
		for (const n of notUsefulNumbers) {
			const entry = session.registry.get(n);
			if (entry) feedback.push({ source: entry.source, useful: false });
		}
		if (feedback.length > 0) {
			this.recordResultFeedback(feedback);
		}
	}

	getRegistry(): RetrievalMethodRegistry {
		return this.registry;
	}

	getResultRegistry(sessionId?: string): ReadonlyMap<number, CuratedResult> {
		const sid = sessionId ?? this.lastSessionId;
		const session = sid ? this.sessions.get(sid) : undefined;
		return session?.registry ?? new Map();
	}

	getSystemPrompt(): string {
		return this.innerAgent.state.systemPrompt;
	}
}

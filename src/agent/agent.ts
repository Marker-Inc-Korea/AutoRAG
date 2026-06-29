import { randomUUID } from "node:crypto";
import { homedir } from "node:os";
import { join } from "node:path";
import { Agent, type AgentTool } from "@earendil-works/pi-agent-core";
import type { Api, Model } from "@earendil-works/pi-ai";
import { JikjiClient, type JikjiOptions, type JikjiPrepareResult } from "../jikji/index.ts";
import { loadManifests } from "../manifest/loader.ts";
import { createCheckMemoryTool } from "../memory/check-memory-tool.ts";
import type { ResultFeedback } from "../memory/memory.ts";
import { RetrievalMemory } from "../memory/memory.ts";
import { renderMemoryContext } from "../memory/renderer.ts";
import { type MinSyncSyncResult, MinSyncVectorMethod, type MinSyncVectorMethodOptions } from "../minsync/index.ts";
import { type ParsedMirrorSyncResult, syncParsedMirrors } from "../mirror/sync.ts";
import type { DefaultParserRegistryOptions } from "../parser/index.ts";
import { ParallelRetriever, ResultMerger } from "../retrieval/merger.ts";
import { PosixMethod } from "../retrieval/methods/posix.ts";
import { RetrievalMethodRegistry } from "../retrieval/registry.ts";
import type { CuratedResult, RetrievalOptions, RetrievalResult } from "../retrieval/types.ts";
import {
	type AutoRAGResultsDetails,
	createEmitResultsTool,
	EMIT_AUTORAG_RESULTS_TOOL_NAME,
} from "./emit-results-tool.ts";
import {
	createEmptySearchDocumentsResponse,
	recordNumberedFeedback,
	recordStructuredResultsSession,
	type SearchDocumentsResponse,
} from "./search-documents.ts";
import { buildSystemPrompt } from "./system-prompt.ts";

const SEARCH_TOOLS = ["grep", "find"] as const;

export interface AutoRefreshOptions {
	readonly intervalMs: number;
	readonly immediate?: boolean;
}

export interface AutoRAGAgentOptions {
	model?: Model<Api>;
	searchPaths: string[];
	manifestDir?: string;
	memoryPath?: string;
	workspacePath?: string;
	tools?: AgentTool[];
	minSync?: Omit<MinSyncVectorMethodOptions, "root">;
	jikji?: JikjiOptions;
	autoRefresh?: AutoRefreshOptions;
	parserOptions?: DefaultParserRegistryOptions;
}

export class AutoRAGAgent {
	private readonly innerAgent: Agent;
	private readonly memory: RetrievalMemory;
	private lastQuery: string | undefined;
	private lastSessionId: string | undefined;
	private readonly sessions = new Map<string, { query: string; registry: Map<number, CuratedResult> }>();
	private activeRun = false;
	private resultCapture: ((details: AutoRAGResultsDetails) => void) | undefined;
	private autoRefreshTimer: NodeJS.Timeout | undefined;
	private refreshing = false;

	private readonly searchPaths: string[];
	private readonly workspaceProjectRoot: string;
	private readonly methodRegistry = new RetrievalMethodRegistry();
	private readonly retriever = new ParallelRetriever();
	private readonly merger = new ResultMerger();
	private readonly minSyncMethod: MinSyncVectorMethod | undefined;
	private readonly jikjiClient: JikjiClient | undefined;
	private readonly parserOptions: DefaultParserRegistryOptions | undefined;

	constructor(options: AutoRAGAgentOptions) {
		const { manifestDir, memoryPath } = options;
		const manifests = manifestDir ? loadManifests(manifestDir) : [];

		this.searchPaths = options.searchPaths;
		this.workspaceProjectRoot = options.workspacePath ?? process.cwd();
		this.parserOptions = options.parserOptions;
		this.methodRegistry.register(new PosixMethod({ root: this.workspaceProjectRoot, searchPaths: this.searchPaths }));
		if (options.minSync) {
			this.minSyncMethod = new MinSyncVectorMethod({ ...options.minSync, root: this.workspaceProjectRoot });
			this.methodRegistry.register(this.minSyncMethod);
		}
		if (options.jikji) {
			this.jikjiClient = new JikjiClient(options.jikji);
		}

		const memPath = memoryPath ?? join(homedir(), ".autorag", "memory.json");
		this.memory = new RetrievalMemory({ storagePath: memPath });
		this.memory.load();

		const checkMemoryTool = createCheckMemoryTool(this.memory);
		const emitResultsTool = createEmitResultsTool((details) => this.resultCapture?.(details));
		const tools = [...(options.tools ?? []), checkMemoryTool, emitResultsTool];
		const toolNames = tools.map((tool) => tool.name);
		const systemPrompt = buildSystemPrompt({
			toolNames,
			memoryEntries: this.memory.getEntries(),
			manifests,
			jikjiIndexingEnabled: options.jikji !== undefined,
		});

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
				if (!this.lastQuery || !(SEARCH_TOOLS as readonly string[]).includes(toolName)) return undefined;

				const details = context.result.details as
					| { resultCount?: number; sources?: string[]; method?: string }
					| undefined;
				const resultCount = details?.resultCount ?? 0;
				const sources = details?.sources ?? [];
				const method = details?.method ?? toolName;
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
				return undefined;
			},
		});

		if (options.autoRefresh) {
			this.startAutoRefresh(options.autoRefresh.intervalMs, { immediate: options.autoRefresh.immediate });
		}
	}

	subscribe(listener: Parameters<Agent["subscribe"]>[0]): () => void {
		return this.innerAgent.subscribe(listener);
	}

	abort(): void {
		this.innerAgent.abort();
	}

	/**
	 * Periodically re-runs the incremental {@link refresh} so parsed mirrors and
	 * indexes stay current. Re-parsing is incremental (mtime/size) via the
	 * existing mirror sync; this only schedules it. Opt-in and stoppable.
	 */
	startAutoRefresh(intervalMs: number, options: { immediate?: boolean } = {}): void {
		this.stopAutoRefresh();
		const tick = () => {
			void this.runAutoRefreshTick();
		};
		this.autoRefreshTimer = setInterval(tick, intervalMs);
		this.autoRefreshTimer.unref();
		if (options.immediate) tick();
	}

	stopAutoRefresh(): void {
		if (this.autoRefreshTimer === undefined) return;
		clearInterval(this.autoRefreshTimer);
		this.autoRefreshTimer = undefined;
	}

	private async runAutoRefreshTick(): Promise<void> {
		if (this.refreshing) return;
		this.refreshing = true;
		try {
			await this.refresh(false);
		} catch {
			// Background auto-refresh is best-effort; keep the interval alive.
		} finally {
			this.refreshing = false;
		}
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
		recordNumberedFeedback(this.sessions, this.memory, sessionId, usefulNumbers, notUsefulNumbers);
	}

	getResultRegistry(sessionId?: string): ReadonlyMap<number, CuratedResult> {
		const sid = sessionId ?? this.lastSessionId;
		const session = sid ? this.sessions.get(sid) : undefined;
		return session?.registry ?? new Map();
	}

	async searchDocuments(query: string, options: RetrievalOptions = {}): Promise<SearchDocumentsResponse> {
		if (this.activeRun) {
			throw new Error("AutoRAG agent is busy; await the in-flight searchDocuments() call before starting another");
		}

		const sessionId = randomUUID();
		const trimmedQuery = query.trim();
		if (trimmedQuery.length === 0) {
			this.lastQuery = trimmedQuery;
			this.lastSessionId = sessionId;
			return createEmptySearchDocumentsResponse(sessionId, trimmedQuery, this.sessions);
		}

		this.activeRun = true;
		this.lastQuery = trimmedQuery;
		this.lastSessionId = sessionId;
		let captured: AutoRAGResultsDetails | undefined;
		this.resultCapture = (details) => {
			captured = details;
		};
		try {
			await this.innerAgent.prompt(this.buildSearchPrompt(trimmedQuery, options));
		} finally {
			this.resultCapture = undefined;
			this.activeRun = false;
		}

		if (captured === undefined) {
			throw new Error("AutoRAG agent completed without emitting structured results");
		}
		return recordStructuredResultsSession(sessionId, trimmedQuery, captured, this.sessions, this.memory);
	}

	private buildSearchPrompt(query: string, options: RetrievalOptions): string {
		const limit = typeof options.topK === "number" ? ` Return at most ${options.topK} curated results.` : "";
		return (
			`Find and curate information for this query: ${query}${limit}\n\n` +
			`When finished, call ${EMIT_AUTORAG_RESULTS_TOOL_NAME} exactly once as your final action with the curated ` +
			`results and the internal number-to-source mapping.`
		);
	}

	async refresh(force = false): Promise<ParsedMirrorSyncResult> {
		const summary = await this.syncParsedMirrors(force);
		await this.syncMinSync();
		await this.prepareJikji();
		return summary;
	}

	async syncParsedMirrors(force = false): Promise<ParsedMirrorSyncResult> {
		return syncParsedMirrors({
			root: this.workspaceProjectRoot,
			searchPaths: this.searchPaths,
			force,
			parserOptions: this.parserOptions,
		});
	}

	async syncMinSync(): Promise<MinSyncSyncResult | undefined> {
		return this.minSyncMethod?.sync();
	}

	async prepareJikji(): Promise<readonly JikjiPrepareResult[] | undefined> {
		if (this.jikjiClient === undefined) return undefined;
		const results: JikjiPrepareResult[] = [];
		for (const sourcePath of this.searchPaths) {
			results.push(await this.jikjiClient.prepare(sourcePath));
		}
		return results;
	}

	/**
	 * Programmatic retrieval across all registered methods (currently the
	 * real-directory `posix` method), merged via min-max normalization + source
	 * dedup. Activates the RetrievalMethodRegistry / ParallelRetriever /
	 * ResultMerger pipeline. Returns opaque root-relative sourced results.
	 */
	async retrieve(query: string, options: RetrievalOptions = {}): Promise<RetrievalResult[]> {
		const byMethod = await this.retriever.retrieve(this.methodRegistry.list(), query, options);
		return this.merger.merge(byMethod, { topK: options.topK ?? 20, dedup: true });
	}

	/** The retrieval method registry (posix active; vector/bm25/hybrid pluggable). */
	getMethodRegistry(): RetrievalMethodRegistry {
		return this.methodRegistry;
	}

	getSystemPrompt(): string {
		return this.innerAgent.state.systemPrompt;
	}
}

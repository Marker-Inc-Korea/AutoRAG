import { randomUUID } from "node:crypto";
import { watch as fsWatch } from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";
import { Agent, type AgentTool } from "@earendil-works/pi-agent-core";
import type { Api, Model } from "@earendil-works/pi-ai";
import { jikjiPrepareDiagnostic } from "../jikji/diagnostics.ts";
import { JikjiClient, type JikjiDiagnostic, type JikjiOptions, type JikjiPrepareResult } from "../jikji/index.ts";
import { loadManifests } from "../manifest/loader.ts";
import { createCheckMemoryTool } from "../memory/check-memory-tool.ts";
import type { ResultFeedback } from "../memory/memory.ts";
import { RetrievalMemory } from "../memory/memory.ts";
import { renderMemoryContext } from "../memory/renderer.ts";
import { type MinSyncSyncResult, MinSyncVectorMethod, type MinSyncVectorMethodOptions } from "../minsync/index.ts";
import {
	detectMirrorStaleness,
	type ParsedMirrorDiagnostic,
	type ParsedMirrorSyncResult,
	syncParsedMirrors,
} from "../mirror/sync.ts";
import type { DefaultParserRegistryOptions } from "../parser/index.ts";
import { ParallelRetriever, ResultMerger } from "../retrieval/merger.ts";
import { BM25Method, type BM25MethodOptions, type BM25SyncResult } from "../retrieval/methods/bm25.ts";
import { PosixMethod } from "../retrieval/methods/posix.ts";
import { RetrievalMethodRegistry } from "../retrieval/registry.ts";
import type { CuratedResult, RetrievalDiagnostic, RetrievalOptions, RetrievalResult } from "../retrieval/types.ts";
import {
	type AutoRAGResultsDetails,
	createEmitResultsTool,
	EMIT_AUTORAG_RESULTS_TOOL_NAME,
} from "./emit-results-tool.ts";
import { createSearchBM25DocumentsTool, SEARCH_BM25_DOCUMENTS_TOOL_NAME } from "./search-bm25-tool.ts";
import {
	createEmptySearchDocumentsResponse,
	recordNumberedFeedback,
	recordStructuredResultsSession,
	type SearchDocumentDiagnostic,
	type SearchDocumentsResponse,
} from "./search-documents.ts";
import { buildSystemPrompt } from "./system-prompt.ts";
import {
	createWatchRefresh,
	type WatcherFactory,
	type WatchRefreshHandle,
	type WatchWatcher,
} from "./watch-refresh.ts";

const SEARCH_TOOLS = ["grep", "find", SEARCH_BM25_DOCUMENTS_TOOL_NAME] as const;

export interface AutoRefreshOptions {
	readonly intervalMs: number;
	readonly immediate?: boolean;
}

export interface AutoRAGRefreshResult extends ParsedMirrorSyncResult {
	readonly bm25?: BM25SyncResult;
}

export interface AutoRAGRefreshComponentStatus {
	readonly bm25?: string;
	readonly minsync?: string;
	readonly jikji?: string;
}

/** Path-opaque snapshot of corpus freshness and the last refresh outcome. */
export interface AutoRAGRefreshStatus {
	readonly state: "idle" | "indexing" | "success" | "failed";
	readonly inFlight: boolean;
	readonly lastStartedAt?: string;
	readonly lastFinishedAt?: string;
	readonly counts?: {
		readonly scanned: number;
		readonly written: number;
		readonly deleted: number;
		readonly skipped: number;
	};
	readonly stale: boolean;
	readonly diagnostics: readonly SearchDocumentDiagnostic[];
	readonly components: AutoRAGRefreshComponentStatus;
	/** Path-free failure summary of the last refresh, if it failed. */
	readonly lastError?: string;
}

export interface AutoRAGWatchRefreshOptions {
	readonly debounceMs?: number;
	readonly force?: boolean;
	readonly maxWatchers?: number;
	/** Injectable watcher factory (defaults to a recursive fs.watch). Primarily for tests. */
	readonly watcherFactory?: WatcherFactory;
}

export type AutoRAGWatchRefreshHandle = WatchRefreshHandle;

interface RefreshState {
	inFlight: boolean;
	lastStartedAt?: string;
	lastFinishedAt?: string;
	lastOutcome: "never" | "success" | "failed";
	counts?: { scanned: number; written: number; deleted: number; skipped: number };
	mirrorDiagnostics: readonly ParsedMirrorDiagnostic[];
	jikjiDiagnostics: readonly JikjiDiagnostic[];
	minsync?: MinSyncSyncResult;
	lastError?: string;
	watchLimited: boolean;
	watchFailed: boolean;
}

export interface AutoRAGAgentOptions {
	model?: Model<Api>;
	searchPaths: string[];
	manifestDir?: string;
	memoryPath?: string;
	workspacePath?: string;
	tools?: AgentTool[];
	minSync?: Omit<MinSyncVectorMethodOptions, "root">;
	bm25?: Omit<BM25MethodOptions, "root">;
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
	private refreshState: RefreshState = {
		inFlight: false,
		lastOutcome: "never",
		mirrorDiagnostics: [],
		jikjiDiagnostics: [],
		watchLimited: false,
		watchFailed: false,
	};

	private readonly searchPaths: string[];
	private readonly workspaceProjectRoot: string;
	private readonly methodRegistry = new RetrievalMethodRegistry();
	private readonly retriever = new ParallelRetriever();
	private readonly merger = new ResultMerger();
	private readonly minSyncMethod: MinSyncVectorMethod | undefined;
	private readonly bm25Method: BM25Method | undefined;
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
		if (options.bm25) {
			this.bm25Method = new BM25Method({ ...options.bm25, root: this.workspaceProjectRoot });
			this.methodRegistry.register(this.bm25Method);
		}
		if (options.jikji) {
			this.jikjiClient = new JikjiClient(options.jikji);
		}

		const memPath = memoryPath ?? join(homedir(), ".autorag", "memory.json");
		this.memory = new RetrievalMemory({ storagePath: memPath });
		this.memory.load();

		const checkMemoryTool = createCheckMemoryTool(this.memory);
		const searchBM25Tool = createSearchBM25DocumentsTool(() => this.bm25Method);
		const emitResultsTool = createEmitResultsTool((details) => this.resultCapture?.(details));
		const tools = [...(options.tools ?? []), checkMemoryTool, searchBM25Tool, emitResultsTool];
		const toolNames = tools.map((tool) => tool.name);
		const systemPrompt = buildSystemPrompt({
			toolNames,
			memorySignalCount: this.memory.getSignalCount(),
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
				const hints = this.lastQuery ? this.memory.getMethodHints(this.lastQuery) : [];
				const insights = this.lastQuery ? this.memory.getInsights(this.lastQuery) : [];
				if (hints.length === 0 && insights.length === 0) return messages;
				const summary = renderMemoryContext(hints, { insights });
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
				const method = details?.method ?? toolName;
				this.memory.recordWeakSignal(this.lastQuery, method, "followup");
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
		return recordStructuredResultsSession(
			sessionId,
			trimmedQuery,
			captured,
			this.sessions,
			this.memory,
			{ workspaceRoot: this.workspaceProjectRoot, searchPaths: this.searchPaths },
			this.collectComponentDiagnostics(),
		);
	}

	/** Path-opaque component diagnostics (e.g. BM25 readiness) for the search response. */
	private collectComponentDiagnostics(): SearchDocumentDiagnostic[] {
		const diagnostics: SearchDocumentDiagnostic[] = [];
		const bm25 = this.bm25Method?.getStatus();
		if (bm25 !== undefined) {
			if (bm25.readiness === "degraded_fallback") {
				diagnostics.push({
					code: "bm25-degraded-fallback",
					severity: "warning",
					message: "BM25 is running in the TypeScript fallback engine; lexical ranking may be lower quality.",
					source: "bm25",
				});
			} else if (
				bm25.readiness === "dependency_unavailable" ||
				bm25.readiness === "index_missing" ||
				bm25.readiness === "error"
			) {
				diagnostics.push({
					code: "bm25-unavailable",
					severity: "warning",
					message: "BM25 lexical search is unavailable; results rely on other retrieval paths.",
					source: "bm25",
				});
			}
		}
		if (this.minSyncMethod?.isBinaryMissing()) {
			diagnostics.push({
				code: "minsync-unavailable",
				severity: "warning",
				message: "MinSync semantic search is unavailable; results rely on other retrieval paths.",
				source: "minsync",
			});
		}
		return diagnostics;
	}

	buildSearchPrompt(query: string, options: RetrievalOptions): string {
		const limit = typeof options.topK === "number" ? ` Return at most ${options.topK} curated results.` : "";
		const scope = options.scope ? ` Restrict search to virtual path scope ${options.scope}.` : "";
		return (
			`Find and curate information for this query: ${query}${limit}${scope}\n\n` +
			`When finished, call ${EMIT_AUTORAG_RESULTS_TOOL_NAME} exactly once as your final action with the curated ` +
			`results and the internal number-to-source mapping.`
		);
	}

	async refresh(force = false): Promise<AutoRAGRefreshResult> {
		this.refreshState = {
			...this.refreshState,
			inFlight: true,
			lastStartedAt: new Date().toISOString(),
		};
		try {
			const summary = await this.syncParsedMirrors(force);
			const bm25 = await this.syncBM25();
			const minsync = await this.syncMinSync();
			const jikji = await this.prepareJikji();
			const jikjiDiagnostics = (jikji ?? [])
				.map((result) => jikjiPrepareDiagnostic(result))
				.filter((diag): diag is JikjiDiagnostic => diag !== undefined);
			this.refreshState = {
				...this.refreshState,
				lastOutcome: "success",
				counts: {
					scanned: summary.scanned,
					written: summary.written,
					deleted: summary.deleted,
					skipped: summary.skipped,
				},
				mirrorDiagnostics: summary.diagnostics,
				jikjiDiagnostics,
				minsync,
				lastError: undefined,
			};
			return bm25 ? { ...summary, bm25 } : summary;
		} catch (error) {
			this.refreshState = {
				...this.refreshState,
				lastOutcome: "failed",
				lastError: error instanceof Error ? `Refresh failed: ${error.name}` : "Refresh failed.",
			};
			throw error;
		} finally {
			this.refreshState = {
				...this.refreshState,
				inFlight: false,
				lastFinishedAt: new Date().toISOString(),
			};
		}
	}

	/**
	 * Path-opaque snapshot of corpus freshness and the last refresh outcome. Runs
	 * a cheap parse-free staleness scan (stat only); never parses in this path.
	 */
	async getRefreshStatus(): Promise<AutoRAGRefreshStatus> {
		const staleDiagnostics = await detectMirrorStaleness({
			root: this.workspaceProjectRoot,
			searchPaths: this.searchPaths,
			parserOptions: this.parserOptions,
		});
		const diagnostics: SearchDocumentDiagnostic[] = [
			...this.refreshState.mirrorDiagnostics.map(toSearchDiagnostic),
			...staleDiagnostics.map(toSearchDiagnostic),
			...this.refreshState.jikjiDiagnostics.map((d) => ({
				code: d.code,
				severity: d.severity,
				message: d.message,
				source: d.source,
			})),
		];
		if (this.refreshState.watchLimited) {
			diagnostics.push({
				code: "watch-limited",
				severity: "warning",
				message: "Filesystem watch hit its watcher cap; some directories fall back to manual/polling refresh.",
				source: "watch",
			});
		}
		if (this.refreshState.watchFailed) {
			diagnostics.push({
				code: "watch-failed",
				severity: "warning",
				message: "A filesystem watcher could not be established for a configured search path.",
				source: "watch",
			});
		}
		const state: AutoRAGRefreshStatus["state"] = this.refreshState.inFlight
			? "indexing"
			: this.refreshState.lastOutcome === "never"
				? "idle"
				: this.refreshState.lastOutcome;
		return {
			state,
			inFlight: this.refreshState.inFlight,
			lastStartedAt: this.refreshState.lastStartedAt,
			lastFinishedAt: this.refreshState.lastFinishedAt,
			counts: this.refreshState.counts,
			stale: this.refreshState.lastOutcome === "never" || staleDiagnostics.length > 0,
			diagnostics,
			components: this.refreshComponentStatus(),
			lastError: this.refreshState.lastError,
		};
	}

	private refreshComponentStatus(): AutoRAGRefreshComponentStatus {
		const status: { bm25?: string; minsync?: string; jikji?: string } = {};
		const bm25 = this.bm25Method?.getStatus();
		if (bm25 !== undefined) status.bm25 = bm25.readiness;
		if (this.minSyncMethod !== undefined) {
			status.minsync = this.minSyncMethod.isBinaryMissing()
				? "unavailable"
				: this.refreshState.minsync?.ok === false
					? "degraded"
					: this.refreshState.minsync?.ok
						? "ready"
						: "configured";
		}
		if (this.jikjiClient !== undefined) {
			status.jikji = this.refreshState.jikjiDiagnostics.length > 0 ? "degraded" : "configured";
		}
		return status;
	}

	/**
	 * Opt-in filesystem watch that keeps parsed mirrors and configured indexes
	 * current. Debounced, backpressure-limited (one in-flight refresh plus one
	 * coalesced rerun), stoppable, and safe under rapid change bursts. Excludes
	 * `.autorag`/`.git`/`node_modules` and does not follow symlinks. Coexists with
	 * the polling {@link startAutoRefresh}. Returns a handle whose stop() closes
	 * every watcher and prevents any further scheduled refresh.
	 */
	startWatchRefresh(options: AutoRAGWatchRefreshOptions = {}): AutoRAGWatchRefreshHandle {
		this.refreshState = { ...this.refreshState, watchLimited: false, watchFailed: false };
		const dirs = this.searchPaths.map((searchPath) => resolve(searchPath));
		const watcherFactory = options.watcherFactory ?? this.defaultWatcherFactory();
		return createWatchRefresh({
			dirs,
			debounceMs: options.debounceMs ?? 200,
			maxWatchers: options.maxWatchers ?? 64,
			watcherFactory,
			runRefresh: async () => {
				await this.refresh(options.force ?? false);
			},
			onLimit: () => {
				this.refreshState = { ...this.refreshState, watchLimited: true };
			},
		});
	}

	private defaultWatcherFactory(): WatcherFactory {
		return (dir, onChange): WatchWatcher => {
			try {
				const watcher = fsWatch(dir, { recursive: true, persistent: false }, (_event, filename) => {
					onChange(typeof filename === "string" ? filename : null);
				});
				watcher.on("error", () => {
					this.refreshState = { ...this.refreshState, watchFailed: true };
				});
				return { close: () => watcher.close() };
			} catch {
				this.refreshState = { ...this.refreshState, watchFailed: true };
				return { close: () => {} };
			}
		};
	}

	async syncParsedMirrors(force = false): Promise<ParsedMirrorSyncResult> {
		return syncParsedMirrors({
			root: this.workspaceProjectRoot,
			searchPaths: this.searchPaths,
			force,
			parserOptions: this.parserOptions,
		});
	}

	async syncBM25(): Promise<BM25SyncResult | undefined> {
		return this.bm25Method?.sync();
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
	 * Programmatic retrieval across all registered methods, merged via min-max
	 * normalization + source dedup. Activates the RetrievalMethodRegistry /
	 * ParallelRetriever / ResultMerger pipeline. Returns opaque root-relative
	 * sourced results.
	 */
	async retrieve(query: string, options: RetrievalOptions = {}): Promise<RetrievalResult[]> {
		return (await this.retrieveWithDiagnostics(query, options)).results;
	}

	/**
	 * Programmatic retrieval that also returns path-opaque diagnostics for any
	 * retrieval method that failed (e.g. MinSync binary missing). Healthy method
	 * results are preserved. The legacy {@link retrieve} return shape is unchanged.
	 */
	async retrieveWithDiagnostics(
		query: string,
		options: RetrievalOptions = {},
	): Promise<{ results: RetrievalResult[]; diagnostics: RetrievalDiagnostic[] }> {
		const { results: byMethod, diagnostics } = await this.retriever.retrieveWithDiagnostics(
			this.methodRegistry.list(),
			query,
			options,
		);
		if (this.minSyncMethod?.isBinaryMissing() && !diagnostics.some((d) => d.source === "minsync")) {
			diagnostics.push({
				code: "minsync-unavailable",
				severity: "warning",
				message: "MinSync semantic search is unavailable; results rely on other retrieval paths.",
				source: "minsync",
			});
		}
		return {
			results: this.merger.merge(byMethod, { topK: options.topK ?? 20, dedup: true }),
			diagnostics,
		};
	}

	/** The retrieval method registry (posix active; vector/bm25/hybrid pluggable). */
	getMethodRegistry(): RetrievalMethodRegistry {
		return this.methodRegistry;
	}

	getSystemPrompt(): string {
		return this.innerAgent.state.systemPrompt;
	}
}

function toSearchDiagnostic(diagnostic: ParsedMirrorDiagnostic): SearchDocumentDiagnostic {
	return {
		code: diagnostic.code,
		severity: diagnostic.severity,
		message: diagnostic.message,
		source: diagnostic.source,
	};
}

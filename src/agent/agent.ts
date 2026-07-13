import { randomUUID } from "node:crypto";
import { watch as fsWatch } from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";
import { Agent, type AgentEvent, type AgentMessage, type AgentTool, type Skill } from "@earendil-works/pi-agent-core";
import type { Api, Model } from "@earendil-works/pi-ai";
import { DatasourceAccessContext, type DatasourceAccessContextOptions } from "../datasource/access-context.ts";
import { mapDatasourceDiagnostics } from "../datasource/diagnostics.ts";
import { DatasourceResultFilter } from "../datasource/result-filter.ts";
import type { DatasourceIndexResult, DatasourceSkill } from "../datasource/types.ts";
import { jikjiFindDiagnostic, jikjiPrepareDiagnostic } from "../jikji/diagnostics.ts";
import {
	type JikjiAnswerPack,
	type JikjiCandidate,
	JikjiClient,
	type JikjiDiagnostic,
	type JikjiEvidence,
	type JikjiFailureReason,
	type JikjiFindOptions,
	type JikjiFindResult,
	type JikjiHandoffAction,
	type JikjiOptions,
	type JikjiPrepareResult,
	normalizeJikjiAnswerPath,
	planJikjiSourceRoots,
} from "../jikji/index.ts";
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

import { RetrievalMethodRegistry } from "../retrieval/registry.ts";
import type { CuratedResult, RetrievalDiagnostic, RetrievalOptions, RetrievalResult } from "../retrieval/types.ts";
import { loadLocalAutoRAGModels } from "../subagents/local-models.ts";
import { EXPLORER_MODEL_ID, ORCHESTRATOR_MODEL_ID } from "../subagents/model-policy.ts";
import { createMandatorySubagentSession, type MandatorySubagentSessionOptions } from "../subagents/runtime.ts";
import { BASH_TOOL_NAME, createBashTool } from "./bash-tool.ts";
import {
	createLoadDatasourceSkillTool,
	LOAD_DATASOURCE_SKILL_TOOL_NAME,
	toDatasourceAgentSkill,
} from "./datasource-skill.ts";
import {
	type AutoRAGResultsDetails,
	createEmitResultsTool,
	EMIT_AUTORAG_RESULTS_TOOL_NAME,
} from "./emit-results-tool.ts";
import {
	createJikjiFindTool,
	JIKJI_FIND_TOOL_NAME,
	type JikjiFindPerRootPolicy,
	type JikjiFindProviderResult,
	type MergedJikjiPolicy,
} from "./jikji-find-tool.ts";
import { createSearchAllDocumentsTool, SEARCH_ALL_DOCUMENTS_TOOL_NAME } from "./search-all-tool.ts";
import { createSearchBM25DocumentsTool, SEARCH_BM25_DOCUMENTS_TOOL_NAME } from "./search-bm25-tool.ts";
import {
	createSearchDatasourceDocumentsTool,
	SEARCH_DATASOURCE_DOCUMENTS_TOOL_NAME,
} from "./search-datasource-tool.ts";
import {
	createEmptySearchDocumentsResponse,
	recordNumberedFeedback,
	recordStructuredResultsSession,
	type SearchDocumentDiagnostic,
	type SearchDocumentsResponse,
} from "./search-documents.ts";
import { createSearchMinSyncDocumentsTool, SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME } from "./search-minsync-tool.ts";

import { buildSystemPrompt, type SystemPromptConfig } from "./system-prompt.ts";
import {
	createWatchRefresh,
	type WatcherFactory,
	type WatchRefreshHandle,
	type WatchWatcher,
} from "./watch-refresh.ts";

const SEARCH_TOOLS = [
	BASH_TOOL_NAME,
	SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME,
	SEARCH_ALL_DOCUMENTS_TOOL_NAME,
	SEARCH_BM25_DOCUMENTS_TOOL_NAME,
	SEARCH_DATASOURCE_DOCUMENTS_TOOL_NAME,
	JIKJI_FIND_TOOL_NAME,
] as const;

export interface AutoRefreshOptions {
	readonly intervalMs: number;
	readonly immediate?: boolean;
}

export interface AutoRAGRefreshResult extends ParsedMirrorSyncResult {
	readonly bm25?: BM25SyncResult;
	readonly datasources?: readonly DatasourceIndexResult[];
}

export interface AutoRAGRefreshComponentStatus {
	readonly bm25?: string;
	readonly minsync?: string;
	readonly jikji?: string;
	readonly datasources?: string;
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
	datasources: readonly DatasourceIndexResult[];
	lastError?: string;
	watchLimited: boolean;
	watchFailed: boolean;
}

export interface AutoRAGAgentOptions {
	model?: Model<Api>;
	apiKey?: string;
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
	datasourceSkills?: readonly DatasourceSkill[];
	datasourceAccess?: DatasourceAccessContextOptions;
	sessionFactory?: AutoRAGSessionFactory;
}

export interface AutoRAGSearchSession {
	readonly agent: Agent;
	prompt(text: string): Promise<void>;
	abort(): Promise<void> | void;
	dispose(): void;
}

export type AutoRAGSessionFactory = (options: MandatorySubagentSessionOptions) => Promise<AutoRAGSearchSession>;

export type AutoRAGJikjiPrepareResult =
	| {
			readonly ok: true;
			readonly code: number;
			readonly diagnostics: readonly string[];
	  }
	| {
			readonly ok: false;
			readonly reason: JikjiFailureReason;
			readonly code: number | null;
			readonly diagnostics: readonly string[];
	  };

export class AutoRAGAgent {
	private readonly innerAgent: Agent;
	private readonly tools: readonly AgentTool[];
	private readonly configuredModel: Model<Api> | undefined;
	private readonly apiKey: string | undefined;
	private readonly sessionFactory: AutoRAGSessionFactory;
	private readonly usesDefaultSessionFactory: boolean;
	private readonly listeners = new Set<Parameters<Agent["subscribe"]>[0]>();
	private activeSession: AutoRAGSearchSession | undefined;
	private readonly pendingSubagentCalls = new Map<string, unknown>();
	private successfulExplorerCalls = 0;
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
		datasources: [],
		watchLimited: false,
		watchFailed: false,
	};

	private readonly searchPaths: string[];
	private readonly workspaceProjectRoot: string;
	private readonly methodRegistry = new RetrievalMethodRegistry();
	private readonly retriever = new ParallelRetriever();
	private readonly merger = new ResultMerger();
	private readonly datasourceFilter = new DatasourceResultFilter();

	private readonly minSyncMethod: MinSyncVectorMethod | undefined;
	private readonly bm25Method: BM25Method | undefined;
	private readonly jikjiClient: JikjiClient | undefined;
	private readonly datasourceSkills: readonly DatasourceSkill[];
	private readonly datasourceAccessOptions: DatasourceAccessContextOptions;
	private readonly datasourceAgentSkills: readonly Skill[];
	private readonly parserOptions: DefaultParserRegistryOptions | undefined;
	private readonly baseSystemPromptConfig: SystemPromptConfig;
	/** Run-scoped merged Jikji policy, set during searchDocuments; cleared after. */
	private activeJikjiPolicy: MergedJikjiPolicy | undefined;
	/** Run-scoped jikji_find call count for the two-phase raw-fallback gate. */
	private jikjiFindCallCount = 0;
	private readonly droppedCallerToolNames: readonly string[];

	constructor(options: AutoRAGAgentOptions) {
		const { manifestDir, memoryPath } = options;
		this.configuredModel = options.model;
		this.apiKey = options.apiKey;
		this.usesDefaultSessionFactory = options.sessionFactory === undefined;
		this.sessionFactory =
			options.sessionFactory ??
			(async (sessionOptions) => (await createMandatorySubagentSession(sessionOptions)).session);
		const manifests = manifestDir ? loadManifests(manifestDir) : [];
		this.datasourceSkills = options.datasourceSkills ?? [];
		this.datasourceAccessOptions = options.datasourceAccess ?? {};
		this.datasourceAgentSkills = this.buildAuthorizedDatasourceSkills();

		this.searchPaths = options.searchPaths;
		this.workspaceProjectRoot = options.workspacePath ?? process.cwd();
		this.parserOptions = options.parserOptions;

		if (options.minSync) {
			this.minSyncMethod = new MinSyncVectorMethod({ ...options.minSync, root: this.workspaceProjectRoot });
			this.methodRegistry.register(this.minSyncMethod);
		}
		if (options.bm25) {
			this.bm25Method = new BM25Method({ ...options.bm25, root: this.workspaceProjectRoot });
			this.methodRegistry.register(this.bm25Method);
		}
		for (const skill of this.datasourceSkills) {
			for (const method of skill.retrievalMethods()) this.methodRegistry.register(method);
		}
		if (options.jikji) {
			this.jikjiClient = new JikjiClient(options.jikji);
		}

		const memPath = memoryPath ?? join(homedir(), ".autorag", "memory.json");
		this.memory = new RetrievalMemory({ storagePath: memPath });
		this.memory.load();

		const checkMemoryTool = createCheckMemoryTool(this.memory);
		const searchBM25Tool = createSearchBM25DocumentsTool(() => this.bm25Method);
		const searchDatasourceTool = createSearchDatasourceDocumentsTool(this);

		const searchMinSyncTool = createSearchMinSyncDocumentsTool(() => this.minSyncMethod);
		const searchAllTool = createSearchAllDocumentsTool(this);
		const loadDatasourceSkillTool = createLoadDatasourceSkillTool(this);
		const emitResultsTool = createEmitResultsTool((details) => this.resultCapture?.(details));

		const bashTool = createBashTool({ cwd: this.workspaceProjectRoot, gate: () => this.bashGate() });

		const jikjiFindTool = this.jikjiClient !== undefined ? createJikjiFindTool(this) : undefined;

		// Reserved AutoRAG tool names the agent always owns. Caller tools with
		// these names are dropped (reserved wins), never rejected.
		const reservedNames = new Set<string>([
			BASH_TOOL_NAME,
			"check_memory",
			SEARCH_BM25_DOCUMENTS_TOOL_NAME,
			SEARCH_DATASOURCE_DOCUMENTS_TOOL_NAME,
			LOAD_DATASOURCE_SKILL_TOOL_NAME,
			EMIT_AUTORAG_RESULTS_TOOL_NAME,
			SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME,
			SEARCH_ALL_DOCUMENTS_TOOL_NAME,
			JIKJI_FIND_TOOL_NAME,
		]);
		const droppedCallerToolNames: string[] = [];
		const callerTools = (options.tools ?? []).filter((tool) => {
			if (reservedNames.has(tool.name)) {
				droppedCallerToolNames.push(tool.name);
				return false;
			}
			return true;
		});
		this.droppedCallerToolNames = [...new Set(droppedCallerToolNames)];

		// Deterministic, duplicate-free ordering: bash first, then surviving
		// caller tools, then AutoRAG-internal tools.
		const orderedTools: AgentTool[] = [
			bashTool,
			...callerTools,
			checkMemoryTool,
			searchBM25Tool,
			searchMinSyncTool,
			searchAllTool,
			searchDatasourceTool,
			loadDatasourceSkillTool,
			emitResultsTool,
			...(jikjiFindTool !== undefined ? [jikjiFindTool] : []),
		];
		const seenToolNames = new Set<string>();
		const tools = orderedTools.filter((tool) => {
			if (seenToolNames.has(tool.name)) return false;
			seenToolNames.add(tool.name);
			return true;
		});
		this.tools = tools;
		const toolNames = tools.map((tool) => tool.name);
		this.baseSystemPromptConfig = {
			toolNames,
			memorySignalCount: this.memory.getSignalCount(),
			manifests,
			datasourceSkills: this.datasourceAgentSkills,
			jikjiIndexingEnabled: options.jikji !== undefined,
		};
		const systemPrompt = buildSystemPrompt(this.currentSystemPromptConfig());

		this.innerAgent = new Agent({
			initialState: {
				systemPrompt,
				model: options.model as Model<Api>,
				tools,
			},
			convertToLlm: (messages) =>
				messages.filter((m) => m.role === "user" || m.role === "assistant" || m.role === "toolResult"),
			transformContext: async (messages) => this.withMemoryContext(messages),
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

	private async withMemoryContext(messages: AgentMessage[]): Promise<AgentMessage[]> {
		const hints = this.lastQuery ? this.memory.getMethodHints(this.lastQuery) : [];
		const insights = this.lastQuery ? this.memory.getInsights(this.lastQuery) : [];
		if (hints.length === 0 && insights.length === 0) return messages;
		const summary = renderMemoryContext(hints, { insights });
		return [
			{
				role: "user",
				content: [{ type: "text", text: `<memory_context>\n${summary}\n</memory_context>` }],
				timestamp: Date.now(),
			},
			...messages,
		];
	}

	private resolveSessionModel(): {
		readonly model: Model<Api>;
		readonly explorerModel: string;
		readonly apiKey?: string;
	} {
		if (this.configuredModel !== undefined) {
			if (this.usesDefaultSessionFactory && this.configuredModel.id !== ORCHESTRATOR_MODEL_ID) {
				throw new Error(
					`AutoRAG orchestrator must use ${ORCHESTRATOR_MODEL_ID}; received ${this.configuredModel.id}`,
				);
			}
			return {
				model: this.configuredModel,
				explorerModel: `${this.configuredModel.provider}/${EXPLORER_MODEL_ID}`,
				...(this.apiKey !== undefined ? { apiKey: this.apiKey } : {}),
			};
		}
		const local = loadLocalAutoRAGModels();
		return {
			model: local.orchestrator,
			explorerModel: `${local.provider}/${local.explorer.id}`,
			apiKey: local.apiKey,
		};
	}

	private configureSearchSession(session: AutoRAGSearchSession): readonly (() => void)[] {
		const extensionTransform = session.agent.transformContext;
		session.agent.transformContext = async (messages, signal) => {
			const transformed = extensionTransform === undefined ? messages : await extensionTransform(messages, signal);
			return this.withMemoryContext(transformed);
		};
		const unsubscribers = [...this.listeners].map((listener) => session.agent.subscribe(listener));
		unsubscribers.push(
			session.agent.subscribe((event) => {
				this.recordSearchToolEvent(event);
			}),
		);
		return unsubscribers;
	}

	private recordSearchToolEvent(event: AgentEvent): void {
		if (event.type === "tool_execution_start" && event.toolName === "subagent") {
			this.pendingSubagentCalls.set(event.toolCallId, event.args);
			return;
		}
		if (event.type === "tool_execution_end" && event.toolName === "subagent") {
			const args = this.pendingSubagentCalls.get(event.toolCallId);
			this.pendingSubagentCalls.delete(event.toolCallId);
			if (!event.isError && isRequiredExplorerInvocation(args)) this.successfulExplorerCalls += 1;
			return;
		}
		if (event.type !== "tool_execution_end" || !this.lastQuery) return;
		if (!(SEARCH_TOOLS as readonly string[]).includes(event.toolName)) return;
		const details = event.result.details as { method?: string } | undefined;
		this.memory.recordWeakSignal(this.lastQuery, details?.method ?? event.toolName, "followup");
		this.memory.save();
	}

	private currentSystemPromptConfig(): SystemPromptConfig {
		return {
			...this.baseSystemPromptConfig,
			memorySignalCount: this.memory.getSignalCount(),
		};
	}

	subscribe(listener: Parameters<Agent["subscribe"]>[0]): () => void {
		this.listeners.add(listener);
		return () => {
			this.listeners.delete(listener);
		};
	}

	abort(): void {
		void this.activeSession?.abort();
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
		this.activeJikjiPolicy = undefined;
		this.jikjiFindCallCount = 0;
		this.pendingSubagentCalls.clear();
		this.successfulExplorerCalls = 0;
		this.lastQuery = trimmedQuery;
		this.lastSessionId = sessionId;
		let captured: AutoRAGResultsDetails | undefined;
		let session: AutoRAGSearchSession | undefined;
		let unsubscribers: readonly (() => void)[] = [];
		this.resultCapture = (details) => {
			captured = details;
		};
		try {
			const resolved = this.resolveSessionModel();
			session = await this.sessionFactory({
				cwd: this.workspaceProjectRoot,
				model: resolved.model,
				systemPrompt: buildSystemPrompt(this.currentSystemPromptConfig()),
				tools: this.tools,
				...(resolved.apiKey !== undefined ? { apiKey: resolved.apiKey } : {}),
			});
			this.activeSession = session;
			unsubscribers = this.configureSearchSession(session);
			await session.prompt(this.buildSearchPrompt(trimmedQuery, options, resolved.explorerModel));
		} finally {
			for (const unsubscribe of unsubscribers) unsubscribe();
			session?.dispose();
			this.activeSession = undefined;
			this.resultCapture = undefined;
			this.activeRun = false;
			this.activeJikjiPolicy = undefined;
			this.jikjiFindCallCount = 0;
		}

		if (captured === undefined) {
			throw new Error("AutoRAG agent completed without emitting structured results");
		}
		if (this.successfulExplorerCalls === 0) {
			throw new Error(
				`AutoRAG requires a successful autorag-explorer subagent call using ${EXPLORER_MODEL_ID} before final curation`,
			);
		}
		return recordStructuredResultsSession(
			sessionId,
			trimmedQuery,
			captured,
			this.sessions,
			this.memory,
			this.collectComponentDiagnostics(),
		);
	}

	private datasourceAccessContext(options: RetrievalOptions = {}): DatasourceAccessContext {
		return new DatasourceAccessContext({
			allowedTags: options.allowedTags ?? this.datasourceAccessOptions.allowedTags,
			allowedScopes: options.allowedScopes ?? this.datasourceAccessOptions.allowedScopes,
		});
	}

	/**
	 * Build the Pi agent-skill list for datasource skills authorized by the
	 * trusted, server-bound access context. Only authorized skills become
	 * model-visible; unauthorized skills are omitted entirely (default-deny).
	 */
	private buildAuthorizedDatasourceSkills(): Skill[] {
		const ctx = this.datasourceAccessContext();
		const skills: Skill[] = [];
		for (const skill of this.datasourceSkills) {
			if (!ctx.isAccessible(skill.describe())) continue;
			skills.push(toDatasourceAgentSkill(skill.skillManifest()));
		}
		return skills;
	}

	/**
	 * Resolve an authorized datasource agent skill by model-visible name for the
	 * `load_datasource_skill` tool. Returns `undefined` for unknown or
	 * unauthorized names — model/tool input can never widen authorization.
	 */
	loadDatasourceSkill(name: string): Skill | undefined {
		return this.datasourceAgentSkills.find((skill) => skill.name === name);
	}

	private async indexDatasources(): Promise<readonly DatasourceIndexResult[]> {
		const results: DatasourceIndexResult[] = [];
		for (const skill of this.datasourceSkills) {
			try {
				results.push(await skill.index());
			} catch (error) {
				const descriptor = skill.describe();
				results.push({
					ok: false,
					instanceId: descriptor.instanceId ?? "default",
					skill: descriptor.name,
					indexedAt: Date.now(),
					diagnostics: [
						{
							code: "datasource-index-failed",
							severity: "error",
							message: error instanceof Error ? error.message : "Datasource indexing failed.",
							source: descriptor.name,
							instanceId: descriptor.instanceId,
						},
					],
					error: "datasource-index-failed",
					code: "datasource-index-failed",
					message: error instanceof Error ? error.message : "Datasource indexing failed.",
				});
			}
		}
		return results;
	}

	/** Path-opaque component diagnostics (e.g. BM25 readiness) for the search response. */
	private collectComponentDiagnostics(): SearchDocumentDiagnostic[] {
		const diagnostics: SearchDocumentDiagnostic[] = [];
		if (this.droppedCallerToolNames.length > 0) {
			diagnostics.push({
				code: "caller-tool-dropped",
				severity: "info",
				message:
					"One or more caller-provided tools were ignored because AutoRAG reserves read-only search tool names.",
				source: "tools",
			});
		}
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
		for (const result of this.refreshState.datasources) {
			diagnostics.push(...mapDatasourceDiagnostics(result.diagnostics));
		}
		return diagnostics;
	}

	buildSearchPrompt(query: string, options: RetrievalOptions, explorerModel?: string): string {
		const limit = typeof options.topK === "number" ? ` Return at most ${options.topK} curated results.` : "";
		const scope = options.scope ? ` Restrict search to virtual path scope ${options.scope}.` : "";
		const resolvedExplorerModel =
			explorerModel ?? `${this.configuredModel?.provider ?? "provider"}/${EXPLORER_MODEL_ID}`;
		return (
			`Find and curate information for this original query: ${query}${limit}${scope}\n\n` +
			`You must use the subagent tool before judging or emitting results; there is no single-agent fallback. ` +
			`For process-bound BM25, MinSync, Jikji, or datasource methods, call the matching AutoRAG tool only to create a bounded seed pack, then give that pack to an explorer for document reading; POSIX/bash discovery runs in the explorer. ` +
			`Dispatch one or more explorer tasks with agent autorag-explorer and model ${resolvedExplorerModel}. Each task must repeat the original query verbatim, ` +
			`name at least one selected retrieval method, provide multiple query variants, and request broad evidence coverage including weakly relevant candidates. ` +
			`Each explorer must return source-level evidence, location context, retrievedAt, source temporal metadata (or explicit unknown), and uncertainty. ` +
			`Explorers must not decide sufficiency, resolve conflicts, assign follow-ups, curate the final answer, or call ${EMIT_AUTORAG_RESULTS_TOOL_NAME}. ` +
			`The orchestrator alone performs final judgment, freshness checks, follow-up decisions, and final curation.\n\n` +
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
			const datasources = await this.indexDatasources();
			const jikji = await this.executeJikjiPrepare();
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
				datasources,
				lastError: undefined,
			};
			return { ...(bm25 ? { ...summary, bm25 } : summary), datasources };
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
		for (const result of this.refreshState.datasources) {
			diagnostics.push(...mapDatasourceDiagnostics(result.diagnostics));
		}
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
		const status: { bm25?: string; minsync?: string; jikji?: string; datasources?: string } = {};
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
		if (this.datasourceSkills.length > 0) {
			status.datasources = this.refreshState.datasources.some((result) => !result.ok) ? "degraded" : "configured";
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

	async prepareJikji(): Promise<readonly AutoRAGJikjiPrepareResult[] | undefined> {
		const results = await this.executeJikjiPrepare();
		return results?.map((result) => this.sanitizeJikjiPrepareResult(result));
	}

	private async executeJikjiPrepare(): Promise<readonly JikjiPrepareResult[] | undefined> {
		if (this.jikjiClient === undefined) return undefined;
		const results: JikjiPrepareResult[] = [];
		for (const sourcePath of this.searchPaths) {
			results.push(await this.jikjiClient.prepare(sourcePath));
		}
		return results;
	}

	private sanitizeJikjiPrepareResult(result: JikjiPrepareResult): AutoRAGJikjiPrepareResult {
		if (result.ok) {
			return {
				ok: true,
				code: result.code,
				diagnostics: [],
			};
		}
		return {
			ok: false,
			reason: result.reason,
			code: result.code,
			diagnostics: [],
		};
	}

	/**
	 * Provider method for the jikji_find tool. Runs JikjiClient.find over all
	 * configured search roots, normalizes answer paths against planned source
	 * roots, merges per-root answer packs using least-privilege (restrictive-
	 * wins) semantics, and sets the run-scoped activeJikjiPolicy. Only mutates
	 * run state (activeJikjiPolicy / jikjiFindCallCount) when a searchDocuments
	 * run is active; direct out-of-run calls compute and return the result
	 * without persisting. When jikji is unavailable or all roots fail, returns
	 * an unavailable result and leaves the policy undefined (bash is allowed).
	 */
	async findJikji(
		query: string,
		opts?: { readonly topK?: number; readonly first?: boolean },
	): Promise<JikjiFindProviderResult> {
		if (this.jikjiClient === undefined) {
			return { answerPack: undefined, policy: undefined, diagnostics: [], roots: [], perRoot: [] };
		}
		// Run-scoped state: only persist policy/count when a searchDocuments run
		// is active. Direct (out-of-run) calls compute and return the result
		// without mutating run state. The call count is incremented on EVERY
		// find attempt (success or failure) under an active run, so the
		// find-first bash gate releases after the first jikji_find — even when
		// all roots fail (policy stays undefined → bash allowed as fallback).
		const runScoped = this.activeRun === true;
		if (runScoped) {
			this.jikjiFindCallCount += 1;
		}
		const effectiveCount = this.jikjiFindCallCount;
		const sourceRoots = planJikjiSourceRoots(this.searchPaths);
		const findOpts: JikjiFindOptions = {
			topK: opts?.topK,
			first: opts?.first,
		};
		const diagnostics: JikjiDiagnostic[] = [];
		const okPacks: { pack: JikjiAnswerPack; root: string }[] = [];
		for (const sourcePath of this.searchPaths) {
			const result: JikjiFindResult = await this.jikjiClient.find(sourcePath, query, findOpts);
			if (result.ok) {
				okPacks.push({ pack: result.answerPack, root: sourcePath });
			} else {
				const diag = jikjiFindDiagnostic(result);
				if (diag !== undefined) diagnostics.push(diag);
			}
		}
		if (okPacks.length === 0) {
			return { answerPack: undefined, policy: undefined, diagnostics, roots: this.searchPaths, perRoot: [] };
		}

		// Per-root policy summaries, captured BEFORE the least-privilege merge.
		const perRoot: JikjiFindPerRootPolicy[] = okPacks.map((entry) => ({
			root: entry.root,
			handoffAction: entry.pack.handoffAction,
			stopAfterFind: entry.pack.toolCallPolicy.stopAfterFind,
			forbiddenTools: [...entry.pack.toolCallPolicy.forbiddenTools],
			allowedFollowups: [...entry.pack.toolCallPolicy.allowedFollowups],
			agentShouldNotRerank: entry.pack.agentShouldNotRerank,
		}));

		const policy = this.mergePolicy(
			okPacks.map((entry) => entry.pack),
			effectiveCount,
		);
		const merged = this.mergeAnswerPacks(okPacks, sourceRoots, policy);
		if (runScoped) {
			this.activeJikjiPolicy = policy;
		}
		return { answerPack: merged, policy, diagnostics, roots: this.searchPaths, perRoot };
	}

	/**
	 * Merge per-root answer packs into one. Concatenates answer_paths/candidates
	 * preserving per-root order; dedupes by normalized path. Does NOT cross-root
	 * rerank when any root has agentShouldNotRerank=true.
	 */
	private mergeAnswerPacks(
		entries: readonly { pack: JikjiAnswerPack; root: string }[],
		sourceRoots: ReturnType<typeof planJikjiSourceRoots>,
		policy: MergedJikjiPolicy,
	): JikjiAnswerPack {
		const seenPaths = new Set<string>();
		const answerPaths: string[] = [];
		const candidates: JikjiCandidate[] = [];
		const evidencePack: JikjiEvidence[] = [];
		const allPaths: string[] = [];

		for (const entry of entries) {
			// Root-provenance: normalize each entry's paths ONLY against that
			// entry's ORIGIN root, so a relative path from root B never resolves
			// against root A. If the origin root can't be resolved, skip the
			// entry's paths entirely. Global dedupe by normalized path remains.
			const originRoot = sourceRoots.find((sr) => sr.rootPath === resolve(entry.root));
			if (originRoot === undefined) continue;
			const originRoots = [originRoot];
			for (const rawPath of entry.pack.answerPaths) {
				const norm = normalizeJikjiAnswerPath(rawPath, originRoots);
				if (norm !== undefined && !seenPaths.has(norm)) {
					seenPaths.add(norm);
					answerPaths.push(norm);
				}
			}
			for (const rawPath of entry.pack.paths) {
				const norm = normalizeJikjiAnswerPath(rawPath, originRoots);
				if (norm !== undefined && !allPaths.includes(norm)) {
					allPaths.push(norm);
				}
			}
			for (const cand of entry.pack.candidates) {
				const norm = normalizeJikjiAnswerPath(cand.path, originRoots);
				if (norm !== undefined && !candidates.some((c) => c.path === norm)) {
					candidates.push({
						path: norm,
						nextRead: cand.nextRead,
						...(cand.label !== undefined ? { label: cand.label } : {}),
						...(cand.score !== undefined ? { score: cand.score } : {}),
					});
				}
			}
			for (const ev of entry.pack.evidencePack) {
				const norm = normalizeJikjiAnswerPath(ev.path, originRoots);
				if (norm !== undefined && !evidencePack.some((e) => e.path === norm)) {
					evidencePack.push({ path: norm, nextRead: ev.nextRead });
				}
			}
		}

		// Concatenation preserves per-root candidate order; no cross-root rerank.
		return {
			answerPaths,
			paths: allPaths,
			candidates,
			evidencePack,
			handoffAction: policy.handoffAction,
			toolCallPolicy: {
				stopAfterFind: policy.stopAfterFind,
				forbiddenTools: policy.forbiddenTools,
				allowedFollowups: policy.allowedFollowups,
			},
			agentShouldNotRerank: policy.agentShouldNotRerank,
		};
	}

	/**
	 * Least-privilege (restrictive-wins) merge of per-root policies.
	 * - forbiddenTools: UNION
	 * - allowedFollowups: INTERSECTION
	 * - stopAfterFind: OR
	 * - agentShouldNotRerank: OR
	 * - handoffAction: MOST RESTRICTIVE (direct_use < jikji_retry < raw_fallback_after_retry)
	 * - rawFallbackAllowed: handoffAction===raw_fallback_after_retry AND callCount>=2
	 */
	private mergePolicy(packs: readonly JikjiAnswerPack[], callCount: number): MergedJikjiPolicy {
		const HANDOFF_RANK: Record<JikjiHandoffAction, number> = {
			direct_use: 0,
			jikji_retry: 1,
			raw_fallback_after_retry: 2,
		};
		let handoff: JikjiHandoffAction = "raw_fallback_after_retry";
		let stopAfterFind = false;
		let agentShouldNotRerank = false;
		const forbidden = new Set<string>();
		let allowedFollowups: Set<string> | undefined;
		for (const pack of packs) {
			if (HANDOFF_RANK[pack.handoffAction] < HANDOFF_RANK[handoff]) {
				handoff = pack.handoffAction;
			}
			stopAfterFind = stopAfterFind || pack.toolCallPolicy.stopAfterFind;
			agentShouldNotRerank = agentShouldNotRerank || pack.agentShouldNotRerank;
			for (const tool of pack.toolCallPolicy.forbiddenTools) forbidden.add(tool);
			if (allowedFollowups === undefined) {
				allowedFollowups = new Set(pack.toolCallPolicy.allowedFollowups);
			} else {
				const next = new Set<string>();
				for (const f of pack.toolCallPolicy.allowedFollowups) {
					if (allowedFollowups.has(f)) next.add(f);
				}
				allowedFollowups = next;
			}
		}
		const rawFallbackAllowed = handoff === "raw_fallback_after_retry" && callCount >= 2;
		return {
			handoffAction: handoff,
			stopAfterFind,
			forbiddenTools: [...forbidden],
			allowedFollowups: allowedFollowups ? [...allowedFollowups] : [],
			agentShouldNotRerank,
			rawFallbackAllowed,
		};
	}

	/**
	 * Deny-by-default bash gate. When jikji is configured but no jikji_find has
	 * run yet this run (count===0), bash is blocked so the agent discovers local
	 * files via jikji_find first. After a find, the run-scoped activeJikjiPolicy
	 * applies: when no policy is active, bash behaves exactly as before
	 * (allowed). Under an active policy, bash is denied unless handoffAction is
	 * raw_fallback_after_retry AND rawFallbackAllowed is true (after a second
	 * jikji_find). When jikji is not configured, the find-first branch is
	 * skipped entirely (bash unchanged).
	 */
	private bashGate(): { allowed: boolean; message: string } {
		// Find-first: when jikji is configured but no jikji_find has run this run,
		// bash is blocked so the agent uses jikji_find for local discovery first.
		// After a jikji_find (success or failure), count>0 and the policy checks
		// below apply. If jikji was unavailable/all-failed, policy stays undefined
		// and the "no policy → allowed" path lets bash run (fallback). When jikji
		// is not configured, this branch is skipped (bash unchanged).
		if (this.jikjiClient !== undefined && this.jikjiFindCallCount === 0) {
			return {
				allowed: false,
				message:
					"Call jikji_find first for local file discovery (jikji is configured). Use bash only after jikji_find, per its policy.",
			};
		}
		const policy = this.activeJikjiPolicy;
		if (policy === undefined) return { allowed: true, message: "" };
		if (policy.forbiddenTools.includes("bash")) {
			return {
				allowed: false,
				message:
					"Bash is forbidden by the active Jikji policy (forbidden_tools includes bash). Use jikji_find answer_paths to answer directly.",
			};
		}
		if (policy.stopAfterFind) {
			return {
				allowed: false,
				message: "stop_after_find is active — answer from the jikji_find answer_paths. Raw shell is disallowed.",
			};
		}
		if (policy.handoffAction === "direct_use") {
			return {
				allowed: false,
				message: "Jikji policy is direct_use — use the jikji_find answer_paths directly. Raw shell is disallowed.",
			};
		}
		if (policy.handoffAction === "jikji_retry") {
			return {
				allowed: false,
				message: "Jikji policy is jikji_retry — retry jikji_find with a refined query. Raw shell is disallowed.",
			};
		}
		// handoffAction === "raw_fallback_after_retry"
		if (policy.rawFallbackAllowed) return { allowed: true, message: "" };
		return {
			allowed: false,
			message: "Raw fallback is allowed only after a second jikji_find. Retry jikji_find first before using bash.",
		};
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
	 * Programmatic retrieval that also returns diagnostics for any
	 * retrieval method that failed (e.g. MinSync binary missing). Healthy method
	 * results are preserved. The legacy {@link retrieve} return shape is unchanged.
	 */
	async retrieveWithDiagnostics(
		query: string,
		options: RetrievalOptions = {},
	): Promise<{ results: RetrievalResult[]; diagnostics: RetrievalDiagnostic[] }> {
		const methods = this.methodRegistry.list();
		const { results: byMethod, diagnostics } = await this.retriever.retrieveWithDiagnostics(methods, query, options);
		const filteredByMethod = this.datasourceFilter.filter(
			byMethod,
			methods,
			this.datasourceAccessContext(options),
			options.scope,
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
			results: this.merger.merge(filteredByMethod, { topK: options.topK ?? 20, dedup: true }),
			diagnostics,
		};
	}

	async searchAllDocuments(
		query: string,
		options: { readonly topK?: number; readonly scope?: string } = {},
	): Promise<{ results: RetrievalResult[]; diagnostics: RetrievalDiagnostic[] }> {
		return this.retrieveWithDiagnostics(query, { topK: options.topK, scope: options.scope });
	}

	async searchDatasourceDocuments(
		query: string,
		options: { readonly topK?: number; readonly scope?: string } = {},
	): Promise<{ results: RetrievalResult[]; diagnostics: RetrievalDiagnostic[] }> {
		const retrievalOptions: RetrievalOptions = { topK: options.topK, scope: options.scope };
		const ctx = this.datasourceAccessContext(retrievalOptions);
		const methods = this.methodRegistry.list().filter((method) => {
			const descriptor = method.describe();
			return descriptor.datasourceId !== undefined && ctx.isAccessible(descriptor);
		});
		if (methods.length === 0) return { results: [], diagnostics: [] };
		const { results: byMethod, diagnostics } = await this.retriever.retrieveWithDiagnostics(
			methods,
			query,
			retrievalOptions,
		);
		const filteredByMethod = this.datasourceFilter.filter(byMethod, methods, ctx, options.scope);
		return {
			results: this.merger.merge(filteredByMethod, { topK: options.topK ?? 20, dedup: true }),
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

function isRequiredExplorerInvocation(value: unknown): boolean {
	if (typeof value !== "object" || value === null) return false;
	const args = value as Record<string, unknown>;
	const matches = (task: unknown): boolean => {
		if (typeof task !== "object" || task === null) return false;
		const record = task as Record<string, unknown>;
		return (
			record.agent === "autorag-explorer" &&
			typeof record.model === "string" &&
			record.model.split(":", 1)[0]?.endsWith(`/${EXPLORER_MODEL_ID}`) === true
		);
	};
	if (matches(args)) return true;
	if (Array.isArray(args.tasks) && args.tasks.some(matches)) return true;
	if (Array.isArray(args.chain) && args.chain.some(matches)) return true;
	return false;
}

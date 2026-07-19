import { randomUUID } from "node:crypto";
import { watch as fsWatch, realpathSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { Agent, type AgentEvent, type AgentMessage, type AgentTool, type Skill } from "@earendil-works/pi-agent-core";
import type { Api, Model } from "@earendil-works/pi-ai";
import { resolveAutoRAGHome } from "../config/home.ts";
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
import { PARSED_MIRROR_SUBDIR } from "../mirror/paths.ts";
import {
	detectMirrorStaleness,
	type ParsedMirrorDiagnostic,
	type ParsedMirrorSyncResult,
	syncParsedMirrors,
} from "../mirror/sync.ts";
import { AutoRAGRunLogger } from "../observability/run-log.ts";
import type { DefaultParserRegistryOptions } from "../parser/index.ts";
import { ParallelRetriever, ResultMerger } from "../retrieval/merger.ts";
import { BM25Method, type BM25MethodOptions, type BM25SyncResult } from "../retrieval/methods/bm25.ts";

import { RetrievalMethodRegistry } from "../retrieval/registry.ts";
import type { CuratedResult, RetrievalDiagnostic, RetrievalOptions, RetrievalResult } from "../retrieval/types.ts";
import { safeParseExplorerReport } from "../subagents/contracts.ts";
import { buildDispatchTemplatesPromptSection } from "../subagents/dispatch-templates.ts";
import {
	autoragPrepare,
	classifyDispatch,
	createDispatchRejectionError,
	DispatchRejectionError,
	FORCE_CORRECTABLE_MAP,
	type PrepareContext,
	readAutofilled,
	validateLaunchPostSchema,
} from "../subagents/dispatch-validation.ts";
import { loadLocalAutoRAGModels } from "../subagents/local-models.ts";
import { EXPLORER_MODEL_ID } from "../subagents/model-policy.ts";
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

/** Methods that `refresh` can selectively run. Defaults to all when omitted. */
export type RefreshMethod = "parsed" | "bm25" | "minsync" | "datasources" | "jikji";

export interface AutoRAGRefreshOptions {
	/** Restrict refresh to specific methods. Defaults to all when undefined. */
	readonly methods?: readonly RefreshMethod[];
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
	explorerModel?: Model<Api>;
	apiKey?: string;
	providerApiKeys?: Readonly<Record<string, string>>;
	searchPaths: string[];
	manifestDir?: string;
	memoryPath?: string;
	workspacePath?: string;
	tools?: AgentTool[];
	minSync?: Omit<MinSyncVectorMethodOptions, "root"> | false;
	bm25?: Omit<BM25MethodOptions, "root"> | false;
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
	private readonly configuredExplorerModel: Model<Api> | undefined;
	private readonly apiKey: string | undefined;
	private readonly providerApiKeys: Readonly<Record<string, string>> | undefined;
	private readonly sessionFactory: AutoRAGSessionFactory;
	private readonly listeners = new Set<Parameters<Agent["subscribe"]>[0]>();
	private activeSession: AutoRAGSearchSession | undefined;
	private readonly pendingSubagentCalls = new Map<string, { readonly invocation: unknown; readonly query: string }>();
	private readonly dispatchSequenceBySession = new Map<string, number>();
	private readonly dispatchArgsTracker = new WeakMap<object, { toolCallId: string; rejected: boolean }>();
	private dispatchSyntheticCounter = 0;
	private successfulExplorerCalls = 0;
	private readonly memory: RetrievalMemory;
	private readonly runLogger: AutoRAGRunLogger;
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
	private readonly allowedExplorerRoots: readonly string[];
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
	private activeExplorerModel: string | undefined;
	/** Run-scoped jikji_find call count for the two-phase raw-fallback gate. */
	private jikjiFindCallCount = 0;
	private readonly droppedCallerToolNames: readonly string[];

	constructor(options: AutoRAGAgentOptions) {
		const { manifestDir, memoryPath } = options;
		this.configuredModel = options.model;
		this.configuredExplorerModel = options.explorerModel;
		this.apiKey = options.apiKey;
		this.providerApiKeys = options.providerApiKeys;
		this.sessionFactory =
			options.sessionFactory ??
			(async (sessionOptions) => (await createMandatorySubagentSession(sessionOptions)).session);
		const manifests = manifestDir ? loadManifests(manifestDir) : [];
		this.datasourceSkills = options.datasourceSkills ?? [];
		this.datasourceAccessOptions = options.datasourceAccess ?? {};
		this.datasourceAgentSkills = this.buildAuthorizedDatasourceSkills();

		this.searchPaths = options.searchPaths.map(pinSearchRoot);
		this.workspaceProjectRoot = options.workspacePath ?? process.cwd();
		this.allowedExplorerRoots = [...new Set(this.searchPaths)].sort();
		this.parserOptions = options.parserOptions;

		if (options.minSync !== false) {
			const minSyncOpts = options.minSync ?? { autoInstall: true };
			this.minSyncMethod = new MinSyncVectorMethod({ ...minSyncOpts, root: this.workspaceProjectRoot });
			this.methodRegistry.register(this.minSyncMethod);
		}
		if (options.bm25 !== false) {
			const bm25Opts = options.bm25 ?? {};
			this.bm25Method = new BM25Method({ ...bm25Opts, root: this.workspaceProjectRoot });
			this.methodRegistry.register(this.bm25Method);
		}
		for (const skill of this.datasourceSkills) {
			for (const method of skill.retrievalMethods()) this.methodRegistry.register(method);
		}
		if (options.jikji) {
			this.jikjiClient = new JikjiClient({ ...options.jikji, root: this.workspaceProjectRoot });
		}

		const memPath = memoryPath ?? join(resolveAutoRAGHome(), "memory.json");
		this.memory = new RetrievalMemory({ storagePath: memPath });
		this.memory.load();
		this.runLogger = new AutoRAGRunLogger(join(dirname(memPath), "logs", "runs.jsonl"));

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
			orchestratorModelId: options.model?.id,
			explorerModelId: options.explorerModel?.id,
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
		readonly explorerModel: Model<Api>;
		readonly apiKey?: string;
		readonly providerApiKeys?: Readonly<Record<string, string>>;
	} {
		if (this.configuredModel !== undefined) {
			const explorerModel =
				this.configuredExplorerModel ??
				({ ...this.configuredModel, id: EXPLORER_MODEL_ID, name: "GPT-5.6 Luna" } satisfies Model<Api>);
			return {
				model: this.configuredModel,
				explorerModel,
				...(this.apiKey !== undefined ? { apiKey: this.apiKey } : {}),
				...(this.providerApiKeys !== undefined ? { providerApiKeys: this.providerApiKeys } : {}),
			};
		}
		const local = loadLocalAutoRAGModels();
		return {
			model: local.orchestrator,
			explorerModel: local.explorer,
			apiKey: local.apiKey,
			providerApiKeys: { [local.provider]: local.apiKey },
		};
	}

	private configureSearchSession(session: AutoRAGSearchSession): readonly (() => void)[] {
		const extensionTransform = session.agent.transformContext;
		session.agent.transformContext = async (messages, signal) => {
			const transformed = extensionTransform === undefined ? messages : await extensionTransform(messages, signal);
			return this.withMemoryContext(transformed);
		};

		// Shallow-clone the subagent AgentTool with a composed prepareArguments.
		// The clone has the same name/schema/execute but adds our pre-schema
		// validation (classify → launch-only defaults → error catalog).
		// We replace the entry in agent.state.tools so the agent loop calls
		// our composed prepare before schema validation.
		const subagentIndex = session.agent.state.tools.findIndex((tool) => tool.name === "subagent");
		if (subagentIndex !== -1) {
			const originalTool = session.agent.state.tools[subagentIndex];
			const prepareCtx: PrepareContext = {
				configuredModel: this.activeExplorerModel,
			};
			const clonedTool: AgentTool = {
				...originalTool,
				prepareArguments: (rawArgs: unknown) => {
					// Track raw args by object identity for correlation.
					// The tracker keys on the original rawArgs object so that
					// beforeToolCall and tool_execution_start can resolve the
					// same correlation entry.
					if (rawArgs !== null && typeof rawArgs === "object" && !Array.isArray(rawArgs)) {
						const existing = this.dispatchArgsTracker.get(rawArgs as object);
						if (existing === undefined) {
							const syntheticId = `synthetic-${++this.dispatchSyntheticCounter}`;
							this.dispatchArgsTracker.set(rawArgs as object, {
								toolCallId: syntheticId,
								rejected: false,
							});
						}
					}

					// Compose the original tool's prepareArguments first (if it
					// exists), then pass its returned args into autoragPrepare.
					// This preserves any upstream argument preparation while
					// layering our dispatch validation on top. When the original
					// has no prepareArguments, fall back to the raw args.
					const baseArgs =
						originalTool.prepareArguments !== undefined ? originalTool.prepareArguments(rawArgs) : rawArgs;

					try {
						const prepared = autoragPrepare(baseArgs, prepareCtx);
						// Emit dispatch_autofilled if fields changed
						const autofilled = readAutofilled(prepared);
						if (autofilled !== undefined) {
							const changed = autofilled.artifacts || autofilled.agentScope || autofilled.leafModelFillCount > 0;
							if (changed) {
								this.emitDispatchAutofilled(rawArgs, autofilled);
							}
						}
						return prepared;
					} catch (error) {
						if (error instanceof DispatchRejectionError) {
							// Mark tracker as rejected so beforeToolCall doesn't duplicate
							if (rawArgs !== null && typeof rawArgs === "object" && !Array.isArray(rawArgs)) {
								const tracker = this.dispatchArgsTracker.get(rawArgs as object);
								if (tracker !== undefined) {
									tracker.rejected = true;
								}
							}
							this.emitDispatchRejected(error, rawArgs);
						}
						throw error;
					}
				},
			};
			const newTools = [...session.agent.state.tools];
			newTools[subagentIndex] = clonedTool;
			session.agent.state.tools = newTools;
		}

		const extensionBeforeToolCall = session.agent.beforeToolCall;
		session.agent.beforeToolCall = async (context, signal) => {
			if (context.toolCall.name === "subagent") {
				// Check if prepare already rejected (avoid duplicate reject)
				const rawArgs = context.toolCall.arguments;
				let alreadyRejected = false;
				if (rawArgs !== null && typeof rawArgs === "object" && !Array.isArray(rawArgs)) {
					const tracker = this.dispatchArgsTracker.get(rawArgs as object);
					if (tracker?.rejected) alreadyRejected = true;
				}

				if (!alreadyRejected) {
					const args = context.args;
					if (
						typeof args === "object" &&
						args !== null &&
						!Array.isArray(args) &&
						classifyDispatch(args as Record<string, unknown>) === "launch"
					) {
						const rejection = validateLaunchPostSchema(args as Record<string, unknown>, {
							configuredModel: this.activeExplorerModel,
							currentQuery: this.lastQuery,
							allowedRoots: this.allowedExplorerRoots,
							workspaceRoot: this.workspaceProjectRoot,
						});
						if (rejection !== undefined) {
							const error = createDispatchRejectionError(rejection, args as Record<string, unknown>);
							this.emitDispatchRejectedFromRejection(rejection, rawArgs);
							return { block: true, reason: error.message };
						}

						// Store normalized pending for success credit (structuredClone + deep-freeze).
						// Capture the query bound at dispatch time, never a later mutable value.
						const activeQuery = this.lastQuery;
						if (activeQuery !== undefined) {
							const normalizedSnapshot = deepFreeze(structuredClone(args));
							this.pendingSubagentCalls.set(context.toolCall.id, {
								invocation: normalizedSnapshot,
								query: activeQuery,
							});
						}
					}
				}
			}
			return extensionBeforeToolCall?.(context, signal);
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
			// Set the tracker with the real toolCallId from the start event.
			// This runs before prepareArguments (which looks up the tracker for telemetry).
			// If the tracker was already set in prepareArguments with a synthetic id,
			// update it with the real id.
			const rawArgs = event.args;
			if (rawArgs !== null && typeof rawArgs === "object" && !Array.isArray(rawArgs)) {
				const existing = this.dispatchArgsTracker.get(rawArgs as object);
				if (existing === undefined) {
					this.dispatchArgsTracker.set(rawArgs as object, {
						toolCallId: event.toolCallId,
						rejected: false,
					});
				} else if (existing.toolCallId.startsWith("synthetic-")) {
					existing.toolCallId = event.toolCallId;
				}
			}
			return;
		}
		if (event.type === "tool_execution_end" && event.toolName === "subagent") {
			const pending = this.pendingSubagentCalls.get(event.toolCallId);
			this.pendingSubagentCalls.delete(event.toolCallId);
			// Success credit uses only the frozen normalized snapshot and the
			// query captured at beforeToolCall, never the mutable `this.lastQuery`.
			if (
				!event.isError &&
				pending !== undefined &&
				isGroundedExplorerResult(event.result, pending.invocation, pending.query)
			) {
				this.successfulExplorerCalls += 1;
			}
			return;
		}
		if (event.type !== "tool_execution_end" || !this.lastQuery) return;
		if (!(SEARCH_TOOLS as readonly string[]).includes(event.toolName)) return;
		const details = event.result.details as { method?: string } | undefined;
		this.memory.recordWeakSignal(this.lastQuery, details?.method ?? event.toolName, "followup");
		this.memory.save();
	}

	private nextDispatchSequence(): number {
		const sid = this.lastSessionId ?? "unknown";
		const next = (this.dispatchSequenceBySession.get(sid) ?? 0) + 1;
		this.dispatchSequenceBySession.set(sid, next);
		return next;
	}

	private resolveToolCallId(rawArgs: unknown): string | null {
		if (rawArgs !== null && typeof rawArgs === "object" && !Array.isArray(rawArgs)) {
			const tracker = this.dispatchArgsTracker.get(rawArgs as object);
			if (tracker?.toolCallId.startsWith("synthetic-")) return null;
			return tracker?.toolCallId ?? null;
		}
		return null;
	}

	private emitDispatchAutofilled(
		rawArgs: unknown,
		autofilled: { readonly artifacts: boolean; readonly agentScope: boolean; readonly leafModelFillCount: number },
	): void {
		try {
			this.runLogger.write({
				event: "dispatch_autofilled",
				schemaVersion: 1,
				sessionId: this.lastSessionId ?? "unknown",
				toolCallId: this.resolveToolCallId(rawArgs),
				sequence: this.nextDispatchSequence(),
				timestamp: new Date().toISOString(),
				dispatchKind: "launch",
				fields: {
					artifacts: autofilled.artifacts,
					agentScope: autofilled.agentScope,
					leafModelFillCount: autofilled.leafModelFillCount,
				},
			});
		} catch {
			// Logger failures are nonfatal.
		}
	}

	private emitDispatchRejected(error: DispatchRejectionError, rawArgs: unknown): void {
		try {
			this.runLogger.write({
				event: "dispatch_rejected",
				schemaVersion: 1,
				sessionId: this.lastSessionId ?? "unknown",
				toolCallId: this.resolveToolCallId(rawArgs),
				sequence: this.nextDispatchSequence(),
				timestamp: new Date().toISOString(),
				dispatchKind: error.dispatchKind,
				code: error.code,
				field: error.field,
				forceCorrectable: error.forceCorrectable,
			});
		} catch {
			// Logger failures are nonfatal.
		}
	}

	private emitDispatchRejectedFromRejection(
		rejection: { readonly code: string; readonly field: string; readonly dispatchKind: string },
		rawArgs: unknown,
	): void {
		const code = rejection.code as keyof typeof FORCE_CORRECTABLE_MAP;
		try {
			this.runLogger.write({
				event: "dispatch_rejected",
				schemaVersion: 1,
				sessionId: this.lastSessionId ?? "unknown",
				toolCallId: this.resolveToolCallId(rawArgs),
				sequence: this.nextDispatchSequence(),
				timestamp: new Date().toISOString(),
				dispatchKind: rejection.dispatchKind as
					| "launch"
					| "admin"
					| "control"
					| "mutation"
					| "schedule"
					| "hybrid"
					| "unknown",
				code: rejection.code,
				field: rejection.field,
				forceCorrectable: FORCE_CORRECTABLE_MAP[code] ?? false,
			});
		} catch {
			// Logger failures are nonfatal.
		}
	}

	private currentSystemPromptConfig(models: Partial<SystemPromptConfig> = {}): SystemPromptConfig {
		return {
			...this.baseSystemPromptConfig,
			memorySignalCount: this.memory.getSignalCount(),
			...models,
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
		this.dispatchSequenceBySession.delete(sessionId);
		this.activeExplorerModel = undefined;
		this.lastQuery = trimmedQuery;
		this.lastSessionId = sessionId;
		let captured: AutoRAGResultsDetails | undefined;
		let session: AutoRAGSearchSession | undefined;
		let unsubscribers: readonly (() => void)[] = [];
		this.resultCapture = (details) => {
			captured = details;
		};
		let searchStarted = false;
		try {
			const resolved = this.resolveSessionModel();
			this.activeExplorerModel = modelReference(resolved.explorerModel);
			this.runLogger.write({
				event: "search_started",
				timestamp: new Date().toISOString(),
				sessionId,
				queryLength: trimmedQuery.length,
				orchestratorModel: resolved.model.id,
				explorerModel: resolved.explorerModel.id,
			});
			searchStarted = true;
			session = await this.sessionFactory({
				cwd: this.workspaceProjectRoot,
				model: resolved.model,
				systemPrompt: buildSystemPrompt(
					this.currentSystemPromptConfig({
						orchestratorModelId: resolved.model.id,
						explorerModelId: resolved.explorerModel.id,
					}),
				),
				tools: this.tools.filter((tool) => tool.name !== BASH_TOOL_NAME),
				...(resolved.apiKey !== undefined ? { apiKey: resolved.apiKey } : {}),
				...(resolved.providerApiKeys !== undefined ? { providerApiKeys: resolved.providerApiKeys } : {}),
				explorerModel: resolved.explorerModel,
			});
			this.activeSession = session;
			unsubscribers = this.configureSearchSession(session);
			await session.prompt(this.buildSearchPrompt(trimmedQuery, options, modelReference(resolved.explorerModel)));

			if (captured === undefined) {
				throw new Error("AutoRAG agent completed without emitting structured results");
			}
			if (this.successfulExplorerCalls === 0) {
				throw new Error(
					`AutoRAG requires a successful autorag-explorer subagent call using ${modelReference(resolved.explorerModel)} before final curation`,
				);
			}
			const response = recordStructuredResultsSession(
				sessionId,
				trimmedQuery,
				captured,
				this.sessions,
				this.memory,
				this.collectComponentDiagnostics(),
			);
			this.runLogger.write({
				event: "search_completed",
				timestamp: new Date().toISOString(),
				sessionId,
				resultCount: response.results.length,
			});
			return response;
		} catch (error) {
			if (searchStarted) {
				this.runLogger.write({
					event: "search_failed",
					timestamp: new Date().toISOString(),
					sessionId,
					errorType: error instanceof Error ? error.name : "UnknownError",
				});
			}
			throw error;
		} finally {
			const cleanupActions: readonly (() => void)[] = [
				...unsubscribers,
				() => {
					session?.dispose();
				},
			];
			const cleanupResults = await Promise.allSettled(
				cleanupActions.map((cleanup) => Promise.resolve().then(cleanup)),
			);
			const cleanupFailures = cleanupResults.filter(
				(result): result is PromiseRejectedResult => result.status === "rejected",
			);
			if (cleanupFailures.length > 0) {
				this.runLogger.write({
					event: "cleanup_failed",
					timestamp: new Date().toISOString(),
					sessionId,
					failureCount: cleanupFailures.length,
					errorTypes: [
						...new Set(
							cleanupFailures.map(({ reason }) => (reason instanceof Error ? reason.name : "UnknownError")),
						),
					],
				});
			}
			this.activeSession = undefined;
			this.resultCapture = undefined;
			this.activeRun = false;
			this.activeJikjiPolicy = undefined;
			this.jikjiFindCallCount = 0;
			this.activeExplorerModel = undefined;
		}
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
		const allowedRoots =
			this.allowedExplorerRoots.length > 0
				? this.allowedExplorerRoots.map((root) => `- ${root}`).join("\n")
				: "- (none configured)";
		const dispatchTemplates = buildDispatchTemplatesPromptSection(resolvedExplorerModel);
		return (
			`Find and curate information for this original query: ${query}${limit}${scope}\n\n` +
			`${dispatchTemplates}\n\n` +
			`You must use the subagent tool before judging or emitting results; there is no single-agent fallback. ` +
			`For process-bound BM25, MinSync, Jikji, or datasource methods, call the matching AutoRAG tool only to create a bounded seed pack, then give that pack to an explorer for document reading; POSIX/bash discovery runs in the explorer. ` +
			`Dispatch one or more explorer tasks with agent autorag-explorer and model ${resolvedExplorerModel}. Prefer the canonical AUTORAG_ASSIGNMENT_V1 block shown above; the legacy three-label assignment remains compatible. The structured originalQuery must equal the caller query exactly, query variants must cover the intent, and explorers must retain weakly relevant candidates that may explain conflicts or gaps. ` +
			`Missing or null top-level artifacts, agentScope, and missing explorer models are safely autofilled for launch dispatches only. Explicit wrong values remain rejected. Set artifacts and agentScope only at the top level; nested task items must omit them. Read-only diagnostic actions are validated separately from explorer launches. ` +
			`Allowed explorer roots (normalized):\n${allowedRoots}\n` +
			`Every autorag-explorer task must set an explicit cwd to exactly one allowed root above. If multiple roots are needed, dispatch one task per root and set each task's cwd to that root. ` +
			`Never use the workspace root or any path outside these roots for discovery unless that workspace root is explicitly listed above. ` +
			`Each explorer must return source-level evidence, location context, retrievedAt, source temporal metadata (or explicit unknown), and uncertainty. ` +
			`Explorers must not decide sufficiency, resolve conflicts, assign follow-ups, curate the final answer, or call ${EMIT_AUTORAG_RESULTS_TOOL_NAME}. ` +
			`The orchestrator alone performs final judgment, freshness checks, follow-up decisions, and final curation.\n\n` +
			`When finished, call ${EMIT_AUTORAG_RESULTS_TOOL_NAME} exactly once as your final action with the curated ` +
			`results and the internal number-to-source mapping.`
		);
	}

	async refresh(force = false, opts?: AutoRAGRefreshOptions): Promise<AutoRAGRefreshResult> {
		const methods = opts?.methods;
		const allMethods = methods === undefined;
		const wants = (m: RefreshMethod): boolean => allMethods || (methods as readonly RefreshMethod[]).includes(m);
		// Parsed mirror is required when any indexing method (bm25/minsync) runs,
		// since they index over the parsed mirrors. Also run it when explicitly
		// requested or when all methods are selected.
		const needsParsed = allMethods || wants("parsed") || wants("bm25") || wants("minsync");
		this.refreshState = {
			...this.refreshState,
			inFlight: true,
			lastStartedAt: new Date().toISOString(),
		};
		try {
			const summary = needsParsed ? await this.syncParsedMirrors(force) : await this.scanMirrorStaleness();
			const bm25 = wants("bm25") ? await this.syncBM25() : undefined;
			const minsync = wants("minsync") ? await this.syncMinSync() : undefined;
			const datasources = wants("datasources") ? await this.indexDatasources() : [];
			const jikji = wants("jikji") ? await this.executeJikjiPrepare() : undefined;
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

	/**
	 * Lightweight stat-only staleness scan used when `refresh` is called with
	 * methods that exclude parsed mirrors (e.g. only `datasources` or `jikji`).
	 * Returns a zero-count `ParsedMirrorSyncResult` carrying fresh diagnostics
	 * so the refresh result and status remain consistent.
	 */
	private async scanMirrorStaleness(): Promise<ParsedMirrorSyncResult> {
		const diagnostics = await detectMirrorStaleness({
			root: this.workspaceProjectRoot,
			searchPaths: this.searchPaths,
			parserOptions: this.parserOptions,
		});
		return {
			scanned: 0,
			written: 0,
			deleted: 0,
			skipped: 0,
			indexPath: join(this.workspaceProjectRoot, PARSED_MIRROR_SUBDIR),
			diagnostics,
		};
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

function pinSearchRoot(searchPath: string): string {
	const resolvedPath = resolve(searchPath);
	let canonicalPath: string;
	try {
		canonicalPath = realpathSync(resolvedPath);
	} catch (error) {
		if (hasFileSystemErrorCode(error, "ENOENT")) {
			throw new Error(`AutoRAG search root does not exist: ${resolvedPath}`, { cause: error });
		}
		if (hasFileSystemErrorCode(error, "ENOTDIR")) {
			throw new Error(`AutoRAG search root is not a directory: ${resolvedPath}`, { cause: error });
		}
		throw new Error(`AutoRAG search root could not be resolved: ${resolvedPath}`, { cause: error });
	}
	let isDirectory: boolean;
	try {
		isDirectory = statSync(canonicalPath).isDirectory();
	} catch (error) {
		if (hasFileSystemErrorCode(error, "ENOENT")) {
			throw new Error(`AutoRAG search root does not exist: ${resolvedPath}`, { cause: error });
		}
		throw new Error(`AutoRAG search root could not be inspected: ${resolvedPath}`, { cause: error });
	}
	if (!isDirectory) {
		throw new Error(`AutoRAG search root is not a directory: ${resolvedPath}`);
	}
	return canonicalPath;
}

function hasFileSystemErrorCode(error: unknown, code: string): boolean {
	return error instanceof Error && "code" in error && error.code === code;
}

function collectSubagentTasks(value: unknown): {
	readonly tasks: Record<string, unknown>[];
	readonly malformed: boolean;
} {
	const tasks: Record<string, unknown>[] = [];
	let malformed = false;
	const collectTasks = (valueToInspect: unknown, isInvocationRoot = false): void => {
		if (typeof valueToInspect !== "object" || valueToInspect === null || Array.isArray(valueToInspect)) {
			malformed = true;
			return;
		}
		const record = valueToInspect as Record<string, unknown>;
		const hasAgent = Object.hasOwn(record, "agent");
		const nestedValues = ["tasks", "chain", "parallel"]
			.map((key) => record[key])
			.filter((nested) => nested !== undefined);
		if (hasAgent) tasks.push(record);
		for (const nested of nestedValues) {
			if (Array.isArray(nested)) {
				if (nested.length === 0) malformed = true;
				for (const task of nested) collectTasks(task);
				continue;
			}
			if (typeof nested === "object" && nested !== null) {
				collectTasks(nested);
				continue;
			}
			malformed = true;
		}
		if (!hasAgent && nestedValues.length === 0) {
			if (isInvocationRoot) malformed = true;
			else tasks.push(record);
		}
	};
	collectTasks(value, true);
	return { tasks, malformed };
}

function isPlaceholderOriginalQuery(value: string): boolean {
	const normalized = value.trim().toLowerCase().replace(/\s+/g, " ");
	if (!isSubstantiveTextValue(normalized)) return true;
	return (
		/^(?:<|\[|\{)?(?:the )?(?:(?:original|user|active|current|caller|search) )?query(?: here)?(?:>|\]|\})?$/.test(
			normalized,
		) ||
		/^(?:the )?(?:(?:original|user|active|current|caller|search) )?query(?:\s+(?:goes?\s+(?:here|below)|placeholder|template|value|text|pending|missing|to be (?:provided|filled(?: in)?|inserted|added|included|supplied)))?$/.test(
			normalized,
		) ||
		/^(?:placeholder|tbd|todo|same query|same as above|not provided|omitted)$/.test(normalized)
	);
}

function modelReference(model: Model<Api>): string {
	return `${model.provider}/${model.id}`;
}

function isGroundedExplorerResult(value: unknown, invocation: unknown, currentQuery: string): boolean {
	if (!isRecord(value) || !isRecord(value.details) || !Array.isArray(value.details.results)) return false;
	const childSet = collectSubagentTasks(invocation);
	if (childSet.malformed || childSet.tasks.length === 0) return false;
	return value.details.results.some(
		(child, index) => childSet.tasks[index] !== undefined && isGroundedExplorerChild(child, currentQuery),
	);
}

function isGroundedExplorerChild(value: unknown, currentQuery: string): boolean {
	if (!isRecord(value) || value.agent !== "autorag-explorer" || value.exitCode !== 0) return false;
	const structuredReport = classifyExplorerReportGrounding(value.structuredOutput, currentQuery);
	if (structuredReport !== "absent") return structuredReport === "grounded";
	return typeof value.finalOutput === "string" && hasGroundedFinalOutput(value.finalOutput, currentQuery);
}

type ExplorerReportGrounding = "absent" | "grounded" | "rejected";

function classifyExplorerReportGrounding(value: unknown, currentQuery: string): ExplorerReportGrounding {
	const parsedValue = typeof value === "string" ? parseJsonValue(value) : value;
	if (!isRecord(parsedValue)) return "absent";
	if (
		!Object.hasOwn(parsedValue, "assignment") &&
		!Object.hasOwn(parsedValue, "evidenceCandidates") &&
		!Object.hasOwn(parsedValue, "candidates")
	) {
		return "absent";
	}
	const report = safeParseExplorerReport(parsedValue);
	if (report === undefined) return "rejected";
	const grounded =
		report.assignment.originalQuery === currentQuery &&
		isSubstantiveTextValue(report.assignment.method) &&
		isSubstantiveTextValue(report.assignment.queryVariant) &&
		report.evidenceCandidates.some(
			(candidate) =>
				isSubstantiveTextValue(candidate.source) &&
				isSubstantiveTextValue(candidate.method) &&
				isSubstantiveTextValue(candidate.evidence) &&
				isTimestampTextValue(candidate.retrievedAt),
		);
	return grounded ? "grounded" : "rejected";
}

function parseJsonValue(value: string): unknown | undefined {
	try {
		const parsed: unknown = JSON.parse(value);
		return parsed;
	} catch (error) {
		if (error instanceof SyntaxError) return undefined;
		throw error;
	}
}

function hasGroundedFinalOutput(value: string, currentQuery: string): boolean {
	const directReport = classifyExplorerReportGrounding(value, currentQuery);
	if (directReport !== "absent") return directReport === "grounded";
	let rejectedCanonicalReport = false;
	let start = value.indexOf("{");
	while (start >= 0) {
		const end = findJsonObjectEnd(value, start);
		if (end === undefined) {
			start = value.indexOf("{", start + 1);
			continue;
		}
		const embeddedReport = classifyExplorerReportGrounding(value.slice(start, end + 1), currentQuery);
		if (embeddedReport === "grounded") return true;
		if (embeddedReport === "rejected") rejectedCanonicalReport = true;
		start = value.indexOf("{", end + 1);
	}
	return !rejectedCanonicalReport && hasGroundedTextHandoff(value, currentQuery);
}

function findJsonObjectEnd(value: string, start: number): number | undefined {
	if (value[start] !== "{") return undefined;
	let depth = 0;
	let inString = false;
	let escaped = false;
	for (let index = start; index < value.length; index += 1) {
		const character = value[index];
		if (inString) {
			if (escaped) {
				escaped = false;
			} else if (character === "\\") {
				escaped = true;
			} else if (character === '"') {
				inString = false;
			}
			continue;
		}
		if (character === '"') {
			inString = true;
		} else if (character === "{") {
			depth += 1;
		} else if (character === "}") {
			depth -= 1;
			if (depth === 0) return index;
			if (depth < 0) return undefined;
		}
	}
	return undefined;
}

type GroundingTextField = "source" | "evidence" | "retrievedAt" | "temporalMetadata";

const GROUNDING_TEXT_FIELD_PATTERN =
	/(?:^|[|\n]|\s)(source\s*temporal\s*metadata|temporal\s*metadata|retrieval\s*timestamp|retrieved\s*at|exact\s+source\s+path|exact\s+supporting\s+excerpt|verbatim\s+evidence|evidence\s*(?:excerpt|items?)?|sources?(?:\s*(?:path|id))?|as\s*of)\s*(?::|=|-|\|)\s*(.*?)(?=\s+(?:source\s*temporal\s*metadata|temporal\s*metadata|retrieval\s*timestamp|retrieved\s*at|exact\s+source\s+path|exact\s+supporting\s+excerpt|verbatim\s+evidence|evidence\s*(?:excerpt|items?)?|sources?(?:\s*(?:path|id))?|as\s*of)\s*(?::|=|-|\|)|\s*[|\n]|$)/gim;

function hasGroundedTextHandoff(value: string, currentQuery: string): boolean {
	const handoffBody = textBeforeDiagnosticsSection(value);
	if (handoffBody.trim().length === 0 || hasExplicitNoHandoffContext(handoffBody)) return false;
	// Real Luna prose handoffs may omit Original query; the invocation-bound currentQuery is the governing binding.
	if (hasInvalidDeclaredOriginalQuery(handoffBody, currentQuery)) return false;
	if (
		hasGroundedMarkdownTable(handoffBody) ||
		hasGroundedFieldValueTable(handoffBody) ||
		hasGroundedProseCandidateHandoff(handoffBody)
	) {
		return true;
	}

	let fields = createGroundingTextFields();
	for (const line of handoffBody.split(/\r?\n/)) {
		const lineFields = collectGroundingTextFields(line);
		if (lineFields === undefined) {
			if (isDocumentedHandoffMetadataLine(line)) continue;
			fields = createGroundingTextFields();
			continue;
		}
		for (const field of Object.keys(lineFields) as GroundingTextField[]) {
			fields[field].push(...lineFields[field]);
		}
		if (hasSubstantiveGroundingTextFields(fields)) return true;
	}
	return false;
}

function textBeforeDiagnosticsSection(value: string): string {
	const lines: string[] = [];
	for (const line of value.split(/\r?\n/)) {
		if (/^\s*(?:#{1,6}\s*)?diagnostics?\b/i.test(normalizeGroundingTextLine(line))) break;
		lines.push(line);
	}
	return lines.join("\n");
}

function hasInvalidDeclaredOriginalQuery(value: string, currentQuery: string): boolean {
	for (const line of value.split(/\r?\n/)) {
		const declaredQuery = parseDeclaredOriginalQuery(line);
		if (declaredQuery === undefined) continue;
		if (isPlaceholderOriginalQuery(declaredQuery) || declaredQuery !== currentQuery) return true;
	}
	return false;
}

function parseDeclaredOriginalQuery(line: string): string | undefined {
	const normalizedLine = normalizeGroundingTextLine(line);
	const markdownRow =
		/^\s*\|\s*(?:\*\*|__)?original[\s_]query(?:\*\*|__)?\s*:?\s*(?:\*\*|__)?\s*\|\s*(.*?)\s*\|?\s*$/i.exec(
			normalizedLine,
		);
	if (markdownRow !== null) return unwrapDeclaredQuery(markdownRow[1] ?? "");
	const proseDeclaration = /^\s*(?:\*\*|__)?original[\s_]query(?:\*\*|__)?\s*(?::|=|-|\|)\s*(.*?)\s*$/i.exec(
		normalizedLine,
	);
	if (proseDeclaration === null) return undefined;
	return unwrapDeclaredQuery(proseDeclaration[1] ?? "");
}

function unwrapDeclaredQuery(value: string): string {
	const trimmed = value.trim();
	const unquoted = trimmed
		.replace(/^`+|`+$/g, "")
		.replace(/^"+|"+$/g, "")
		.replace(/^'+|'+$/g, "")
		.trim();
	return unquoted;
}

function isDocumentedHandoffMetadataLine(value: string): boolean {
	return /^\s*(?:original[\s_]query|(?:selected[\s_]+)?(?:retrieval[\s_]+)?method|query[\s_]+variants?(?:[\s_]+used)?|relevance|location[\s_]context|locator|temporal[\s_]basis|uncertainty|discovery[\s_]result)\s*(?::|=|-|\|)\s*\S.*$/i.test(
		normalizeGroundingTextLine(value),
	);
}

function createGroundingTextFields(): Record<GroundingTextField, string[]> {
	return { source: [], evidence: [], retrievedAt: [], temporalMetadata: [] };
}

function collectGroundingTextFields(value: string): Record<GroundingTextField, string[]> | undefined {
	const fields = createGroundingTextFields();
	let matched = false;
	for (const match of normalizeGroundingTextLine(value).matchAll(GROUNDING_TEXT_FIELD_PATTERN)) {
		const label = match[1];
		const fieldValue = match[2];
		if (typeof label !== "string" || typeof fieldValue !== "string") continue;
		const field = classifyGroundingTextLabel(label);
		if (field === undefined) continue;
		fields[field].push(fieldValue);
		matched = true;
	}
	return matched ? fields : undefined;
}

function hasSubstantiveGroundingTextFields(fields: Record<GroundingTextField, string[]>): boolean {
	return (
		fields.source.some((fieldValue) => isSubstantiveTextValue(fieldValue)) &&
		fields.evidence.some((fieldValue) => isSubstantiveTextValue(fieldValue)) &&
		fields.retrievedAt.some(isTimestampTextValue) &&
		fields.temporalMetadata.some((fieldValue) => isSubstantiveTextValue(fieldValue, true))
	);
}

function hasExplicitNoHandoffContext(value: string): boolean {
	const normalizedValue = value.split(/\r?\n/).map(normalizeGroundingTextLine).join("\n");
	return (
		/^\s*diagnostics?\b/im.test(normalizedValue) ||
		/\b(?:no|without)\s+(?:(?:an?|the)\s+)?(?:(?:explorer|source|evidence)\s+)?handoff\b/i.test(normalizedValue) ||
		/\bhandoff\s+(?:was\s+)?(?:not|never)\s+(?:returned|provided)\b/i.test(normalizedValue)
	);
}

function hasGroundedMarkdownTable(value: string): boolean {
	const lines = value.split(/\r?\n/);
	for (let headerIndex = 0; headerIndex < lines.length; headerIndex += 1) {
		const headers = splitMarkdownRow(lines[headerIndex]);
		if (headers.length === 0) continue;
		const indexes = {
			source: headers.findIndex((header) => classifyGroundingTextLabel(header) === "source"),
			evidence: headers.findIndex((header) => classifyGroundingTextLabel(header) === "evidence"),
			retrievedAt: headers.findIndex((header) => classifyGroundingTextLabel(header) === "retrievedAt"),
			temporalMetadata: headers.findIndex((header) => classifyGroundingTextLabel(header) === "temporalMetadata"),
		};
		if (Object.values(indexes).some((index) => index < 0)) continue;
		const requiredColumn = Math.max(...Object.values(indexes));
		const delimiter = splitMarkdownRow(lines[headerIndex + 1] ?? "");
		if (delimiter.length !== headers.length || !delimiter.every(isMarkdownSeparatorCell)) continue;
		const cells = splitMarkdownRow(lines[headerIndex + 2] ?? "");
		if (cells.length !== headers.length || cells.length <= requiredColumn) continue;
		if (
			isSubstantiveTextValue(cells[indexes.source], false) &&
			isSubstantiveTextValue(cells[indexes.evidence], false) &&
			isTimestampTextValue(cells[indexes.retrievedAt]) &&
			isSubstantiveTextValue(cells[indexes.temporalMetadata], true)
		) {
			return true;
		}
	}
	return false;
}
function hasGroundedFieldValueTable(value: string): boolean {
	const lines = value.split(/\r?\n/);
	// Document-level RetrievedAt / temporal lines can complete Field/Value tables that omit them.
	const preamble = createGroundingTextFields();
	for (const line of lines) {
		const lineFields = collectGroundingTextFields(line);
		if (lineFields === undefined) continue;
		for (const field of Object.keys(lineFields) as GroundingTextField[]) {
			preamble[field].push(...lineFields[field]);
		}
	}

	let fields = createGroundingTextFields();
	let inFieldValueTable = false;
	for (let index = 0; index < lines.length; index += 1) {
		const headers = splitMarkdownRow(lines[index] ?? "").map((header) =>
			header
				.replace(/\*\*|__/g, "")
				.toLowerCase()
				.trim(),
		);
		if (headers.length >= 2 && headers[0] === "field" && headers[1] === "value") {
			const delimiter = splitMarkdownRow(lines[index + 1] ?? "");
			if (delimiter.length >= 2 && delimiter.every(isMarkdownSeparatorCell)) {
				inFieldValueTable = true;
				fields = createGroundingTextFields();
				for (const field of Object.keys(preamble) as GroundingTextField[]) {
					fields[field].push(...preamble[field]);
				}
				index += 1; // skip delimiter row
				continue;
			}
		}
		if (!inFieldValueTable) continue;
		const cells = splitMarkdownRow(lines[index] ?? "").map((cell) => cell.replace(/\*\*|__/g, "").trim());
		if (cells.length < 2) {
			if (hasSubstantiveGroundingTextFields(fields)) return true;
			inFieldValueTable = false;
			continue;
		}
		const field = classifyGroundingTextLabel(cells[0] ?? "");
		if (field === undefined) continue;
		fields[field].push(cells.slice(1).join(" | "));
		if (hasSubstantiveGroundingTextFields(fields)) return true;
	}
	return hasSubstantiveGroundingTextFields(fields);
}
function hasGroundedProseCandidateHandoff(value: string): boolean {
	// Real explorers often emit bullet/prose candidate blocks rather than a single
	// field-complete line or wide markdown table. Accept a handoff when it has a
	// real source path, temporal status, and non-trivial evidence. retrievedAt is
	// preferred but document-internal dates in temporal metadata are accepted when
	// explorers omit an explicit retrieval timestamp.
	const body = value;
	const sourceMatch =
		/(?:^|\n)\s*(?:[-*+]\s+)?(?:\*\*|__|`)?source(?:\s*path)?(?:\*\*|__|`)?\s*(?::|=|-|\|)\s*`?(\/[^`\n|]+)`?/im.exec(
			body,
		) ??
		/(?:^|\n)\s*(?:[-*+]\s+)?(?:\*\*|__)?(?:file|path|document)(?:\*\*|__)?\s*(?::|=|-|\|)\s*`?(\/[^`\n|]+)`?/im.exec(
			body,
		) ??
		/`(\/(?:Users|home|docs)\/[^`\n]+)`/.exec(body) ??
		/(?:^|\n)\s*[-*+]\s+(\/(?:Users|home|docs)\/[^\n]+)/.exec(body);
	const source = sourceMatch?.[1]?.trim();
	if (!isSubstantiveTextValue(source, false) || !source?.startsWith("/")) return false;

	const retrievedMatch =
		/(?:retrieved\s*at|retrievedat|retrieval\s*timestamp)(?:\s*\([^)]*\))?\s*(?::|=|-|\|)\s*`?\**([0-9]{4}-[0-9]{2}-[0-9]{2}[^`\n|]*)/im.exec(
			body,
		);
	const temporalDateMatch =
		/(?:source\s*temporal(?:\s*metadata)?|temporal\s*metadata|temporal\s*basis|as\s*of|사업기간|참여기간)\s*(?::|=|-|\|)\s*[^\n]*?(\d{4}[./-]\d{2}(?:[./-]\d{2})?)/im.exec(
			body,
		);
	const retrievedCandidate =
		retrievedMatch?.[1] ??
		(temporalDateMatch?.[1] !== undefined
			? temporalDateMatch[1].replace(/\./g, "-").replace(/(\d{4}-\d{2})$/, "$1-01")
			: undefined);
	if (!isTimestampTextValue(retrievedCandidate)) return false;

	const temporalMatch =
		/(?:source\s*temporal(?:\s*metadata)?|temporal\s*metadata|temporal\s*basis|as\s*of|사업기간|참여기간)\s*(?::|=|-|\|)\s*(.+)/im.exec(
			body,
		);
	const temporal = temporalMatch?.[1]?.trim() ?? (/\bunknown\b/i.test(body) ? "unknown" : undefined);
	if (!isSubstantiveTextValue(temporal, true)) return false;

	const evidenceMatch =
		/(?:evidence(?:\s*\/\s*location)?|evidence\s*excerpt|verbatim\s*evidence|excerpt)\s*(?::|=|-|\|)\s*(.+)/im.exec(
			body,
		);
	const evidence = evidenceMatch?.[1]?.trim() ?? "";
	const evidenceBlock =
		evidence.length > 0
			? evidence
			: (/(?:evidence(?:\s*\/\s*location)?|excerpt)\s*(?::|=|-|\|)\s*\n([\s\S]{20,800})/im.exec(body)?.[1]?.trim() ??
				"");
	if (isSubstantiveTextValue(evidenceBlock, false)) return true;
	const multiLineEvidence =
		/(?:evidence(?:\s*\/\s*location)?|excerpt)\s*(?::|=|-|\|)\s*\n([\s\S]{40,1200})/im.exec(body)?.[1]?.trim() ?? "";
	if (isSubstantiveTextValue(multiLineEvidence, false)) return true;
	const nearSource =
		body.includes(source) &&
		/(?:pdf|docx|pptx|xlsx|html|md|hwp|filename|grep|read|invoice|receipt|청구서|영수증|채용|참여|인건비)/i.test(
			body,
		) &&
		body.length >= 200 &&
		!GROUNDING_TEXT_PLACEHOLDER_PATTERN.test(evidenceBlock.toLowerCase().replace(/\s+/g, " "));
	return nearSource;
}

function splitMarkdownRow(value: string): string[] {
	if (!value.includes("|")) return [];
	const row = value.trim().replace(/^\|/, "").replace(/\|$/, "");
	return row.split("|").map((cell) => cell.trim());
}

function isMarkdownSeparatorCell(value: string): boolean {
	return /^:?-{3,}:?$/.test(value.trim());
}

function classifyGroundingTextLabel(label: string): GroundingTextField | undefined {
	const normalized = label.toLowerCase().replace(/[_/]+/g, " ").replace(/\s+/g, " ").trim();
	if (
		[
			"source",
			"sources",
			"source path",
			"source id",
			"sourcepath",
			"sourceid",
			"exact source path",
			"path",
			"file",
			"document",
		].includes(normalized)
	) {
		return "source";
	}
	if (
		[
			"evidence",
			"evidence excerpt",
			"evidence item",
			"evidence items",
			"evidence / location",
			"evidence location",
			"exact supporting excerpt",
			"verbatim evidence",
		].includes(normalized)
	) {
		return "evidence";
	}
	if (["retrieved at", "retrievedat", "retrieval timestamp", "retrievaltimestamp"].includes(normalized)) {
		return "retrievedAt";
	}
	if (
		[
			"source temporal metadata",
			"source temporalmetadata",
			"sourcetemporalmetadata",
			"source temporal",
			"temporal metadata",
			"temporalmetadata",
			"temporal",
			"as of",
			"asof",
		].includes(normalized)
	) {
		return "temporalMetadata";
	}
	return undefined;
}

function normalizeGroundingTextLine(value: string): string {
	return value
		.replace(/^\s*[-+*]\s+/, "")
		.replace(/^(\s*(?:#{1,6}\s+)?)(?:\*\*|__)([^*_|\n]+?)(?:\*\*|__)\s*:\s*/, "$1$2: ")
		.replace(/^(\s*(?:#{1,6}\s+)?)(?:\*\*|__)([^*_|\n]+?)\s*:\s*(?:\*\*|__)\s*/, "$1$2: ")
		.replace(/^(\s*(?:#{1,6}\s+)?)(?:\*\*|__)([^*_|\n]+?)(?:\*\*|__)\s*$/, "$1$2")
		.trim();
}

function cleanTextValue(value: string): string {
	return value
		.trim()
		.replace(/^[|`*_"' ]+|[|`*_"'.,; ]+$/g, "")
		.trim();
}

function isSubstantiveTextValue(value: string | undefined, allowUnknown = false): boolean {
	if (value === undefined) return false;
	const cleaned = cleanTextValue(value);
	if (cleaned.length === 0) return false;
	if (cleaned.toLowerCase() === "unknown") return allowUnknown;
	if (/^(?:none|n\/a|na|null|undefined|true|false|-|—|\?)$/i.test(cleaned)) return false;
	const normalized = cleaned.toLowerCase().replace(/\s+/g, " ");
	if (
		(/^(?:<[^>]+>|\[[^\]]+\]|\{[^}]+\})$/.test(normalized) &&
			/\b(?:source|evidence|excerpt|path|id)\b/.test(normalized)) ||
		/^(?:the )?(?:source|evidence|exact supporting excerpt)(?:\s+(?:path|id|excerpt|here|placeholder|goes here|goes below))?$/.test(
			normalized,
		) ||
		/^(?:(?:source|evidence)(?:\s+(?:path|id|excerpt))?|(?:path|excerpt)|exact supporting excerpt|verbatim evidence)\s+(?:here|below)$/i.test(
			normalized,
		) ||
		GROUNDING_TEXT_PLACEHOLDER_PATTERN.test(normalized)
	) {
		return false;
	}
	return !/^-?\d+(?:\.\d+)?$/.test(cleaned);
}

const GROUNDING_TEXT_PLACEHOLDER_SUBJECT =
	"(?:the )?(?:source(?: (?:path|id))?|evidence(?: (?:excerpt|item(?:s)?))?|exact supporting excerpt|verbatim evidence|excerpt)";
const GROUNDING_TEXT_PLACEHOLDER_LOCATION =
	"(?:here|below|above|provided|filled(?: in)?|inserted|added|included|supplied|quoted|located)(?: (?:here|below|above))?";
const GROUNDING_TEXT_PLACEHOLDER_PATTERN = new RegExp(
	`^(?:${GROUNDING_TEXT_PLACEHOLDER_SUBJECT}\\s+(?:(?:goes?|belongs|is|will be|should be|must be|needs to be|to be)\\s+${GROUNDING_TEXT_PLACEHOLDER_LOCATION}|(?:should|must|needs to) go\\s+${GROUNDING_TEXT_PLACEHOLDER_LOCATION}|(?:placeholder|template|value|text|pending|missing)|to\\s+(?:provide|fill(?: in)?|insert|add|include|supply|quote))|(?:provide|insert|add|include|quote|supply)\\s+${GROUNDING_TEXT_PLACEHOLDER_SUBJECT}(?:\\s+${GROUNDING_TEXT_PLACEHOLDER_LOCATION})?)$`,
	"i",
);

const TIMESTAMP_TEXT_PATTERN =
	/^(\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?)(?:\s*;\s*.+)?$/;

function isTimestampTextValue(value: string | undefined): boolean {
	if (value === undefined) return false;
	const cleaned = cleanTextValue(value);
	// Explorers often append parenthetical notes: `2026-07-16 (session clock; ...)`.
	const leading =
		/^(\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?)/.exec(cleaned)?.[1] ??
		undefined;
	const match = TIMESTAMP_TEXT_PATTERN.exec(cleaned);
	const timestamp = match?.[1] ?? leading;
	if (timestamp === undefined) return false;
	const dateMatch = /^(\d{4})-(\d{2})-(\d{2})/.exec(timestamp);
	if (dateMatch === null) return false;
	const year = Number(dateMatch[1]);
	const month = Number(dateMatch[2]);
	const day = Number(dateMatch[3]);
	const calendarDate = new Date(Date.UTC(year, month - 1, day));
	if (
		calendarDate.getUTCFullYear() !== year ||
		calendarDate.getUTCMonth() !== month - 1 ||
		calendarDate.getUTCDate() !== day
	) {
		return false;
	}
	const timeMatch = /[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?/.exec(timestamp);
	if (timeMatch !== null) {
		const hours = Number(timeMatch[1]);
		const minutes = Number(timeMatch[2]);
		const seconds = timeMatch[3] === undefined ? 0 : Number(timeMatch[3]);
		if (hours > 23 || minutes > 59 || seconds > 59) return false;
	}
	const timezoneMatch = /([+-])(\d{2}):?(\d{2})$/.exec(timestamp);
	if (timezoneMatch !== null && (Number(timezoneMatch[2]) > 23 || Number(timezoneMatch[3]) > 59)) return false;
	return Number.isFinite(Date.parse(timestamp));
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function deepFreeze<T>(value: T): T {
	if (value !== null && typeof value === "object") {
		if (Array.isArray(value)) {
			for (const item of value) deepFreeze(item);
		} else {
			for (const key of Object.keys(value)) {
				deepFreeze((value as Record<string, unknown>)[key]);
			}
		}
		Object.freeze(value);
	}
	return value;
}

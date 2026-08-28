import { randomUUID } from "node:crypto";
import { watch as fsWatch, realpathSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { Agent, type AgentEvent, type AgentMessage, type AgentTool, type Skill } from "@earendil-works/pi-agent-core";
import type { Api, Model } from "@earendil-works/pi-ai";
import { streamSimple } from "@earendil-works/pi-ai/compat";
import { ManagedCliConfigManager, ManagedCliRegistry } from "../cli/managed-cli-config.ts";
import { resolveAutoRAGHome } from "../config/home.ts";
import { DatasourceAccessContext, type DatasourceAccessContextOptions } from "../datasource/access-context.ts";
import { createCrawlerManagedCliProvider } from "../datasource/crawler-managed-config.ts";
import { mapDatasourceDiagnostics } from "../datasource/diagnostics.ts";
import { DatasourceResultFilter } from "../datasource/result-filter.ts";
import { createDiscrawlManagedCliProvider } from "../datasource/skills/discrawl/config.ts";
import { createRcloneManagedCliProvider } from "../datasource/skills/gdrive/rclone-managed-config.ts";
import { createHimalayaManagedCliProvider } from "../datasource/skills/gmail/himalaya-managed-config.ts";
import { createKatokManagedCliProvider } from "../datasource/skills/katok/config.ts";
import { createQmdManagedCliProvider } from "../datasource/skills/obsidian/config.ts";
import type { DatasourceIndexResult, DatasourceSkill } from "../datasource/types.ts";
import { DupeyCliError, type DupeyCliOptions, scanWithDupey, selectExactDuplicateExclusions } from "../dupey/index.ts";
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
import {
	MinSyncBM25Method,
	type MinSyncBM25MethodOptions,
	type MinSyncSyncResult,
	MinSyncVectorMethod,
	type MinSyncVectorMethodOptions,
} from "../minsync/index.ts";
import { PARSED_MIRROR_SUBDIR } from "../mirror/paths.ts";
import {
	detectMirrorStaleness,
	type ParsedMirrorDiagnostic,
	type ParsedMirrorSyncResult,
	syncParsedMirrors,
} from "../mirror/sync.ts";
import { AutoRAGRunLogger } from "../observability/run-log.ts";
import type { DefaultParserRegistryOptions } from "../parser/index.ts";
import { ManagedRetrievalRuntime } from "../retrieval/managed-runtime.ts";
import { ParallelRetriever, ResultMerger } from "../retrieval/merger.ts";
import { BM25Method, type BM25MethodOptions, type BM25SyncResult } from "../retrieval/methods/bm25.ts";

import { RetrievalMethodRegistry } from "../retrieval/registry.ts";
import {
	buildRetrievalScopeBindings,
	normalizeVirtualPath,
	type RetrievalScopeBinding,
	resolveRetrievalScope,
} from "../retrieval/scope.ts";
import type { CuratedResult, RetrievalDiagnostic, RetrievalOptions, RetrievalResult } from "../retrieval/types.ts";
import { BASH_TOOL_NAME, createBashTool } from "./bash-tool.ts";
import {
	createLoadDatasourceSkillTool,
	LOAD_DATASOURCE_SKILL_TOOL_NAME,
	toDatasourceAgentSkill,
} from "./datasource-skill.ts";
import {
	createScanDuplicateDocumentsTool,
	SCAN_DUPLICATE_DOCUMENTS_TOOL_NAME,
	type ScanDuplicateDocumentsDetails,
} from "./dupey-tool.ts";
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
import { loadLocalAutoRAGModel } from "./local-model.ts";
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

export interface AutoRAGMinSyncRefreshResult {
	readonly ok: boolean;
	readonly synced: number;
	readonly reason?: string;
}

export interface AutoRAGRefreshResult extends ParsedMirrorSyncResult {
	readonly bm25?: BM25SyncResult;
	readonly minsync?: AutoRAGMinSyncRefreshResult;
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
	providerApiKeys?: Readonly<Record<string, string>>;
	searchPaths: string[];
	manifestDir?: string;
	memoryPath?: string;
	workspacePath?: string;
	tools?: AgentTool[];
	minSync?: Omit<MinSyncVectorMethodOptions, "root"> | false;
	bm25?: Omit<MinSyncBM25MethodOptions, "root"> | Omit<BM25MethodOptions, "root"> | false;
	jikji?: JikjiOptions;
	autoRefresh?: AutoRefreshOptions;
	parserOptions?: DefaultParserRegistryOptions;
	dupey?: DupeyCliOptions | false;
	excludeExactDuplicates?: boolean;
	datasourceSkills?: readonly DatasourceSkill[];
	datasourceAccess?: DatasourceAccessContextOptions;
	managedCliRegistry?: ManagedCliRegistry;
	managedCliConfigManager?: ManagedCliConfigManager;
	managedRetrievalRuntime?: ManagedRetrievalRuntime;
	/** Non-fatal diagnostics from config/agent construction (e.g. skipped unknown datasources). */
	startupDiagnostics?: readonly SearchDocumentDiagnostic[];
	/** Maximum time a model/tool search may run before it is aborted. */
	searchTimeoutMs?: number;
	/** Maximum number of retrieval/tool executions allowed in one search. */
	maxSearchToolCalls?: number;
}

export interface AutoRAGSearchSession {
	readonly agent: Agent;
	prompt(text: string): Promise<void>;
	abort(): Promise<void> | void;
	dispose(): void;
}

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
	private readonly providerApiKeys: Readonly<Record<string, string>> | undefined;
	private readonly listeners = new Set<Parameters<Agent["subscribe"]>[0]>();
	private activeSession: AutoRAGSearchSession | undefined;
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
	private readonly configuredSearchPaths: readonly string[];
	private retrievalScopeBindings: readonly RetrievalScopeBinding[];
	private readonly datasourceVirtualScopePrefixes: readonly string[];
	private readonly workspaceProjectRoot: string;
	private readonly methodRegistry = new RetrievalMethodRegistry();
	private readonly retriever = new ParallelRetriever();
	private readonly merger = new ResultMerger();
	private readonly datasourceFilter = new DatasourceResultFilter();

	private readonly minSyncMethod: MinSyncVectorMethod | undefined;
	private readonly bm25Method: MinSyncBM25Method | BM25Method | undefined;
	private readonly jikjiClient: JikjiClient | undefined;
	private readonly datasourceSkills: readonly DatasourceSkill[];
	private readonly managedCliRegistry: ManagedCliRegistry;
	private readonly managedCliConfigManager: ManagedCliConfigManager;
	private readonly managedRetrievalRuntime: ManagedRetrievalRuntime;
	private readonly datasourceAccessOptions: DatasourceAccessContextOptions;
	private readonly startupDiagnostics: readonly SearchDocumentDiagnostic[];
	private readonly datasourceAgentSkills: readonly Skill[];
	private readonly parserOptions: DefaultParserRegistryOptions | undefined;
	private readonly dupeyOptions: DupeyCliOptions | false;
	private readonly excludeExactDuplicates: boolean;
	private readonly baseSystemPromptConfig: SystemPromptConfig;
	private readonly droppedCallerToolNames: readonly string[];
	private readonly searchTimeoutMs: number;
	private readonly maxSearchToolCalls: number;
	private searchToolCallCount = 0;

	constructor(options: AutoRAGAgentOptions) {
		const { manifestDir, memoryPath } = options;
		this.configuredModel = options.model;
		this.searchTimeoutMs = options.searchTimeoutMs ?? 10 * 60 * 1000;
		this.maxSearchToolCalls = options.maxSearchToolCalls ?? 32;
		if (!Number.isFinite(this.searchTimeoutMs) || this.searchTimeoutMs <= 0) {
			throw new Error("searchTimeoutMs must be a positive finite number");
		}
		if (!Number.isInteger(this.maxSearchToolCalls) || this.maxSearchToolCalls <= 0) {
			throw new Error("maxSearchToolCalls must be a positive integer");
		}
		this.apiKey = options.apiKey;
		this.providerApiKeys = options.providerApiKeys;
		const manifests = manifestDir ? loadManifests(manifestDir) : [];
		this.datasourceSkills = options.datasourceSkills ?? [];
		this.datasourceVirtualScopePrefixes = this.datasourceSkills.map((skill) =>
			normalizeVirtualPath(`/${skill.describe().name}`),
		);
		this.datasourceAccessOptions = options.datasourceAccess ?? {};
		this.startupDiagnostics = options.startupDiagnostics ?? [];
		this.datasourceAgentSkills = this.buildAuthorizedDatasourceSkills();
		this.managedCliRegistry = options.managedCliRegistry ?? new ManagedCliRegistry();
		if (this.datasourceSkills.some((skill) => skill.describe().name === "discord")) {
			try {
				this.managedCliRegistry.register(createDiscrawlManagedCliProvider());
			} catch (error) {
				if (!(error instanceof Error) || !error.message.includes("already registered")) throw error;
			}
		}
		if (this.datasourceSkills.some((skill) => skill.describe().name === "kakao")) {
			try {
				this.managedCliRegistry.register(createKatokManagedCliProvider());
			} catch (error) {
				if (!(error instanceof Error) || !error.message.includes("already registered")) throw error;
			}
		}
		for (const [datasource, binary] of [
			["whatsapp", "wacrawl"],
			["telegram", "telecrawl"],
			["slack", "slacrawl"],
			["notion", "notcrawl"],
		] as const) {
			if (!this.datasourceSkills.some((skill) => skill.describe().name === datasource)) continue;
			try {
				this.managedCliRegistry.register(createCrawlerManagedCliProvider(binary));
			} catch (error) {
				if (!(error instanceof Error) || !error.message.includes("already registered")) throw error;
			}
		}
		if (this.datasourceSkills.some((skill) => skill.describe().name === "obsidian")) {
			try {
				this.managedCliRegistry.register(createQmdManagedCliProvider());
			} catch (error) {
				if (!(error instanceof Error) || !error.message.includes("already registered")) throw error;
			}
		}
		if (this.datasourceSkills.some((skill) => ["gdrive", "cloud-drive"].includes(skill.describe().name))) {
			try {
				this.managedCliRegistry.register(createRcloneManagedCliProvider());
			} catch (error) {
				if (!(error instanceof Error) || !error.message.includes("already registered")) throw error;
			}
		}
		if (this.datasourceSkills.some((skill) => skill.describe().name === "gmail")) {
			try {
				this.managedCliRegistry.register(createHimalayaManagedCliProvider());
			} catch (error) {
				if (!(error instanceof Error) || !error.message.includes("already registered")) throw error;
			}
		}
		this.configuredSearchPaths = options.searchPaths.map((searchPath) => resolve(searchPath));
		this.searchPaths = options.searchPaths.map(pinSearchRoot);
		this.workspaceProjectRoot = options.workspacePath ?? process.cwd();
		this.managedCliConfigManager =
			options.managedCliConfigManager ??
			new ManagedCliConfigManager({ workspace: this.workspaceProjectRoot, registry: this.managedCliRegistry });
		this.managedRetrievalRuntime =
			options.managedRetrievalRuntime ??
			new ManagedRetrievalRuntime(this.workspaceProjectRoot, {
				minSync: options.minSync !== false,
				minSyncBinaryPath: options.minSync !== false ? options.minSync?.binaryPath : undefined,
				jikji: options.jikji !== undefined,
				jikjiBinaryPath: options.jikji?.binaryPath,
			});
		this.retrievalScopeBindings = buildRetrievalScopeBindings(
			this.workspaceProjectRoot,
			this.searchPaths,
			this.configuredSearchPaths,
		);
		this.parserOptions = options.parserOptions;
		this.dupeyOptions = options.dupey ?? {};
		this.excludeExactDuplicates = options.excludeExactDuplicates ?? true;

		if (options.minSync !== false) {
			const minSyncOpts = options.minSync ?? { autoInstall: false };
			this.minSyncMethod = new MinSyncVectorMethod({
				...minSyncOpts,
				root: this.workspaceProjectRoot,
				managedCliConfigManager: this.managedRetrievalRuntime.manager,
			});
			this.methodRegistry.register(this.minSyncMethod);
		}
		if (options.bm25 !== false) {
			const bm25Opts = { autoInstall: false, ...(options.bm25 ?? {}) };
			this.bm25Method =
				options.minSync === false || hasLegacyBM25Options(bm25Opts)
					? new BM25Method({ ...bm25Opts, root: this.workspaceProjectRoot } as BM25MethodOptions)
					: new MinSyncBM25Method({ ...bm25Opts, root: this.workspaceProjectRoot } as MinSyncBM25MethodOptions);
			this.methodRegistry.register(this.bm25Method);
		}
		for (const skill of this.datasourceSkills) {
			for (const method of skill.retrievalMethods()) this.methodRegistry.register(method);
		}
		if (options.jikji) {
			this.jikjiClient = new JikjiClient({
				...options.jikji,
				root: this.workspaceProjectRoot,
				managedCliConfigManager: this.managedRetrievalRuntime.manager,
			});
		}

		const memPath = memoryPath ?? join(resolveAutoRAGHome(), "memory.json");
		this.memory = new RetrievalMemory({ storagePath: memPath });
		this.memory.load();
		this.runLogger = new AutoRAGRunLogger(join(dirname(memPath), "logs", "runs.jsonl"));

		const checkMemoryTool = createCheckMemoryTool(this.memory);
		const searchBM25Tool = createSearchBM25DocumentsTool(
			() => this.bm25Method,
			(scope) => this.resolveRetrievalScope(scope),
		);
		const searchDatasourceTool = createSearchDatasourceDocumentsTool(this);

		const searchMinSyncTool = createSearchMinSyncDocumentsTool(
			() => this.minSyncMethod,
			(scope) => this.resolveRetrievalScope(scope),
		);
		const searchAllTool = createSearchAllDocumentsTool(this);
		const loadDatasourceSkillTool = createLoadDatasourceSkillTool(this);
		const emitResultsTool = createEmitResultsTool((details) => this.resultCapture?.(details));
		const scanDuplicateDocumentsTool =
			this.dupeyOptions === false ? undefined : createScanDuplicateDocumentsTool(this);

		const bashTool = createBashTool({
			cwd: this.workspaceProjectRoot,
			managedCliRegistry: this.managedCliRegistry,
			managedCliRegistries: [this.managedRetrievalRuntime.registry],
		});

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
			SCAN_DUPLICATE_DOCUMENTS_TOOL_NAME,
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
			...(scanDuplicateDocumentsTool !== undefined ? [scanDuplicateDocumentsTool] : []),
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
			modelId: options.model?.id,
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
			streamFn: streamSimple,
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

	async scanDuplicateDocuments(): Promise<ScanDuplicateDocumentsDetails> {
		if (this.dupeyOptions === false) {
			return { scans: [], familyCount: 0, exactDuplicateCount: 0 };
		}
		const scans = await Promise.all(
			this.searchPaths.map((searchPath) => scanWithDupey(searchPath, this.dupeyOptions || {})),
		);
		const familyCount = scans.reduce((count, scan) => count + scan.families.length, 0);
		const exactDuplicateCount = scans.reduce((count, scan) => {
			const hashes = new Map<string, number>();
			for (const file of scan.files) {
				if (typeof file.content_hash !== "string") continue;
				hashes.set(file.content_hash, (hashes.get(file.content_hash) ?? 0) + 1);
			}
			return count + [...hashes.values()].reduce((sum, size) => sum + Math.max(0, size - 1), 0);
		}, 0);
		return { scans, familyCount, exactDuplicateCount };
	}

	private async withMemoryContext(messages: AgentMessage[]): Promise<AgentMessage[]> {
		const hints = this.lastQuery ? this.memory.getMethodHints(this.lastQuery) : [];
		const insights = this.lastQuery ? this.memory.getInsights(this.lastQuery) : [];
		const contextHints = this.lastQuery ? this.memory.getContextHints(this.lastQuery) : undefined;
		const contextHintCount = contextHints
			? Object.values(contextHints).reduce((count, values) => count + values.length, 0)
			: 0;
		if (hints.length === 0 && insights.length === 0 && contextHintCount === 0) return messages;
		const summary = renderMemoryContext(hints, { insights, contextHints });
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
		readonly apiKey?: string;
		readonly providerApiKeys?: Readonly<Record<string, string>>;
	} {
		if (this.configuredModel !== undefined) {
			return {
				model: this.configuredModel,
				...(this.apiKey !== undefined ? { apiKey: this.apiKey } : {}),
				...(this.providerApiKeys !== undefined ? { providerApiKeys: this.providerApiKeys } : {}),
			};
		}
		const local = loadLocalAutoRAGModel();
		return {
			model: local.model,
			apiKey: local.apiKey,
			providerApiKeys: { [local.provider]: local.apiKey },
		};
	}

	private createSearchSession(model: Model<Api>, systemPrompt: string): AutoRAGSearchSession {
		const agent = new Agent({
			initialState: { systemPrompt, model, tools: [...this.tools] },
			streamFn: streamSimple,
			getApiKey: (provider) =>
				this.providerApiKeys?.[provider] ?? (provider === model.provider ? this.apiKey : undefined),
			convertToLlm: (messages) =>
				messages.filter(
					(message) => message.role === "user" || message.role === "assistant" || message.role === "toolResult",
				),
			transformContext: async (messages) => this.withMemoryContext(messages),
		});
		return {
			agent,
			prompt: async (prompt) => agent.prompt(prompt),
			abort: async () => agent.abort(),
			dispose: () => {},
		};
	}

	private configureSearchSession(session: AutoRAGSearchSession): readonly (() => void)[] {
		const unsubscribers = [...this.listeners].map((listener) => session.agent.subscribe(listener));
		unsubscribers.push(
			session.agent.subscribe((event) => {
				this.recordSearchToolEvent(event);
			}),
		);
		return unsubscribers;
	}

	private recordSearchToolEvent(event: AgentEvent): void {
		if (event.type !== "tool_execution_end" || !this.lastQuery) return;
		if (!(SEARCH_TOOLS as readonly string[]).includes(event.toolName)) return;
		this.searchToolCallCount += 1;
		if (this.searchToolCallCount >= this.maxSearchToolCalls) {
			void this.activeSession?.abort();
		}
		const details = event.result.details as { method?: string } | undefined;
		this.memory.recordWeakSignal(this.lastQuery, details?.method ?? event.toolName, "followup");
		this.memory.save();
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

	recordFeedbackByIds(usefulFeedbackIds: readonly string[], notUsefulFeedbackIds: readonly string[] = []): void {
		const feedback = [
			...usefulFeedbackIds.map((feedbackId) => ({ feedbackId, useful: true })),
			...notUsefulFeedbackIds.map((feedbackId) => ({ feedbackId, useful: false })),
		];
		if (this.memory.recordFeedbackByIds(feedback)) this.memory.save();
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
		options = this.normalizeRetrievalOptions(options);

		this.activeRun = true;
		this.searchToolCallCount = 0;
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
			this.runLogger.write({
				event: "search_started",
				timestamp: new Date().toISOString(),
				sessionId,
				queryLength: trimmedQuery.length,
				model: resolved.model.id,
			});
			searchStarted = true;
			session = this.createSearchSession(
				resolved.model,
				buildSystemPrompt(this.currentSystemPromptConfig({ modelId: resolved.model.id })),
			);
			this.activeSession = session;
			unsubscribers = this.configureSearchSession(session);
			let timeout: NodeJS.Timeout | undefined;
			try {
				await Promise.race([
					session.prompt(this.buildSearchPrompt(trimmedQuery, options)),
					new Promise<never>((_, reject) => {
						timeout = setTimeout(() => {
							void Promise.resolve(session?.abort());
							reject(new Error(`search timed out after ${this.searchTimeoutMs}ms`));
						}, this.searchTimeoutMs);
					}),
				]);
			} finally {
				if (timeout !== undefined) clearTimeout(timeout);
			}

			if (captured === undefined) {
				throw new Error("AutoRAG agent completed without emitting structured results");
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
		const diagnostics: SearchDocumentDiagnostic[] = [...this.startupDiagnostics];
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

	buildSearchPrompt(query: string, options: RetrievalOptions): string {
		const limit = typeof options.topK === "number" ? ` Return at most ${options.topK} curated results.` : "";
		const scope = options.scope ? ` Restrict search to virtual path scope ${options.scope}.` : "";
		return (
			`Find and curate information for this original query: ${query}${limit}${scope}\n\n` +
			`Use the available retrieval tools to find candidates, then use bash to read and verify the relevant source files directly. ` +
			`Judge relevance, conflicts, freshness, and sufficiency in this agent loop. Preserve real source paths and evidence excerpts in the result mapping.\n\n` +
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
			const minsync = wants("minsync") ? await this.syncMinSync() : undefined;
			const bm25 = wants("bm25") ? await this.syncBM25(minsync) : undefined;
			const datasources = wants("datasources") ? await this.indexDatasources() : [];
			const jikji = wants("jikji") ? await this.executeJikjiPrepare() : undefined;
			this.retrievalScopeBindings = buildRetrievalScopeBindings(
				this.workspaceProjectRoot,
				this.searchPaths,
				this.configuredSearchPaths,
			);
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
			const publicMinsync = minsync
				? {
						ok: minsync.ok,
						synced: minsync.synced,
						...(minsync.reason !== undefined ? { reason: minsync.reason } : {}),
					}
				: undefined;
			return { ...(bm25 ? { ...summary, bm25 } : summary), minsync: publicMinsync, datasources };
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
			...this.startupDiagnostics,
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
		const duplicateFilter = await this.exactDuplicateExclusions();
		return syncParsedMirrors({
			root: this.workspaceProjectRoot,
			searchPaths: this.searchPaths,
			force,
			parserOptions: this.parserOptions,
			excludeSourcePaths: duplicateFilter.excluded,
		});
	}

	/**
	 * Lightweight stat-only staleness scan used when `refresh` is called with
	 * methods that exclude parsed mirrors (e.g. only `datasources` or `jikji`).
	 * Returns a zero-count `ParsedMirrorSyncResult` carrying fresh diagnostics
	 * so the refresh result and status remain consistent.
	 */
	private async scanMirrorStaleness(): Promise<ParsedMirrorSyncResult> {
		const duplicateFilter = await this.exactDuplicateExclusions();
		const diagnostics = await detectMirrorStaleness({
			root: this.workspaceProjectRoot,
			searchPaths: this.searchPaths,
			parserOptions: this.parserOptions,
			excludeSourcePaths: duplicateFilter.excluded,
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

	private async exactDuplicateExclusions(): Promise<{ readonly excluded: ReadonlySet<string> }> {
		if (!this.excludeExactDuplicates || this.dupeyOptions === false) return { excluded: new Set() };
		const excluded = new Set<string>();
		for (const searchPath of this.searchPaths) {
			try {
				const scan = await scanWithDupey(searchPath, this.dupeyOptions || {});
				const selected = await selectExactDuplicateExclusions(searchPath, scan);
				for (const path of selected.excluded) excluded.add(path);
			} catch (error) {
				if (!(error instanceof DupeyCliError)) throw error;
				// Optional optimizer: missing/broken dupey must not make the corpus unavailable.
			}
		}
		return { excluded };
	}

	async syncBM25(minsync?: MinSyncSyncResult): Promise<BM25SyncResult | undefined> {
		if (this.bm25Method === undefined) return undefined;
		if (this.bm25Method instanceof MinSyncBM25Method) {
			return this.bm25Method.syncFromMinSync(minsync ?? (await this.bm25Method.sync()));
		}
		return this.bm25Method.sync();
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
	 * roots, and merges per-root answer packs using least-privilege
	 * (restrictive-wins) semantics. Jikji policy metadata remains visible to the
	 * model, but it does not gate the librarian's direct file-reading tools.
	 */
	async findJikji(
		query: string,
		opts?: { readonly topK?: number; readonly first?: boolean },
	): Promise<JikjiFindProviderResult> {
		if (this.jikjiClient === undefined) {
			return { answerPack: undefined, policy: undefined, diagnostics: [], roots: [], perRoot: [] };
		}
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

		const policy = this.mergePolicy(okPacks.map((entry) => entry.pack));
		const merged = this.mergeAnswerPacks(okPacks, sourceRoots, policy);
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
	 * - rawFallbackAllowed: handoffAction===raw_fallback_after_retry
	 */
	private mergePolicy(packs: readonly JikjiAnswerPack[]): MergedJikjiPolicy {
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
		const rawFallbackAllowed = handoff === "raw_fallback_after_retry";
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
		options = this.normalizeRetrievalOptions(options);
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
			results: this.rerankWithMemory(
				query,
				this.merger.merge(filteredByMethod, { topK: options.topK ?? 20, dedup: true }),
			),
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
			results: this.rerankWithMemory(
				query,
				this.merger.merge(filteredByMethod, { topK: options.topK ?? 20, dedup: true }),
			),
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

	private rerankWithMemory(query: string, results: readonly RetrievalResult[]): RetrievalResult[] {
		const methodScores = new Map(this.memory.getMethodHints(query).map((hint) => [hint.method, hint.score]));
		const context = this.memory.getContextHints(query);
		const scoreMap = (
			hints: readonly { readonly value: string; readonly score: number }[],
		): ReadonlyMap<string, number> => new Map(hints.map((hint) => [hint.value, hint.score]));
		const contextScores = {
			documentArea: scoreMap(context.documentAreas),
			documentType: scoreMap(context.documentTypes),
			evidenceType: scoreMap(context.evidenceTypes),
			evidenceLocation: scoreMap(context.evidenceLocations),
			parserType: scoreMap(context.parserTypes),
			retrieverMix: scoreMap(context.retrieverMix),
		};
		return results
			.map((result, index) => {
				const method = typeof result.metadata.method === "string" ? result.metadata.method : undefined;
				let preference = method ? (methodScores.get(method) ?? 0) : 0;
				for (const key of [
					"documentArea",
					"documentType",
					"evidenceType",
					"evidenceLocation",
					"parserType",
				] as const) {
					const value = result.metadata[key];
					if (typeof value === "string") preference += contextScores[key].get(value) ?? 0;
				}
				const retrievers = Array.isArray(result.metadata.retrieverMix)
					? result.metadata.retrieverMix
					: method
						? [method]
						: [];
				for (const retriever of retrievers) {
					if (typeof retriever === "string") preference += contextScores.retrieverMix.get(retriever) ?? 0;
				}
				const adjustment = Math.max(-0.25, Math.min(0.25, preference * 0.05));
				return { result, index, rankScore: result.score + adjustment };
			})
			.sort((a, b) => b.rankScore - a.rankScore || a.index - b.index)
			.map(({ result }) => result);
	}

	private normalizeRetrievalOptions(options: RetrievalOptions): RetrievalOptions {
		const scope = this.resolveRetrievalScope(options.scope);
		if (scope === undefined) {
			const { scope: _scope, ...rest } = options;
			return rest;
		}
		return { ...options, scope };
	}

	private resolveRetrievalScope(scope: string | undefined): string | undefined {
		return resolveRetrievalScope(
			scope,
			this.retrievalScopeBindings,
			process.platform,
			this.datasourceVirtualScopePrefixes,
		);
	}
}

function hasLegacyBM25Options(options: object): boolean {
	return ["indexPath", "fallback", "forceEngine", "importBinding"].some((key) => Object.hasOwn(options, key));
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

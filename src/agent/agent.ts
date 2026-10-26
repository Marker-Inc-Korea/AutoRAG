import { randomUUID } from "node:crypto";
import { existsSync, watch as fsWatch, mkdirSync, realpathSync, statSync, writeFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import type { Agent, AgentEvent, AgentMessage, AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import type { Api, ImageContent, Model, TextContent } from "@earendil-works/pi-ai";
import { clampThinkingLevel } from "@earendil-works/pi-ai/compat";
import type { ExtensionAPI, ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { getAgentDir, keyText } from "@earendil-works/pi-coding-agent";
import { resolveAutoRAGHome } from "../config/home.ts";
import {
	convertAndRelate,
	type FileeCliOptions,
	type FileeConvertRequest,
	listFileeFormats,
	resolveFileeOptions,
} from "../convert/index.ts";
import { mapDatasourceDiagnostics } from "../datasource/diagnostics.ts";
import { type DatasourceRetryOptions, indexWithRetry } from "../datasource/retry.ts";
import { filterDatasourceScope } from "../datasource/scope.ts";
import type { DatasourceIndexResult, DatasourceSkill } from "../datasource/types.ts";
import { DupeyCliError, type DupeyCliOptions, scanWithDupey, selectExactDuplicateExclusions } from "../dupey/index.ts";
import {
	createEmbedderFromConfig,
	type Embedder,
	type ManagedEmbedder,
	remoteEmbedderFromMinSync,
} from "../embedding/index.ts";
import {
	EverythingClient,
	type EverythingClientOptions,
	type EverythingSearchRequest,
	type EverythingSearchResult,
} from "../everything/index.ts";
import {
	FSearchClient,
	type FSearchClientOptions,
	type FSearchFailureReason,
	type FSearchSearchRequest,
	type FSearchSearchResult,
} from "../fsearch/index.ts";
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
	type JikjiPrepareOptions,
	type JikjiPrepareResult,
	type JikjiSourceRoot,
	normalizeJikjiAnswerPath,
	outermostJikjiRoots,
	planJikjiSourceRoots,
} from "../jikji/index.ts";
import { DEFAULT_LANGUAGES, type LanguageTag } from "../language.ts";
import { createCheckMemoryTool } from "../memory/check-memory-tool.ts";
import { loadMemoryContext, type MemoryContext, type MemoryContextOptions } from "../memory/context.ts";
import { EVIDENCE_SUPPORT_THRESHOLD, type JudgedEvidenceRecord } from "../memory/judged-evidence.ts";
import { RetrievalMemory } from "../memory/memory.ts";
import { memoryVectorStorePath } from "../memory/similar-queries.ts";
import { resolveMinSyncBinary } from "../minsync/binary.ts";
import { MinSyncRequiredError, minSyncBinaryMissingError } from "../minsync/errors.ts";
import {
	type MinSyncDiagnostic,
	type MinSyncEmbedderConfig,
	type MinSyncFileFailure,
	MinSyncHybridMethod,
	type MinSyncSyncResult,
	MinSyncVectorMethod,
	type MinSyncVectorMethodOptions,
} from "../minsync/index.ts";
import { PARSED_MIRROR_SUBDIR, refreshReadinessPath } from "../mirror/paths.ts";
import {
	detectMirrorStaleness,
	isPathExcluded,
	MIRROR_CHECKPOINT_EVERY,
	MirrorWorkspaceError,
	type ParsedMirrorDiagnostic,
	type ParsedMirrorSyncOptions,
	type ParsedMirrorSyncResult,
	syncParsedMirrors,
} from "../mirror/sync.ts";
import { AutoRAGRunLogger } from "../observability/run-log.ts";
import { type DefaultParserRegistryOptions, resolveParserOptions } from "../parser/index.ts";
import { locationSuffix } from "../retrieval/chunk-location.ts";
import { RetrievalEngine } from "../retrieval/engine.ts";
import { ParallelRetriever, ResultMerger } from "../retrieval/merger.ts";
import { RetrievalMethodRegistry } from "../retrieval/registry.ts";
import { createReranker, DEFAULT_RERANK_TOP_N, type Reranker } from "../retrieval/rerank.ts";
import {
	buildRetrievalScopeBindings,
	normalizeVirtualPath,
	type RetrievalScopeBinding,
	resolveRetrievalScope,
} from "../retrieval/scope.ts";
import type { DatasourceCatalogEntry } from "../retrieval/selection.ts";
import type { RetrievalDiagnostic, RetrievalOptions, RetrievalResult } from "../retrieval/types.ts";
import { executeWebSearch } from "../web/search/index.ts";
import { type ModelNativeSearchAuth, modelNativeAuthFromAgentModel } from "../web/search/model-auth.ts";
import { citedChunks, deriveResultsFromAnswer } from "./answer-citations.ts";
import {
	ANSWER_BULLET_RULE,
	ANSWER_CITATION_RULE,
	ANSWER_DELIVERABLE_DELTA_RULE,
	ANSWER_DELIVERABLE_RULE,
	ANSWER_EMPHASIS_RULE,
	ANSWER_IMAGE_DELTA_RULE,
	ANSWER_IMAGE_EMBED_RULE,
	ANSWER_LANGUAGE_RULE,
	ANSWER_LEAD_RULE,
	ANSWER_RELEVANCE_RULE,
	ANSWER_UNCERTAINTY_RULE,
	ANSWER_UNCONFIRMED_FAST_RULE,
	ANSWER_UNCONFIRMED_FINAL_RULE,
} from "./answer-guidelines.ts";
import { ASK_USER_TOOL_NAME, resolveAskUserExtensionPath } from "./ask-user.ts";
import { createManageAutomationsTool, MANAGE_AUTOMATIONS_TOOL_NAME } from "./automation-tool.ts";
import { createCitationLinks } from "./citation-links.ts";
import { CitationNumbers } from "./citation-numbers.ts";
import { CITATION_RELEVANCE_FLOOR, judgeCitationRelevance } from "./citation-relevance.ts";
import { ConversationQueries } from "./conversation-queries.ts";
import {
	CONVERT_FILES_TOOL_NAME,
	type ConvertFilesDetails,
	createConvertFilesTool,
	createListFileConversionsTool,
	LIST_FILE_CONVERSIONS_TOOL_NAME,
} from "./convert-tool.ts";
import {
	createLoadDatasourceSkillTool,
	type DatasourceAgentSkill,
	LOAD_DATASOURCE_SKILL_TOOL_NAME,
	toDatasourceAgentSkill,
} from "./datasource-skill.ts";
import { createDiscordRemoteExtension, DISCORD_REMOTE_TOOL_NAMES } from "./discord-remote.ts";
import {
	createScanDuplicateDocumentsTool,
	SCAN_DUPLICATE_DOCUMENTS_TOOL_NAME,
	type ScanDuplicateDocumentsDetails,
} from "./dupey-tool.ts";
import { createEmbeddingControls } from "./embedding-controls.ts";
import { createEverythingSearchTool, EVERYTHING_SEARCH_TOOL_NAME } from "./everything-search-tool.ts";
import { type EvidenceJudgmentUnit, judgeEvidence } from "./evidence-judgment.ts";
import { EvidenceLedger } from "./evidence-ledger.ts";
import { EvidenceOriginIndex } from "./evidence-origins.ts";
import { createEvidencePersistence } from "./evidence-persistence.ts";
import { createFSearchSearchTool, FSEARCH_SEARCH_TOOL_NAME } from "./fsearch-search-tool.ts";
import { intentOfMessage } from "./intent-tracing.ts";
import {
	createJevExtension,
	createJevJudge,
	JEV_TOOL_NAME,
	type JevJudge,
	type JevToolOptions,
} from "./jev-extension.ts";
import {
	createJikjiFindTool,
	JIKJI_FIND_TOOL_NAME,
	type JikjiFindPerRootPolicy,
	type JikjiFindProviderResult,
	type MergedJikjiPolicy,
} from "./jikji-find-tool.ts";
import { loadLocalAutoRAGModel } from "./local-model.ts";
import { loadPackagedSkill, ORGANIZE_SKILL_NAME, type PackagedAgentSkill, SETUP_SKILL_NAME } from "./packaged-skill.ts";
import { createPermissionGate, PermissionController, type PermissionMode } from "./permission-gate.ts";
import {
	type AutoRAGInputDisposition,
	type AutoRAGInteractiveInput,
	type AutoRAGInteractiveKey,
	type AutoRAGPiInteractiveRuntime,
	type AutoRAGPiSession,
	createAutoRAGPiInteractiveRuntime,
	createAutoRAGPiSession,
	PI_BUILTIN_TOOL_NAMES,
} from "./pi-session.ts";
import { createModelDecompositionCompleter, type DecompositionModel, decomposeQuery } from "./query-decomposition.ts";
import {
	type DatasourceCandidate,
	FALLBACK_QUERY_ROUTE,
	needsFollowUp,
	type PastSearchHint,
	type QueryRoute,
	routeQuery,
	selectDatasources,
	WEB_DATASOURCE_ID,
} from "./query-routing.ts";
import {
	isRefreshOwnerAlive,
	type PersistedRefreshProgress,
	type RefreshProgressCounts,
	type RefreshProgressPhase,
	readRefreshProgress,
	updateRefreshProgress,
	writeRefreshProgress,
} from "./refresh-progress.ts";
import type { AutoRAGResultsDetails } from "./results.ts";
import { createSearchAllDocumentsTool, SEARCH_ALL_DOCUMENTS_TOOL_NAME } from "./search-all-tool.ts";
import {
	createEmptySearchDocumentsResponse,
	createPreliminarySearchDocumentsResponse,
	normalizeEntryEvidenceRefs,
	recordStructuredResultsSession,
	type SearchDocumentDiagnostic,
	type SearchDocumentDiagnosticCode,
	type SearchDocumentRetrievalTraceEntry,
	type SearchDocumentRetrievalTraceResult,
	type SearchDocumentsResponse,
	type SearchDocumentsStreamEvent,
} from "./search-documents.ts";
import { createSearchMinSyncDocumentsTool, SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME } from "./search-minsync-tool.ts";
import { createSearchProgressExtension, SearchProgressTracker } from "./search-progress.ts";
import {
	createSingleDatasourceSearchTools,
	type SingleDatasourceToolSpec,
	singleDatasourceToolName,
} from "./search-single-datasource-tool.ts";
import { buildSelfConfigPrompt, rollbackIfBroken, type SelfConfigOptions, snapshotConfigFile } from "./self-config.ts";
import { formatSkillInvocation } from "./skill-invocation.ts";
import { buildSystemPrompt, type SystemPromptConfig } from "./system-prompt.ts";
import { ToolBurstGuard, type ToolBurstGuardOptions } from "./tool-burst-guard.ts";
import { createUserRequestPersistence, UserRequestLog } from "./user-requests.ts";
import {
	createWatchRefresh,
	type WatcherFactory,
	type WatchRefreshHandle,
	type WatchWatcher,
} from "./watch-refresh.ts";
import { createWebFetchTool, WEB_FETCH_TOOL_NAME, type WebFetchToolOptions } from "./web-fetch-tool.ts";
import { createWebSearchTool, WEB_SEARCH_TOOL_NAME, type WebSearchToolOptions } from "./web-search-tool.ts";

/**
 * Retrieval datasource tools whose executions count toward the per-search
 * tool budget and the run's evidence origins. `bash` is a source-inspection
 * tool, not a search datasource, so it is deliberately excluded from this list.
 */
const SEARCH_TOOLS = [
	SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME,
	SEARCH_ALL_DOCUMENTS_TOOL_NAME,
	JIKJI_FIND_TOOL_NAME,
	EVERYTHING_SEARCH_TOOL_NAME,
	FSEARCH_SEARCH_TOOL_NAME,
] as const;

/**
 * Safety ceiling on merged evidence when the caller names no `topK`.
 *
 * The merged fan-out is bounded by what the methods themselves returned (each
 * method caps its own fetch), so this only exists to stop a pathological
 * registry from producing an unbounded list. It is deliberately far above a
 * realistic fan-out: a live 16-method run over a personal corpus returned ~112
 * chunks (~26k tokens), which every current model holds comfortably, and
 * truncating below that silently hid evidence the librarian had already paid to
 * retrieve.
 */
const MERGED_EVIDENCE_CEILING = 500;

/** Similar past questions shown to the Jev datasource check; enough for a pattern, small enough for one batch. */
const SIMILAR_PAST_SEARCH_LIMIT = 5;
/** Results listed per similar past question; the best-judged evidence carries its answer. */
const SIMILAR_PAST_RESULT_LIMIT = 4;
/** Evidence cited by one answer is truncated to this many characters when stored in memory. */
const MEMORY_EXCERPT_CHARACTERS = 1_500;

/**
 * Safety limits for the parsed-mirror refresh. Every field is optional; an
 * omitted field keeps the shipped default.
 */
export interface AutoRAGRefreshLimits {
	/**
	 * Largest share (0–1) of indexed documents one refresh may delete because
	 * their source disappeared; a bigger deletion above 100 documents is
	 * suppressed unless the refresh is forced. Default 0.3.
	 */
	readonly maxDeleteRatio?: number;
	/** Directory entries examined per search root before the walk stops. Default 2,000,000. */
	readonly maxScannedEntries?: number;
	/** Deepest directory level walked below a search root. Default 64. */
	readonly maxDepth?: number;
	/** Retries for a transient per-file I/O error (EMFILE/EBUSY/EIO/ETIMEDOUT) before the file is skipped. Default 3. */
	readonly transientRetries?: number;
}

/**
 * Hard caps on retrieval, baseline prefetch, and the candidate lists handed to
 * the model. Every field is optional: an omitted field keeps the shipped
 * default below, so this only exists to let an operator tighten or widen a
 * specific cap without rebuilding. Values must be positive integers.
 */
export interface AutoRAGRetrievalLimits {
	/** `search_all_documents` merge ceiling when the model omits `topK`. Default 500. */
	readonly mergedEvidenceCeiling?: number;
	/** `search_datasource_*` merge default when the model omits `topK`. Default 20. */
	readonly singleDatasourceTopK?: number;
	/** MinSync semantic retrieval default `topK`. Default 50. */
	readonly minSyncTopK?: number;
	/** MinSync fetch cap when a scope narrows the query. Default 100. */
	readonly minSyncScopedQueryTopK?: number;
	/** Instance scopes listed in one datasource tool description. Default 8. */
	readonly toolDescriptionInstanceScopes?: number;
	/** Baseline evidence prefetched for the fast answer. */
	readonly prefetch?: {
		/** Jikji find candidate count. Default 30. */
		readonly jikjiTopK?: number;
		/** MinSync retrieve candidate count. Default 100. */
		readonly minSyncTopK?: number;
		/** Max Jikji answer paths rendered into the baseline. Default 100. */
		readonly jikjiPathLimit?: number;
		/** Max results rendered per baseline section. Default 100. */
		readonly sectionLimit?: number;
	};
}

/** Ship defaults for every {@link AutoRAGRetrievalLimits} field. */
const DEFAULT_RETRIEVAL_LIMITS = {
	mergedEvidenceCeiling: MERGED_EVIDENCE_CEILING,
	singleDatasourceTopK: 20,
	minSyncTopK: 50,
	minSyncScopedQueryTopK: 100,
	toolDescriptionInstanceScopes: 8,
	prefetch: { jikjiTopK: 30, minSyncTopK: 100, jikjiPathLimit: 100, sectionLimit: 100 },
} as const;

type ResolvedRetrievalLimits = {
	readonly mergedEvidenceCeiling: number;
	readonly singleDatasourceTopK: number;
	readonly minSyncTopK: number;
	readonly minSyncScopedQueryTopK: number;
	readonly toolDescriptionInstanceScopes: number;
	readonly prefetch: {
		readonly jikjiTopK: number;
		readonly minSyncTopK: number;
		readonly jikjiPathLimit: number;
		readonly sectionLimit: number;
	};
};

/** What the local branch searches alongside Jikji and MinSync: Jev's datasource ids and, when selected, the web. */
interface SearchDatasourceChoice {
	readonly datasources: readonly string[];
	readonly web: boolean;
}

/** One prefetched web search: its candidate hits, or the reason it contributed none. */
interface PrefetchedWebSearch {
	readonly results: readonly RetrievalResult[];
	readonly error?: string;
}

function positiveLimit(value: number | undefined, fallback: number, path: string): number {
	if (value === undefined) return fallback;
	if (!Number.isInteger(value) || value <= 0) throw new Error(`${path} must be a positive integer`);
	return value;
}

function validateRefreshLimits(limits: AutoRAGRefreshLimits | undefined): AutoRAGRefreshLimits {
	if (limits === undefined) return {};
	const { maxDeleteRatio, maxScannedEntries, maxDepth, transientRetries } = limits;
	if (
		maxDeleteRatio !== undefined &&
		!(Number.isFinite(maxDeleteRatio) && maxDeleteRatio >= 0 && maxDeleteRatio <= 1)
	) {
		throw new Error("refresh.maxDeleteRatio must be a number between 0 and 1");
	}
	if (maxScannedEntries !== undefined) positiveLimit(maxScannedEntries, 0, "refresh.maxScannedEntries");
	if (maxDepth !== undefined && (!Number.isInteger(maxDepth) || maxDepth < 0)) {
		throw new Error("refresh.maxDepth must be a non-negative integer");
	}
	if (transientRetries !== undefined && (!Number.isInteger(transientRetries) || transientRetries < 0)) {
		throw new Error("refresh.transientRetries must be a non-negative integer");
	}
	return {
		...(maxDeleteRatio !== undefined ? { maxDeleteRatio } : {}),
		...(maxScannedEntries !== undefined ? { maxScannedEntries } : {}),
		...(maxDepth !== undefined ? { maxDepth } : {}),
		...(transientRetries !== undefined ? { transientRetries } : {}),
	};
}

function resolveRetrievalLimits(limits: AutoRAGRetrievalLimits | undefined): ResolvedRetrievalLimits {
	const prefetch = limits?.prefetch;
	return {
		mergedEvidenceCeiling: positiveLimit(
			limits?.mergedEvidenceCeiling,
			DEFAULT_RETRIEVAL_LIMITS.mergedEvidenceCeiling,
			"limits.mergedEvidenceCeiling",
		),
		singleDatasourceTopK: positiveLimit(
			limits?.singleDatasourceTopK,
			DEFAULT_RETRIEVAL_LIMITS.singleDatasourceTopK,
			"limits.singleDatasourceTopK",
		),
		minSyncTopK: positiveLimit(limits?.minSyncTopK, DEFAULT_RETRIEVAL_LIMITS.minSyncTopK, "limits.minSyncTopK"),
		minSyncScopedQueryTopK: positiveLimit(
			limits?.minSyncScopedQueryTopK,
			DEFAULT_RETRIEVAL_LIMITS.minSyncScopedQueryTopK,
			"limits.minSyncScopedQueryTopK",
		),
		toolDescriptionInstanceScopes: positiveLimit(
			limits?.toolDescriptionInstanceScopes,
			DEFAULT_RETRIEVAL_LIMITS.toolDescriptionInstanceScopes,
			"limits.toolDescriptionInstanceScopes",
		),
		prefetch: {
			jikjiTopK: positiveLimit(
				prefetch?.jikjiTopK,
				DEFAULT_RETRIEVAL_LIMITS.prefetch.jikjiTopK,
				"limits.prefetch.jikjiTopK",
			),
			minSyncTopK: positiveLimit(
				prefetch?.minSyncTopK,
				DEFAULT_RETRIEVAL_LIMITS.prefetch.minSyncTopK,
				"limits.prefetch.minSyncTopK",
			),
			jikjiPathLimit: positiveLimit(
				prefetch?.jikjiPathLimit,
				DEFAULT_RETRIEVAL_LIMITS.prefetch.jikjiPathLimit,
				"limits.prefetch.jikjiPathLimit",
			),
			sectionLimit: positiveLimit(
				prefetch?.sectionLimit,
				DEFAULT_RETRIEVAL_LIMITS.prefetch.sectionLimit,
				"limits.prefetch.sectionLimit",
			),
		},
	};
}

export interface AutoRefreshOptions {
	readonly intervalMs: number;
	readonly immediate?: boolean;
}

/** Methods that `refresh` can selectively run. Defaults to all when omitted. */
export type RefreshMethod = "parsed" | "minsync" | "datasources" | "jikji" | "everything" | "fsearch";

/**
 * Progress of a refresh phase that can count its work. `parsed` counts source files
 * (`currentVirtualPath` is the opaque virtual path of the file about to be processed);
 * `datasources` counts configured datasource skills. Other phases do not report counts.
 */
export interface AutoRAGRefreshProgress {
	readonly phase: "parsed" | "datasources";
	/** Items finished so far. Advances by one per item and ends equal to `total`. */
	readonly processed: number;
	readonly total: number;
	readonly currentVirtualPath?: string;
}

export interface AutoRAGRefreshOptions {
	/** Restrict refresh to specific methods. Defaults to all when undefined. */
	readonly methods?: readonly RefreshMethod[];
	/**
	 * Retry sources previously skipped as parser-failed or source-unreadable once, ignoring their
	 * skip records. Unlike `force`, healthy unchanged mirrors are not re-parsed.
	 */
	readonly retrySkipped?: boolean;
	/**
	 * Cancels the refresh. It is checked between files and between phases, and it kills an in-flight parse
	 * worker and the MinSync, Jikji and datasource child processes. `refresh` then rejects with the signal's
	 * reason and records the run as failed; mirror entries written before the abort are kept.
	 */
	readonly signal?: AbortSignal;
	/** Called while the parse and datasource phases run. A throwing callback is ignored; it never fails the refresh. */
	readonly onProgress?: (progress: AutoRAGRefreshProgress) => void;
	/**
	 * Per-root `jikji prepare` controls: a hard `timeoutMs` wall per root and a
	 * `parseTimeout` (seconds, forwarded to `--parse-timeout`). Unset values fall
	 * back to the `jikji` client options. Cancel through the refresh `signal`.
	 */
	readonly jikjiPrepare?: Omit<JikjiPrepareOptions, "signal">;
}

export interface AutoRAGMinSyncRefreshResult {
	readonly ok: boolean;
	readonly synced: number;
	readonly reason?: string;
	/** Count of parsed documents excluded from the MinSync index by file name. */
	readonly stagingExcludedCount?: number;
	readonly diagnostics?: readonly SearchDocumentDiagnostic[];
}

export interface AutoRAGRefreshResult extends Omit<ParsedMirrorSyncResult, "diagnostics"> {
	readonly diagnostics: readonly SearchDocumentDiagnostic[];
	readonly minsync?: AutoRAGMinSyncRefreshResult;
	readonly datasources?: readonly DatasourceIndexResult[];
	/** Windows-only Everything file-name index outcome; absent on other platforms or when not selected. */
	readonly everything?: AutoRAGEverythingRefreshResult;
	/** macOS/Linux-only FSearch file-name index outcome; absent on other platforms or when not selected. */
	readonly fsearch?: AutoRAGFSearchRefreshResult;
}

export interface AutoRAGEverythingRefreshResult {
	readonly ok: boolean;
	readonly indexedItems?: number;
	readonly reason?: string;
}

export interface AutoRAGFSearchRefreshResult {
	readonly ok: boolean;
	readonly indexedItems?: number;
	/** Stable failure kind (e.g. "binary-missing") when ok is false. */
	readonly reason?: FSearchFailureReason;
	/** Verbatim underlying failure text when ok is false. */
	readonly message?: string;
}

export interface AutoRAGRefreshComponentStatus {
	readonly minsync?: string;
	readonly jikji?: string;
	readonly datasources?: string;
	readonly everything?: string;
	readonly fsearch?: string;
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
	readonly progress?: {
		readonly phase: RefreshProgressPhase;
		readonly sourceFiles?: { readonly total: number };
		readonly parsedCounts?: RefreshProgressCounts;
		readonly minsync?: { readonly synced?: number };
		readonly ownerAlive?: boolean;
	};
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
	everything?: AutoRAGEverythingRefreshResult;
	fsearch?: AutoRAGFSearchRefreshResult;
	lastError?: string;
	watchLimited: boolean;
	watchFailed: boolean;
}

/** Display-only TUI notice that a rerank call failed; stripped from the model's context. */
const RERANK_FAILURE_MESSAGE_TYPE = "autorag.rerank-failed";

/** Thinking level applied per search phase. "off" requests no reasoning. */
export type AutoRAGThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";

/**
 * Per-phase thinking for AutoRAG's two-phase search: the fast phase delivers
 * an immediate first answer with {@link AutoRAGThinkingOptions.fast} thinking
 * (default "minimal"), then the verification phase re-checks and finalizes with
 * {@link AutoRAGThinkingOptions.final} thinking (default "high").
 */
export interface AutoRAGThinkingOptions {
	/** Thinking level for the immediate first answer. Default "minimal": some hosted models reject "off". */
	readonly fast?: AutoRAGThinkingLevel;
	/** Thinking level for the verification/finalization phase. Default "high". */
	readonly final?: AutoRAGThinkingLevel;
}

export interface AutoRAGAgentOptions {
	model?: Model<Api>;
	apiKey?: string;
	providerApiKeys?: Readonly<Record<string, string>>;
	searchPaths: string[];
	languages?: readonly LanguageTag[];
	memoryPath?: string;
	workspacePath?: string;
	tools?: AgentTool[];
	minSync?: Omit<MinSyncVectorMethodOptions, "root">;
	jikji?: JikjiOptions | false;
	/**
	 * Windows-only instant file/folder name search through the bundled
	 * voidtools Everything (portable, user-level, folder-only index over
	 * `searchPaths`). Enabled by default on Windows; ignored elsewhere.
	 * `false` disables it. The remaining fields are test seams.
	 */
	everything?: Omit<EverythingClientOptions, "root" | "folders" | "excludeFolders"> | false;
	/**
	 * macOS/Linux-only instant file/folder name search through the user's
	 * fsearch-cli (FSearch, GPL — spawned as a separate process, never
	 * bundled). Indexes only `searchPaths` into `<workspace>/.autorag/fsearch/`
	 * and keeps a per-workspace `fsearch-cli watch` daemon live; searches fall
	 * back to a bounded slow filesystem walk when fsearch-cli is not
	 * installed. Enabled by default on macOS/Linux; ignored elsewhere.
	 * `false` disables it. The remaining fields are test seams.
	 */
	fsearch?: Omit<FSearchClientOptions, "root" | "folders"> | false;
	/**
	 * Internet web tools (`web_search` + `web_fetch`), ported from oh-my-pi's
	 * provider-chain web module. Default enabled and credential-free: the
	 * chain leads with providers that need no user-issued key — model-native
	 * search reusing the agent's own model credentials (Gemini grounding,
	 * Anthropic/OpenAI/xAI web_search), Parallel's keyless MCP, and the
	 * DuckDuckGo HTML form — then the anonymous Perplexity ask endpoint and
	 * the other scraped engines with headless-browser escalation for bot
	 * challenges. A self-hosted
	 * SEARXNG_ENDPOINT is the only env-gated option. `false` disables both
	 * tools. The `fetch` sub-option tunes or disables `web_fetch` alone.
	 */
	webSearch?: (WebSearchToolOptions & { fetch?: WebFetchToolOptions | false }) | false;
	/**
	 * Post-merge reranking. When set (and not `false`), merged retrieval evidence
	 * is reordered by a dedicated rerank model — OpenRouter by default. A
	 * configured-but-unavailable reranker is reported as a diagnostic and the
	 * merged order is preserved. `false` disables reranking.
	 */
	rerank?: RerankAgentOptions | false;
	autoRefresh?: AutoRefreshOptions;
	parserOptions?: DefaultParserRegistryOptions;
	dupey?: DupeyCliOptions | false;
	/**
	 * Filee command line behind `convert_files` and `list_file_conversions`.
	 * `false` leaves both tools out. The duplicate-family check on converted
	 * copies follows `dupey`.
	 */
	filee?: FileeCliOptions | false;
	/**
	 * Jev (TypeSafe's judgment model: typed questions in, calibrated
	 * probabilities out). When set, the two-phase search asks Jev before the
	 * fast answer which branch handles the question (local search, file
	 * organization, an extra task, or — in a local host — its own configuration)
	 * and whether to decompose it; after the fast answer, whether
	 * verification is needed. It also exposes the `jev` tool.
	 *
	 * The CLI config enables this by default on OpenRouter (`buildAgentOptions`
	 * fills it in). On this programmatic option, absent or `false` keeps Jev
	 * off, so library callers never make paid network calls they did not ask for.
	 */
	jev?: JevToolOptions | false;
	/**
	 * The embedder shared by memory and the datasources. Omitted, it is built from `minSync.embedder`: a
	 * configured remote endpoint, else MinSync's `minsync serve` daemon, else MinSync's in-process engine.
	 * `refresh` brings it up (starting the daemon when none runs); a search only reads from it.
	 */
	embedder?: ManagedEmbedder;
	/**
	 * Embedder used to find similar past questions in memory (semantic half of
	 * the hybrid search). Defaults to the shared {@link AutoRAGAgentOptions.embedder}; `false` keeps
	 * memory search keyword-only.
	 */
	memoryEmbedder?: Embedder | false;
	/**
	 * Agent self-configuration. When set (and Jev is on), Jev gains a `config`
	 * branch for questions about AutoRAG's own settings: the turn skips the
	 * fast answer, receives the full `autorag-setup` skill, edits
	 * `configPath` with the pi tools, and reports what it changed as its final
	 * message.
	 */
	selfConfig?: SelfConfigOptions;
	/**
	 * Override for the packaged `organize` skill file that the Jev `organize`
	 * branch loads into its turn; defaults to the packaged skill. Present for a
	 * host that ships its own organize rules, and for tests.
	 */
	organizeSkillPath?: string;
	/**
	 * Question decomposition used by the Jev query pipeline. `model` (with its
	 * `apiKey`) is the LLM that splits one question into at most five search
	 * queries; omitted, the search session's own model decomposes. The CLI
	 * resolves its configured or default model (`openrouter/qwen/qwen3.7-flash`).
	 */
	queryDecomposition?: { readonly model?: Model<Api>; readonly apiKey?: string };
	excludeExactDuplicates?: boolean;
	excludePaths?: readonly string[];
	/**
	 * Hard caps on retrieval, baseline prefetch, and model-facing candidate
	 * lists. Omitted fields keep their shipped defaults — see
	 * {@link AutoRAGRetrievalLimits}. Values must be positive integers.
	 */
	limits?: AutoRAGRetrievalLimits;
	/** Parsed-mirror refresh limits and retry tuning. See {@link AutoRAGRefreshLimits}. */
	refresh?: AutoRAGRefreshLimits;
	datasourceSkills?: readonly DatasourceSkill[];
	/**
	 * Retry/backoff for datasource `index()` during refresh: 2 retries with
	 * jittered exponential backoff by default, applied only to failures the
	 * skill marks retryable. `maxRetries: 0` disables retrying.
	 */
	datasourceRetry?: DatasourceRetryOptions;
	/** Non-fatal diagnostics from config/agent construction (e.g. skipped unknown datasources). */
	startupDiagnostics?: readonly SearchDocumentDiagnostic[];
	/**
	 * Runaway-loop guard: the run is aborted when one tool is called `maxCalls`
	 * times within `windowMs` (default 60 calls in 60 s). `false` disables it.
	 * There is no limit on a run's duration or its total number of tool calls.
	 */
	toolCallBurstGuard?: ToolBurstGuardOptions | false;
	/**
	 * Control the interactive (TUI) session from Discord through the
	 * `pi-discord-remote` extension: a session channel is created on launch, and
	 * allowed users' messages run as questions; the final answer is posted back. Needs `~/.pi/agent/pi-discord-remote/config.json` (or
	 * `DISCORD_TOKEN` + `DISCORD_GUILD_ID`).
	 * Default false.
	 */
	discordRemote?: boolean;
	/** Per-phase thinking levels for the two-phase (fast → verification) search. */
	thinking?: AutoRAGThinkingOptions;
	/** pi agent directory for auth/models/extensions. Defaults to ~/.pi/agent. */
	piAgentDir?: string;
	/** Optional persistent pi session directory. */
	piSessionDir?: string;
	/** Persist one pi session transcript per AutoRAG search. Defaults true. */
	persistPiSessions?: boolean;
	/**
	 * Best-effort provider for an interactive-only startup notice (e.g. a newer
	 * AutoRAG release). Resolves to the notice text or `undefined`; never fails
	 * a launch.
	 */
	updateNotice?: () => Promise<string | undefined>;
	/** Interactive only: show the AutoRAG welcome banner (with this version) instead of pi's startup header. */
	welcomeVersion?: string;
	/** Extension entry files (installed pi packages) loaded into the interactive session. */
	piExtensionPaths?: readonly string[];
	/** Flag values for the flags those extensions register. */
	piExtensionFlags?: Readonly<Record<string, string | boolean>>;
	/**
	 * Load the `@llblab/pi-telegram` bridge into the interactive session so a
	 * paired Telegram chat can start, steer, and read answers from it. The
	 * bridge adds `/telegram-setup` and `/telegram-connect`; its bot token and
	 * owner pairing live in pi's `telegram.json`, never in AutoRAG config.
	 */
	telegram?: boolean;
	/**
	 * Approval gate for what the agent does to existing files. `full` (the programmatic default) installs a
	 * gate that never asks; `ask` and `auto` approve deletes, moves, renames, and in-place edits of originals
	 * (copying and creating new files never ask). `auto` asks Jev first and only falls back to the user when
	 * Jev will not approve. See {@link PermissionMode}.
	 */
	permissionMode?: PermissionMode;
	/** Share one controller (mode, session grants) with a host that switches modes itself. Wins over `permissionMode`. */
	permissionController?: PermissionController;
}

/** Post-merge reranking options. Mirrors the CLI `RerankConfig` (secrets via env). */
export interface RerankAgentOptions {
	/** Provider id. @default "openrouter" */
	provider?: string;
	/** Wire model id. @default "voyageai/rerank-3-lite" */
	model?: string;
	/** API key. Prefer `apiKeyEnv`; this is a programmatic seam. Not used by the `dazzi` provider. */
	apiKey?: string;
	/** Environment variable holding the provider API key. Not used by the `dazzi` provider, which uses the `/login` credential. */
	apiKeyEnv?: string;
	/** Override the provider base URL. */
	baseUrl?: string;
	/** Return only the top N merged results. */
	topN?: number;
	/** Per-request timeout in milliseconds. */
	timeoutMs?: number;
}

export interface AutoRAGSearchSession {
	readonly agent: Agent;
	readonly piSession?: AutoRAGPiSession["session"];
	prompt(text: string): Promise<void>;
	abort(): Promise<void> | void;
	dispose(): void | Promise<void>;
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
	private readonly innerAgent: {
		readonly state: {
			systemPrompt: string;
			tools: readonly { name: string }[];
			messages: AgentMessage[];
		};
		readonly transformContext: (messages: AgentMessage[]) => Promise<AgentMessage[]>;
	};
	private readonly tools: readonly AgentTool[];
	/** Static SEARCH_TOOLS plus the generated per-datasource tool names. */
	private readonly searchToolNames: ReadonlySet<string>;
	private readonly configuredModel: Model<Api> | undefined;
	private readonly apiKey: string | undefined;
	private readonly providerApiKeys: Readonly<Record<string, string>> | undefined;
	private readonly listeners = new Set<Parameters<Agent["subscribe"]>[0]>();
	private activeSession: AutoRAGSearchSession | undefined;
	private readonly memory: RetrievalMemory;
	private readonly runLogger: AutoRAGRunLogger;
	private lastQuery: string | undefined;
	/** What the user has said in this conversation; Jev's permission gate judges file changes against it. */
	private readonly userRequests = new UserRequestLog();
	private lastSessionId: string | undefined;
	private activeRun = false;
	/**
	 * Steering state of the running search. A TUI query runs detached from pi's
	 * input loop (pi reads the next input only after its input handler returns),
	 * so input typed meanwhile steers this run (Enter) or waits behind it
	 * (Alt+Enter), the way pi steers its own turns. {@link steer} feeds the same
	 * state for `searchDocuments()` / `searchDocumentsStream()` callers.
	 */
	private readonly interactive: {
		/** The detached TUI search in flight, if any. */
		run: Promise<void> | undefined;
		/** Queries asked in order once the running search answers: Alt+Enter follow-ups, unanswered steers. */
		readonly followUps: string[];
		/** Steers this run received; any steer forces verification and is restated to it. */
		readonly steers: string[];
		/**
		 * Steers queued on the session agent's steering queue that no model request has
		 * taken yet; the next request drains them. A TUI turns any left when the search
		 * ends into follow-up queries instead of letting them leak into the next prompt.
		 */
		readonly queuedSteers: string[];
		/** Steers that arrived before the run's pi session existed; queued on it once it does. */
		readonly preSessionSteers: AgentMessage[];
		/** The run still takes steers: false once its last model request settled. */
		acceptingSteers: boolean;
		/** AutoRAG itself aborted the turn (tool-call limit, timeout): not the user stopping the search. */
		harnessAbort: boolean;
		/** The user stopped the search (Esc, {@link abort}): no later phase runs. */
		userAborted: boolean;
		/**
		 * Aborted when a steer arrives and replaced once the steer reaches the model:
		 * in-flight retrieval tools watch it, so a steer cuts a slow search short
		 * instead of waiting for it, like an interruptible wait in pi.
		 */
		steerInterrupt: AbortController;
		/**
		 * Alt+Enter was pressed while no turn streamed. pi then submits the editor
		 * like Enter, with no streaming behavior; this marks that one submit as a
		 * follow-up question.
		 */
		nextSubmitIsFollowUp: boolean;
		/**
		 * Every steer text this agent queued. The transcript and pi's queue keep the
		 * user's words; the model context frames these as corrections.
		 */
		readonly steerTexts: Set<string>;
	} = {
		run: undefined,
		followUps: [],
		steers: [],
		queuedSteers: [],
		preSessionSteers: [],
		acceptingSteers: false,
		harnessAbort: false,
		userAborted: false,
		steerInterrupt: new AbortController(),
		nextSubmitIsFollowUp: false,
		steerTexts: new Set(),
	};
	/** What the running search is doing, shown as a live widget in the TUI. */
	private readonly searchProgress = new SearchProgressTracker();
	/** Aborted by {@link abort} and by the run's timeout: gates publishing and the run's own requests. */
	private activeRunAbort: AbortController | undefined;
	private modelNativeSearchAuth: ModelNativeSearchAuth | undefined;
	private retrievalTrace: SearchDocumentRetrievalTraceEntry[] = [];
	/**
	 * Evidence the retrieval tools returned; the answer's inline `[eN]` citations resolve against it. A bound
	 * interactive session keeps it for the whole conversation and persists it with the transcript; any other
	 * run starts from an empty ledger.
	 */
	private readonly evidenceLedger = new EvidenceLedger();
	/** The result numbers (`[n]`) this conversation has shown, kept next to the ledger's `e<N>` ordinals. */
	private readonly citationNumbers = new CitationNumbers();
	/** Everything the user typed in this conversation; Jev reads it to judge whether a citation relates to what was asked. */
	private readonly conversationQueries = new ConversationQueries();
	private preliminaryCallback: ((response: SearchDocumentsResponse) => void) | undefined;
	/** Per-phase thinking levels of the two-phase search. */
	private readonly fastThinkingLevel: AutoRAGThinkingLevel;
	private readonly finalThinkingLevel: AutoRAGThinkingLevel;
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
	/** Roots handed to Jikji: duplicates and roots nested in another root are dropped (see `outermostJikjiRoots`). */
	private readonly jikjiRoots: readonly string[];
	private readonly configuredSearchPaths: readonly string[];
	readonly languages: readonly LanguageTag[];
	private retrievalScopeBindings: readonly RetrievalScopeBinding[];
	private readonly datasourceVirtualScopePrefixes: readonly string[];
	private readonly workspaceProjectRoot: string;
	private readonly methodRegistry = new RetrievalMethodRegistry();
	private readonly retriever = new ParallelRetriever();
	private readonly merger = new ResultMerger();
	private readonly reranker: Reranker | undefined;
	private readonly rerankTopN: number | undefined;

	private readonly minSyncMethod: MinSyncVectorMethod;
	private readonly minSyncEmbedderConfig: MinSyncEmbedderConfig | undefined;
	private readonly minSyncBinaryPath: string | undefined;
	private readonly jikjiClient: JikjiClient | undefined;
	private readonly everythingClient: EverythingClient | undefined;
	private readonly fsearchClient: FSearchClient | undefined;
	private readonly datasourceSkills: readonly DatasourceSkill[];
	private readonly datasourceRetry: DatasourceRetryOptions;
	private readonly startupDiagnostics: readonly SearchDocumentDiagnostic[];
	private readonly datasourceAgentSkills: readonly DatasourceAgentSkill[];
	private readonly parserOptions: DefaultParserRegistryOptions | undefined;
	private readonly dupeyOptions: DupeyCliOptions | false;
	private readonly fileeOptions: FileeCliOptions | false;
	/** pi extension registering the optional `jev` tool; undefined when disabled. */
	private readonly jevExtension: ExtensionFactory | undefined;
	/** Jev judge shared by the `jev` tool and the query router; undefined when disabled. */
	private readonly jevJudge: JevJudge | undefined;
	/** Mode and grants of the file-change approval gate; shared by every pi session this agent creates. */
	private readonly permissions: PermissionController;
	private readonly permissionGate: ExtensionFactory;
	/** Self-configuration, enabled only with Jev on a local session. */
	private readonly selfConfig: SelfConfigOptions | undefined;
	/** Override for the packaged organize skill the `organize` branch loads; undefined keeps the packaged one. */
	private readonly organizeSkillPath: string | undefined;
	private readonly queryDecompositionModel: DecompositionModel | undefined;
	/** Web search options for the agent's `web_search` / `web_fetch` tools; undefined when web tools are off. */
	private readonly webSearchOptions: WebSearchToolOptions | undefined;
	/** Routing diagnostics for the in-flight search; reset per run. */
	private routingDiagnostics: SearchDocumentDiagnostic[] = [];
	private readonly excludeExactDuplicates: boolean;
	private readonly excludePaths: readonly string[];
	private readonly baseSystemPromptConfig: SystemPromptConfig;
	private readonly droppedCallerToolNames: readonly string[];
	private readonly piAgentDir: string | undefined;
	private readonly piSessionDir: string | undefined;
	private readonly persistPiSessions: boolean;
	private readonly updateNotice: (() => Promise<string | undefined>) | undefined;
	private readonly welcomeVersion: string | undefined;
	private readonly piExtensionPaths: readonly string[] | undefined;
	private readonly piExtensionFlags: Readonly<Record<string, string | boolean>> | undefined;
	private readonly telegram: boolean;
	private boundPiRuntime: AutoRAGPiInteractiveRuntime["runtime"] | undefined;
	/** Embeds past questions for similar-question search; undefined keeps memory search keyword-only. */
	private readonly embedder: ManagedEmbedder;
	private readonly memoryEmbedder: Embedder | undefined;
	private readonly memoryVectorPath: string;
	/** Every retrieval result of the in-flight run with the query/method that surfaced it. */
	private readonly evidenceOrigins = new EvidenceOriginIndex();
	/** Search-tool queries by tool call id, read back when the call ends. */
	private readonly searchToolQueries = new Map<string, string>();
	/** Identifies this agent's conversation: everything judged in it is "current" memory. */
	private readonly conversationId = randomUUID();
	/** The memory context loaded for the in-flight run; cleared whenever a run starts. */
	private memoryContextCache: { readonly query: string; readonly context: Promise<MemoryContext> } | undefined;
	private readonly discordRemote: boolean;
	/** Whether the bound interactive session offers `ask_user`; the phase prompts name it only then. */
	private askUserOffered = false;
	private activeRetrievalOptions: RetrievalOptions | undefined;
	private readonly toolBurstGuard: ToolBurstGuard;
	private readonly limits: ResolvedRetrievalLimits;
	private readonly refreshLimits: AutoRAGRefreshLimits;

	constructor(options: AutoRAGAgentOptions) {
		const { memoryPath } = options;
		this.configuredModel = options.model;
		this.discordRemote = options.discordRemote ?? false;
		this.toolBurstGuard = new ToolBurstGuard(options.toolCallBurstGuard);
		this.fastThinkingLevel = options.thinking?.fast ?? "minimal";
		this.finalThinkingLevel = options.thinking?.final ?? "high";
		this.apiKey = options.apiKey;
		this.providerApiKeys = options.providerApiKeys;
		this.piAgentDir = options.piAgentDir;
		this.piSessionDir = options.piSessionDir;
		this.persistPiSessions = options.persistPiSessions ?? true;
		this.updateNotice = options.updateNotice;
		this.welcomeVersion = options.welcomeVersion;
		this.piExtensionPaths = options.piExtensionPaths;
		this.piExtensionFlags = options.piExtensionFlags;
		this.telegram = options.telegram === true;
		this.datasourceSkills = options.datasourceSkills ?? [];
		this.datasourceRetry = options.datasourceRetry ?? {};
		this.datasourceVirtualScopePrefixes = this.datasourceSkills.map((skill) =>
			normalizeVirtualPath(`/${skill.describe().name}`),
		);
		this.startupDiagnostics = options.startupDiagnostics ?? [];
		this.datasourceAgentSkills = this.buildDatasourceAgentSkills();
		this.configuredSearchPaths = options.searchPaths.map((searchPath) => resolve(searchPath));
		this.languages = options.languages ?? DEFAULT_LANGUAGES;
		this.searchPaths = options.searchPaths.map(pinSearchRoot);
		this.jikjiRoots = outermostJikjiRoots(this.searchPaths);
		this.workspaceProjectRoot = options.workspacePath ?? process.cwd();
		this.retrievalScopeBindings = buildRetrievalScopeBindings(
			this.workspaceProjectRoot,
			this.searchPaths,
			this.configuredSearchPaths,
		);
		// The global language setting selects the OCR engine; the workspace pins Tesseract's model cache.
		this.parserOptions = resolveParserOptions(options.parserOptions, this.languages, this.workspaceProjectRoot);
		this.dupeyOptions = options.dupey ?? {};
		this.fileeOptions = options.filee ?? {};
		this.excludeExactDuplicates = options.excludeExactDuplicates ?? true;
		this.excludePaths = (options.excludePaths ?? []).map(pinExcludedPath);
		this.limits = resolveRetrievalLimits(options.limits);
		this.refreshLimits = validateRefreshLimits(options.refresh);
		this.reranker = createReranker(
			options.rerank === false || options.rerank === undefined
				? false
				: { ...options.rerank, agentDir: options.piAgentDir },
		);
		this.rerankTopN =
			options.rerank === false || options.rerank === undefined
				? undefined
				: (options.rerank.topN ?? DEFAULT_RERANK_TOP_N);

		// MinSync is required: there is no option to turn it off and no degraded mode.
		if ((options.minSync as unknown) === false) {
			throw new MinSyncRequiredError(
				"MinSync is required and cannot be disabled; remove `minSync: false` and install the `minsync` binary.",
			);
		}
		const minSyncDefaults = {
			...(options.minSync ?? { autoInstall: true }),
			root: this.workspaceProjectRoot,
			defaultTopK: this.limits.minSyncTopK,
			scopedTopK: this.limits.minSyncScopedQueryTopK,
		};
		// The vector method owns sync, readiness and the dedicated semantic tool, but is not part of the
		// fan-out: the hybrid method already fuses BM25 and vectors over the same chunks and degrades to
		// BM25 by itself, so registering both would search the workspace twice and let the merger count one
		// index as two corroborating methods.
		this.minSyncMethod = new MinSyncVectorMethod(minSyncDefaults);
		this.methodRegistry.register(new MinSyncHybridMethod(minSyncDefaults));
		const registeredDatasourceIds = new Set<string>();
		for (const skill of this.datasourceSkills) {
			const datasourceId = skill.describe().datasourceId;
			if (datasourceId !== undefined) {
				if (registeredDatasourceIds.has(datasourceId)) continue;
				registeredDatasourceIds.add(datasourceId);
			}
			for (const method of skill.retrievalMethods()) this.methodRegistry.register(method);
		}
		if (options.jikji !== false) {
			this.jikjiClient = new JikjiClient({
				...(options.jikji ?? {}),
				root: this.workspaceProjectRoot,
			});
		}
		if (options.everything !== false) {
			const everythingClient = new EverythingClient({
				...(options.everything ?? {}),
				root: this.workspaceProjectRoot,
				folders: this.searchPaths,
				excludeFolders: this.excludePaths,
			});
			if (everythingClient.isSupported()) this.everythingClient = everythingClient;
		}
		if (options.fsearch !== false) {
			const fsearchClient = new FSearchClient({
				...(options.fsearch ?? {}),
				root: this.workspaceProjectRoot,
				folders: this.searchPaths,
				excludeFolders: this.excludePaths,
			});
			if (fsearchClient.isSupported()) this.fsearchClient = fsearchClient;
		}

		const memPath = memoryPath ?? join(resolveAutoRAGHome(), "memory.json");
		this.memory = new RetrievalMemory({ storagePath: memPath });
		this.memory.load();
		this.memoryVectorPath = memoryVectorStorePath(memPath);
		this.minSyncEmbedderConfig = options.minSync?.embedder;
		this.minSyncBinaryPath = options.minSync?.binaryPath;
		this.embedder = options.embedder ?? createEmbedderFromConfig({ embedder: options.minSync?.embedder });
		this.memoryEmbedder = options.memoryEmbedder === false ? undefined : (options.memoryEmbedder ?? this.embedder);
		this.runLogger = new AutoRAGRunLogger(join(dirname(memPath), "logs", "runs.jsonl"));

		const checkMemoryTool = createCheckMemoryTool(this.memory, () => this.memoryContextOptions());
		// One tool per configured datasource connection, so a question that
		// targets a single connection spawns only that connection's CLIs instead
		// of fanning out to every datasource. Generated from the configured
		// datasource skills, so every connection gets its own tool. These are
		// the only datasource retrieval tools: cross-datasource fan-out lives in
		// search_all_documents.
		const singleDatasourceTools = createSingleDatasourceSearchTools(
			this,
			this.singleDatasourceToolSpecs(),
			this.evidenceLedger,
		);
		this.searchToolNames = new Set([...SEARCH_TOOLS, ...singleDatasourceTools.map((tool) => tool.name)]);

		const searchMinSyncTool = createSearchMinSyncDocumentsTool(
			() => this.remoteFilteredRetrievalMethod(this.minSyncMethod),
			(scope) => this.resolveRetrievalScope(scope),
			this.evidenceLedger,
		);
		const searchAllTool = createSearchAllDocumentsTool(this, this.evidenceLedger);
		const loadDatasourceSkillTool = createLoadDatasourceSkillTool(this);
		const scanDuplicateDocumentsTool =
			this.dupeyOptions === false ? undefined : createScanDuplicateDocumentsTool(this);
		const convertFilesTool = this.fileeOptions === false ? undefined : createConvertFilesTool(this);
		const listFileConversionsTool = this.fileeOptions === false ? undefined : createListFileConversionsTool(this);
		const manageAutomationsTool = createManageAutomationsTool();
		this.jevJudge =
			options.jev === undefined || options.jev === false
				? undefined
				: createJevJudge({ ...options.jev, agentDir: options.piAgentDir });
		this.jevExtension = this.jevJudge === undefined ? undefined : createJevExtension(this.jevJudge);
		this.selfConfig = this.jevJudge === undefined ? undefined : options.selfConfig;
		this.organizeSkillPath = options.organizeSkillPath;
		this.permissions = options.permissionController ?? new PermissionController(options.permissionMode ?? "full");
		this.permissionGate = createPermissionGate({
			controller: this.permissions,
			...(this.jevJudge === undefined ? {} : { judge: this.jevJudge }),
			requests: this.userRequests,
			// The self-configuration turn edits AutoRAG's own config at the user's request; that is never a gated original.
			...(this.selfConfig === undefined ? {} : { trustedPaths: [resolve(this.selfConfig.configPath)] }),
		});
		this.queryDecompositionModel =
			options.queryDecomposition?.model === undefined
				? undefined
				: {
						model: options.queryDecomposition.model,
						...(options.queryDecomposition.apiKey !== undefined
							? { apiKey: options.queryDecomposition.apiKey }
							: {}),
					};

		const jikjiFindTool = this.jikjiClient !== undefined ? createJikjiFindTool(this, this.evidenceLedger) : undefined;
		const everythingSearchTool = this.everythingClient !== undefined ? createEverythingSearchTool(this) : undefined;
		const fsearchSearchTool = this.fsearchClient !== undefined ? createFSearchSearchTool(this) : undefined;

		const webSearchOption = options.webSearch;
		const webToolsEnabled = webSearchOption !== false;
		this.webSearchOptions = webToolsEnabled
			? { ...(webSearchOption ?? {}), modelAuth: () => this.modelNativeSearchAuth }
			: undefined;
		const webSearchTool =
			this.webSearchOptions !== undefined
				? createWebSearchTool(this.webSearchOptions, this.evidenceLedger)
				: undefined;
		const webFetchTool =
			webToolsEnabled && webSearchOption?.fetch !== false
				? createWebFetchTool(webSearchOption?.fetch ?? {}, this.evidenceLedger)
				: undefined;

		// Reserved AutoRAG tool names the agent always owns. Caller tools with
		// these names are dropped (reserved wins), never rejected.
		const reservedNames = new Set<string>([
			...PI_BUILTIN_TOOL_NAMES,
			"check_memory",
			...singleDatasourceTools.map((tool) => tool.name),
			LOAD_DATASOURCE_SKILL_TOOL_NAME,
			SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME,
			SEARCH_ALL_DOCUMENTS_TOOL_NAME,
			JIKJI_FIND_TOOL_NAME,
			EVERYTHING_SEARCH_TOOL_NAME,
			FSEARCH_SEARCH_TOOL_NAME,
			SCAN_DUPLICATE_DOCUMENTS_TOOL_NAME,
			CONVERT_FILES_TOOL_NAME,
			LIST_FILE_CONVERSIONS_TOOL_NAME,
			MANAGE_AUTOMATIONS_TOOL_NAME,
			JEV_TOOL_NAME,
			WEB_SEARCH_TOOL_NAME,
			WEB_FETCH_TOOL_NAME,
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

		// pi owns the built-in tools; AutoRAG contributes caller and domain tools.
		const orderedTools: AgentTool[] = [
			...callerTools,
			checkMemoryTool,
			searchMinSyncTool,
			searchAllTool,
			...singleDatasourceTools,
			loadDatasourceSkillTool,
			...(webSearchTool !== undefined ? [webSearchTool] : []),
			...(webFetchTool !== undefined ? [webFetchTool] : []),
			...(scanDuplicateDocumentsTool !== undefined ? [scanDuplicateDocumentsTool] : []),
			...(convertFilesTool !== undefined ? [convertFilesTool] : []),
			...(listFileConversionsTool !== undefined ? [listFileConversionsTool] : []),
			manageAutomationsTool,
			...(jikjiFindTool !== undefined ? [jikjiFindTool] : []),
			...(everythingSearchTool !== undefined ? [everythingSearchTool] : []),
			...(fsearchSearchTool !== undefined ? [fsearchSearchTool] : []),
		];
		const seenToolNames = new Set<string>();
		const tools = orderedTools.filter((tool) => {
			if (seenToolNames.has(tool.name)) return false;
			seenToolNames.add(tool.name);
			return true;
		});
		this.tools = tools;
		// pi registers the jev tool from its extension; AutoRAG still lists the
		// name so the prompt advertises it and reserved-name checks cover it.
		const toolNames = [
			...PI_BUILTIN_TOOL_NAMES,
			...tools.map((tool) => tool.name),
			...(this.jevExtension !== undefined ? [JEV_TOOL_NAME] : []),
		];
		this.baseSystemPromptConfig = {
			toolNames,
			modelId: options.model?.id,
			datasourceSkills: this.datasourceAgentSkills,
			jikjiIndexingEnabled: options.jikji !== false,
			retrievedContentGuard: false,
		};
		const systemPrompt = buildSystemPrompt(this.currentSystemPromptConfig());

		this.innerAgent = {
			state: {
				systemPrompt,
				tools: [...PI_BUILTIN_TOOL_NAMES.map((name) => ({ name })), ...tools],
				messages: [],
			},
			transformContext: (messages) => this.withMemoryContext(messages),
		};

		if (options.autoRefresh) {
			this.startAutoRefresh(options.autoRefresh.intervalMs, { immediate: options.autoRefresh.immediate });
		}
	}

	async convertFiles(request: FileeConvertRequest, signal?: AbortSignal): Promise<ConvertFilesDetails> {
		if (this.fileeOptions === false) throw new Error("File conversion is disabled.");
		const filee = await resolveFileeOptions(this.fileeOptions, { env: process.env });
		return convertAndRelate(request, {
			filee: { ...filee, ...(signal !== undefined ? { signal } : {}) },
			dupey: this.dupeyOptions,
		});
	}

	async listFileConversions(from?: string, signal?: AbortSignal): Promise<string> {
		if (this.fileeOptions === false) throw new Error("File conversion is disabled.");
		const filee = await resolveFileeOptions(this.fileeOptions, { env: process.env });
		return listFileeFormats(from, { ...filee, ...(signal !== undefined ? { signal } : {}) });
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

	/**
	 * Reference memory for the current question: everything judged earlier in
	 * this conversation, up to 25 similar past questions, and matching
	 * long-term insights. Computed once per run (the datasource check and every
	 * model turn share it) so the similar-question search and its embedder call
	 * are not repeated and a fallback is reported once.
	 */
	private loadMemoryContextFor(query: string): Promise<MemoryContext> {
		if (this.memoryContextCache?.query === query) return this.memoryContextCache.context;
		const context = loadMemoryContext(this.memory, query, this.memoryContextOptions()).then((loaded) => {
			if (loaded.fallbackReason !== undefined) {
				this.routingDiagnostics.push({
					code: "memory-search-fallback",
					severity: "warning",
					message: loaded.fallbackReason,
					source: "memory",
				});
			}
			return loaded;
		});
		this.memoryContextCache = { query, context };
		return context;
	}

	private async withMemoryContext(messages: AgentMessage[]): Promise<AgentMessage[]> {
		if (this.lastQuery === undefined) return messages;
		const context = await this.loadMemoryContextFor(this.lastQuery);
		if (context.empty) return messages;
		return [
			{
				role: "user",
				content: [{ type: "text", text: `<memory_context>\n${context.text}\n</memory_context>` }],
				timestamp: Date.now(),
			},
			...messages,
		];
	}

	/**
	 * Where evidence came from, or undefined when this run may not see it.
	 * Datasource chunks carry a virtual path rooted at the skill name
	 * (`/kakao/default/...`), but the model sometimes maps a datasource hit to a
	 * bare chunk id. The evidence method then still names the datasource
	 * (`search_datasource_kakao_work`, `datasource:kakao`, `kakao-work-lexical`):
	 * both sides collapse every non-alphanumeric to `_` exactly like generated
	 * tool names, and a datasource id that appears as a whole token attributes
	 * it; longer ids are checked first so `kakao-work` wins over `kakao`.
	 * Indexed files are absolute paths under a search path; web evidence is a URL.
	 * Memory is global (shared across workspaces and configs), so evidence from
	 * an unconfigured datasource, and evidence with no recognizable origin, is
	 * never attributed.
	 */
	private evidenceAttribution(
		catalog: readonly DatasourceCatalogEntry[],
	): (evidence: { readonly source: string; readonly method: string }) => string | undefined {
		const datasourceByRoot = new Map(catalog.map((entry) => [entry.name, entry.datasourceId]));
		const nonDatasourceRoots = new Set(
			this.datasourceSkills.map((skill) => skill.describe().name).filter((name) => !datasourceByRoot.has(name)),
		);
		const methodToken = (value: string): string => `_${value.toLowerCase().replace(/[^a-z0-9]+/gu, "_")}_`;
		const idsLongestFirst = catalog.map((entry) => entry.datasourceId).sort((a, b) => b.length - a.length);
		return ({ source, method }) => {
			if (/^https?:\/\//u.test(source)) return "web";
			const normalizedSource = source.replaceAll("\\", "/");
			for (const configuredRoot of this.configuredSearchPaths) {
				const normalizedRoot = configuredRoot.replaceAll("\\", "/").replace(/\/+$/u, "");
				if (normalizedSource === normalizedRoot || normalizedSource.startsWith(`${normalizedRoot}/`))
					return "local files";
			}
			const root = normalizedSource.startsWith("/") ? (normalizedSource.split("/")[1] ?? "") : "";
			if (nonDatasourceRoots.has(root)) return undefined;
			const byRoot = datasourceByRoot.get(root);
			if (byRoot !== undefined) return byRoot;
			const tokens = methodToken(method);
			return idsLongestFirst.find((id) => tokens.includes(methodToken(id)));
		};
	}

	/** Options shared by the prompt injection and the `check_memory` tool; rebuilt on every call because datasource configuration follows live settings. */
	private memoryContextOptions(): MemoryContextOptions {
		const whereFound = this.evidenceAttribution(this.listDatasources());
		const insightSources = new Set(this.listDatasources().map((entry) => entry.datasourceId));
		return {
			conversationId: this.conversationId,
			isVisible: (record) => whereFound(record) !== undefined,
			isInsightVisible: (insight) =>
				insight.recommendedSources.every((source) => whereFound({ source, method: "" }) !== undefined) &&
				insight.recommendedMethods.every(
					(method) =>
						!method.startsWith("search_datasource_") ||
						[...insightSources].some((id) => method.includes(id.replace(/[^a-zA-Z0-9]+/gu, "_"))),
				),
			...(this.memoryEmbedder !== undefined ? { embedder: this.memoryEmbedder } : {}),
			vectorStorePath: this.memoryVectorPath,
		};
	}

	/** Remember which query and method surfaced each retrieval result, so cited evidence can be traced back. */
	private rememberEvidenceOrigins(
		query: string | undefined,
		method: string,
		results: readonly { readonly source?: string; readonly excerpt: string }[],
	): void {
		if (query === undefined) return;
		for (const result of results) {
			this.evidenceOrigins.add({ query, method, source: result.source ?? "", content: result.excerpt });
		}
	}

	/**
	 * After the final answer: map every cited evidence back to the search query
	 * and method that found it, ask Jev (one batched call, evidence judged in
	 * parallel) whether it really supports the answer sentence it backs, and
	 * keep only what clears the threshold. Runs without Jev
	 * store nothing; a failed judgment stores nothing and says why.
	 */
	private async recordJudgedEvidence(
		sessionId: string,
		question: string,
		details: AutoRAGResultsDetails,
	): Promise<SearchDocumentDiagnostic | undefined> {
		if (details.mapping.length === 0) return undefined;
		if (this.jevJudge === undefined) {
			return {
				code: "evidence-judgment-fallback",
				severity: "info",
				message: "Jev is disabled, so this answer's evidence was not judged and nothing was added to memory.",
				source: "memory",
			};
		}
		const resultByNumber = new Map(details.results.map((result) => [result.number, result]));
		const units: (EvidenceJudgmentUnit & {
			readonly record: Omit<JudgedEvidenceRecord, "probability" | "createdAt">;
		})[] = [];
		for (const entry of details.mapping) {
			const result = resultByNumber.get(entry.number);
			const refs = normalizeEntryEvidenceRefs(entry);
			if (result === undefined || refs.length === 0) continue;
			for (const ref of refs) {
				const excerpt = (ref.excerpt ?? ref.content ?? "").slice(0, MEMORY_EXCERPT_CHARACTERS);
				if (excerpt.length === 0) continue;
				const origin = this.evidenceOrigins.resolve(ref);
				const searchQuery = origin?.query ?? question;
				const method = origin?.method ?? ref.method;
				const id = `${entry.number}:${ref.stableEvidenceId}`;
				units.push({
					id,
					resultNumber: entry.number,
					title: result.title,
					summary: result.summary,
					searchQuery,
					method,
					excerpt,
					record: {
						id: `${sessionId}:${ref.stableEvidenceId}`,
						sessionId,
						conversationId: this.conversationId,
						question,
						searchQuery,
						method,
						source: ref.source,
						stableEvidenceId: ref.stableEvidenceId,
						resultNumber: entry.number,
						title: result.title,
						excerpt,
					},
				});
			}
		}
		if (units.length === 0) return undefined;
		const judgment = await judgeEvidence(this.jevJudge, { question, answer: details.answer, units });
		const diagnostic: SearchDocumentDiagnostic =
			judgment.fallbackReason !== undefined
				? {
						code: "evidence-judgment-fallback",
						severity: "warning",
						message: `Jev evidence judgment was unavailable; nothing was added to memory. ${judgment.fallbackReason}`,
						source: "memory",
					}
				: {
						code: "evidence-judged",
						severity: "info",
						message: `Jev judged ${Object.keys(judgment.probabilities).length} of ${units.length} cited evidence; ${judgment.kept.length} supported the question (p >= ${EVIDENCE_SUPPORT_THRESHOLD}) and were added to memory.`,
						source: "memory",
					};
		if (judgment.kept.length === 0) return diagnostic;
		const now = Date.now();
		const keptIds = new Set(judgment.kept);
		this.memory.recordJudgedEvidence(
			units
				.filter((unit) => keptIds.has(unit.id))
				.map((unit) => ({ ...unit.record, probability: judgment.probabilities[unit.id] ?? 0, createdAt: now })),
		);
		this.memory.save();
		return diagnostic;
	}

	private async resolveSessionModel(): Promise<{
		readonly model: Model<Api>;
		readonly apiKey?: string;
		readonly providerApiKeys?: Readonly<Record<string, string>>;
	}> {
		const boundModel = this.boundPiRuntime?.session.model as Model<Api> | undefined;
		if (boundModel !== undefined) {
			const auth = await this.boundPiRuntime?.session.modelRuntime.getAuth(boundModel);
			const apiKey = auth?.auth.apiKey;
			return {
				model: boundModel,
				...(apiKey !== undefined ? { apiKey, providerApiKeys: { [boundModel.provider]: apiKey } } : {}),
			};
		}
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

	private async createSearchSession(
		resolved: {
			readonly model: Model<Api>;
			readonly apiKey?: string;
			readonly providerApiKeys?: Readonly<Record<string, string>>;
		},
		systemPrompt: string,
	): Promise<AutoRAGSearchSession> {
		if (this.boundPiRuntime !== undefined) {
			const session = this.boundPiRuntime.session;
			return {
				agent: session.agent,
				piSession: session,
				prompt: async (prompt) => session.prompt(prompt, { source: "extension" }),
				abort: async () => session.abort(),
				dispose: () => {},
			};
		}
		const piSession = await createAutoRAGPiSession({
			cwd: this.workspaceProjectRoot,
			agentDir: this.piAgentDir,
			sessionDir: this.piSessionDir,
			persistSession: this.persistPiSessions,
			model: resolved.model,
			apiKey: resolved.apiKey,
			providerApiKeys: resolved.providerApiKeys,
			getSystemPrompt: () => systemPrompt,
			customTools: this.steerableTools(),
			// The model sees each steer framed as a correction of the running search.
			contextTransform: (messages) => this.withMemoryContext(frameSteers(messages, this.interactive.steerTexts)),
			extensionFactories: [this.permissionGate, ...(this.jevExtension !== undefined ? [this.jevExtension] : [])],
			extensionToolNames: this.jevExtension !== undefined ? [JEV_TOOL_NAME] : [],
		});
		const agent = piSession.session.agent;
		return {
			agent,
			piSession: piSession.session,
			prompt: async (prompt) => piSession.session.prompt(prompt, { source: "extension" }),
			abort: async () => piSession.session.abort(),
			dispose: () => piSession.dispose(),
		};
	}

	private configureSearchSession(session: AutoRAGSearchSession): readonly (() => void)[] {
		const unsubscribers = [...this.listeners].map((listener) => session.agent.subscribe(listener));
		unsubscribers.push(
			session.agent.subscribe((event) => {
				this.recordSearchToolEvent(event);
				// A user message starts each model request; queued steers go out with it.
				if (event.type === "message_start" && event.message.role === "user") this.steersReachedModel();
			}),
		);
		return unsubscribers;
	}

	private recordSearchToolEvent(event: AgentEvent): void {
		if (event.type === "tool_execution_end" && this.toolBurstGuard.record(event.toolName)) {
			if (!this.interactive.harnessAbort) {
				this.routingDiagnostics.push({
					code: "tool-call-burst",
					severity: "warning",
					message: `Stopped the run: ${event.toolName} was called in a runaway loop (more than the allowed calls per window). Run duration and total tool-call count are not limited.`,
				});
			}
			this.interactive.harnessAbort = true;
			void this.activeSession?.abort();
		}
		if (event.type === "tool_execution_start" && this.searchToolNames.has(event.toolName)) {
			const query: unknown = event.args?.query;
			if (typeof query === "string") this.searchToolQueries.set(event.toolCallId, query);
			return;
		}
		if (event.type !== "tool_execution_end" || !this.lastQuery) return;
		if (!this.searchToolNames.has(event.toolName)) return;
		const details = event.result.details as
			| {
					method?: string;
					sources?: readonly string[];
					resultCount?: number;
					results?: readonly SearchDocumentRetrievalTraceResult[];
			  }
			| undefined;
		if (Array.isArray(details?.results)) {
			const calledQuery = this.searchToolQueries.get(event.toolCallId);
			this.searchToolQueries.delete(event.toolCallId);
			this.retrievalTrace.push({
				tool: event.toolName,
				...(calledQuery !== undefined ? { query: calledQuery } : {}),
				resultCount:
					typeof details?.resultCount === "number" ? details.resultCount : (details?.results?.length ?? 0),
				results: details?.results ?? [],
			});
			this.rememberEvidenceOrigins(
				calledQuery ?? this.lastQuery,
				details?.method ?? event.toolName,
				details?.results ?? [],
			);
		}
	}

	private currentSystemPromptConfig(models: Partial<SystemPromptConfig> = {}): SystemPromptConfig {
		return {
			...this.baseSystemPromptConfig,
			...models,
		};
	}

	/** The approval gate's controller: read or switch the permission mode of this agent's sessions. */
	get permissionControl(): PermissionController {
		return this.permissions;
	}

	subscribe(listener: Parameters<Agent["subscribe"]>[0]): () => void {
		this.listeners.add(listener);
		return () => {
			this.listeners.delete(listener);
		};
	}

	async createPiInteractiveRuntime(): Promise<AutoRAGPiInteractiveRuntime> {
		const resolved = this.configuredModel === undefined ? undefined : await this.resolveSessionModel();
		// Loaded on demand: the bridge is large and most sessions never enable it.
		const telegramBridge = this.telegram ? (await import("@llblab/pi-telegram")).default : undefined;
		// ask_user opens a dialog at the terminal. A session a chat bridge drives has nobody there to answer it
		// (Discord brings its own question tool), so the tool is offered only when the terminal is the user's.
		const askUser = !this.discordRemote && !this.telegram && this.piExtensionPaths === undefined;
		const runtime = await createAutoRAGPiInteractiveRuntime({
			cwd: this.workspaceProjectRoot,
			agentDir: this.piAgentDir,
			sessionDir: this.piSessionDir,
			persistSession: this.persistPiSessions,
			...(resolved === undefined
				? {}
				: {
						model: resolved.model,
						...(resolved.apiKey !== undefined ? { apiKey: resolved.apiKey } : {}),
						...(resolved.providerApiKeys !== undefined ? { providerApiKeys: resolved.providerApiKeys } : {}),
					}),
			getSystemPrompt: () =>
				buildSystemPrompt(
					this.currentSystemPromptConfig({
						modelId: this.boundPiRuntime?.session.model?.id ?? resolved?.model.id,
						...(askUser ? { toolNames: [...this.baseSystemPromptConfig.toolNames, ASK_USER_TOOL_NAME] } : {}),
					}),
				),
			// The model sees each steer framed as a correction of the running search;
			// the transcript, pi's queue display, and Esc/dequeue restores keep the user's words.
			contextTransform: async (messages) =>
				this.withMemoryContext(
					frameSteers(
						messages.filter(
							(message) => !(message.role === "custom" && message.customType === RERANK_FAILURE_MESSAGE_TYPE),
						),
						this.interactive.steerTexts,
					),
				),
			customTools: this.steerableTools(),
			onInput: (input, pi) => this.routeInteractiveInput(input, pi),
			isSearchRunning: () => this.interactive.run !== undefined,
			onKey: (key) => this.handleInteractiveKey(key),
			extensionFactories: [
				this.permissionGate,
				createUserRequestPersistence(this.userRequests),
				createEvidencePersistence(this.evidenceLedger, this.citationNumbers, this.conversationQueries),
				createCitationLinks(this.evidenceLedger),
				createSearchProgressExtension(this.searchProgress, () => keyText("app.interrupt")),
				createEmbeddingControls({
					embedder: this.embedder,
					remote: remoteEmbedderFromMinSync(this.minSyncEmbedderConfig) !== undefined,
					resolveBinary: () =>
						resolveMinSyncBinary({
							root: this.workspaceProjectRoot,
							...(this.minSyncBinaryPath === undefined ? {} : { binaryPath: this.minSyncBinaryPath }),
							autoInstall: false,
						}),
				}),
				...(this.jevExtension !== undefined ? [this.jevExtension] : []),
				...(telegramBridge !== undefined ? [telegramBridge] : []),
				...(this.discordRemote ? [this.discordRemoteExtension()] : []),
			],
			extensionToolNames: [
				...(this.jevExtension !== undefined ? [JEV_TOOL_NAME] : []),
				...(this.discordRemote ? DISCORD_REMOTE_TOOL_NAMES : []),
				...(askUser ? [ASK_USER_TOOL_NAME] : []),
			],
			...(this.updateNotice === undefined ? {} : { updateNotice: this.updateNotice }),
			...(this.welcomeVersion === undefined ? {} : { welcomeVersion: this.welcomeVersion }),
			...(askUser
				? { extensionPaths: [resolveAskUserExtensionPath()] }
				: this.piExtensionPaths === undefined
					? {}
					: { extensionPaths: this.piExtensionPaths }),
			...(this.piExtensionFlags === undefined ? {} : { extensionFlags: this.piExtensionFlags }),
		});
		this.boundPiRuntime = runtime.runtime;
		this.askUserOffered = askUser;
		const dispose = runtime.dispose;
		return {
			...runtime,
			dispose: async () => {
				if (this.boundPiRuntime === runtime.runtime) {
					this.boundPiRuntime = undefined;
					this.askUserOffered = false;
					// A disposed session must not receive later numbers.
					this.citationNumbers.setSink(undefined);
					this.conversationQueries.setSink(undefined);
				}
				await dispose();
			},
		};
	}

	/**
	 * Discord messages enter the same machinery as typed input: a new search when
	 * idle, a steer of the running one otherwise (queued as a follow-up question
	 * once the search can no longer take it).
	 */
	private discordRemoteExtension(): ExtensionFactory {
		return createDiscordRemoteExtension({
			submit: (text, pi) => {
				const state = this.interactive;
				if (state.run === undefined) {
					this.userRequests.record(text);
					this.startInteractiveQuery(text, pi);
				} else if (!this.steer(text)) {
					this.userRequests.record(text);
					state.followUps.push(text);
				}
			},
			isRunning: () => this.interactive.run !== undefined,
			onRunSettled: (callback) => void this.waitForIdle().then(callback),
		});
	}

	/**
	 * The session's custom tools, with retrieval and web reads interruptible by a
	 * steer: stopping them loses no side effect. bash and the file tools are pi
	 * built-ins and never pass through here.
	 */
	private steerableTools(): AgentTool[] {
		return this.tools
			.filter((tool) => !PI_BUILTIN_TOOL_NAMES.includes(tool.name as (typeof PI_BUILTIN_TOOL_NAMES)[number]))
			.map((tool) =>
				this.searchToolNames.has(tool.name) ||
				tool.name === WEB_SEARCH_TOOL_NAME ||
				tool.name === WEB_FETCH_TOOL_NAME
					? this.interruptibleBySteer(tool)
					: tool,
			);
	}

	/**
	 * Lets a steer cut a retrieval tool short, like an interruptible wait in pi:
	 * the tool's signal also aborts on the steer, and the call resolves at once
	 * with a skipped result telling the model to retry if the search still
	 * matters, so the steer reaches the model before the next request.
	 */
	private interruptibleBySteer(tool: AgentTool): AgentTool {
		return {
			...tool,
			execute: async (toolCallId, params, signal, onUpdate) => {
				const steer = this.interactive.steerInterrupt.signal;
				if (steer.aborted) return skippedBySteer(tool.name);
				const combined = signal === undefined ? steer : AbortSignal.any([signal, steer]);
				let onSteer = (): void => {};
				const steered = new Promise<"steered">((resolve) => {
					onSteer = () => resolve("steered");
					steer.addEventListener("abort", onSteer, { once: true });
				});
				try {
					const outcome = await Promise.race([tool.execute(toolCallId, params, combined, onUpdate), steered]);
					return outcome === "steered" ? skippedBySteer(tool.name) : outcome;
				} catch (error) {
					// The steer's abort surfaced as the tool's own error: report the skip, not a failure.
					if (steer.aborted && signal?.aborted !== true) return skippedBySteer(tool.name);
					throw error;
				} finally {
					steer.removeEventListener("abort", onSteer);
				}
			},
		};
	}

	/**
	 * Resolve the model's inline `[eN]` citations against this run's evidence and derive the structured results.
	 * With Jev on, every cited chunk is first checked for relevance to the sentence that cites it and to what
	 * the user typed in this conversation; a chunk Jev finds unrelated (a table of contents, a footer) is dropped
	 * from the citations before the answer is returned or judged for memory.
	 */
	private async deriveAnswer(text: string): Promise<AutoRAGResultsDetails> {
		const options = { allowLocalFiles: true };
		let excludedChunks: ReadonlySet<string> | undefined;
		let removedSources: readonly string[] = [];
		if (this.jevJudge !== undefined) {
			const chunks = citedChunks(text, this.evidenceLedger, options);
			const relevance = await judgeCitationRelevance(this.jevJudge, {
				userMessages: this.conversationQueries.all(),
				chunks,
			});
			if (relevance.fallbackReason !== undefined) {
				this.routingDiagnostics.push({
					code: "citation-relevance-fallback",
					severity: "warning",
					message: `Jev citation relevance check was unavailable; every citation was kept. ${relevance.fallbackReason}`,
					source: "agent",
				});
			} else if (relevance.rejected.size > 0) {
				excludedChunks = relevance.rejected;
				removedSources = [
					...new Set(chunks.filter((chunk) => relevance.rejected.has(chunk.key)).map((chunk) => chunk.source)),
				];
			}
		}
		const derived = deriveResultsFromAnswer(text, this.evidenceLedger, {
			...options,
			numbers: this.citationNumbers,
			...(excludedChunks !== undefined ? { excludedChunks } : {}),
		});
		// The answer is derived after the turn's last session event: persist its new numbers and queries now.
		this.citationNumbers.flush();
		this.conversationQueries.flush();
		this.routingDiagnostics.push(...derived.diagnostics);
		if (excludedChunks !== undefined) {
			this.routingDiagnostics.push({
				code: "citation-irrelevant",
				severity: "info",
				message: `Jev found ${excludedChunks.size} cited chunk(s) unrelated to the sentence that cites them or to what the user asked (P(relevant) < ${CITATION_RELEVANCE_FLOOR}) and removed them from the citations: ${removedSources.join(", ")}.`,
				source: "agent",
			});
		}
		return derived.details;
	}

	/**
	 * Routes one TUI input the way pi routes typed input for its own turns. With
	 * no search running it starts one. During a search, Enter steers it (the
	 * user is correcting a search that is going wrong) and Alt+Enter queues a
	 * follow-up question that runs once this search answers.
	 */
	private routeInteractiveInput(input: AutoRAGInteractiveInput, pi: ExtensionAPI): AutoRAGInputDisposition {
		const state = this.interactive;
		const followUp = input.streamingBehavior === "followUp" || state.nextSubmitIsFollowUp;
		state.nextSubmitIsFollowUp = false;
		if (state.run === undefined) {
			this.userRequests.record(input.text);
			this.startInteractiveQuery(input.text, pi);
			return { kind: "handled" };
		}
		if (followUp) {
			this.userRequests.record(input.text);
			state.followUps.push(input.text);
			return { kind: "handled" };
		}
		if (input.streamingBehavior === "steer") {
			// A turn is streaming: pi queues the user's text as steering for that turn.
			state.steers.push(input.text);
			this.userRequests.record(input.text, "steer");
			state.steerTexts.add(input.text);
			this.conversationQueries.add(input.text);
			// Cut any in-flight retrieval short so the steer reaches the model promptly.
			state.steerInterrupt.abort();
			return { kind: "steer", text: input.text };
		}
		// Between phases no turn streams, so pi would start a new prompt. Queue the
		// steer on the search instead: the next phase's model request takes it. A
		// search past its last request (or between queued queries) answers it next.
		if (!this.steer(input.text, input.images)) {
			this.userRequests.record(input.text);
			state.followUps.push(input.text);
		}
		return { kind: "handled" };
	}

	/**
	 * Esc and Alt+Enter during a search. pi handles both itself while a turn
	 * streams (abort, queue follow-up); between phases it ignores Esc and submits
	 * Alt+Enter like Enter, so AutoRAG acts on them there.
	 */
	private handleInteractiveKey(key: AutoRAGInteractiveKey): boolean {
		const state = this.interactive;
		if (state.run === undefined || this.boundPiRuntime?.session.isStreaming === true) return false;
		if (key === "followUp") {
			// pi still submits the editor text; the next input is routed as a follow-up.
			state.nextSubmitIsFollowUp = true;
			return false;
		}
		state.userAborted = true;
		state.steerInterrupt.abort();
		return true;
	}

	/** Runs one TUI query detached from pi's input loop, then any follow-ups queued behind it. */
	private startInteractiveQuery(query: string, pi: ExtensionAPI): void {
		const state = this.interactive;
		const run = (async () => {
			let next: string | undefined = query;
			while (next !== undefined) {
				try {
					await this.runInteractivePiQuery(next, pi);
				} catch (error) {
					pi.sendMessage(
						{
							customType: "autorag.error",
							content: [{ type: "text", text: error instanceof Error ? error.message : String(error) }],
							display: true,
						},
						{ triggerTurn: false },
					);
				}
				next = state.followUps.shift();
			}
		})().finally(() => {
			state.run = undefined;
		});
		state.run = run;
	}

	/**
	 * Drives one TUI query through the two-phase search. The model's own
	 * assistant messages (progress notes before a tool call, the fast answer,
	 * the final answer) are already recorded and rendered by the bound pi
	 * session, so nothing is re-sent here: a second copy would show every answer
	 * twice and put it in the model's context twice.
	 */
	private async runInteractivePiQuery(query: string, pi: ExtensionAPI): Promise<void> {
		pi.setSessionName(query.slice(0, 80));
		const state = this.interactive;
		const session = this.boundPiRuntime?.session;
		const unsubscribe = session?.subscribe((event) => {
			// Alt+Up (dequeue) and Esc clear the agent's queues, including steers AutoRAG
			// queued between phases: those are gone now, not owed a later answer.
			if (event.type === "queue_update" && state.queuedSteers.length > 0 && !session.agent.hasQueuedMessages()) {
				state.queuedSteers.length = 0;
			}
			// An Esc that aborted a phase ends the search: no later phase may run as if it had finished.
			if (event.type === "agent_settled" && event.aborted && !state.harnessAbort) state.userAborted = true;
		});
		try {
			for await (const _event of this.searchDocumentsStream(query)) {
				// Drained for its side effects: the search turn runs while it is consumed.
			}
		} finally {
			unsubscribe?.();
			// Steers no model request took would land in the next prompt as if typed
			// there; the user still wants them answered, so they run as the next queries.
			if (state.queuedSteers.length > 0) {
				session?.agent.clearSteeringQueue();
				if (!state.userAborted) state.followUps.unshift(...state.queuedSteers);
				state.queuedSteers.length = 0;
			}
		}
	}

	/** Resolves once the TUI search and every follow-up queued behind it finished. */
	async waitForIdle(): Promise<void> {
		while (this.interactive.run !== undefined) await this.interactive.run;
	}

	/**
	 * Steers the running `searchDocuments()` / `searchDocumentsStream()` search,
	 * the way Enter steers a TUI search: `text` is a correction of the search in
	 * progress, not a new question. It reaches the search's next model request
	 * (also between phases, where no turn streams), cuts in-flight retrieval and
	 * the baseline prefetch short so it lands promptly, and forces the
	 * verification phase, whose prompt restates it. A steer that lands after
	 * verification's last request gets one more verification turn in the same
	 * search. The final answer stays the answer to the original question.
	 *
	 * Returns false, and does nothing, when no search is running, the search was
	 * stopped, or it is past its last model request; the caller should then ask
	 * the text as a new question.
	 */
	steer(text: string, images: readonly ImageContent[] = []): boolean {
		const state = this.interactive;
		if (!this.activeRun || !state.acceptingSteers || state.userAborted || text.trim() === "") return false;
		state.steers.push(text);
		this.userRequests.record(text, "steer");
		state.steerTexts.add(text);
		this.conversationQueries.add(text);
		// Cut any in-flight retrieval short so the steer reaches the model promptly.
		state.steerInterrupt.abort();
		const content: (TextContent | ImageContent)[] = [{ type: "text", text }, ...images];
		const message: AgentMessage = { role: "user", content, timestamp: Date.now() };
		const agent = this.activeSession?.agent;
		if (agent === undefined) state.preSessionSteers.push(message);
		else agent.steer(message);
		state.queuedSteers.push(text);
		return true;
	}

	/**
	 * Stops the running search as a whole: the streaming model turn, the
	 * baseline prefetch, in-flight retrieval, and every later phase. The search
	 * then resolves without a verified answer.
	 */
	abort(): void {
		if (!this.activeRun) return;
		const state = this.interactive;
		state.userAborted = true;
		state.steerInterrupt.abort();
		this.activeRunAbort?.abort();
		void this.activeSession?.abort();
	}

	/** The next model request took the queued steers into the transcript; later retrieval runs uninterrupted until the next steer. */
	private steersReachedModel(): void {
		const state = this.interactive;
		state.queuedSteers.length = 0;
		if (state.steerInterrupt.signal.aborted && !state.userAborted) state.steerInterrupt = new AbortController();
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

	async searchDocuments(query: string, options: RetrievalOptions = {}): Promise<SearchDocumentsResponse> {
		if (this.activeRun) {
			throw new Error("AutoRAG agent is busy; await the in-flight searchDocuments() call before starting another");
		}

		// MinSync is required: a run that cannot reach it fails here, before any
		// model call, instead of answering from the retrieval paths that remain.
		if (this.minSyncMethod.isBinaryMissing()) throw minSyncBinaryMissingError();

		const sessionId = randomUUID();
		const trimmedQuery = query.trim();
		if (trimmedQuery.length === 0) {
			this.lastQuery = trimmedQuery;
			this.lastSessionId = sessionId;
			return createEmptySearchDocumentsResponse(sessionId, trimmedQuery, this.startupDiagnostics);
		}
		options = this.normalizeRetrievalOptions(options);
		this.activeRetrievalOptions = options;

		// A later turn of an interactive conversation is a plain agent turn: the
		// user's message reaches the model as typed, and the model answers from
		// the conversation or searches by tool call. Only the first turn runs the
		// Jev gate and the baseline pipeline.
		const conversational = this.isConversationTurn();
		this.activeRun = true;
		this.searchProgress.begin(trimmedQuery);
		this.toolBurstGuard.reset();
		this.retrievalTrace = [];
		// A conversation keeps one evidence ledger for its whole life: the transcript still shows the earlier
		// [eN] ids, and the session rebuilds the ledger on resume, /new, and tree moves. Only a run that owns
		// no transcript starts over, and result numbers follow the same rule: `[n]` accumulates with the conversation.
		if (this.boundPiRuntime === undefined) {
			this.evidenceLedger.clear();
			this.citationNumbers.reset();
			this.conversationQueries.reset();
		}
		if (!conversational) this.evidenceOrigins.clear();
		this.searchToolQueries.clear();
		this.memoryContextCache = undefined;
		this.routingDiagnostics = [];
		this.lastQuery = trimmedQuery;
		// A run that owns no TUI session is its own conversation. In the TUI the log is the session's and is fed
		// as the user types, so a follow-up or a steer is never missing and a phase prompt is never in it.
		if (this.boundPiRuntime === undefined) {
			this.userRequests.reset();
			this.userRequests.record(trimmedQuery);
		}
		this.conversationQueries.add(trimmedQuery);
		this.lastSessionId = sessionId;
		let captured: AutoRAGResultsDetails | undefined;
		let fastCaptured: AutoRAGResultsDetails | undefined;
		/** True when this run took a Jev branch whose report is not retrieved evidence (`config`, `organize`, or an `extra` turn that cited nothing); it never feeds retrieval memory. */
		let isolatedRun = false;
		/**
		 * Without Jev every fast answer goes on to verification, so it is
		 * published as soon as it is written. With Jev, publishing waits for
		 * the follow-up check: an answer that turns out to be final reaches the
		 * caller once, as the complete response.
		 */
		const publishOnCapture = this.jevJudge === undefined;
		let published = false;
		// An abort() cancels this controller. A slow follow-up check or a
		// late tool call can outlive the run, so publishing is gated on it:
		// an aborted run must never push a preliminary into the callback a later
		// run already installed on the instance.
		const planAbort = new AbortController();
		this.activeRunAbort = planAbort;
		// Steering is per search: a steer or stop aimed at an earlier query never reaches this one.
		const steering = this.interactive;
		steering.steers.length = 0;
		steering.queuedSteers.length = 0;
		steering.preSessionSteers.length = 0;
		steering.harnessAbort = false;
		steering.userAborted = false;
		steering.steerInterrupt = new AbortController();
		steering.acceptingSteers = true;
		const publishPreliminary = (details: AutoRAGResultsDetails): void => {
			if (planAbort.signal.aborted || published) return;
			published = true;
			this.preliminaryCallback?.(
				createPreliminarySearchDocumentsResponse(
					sessionId,
					trimmedQuery,
					details,
					this.collectComponentDiagnostics(),
				),
			);
		};
		const emitPreliminary = (details: AutoRAGResultsDetails): void => {
			if (fastCaptured !== undefined) return;
			fastCaptured = details;
			if (publishOnCapture) publishPreliminary(details);
		};
		let session: AutoRAGSearchSession | undefined;
		let unsubscribers: readonly (() => void)[] = [];
		let searchStarted = false;
		try {
			const resolved = await this.resolveSessionModel();
			// Model-native web search rides on the same model credential the
			// agent loop uses — no separate search key (see web/search/model-auth).
			// It lives on the instance, never in module state, so a second agent
			// in the same process cannot take over this agent's credential.
			this.modelNativeSearchAuth = modelNativeAuthFromAgentModel({
				provider: resolved.model.provider,
				...(resolved.apiKey !== undefined ? { apiKey: resolved.apiKey } : {}),
				...(resolved.model.baseUrl !== undefined ? { baseUrl: resolved.model.baseUrl } : {}),
				modelId: resolved.model.id,
			});
			this.runLogger.write({
				event: "search_started",
				timestamp: new Date().toISOString(),
				sessionId,
				queryLength: trimmedQuery.length,
				model: resolved.model.id,
			});
			searchStarted = true;
			session = await this.createSearchSession(
				resolved,
				buildSystemPrompt(this.currentSystemPromptConfig({ modelId: resolved.model.id })),
			);
			this.activeSession = session;
			unsubscribers = this.configureSearchSession(session);
			for (const message of steering.preSessionSteers) session.agent.steer(message);
			steering.preSessionSteers.length = 0;
			await (async () => {
				// Two-phase flow: a fast minimal-thinking answer from the baseline
				// evidence, then a thinking-on verification pass. Each phase ends in
				// plain assistant text; the harness resolves its inline [eN]
				// citations against the evidence ledger.
				// With Jev enabled, Jev first picks the branch that handles the
				// question — local search, file organization, an extra task, or its
				// own configuration — and whether the question needs decomposition.
				const activeSession = session;
				const sessionAgent = activeSession.piSession;
				const transcript = (): readonly AgentMessage[] =>
					sessionAgent?.messages ?? activeSession.agent.state.messages;
				const setThinking = (level: AutoRAGThinkingLevel): void => {
					const clamped = clampThinkingLevel(resolved.model, level);
					if (sessionAgent !== undefined) sessionAgent.setThinkingLevel(clamped);
					else activeSession.agent.state.thinkingLevel = clamped;
				};
				if (conversational) {
					setThinking(this.finalThinkingLevel);
					await this.searchProgress.step("answer", "Answering", () => activeSession.prompt(trimmedQuery));
					// A steer typed after the turn's last model request is still owed an answer.
					while (
						!this.interactive.userAborted &&
						!planAbort.signal.aborted &&
						this.interactive.queuedSteers.length > 0
					) {
						await activeSession.prompt(buildSteerContinuationPrompt(this.interactive.steers));
					}
					steering.acceptingSteers = false;
					if (this.interactive.userAborted) return;
					const turnText = finalAnswerText(transcript());
					if (turnText !== undefined) captured = await this.deriveAnswer(turnText);
					return;
				}
				const plan = await this.planQuery(trimmedQuery, resolved, planAbort.signal);
				if (plan.route === "config" && plan.selfConfigSkill !== undefined && this.selfConfig !== undefined) {
					// Self-configuration: no retrieval and no fast answer. The model
					// gets the full setup skill and edits the config itself, then
					// reports what it changed as its final message. No route changes
					// the active tools, so the loaded skill and the turn prompt scope
					// the work.
					isolatedRun = true;
					// pi's bash tool refuses to run from a missing cwd, and a freshly
					// initialised config has not created its workspace yet.
					mkdirSync(this.workspaceProjectRoot, { recursive: true });
					setThinking(this.finalThinkingLevel);
					const configBefore = snapshotConfigFile(this.selfConfig.configPath);
					await activeSession.prompt(
						buildSelfConfigPrompt({
							query: trimmedQuery,
							configPath: this.selfConfig.configPath,
							agentDir: this.piAgentDir ?? getAgentDir(),
							skill: plan.selfConfigSkill,
							automations: true,
						}),
					);
					const report = finalAnswerText(transcript());
					if (report !== undefined) captured = await this.deriveAnswer(report);
					const rolledBack = await rollbackIfBroken(this.selfConfig, configBefore);
					if (rolledBack !== undefined) {
						this.routingDiagnostics.push({
							code: "self-config-rolled-back",
							severity: "warning",
							message: `The edited config did not validate, so the previous config was restored. ${rolledBack}`,
							source: "self-config",
						});
						if (captured !== undefined) {
							captured = {
								...captured,
								answer: `${captured.answer}\n\nWarning: the edited config did not validate, so the previous config was restored (nothing was changed). Problem: ${rolledBack}`,
							};
						}
					}
					return;
				}
				if (plan.route === "organize" && plan.organizeSkill !== undefined) {
					// Acting on files: no retrieval, no fast answer, no verification.
					// The loaded organize skill carries the rules and the turn prompt
					// names the tools the task uses; both replace the tool swap this
					// branch used to make.
					isolatedRun = true;
					setThinking(this.finalThinkingLevel);
					const organizeReport = await this.runActionTurn(
						activeSession,
						planAbort.signal,
						{
							id: "organize",
							label: "Organizing files and folders",
							prompt: this.buildOrganizePrompt(trimmedQuery, plan.organizeSkill),
						},
						transcript,
					);
					if (organizeReport !== undefined) captured = await this.deriveAnswer(organizeReport);
					return;
				}
				if (plan.route === "extra") {
					// Any other request that needs no search first: no retrieval, no
					// fast answer, no verification. One tool-capable turn with every
					// tool the session has answers or does the work, and its plain
					// final message is the answer or the report.
					isolatedRun = true;
					setThinking(this.finalThinkingLevel);
					const extraReport = await this.runActionTurn(
						activeSession,
						planAbort.signal,
						{ id: "extra", label: "Carrying out the request", prompt: this.buildExtraPrompt(trimmedQuery) },
						transcript,
					);
					if (extraReport !== undefined) captured = await this.deriveAnswer(extraReport);
					// An `extra` turn that looked facts up with a search, datasource, or
					// web tool and cited them did retrieve evidence, so it feeds
					// retrieval memory exactly like a local run; a pure-knowledge answer
					// or an action report that cites nothing stays isolated.
					isolatedRun = captured === undefined || captured.mapping.length === 0;
					return;
				}
				const steered = (): boolean => this.interactive.steers.length > 0;
				// Esc aborted the phase that just ran: the user stopped the search.
				const stopped = (): boolean => this.interactive.userAborted;
				if (stopped()) return;
				// Only `local` reaches here: `config`, `organize`, and `extra` ended
				// the run in their own single turn above.
				const baseline = await this.prefetchInitialRetrievalContext(
					trimmedQuery,
					plan.queries,
					options,
					plan.datasources,
					plan.web,
				);
				if (stopped()) return;
				let fastText: string | undefined;
				setThinking(this.fastThinkingLevel);
				await this.searchProgress.step("fast", "Drafting a first answer", () =>
					activeSession.prompt(this.buildFastAnswerPrompt(trimmedQuery, options, baseline)),
				);
				if (stopped()) return;
				fastText = finalAnswerText(transcript());
				const preliminary = fastText === undefined ? undefined : await this.deriveAnswer(fastText);
				if (preliminary !== undefined) {
					emitPreliminary(preliminary);
					// With Jev enabled, a fast answer that needs no correction,
					// clarification, or further research ends the run here. A steer
					// means the search is going wrong, so it always verifies.
					// Without Jev there is no check to show: `shouldFollowUp` answers true at once.
					const checkFollowUp = (): Promise<boolean> => this.shouldFollowUp(trimmedQuery, preliminary);
					const followUp =
						steered() ||
						(this.jevJudge === undefined
							? await checkFollowUp()
							: await this.searchProgress.step(
									"followup",
									"Checking whether the first answer needs verification",
									checkFollowUp,
									(needed) => (needed ? "verifying" : "final"),
								));
					// The follow-up check is not tied to the session abort, so it
					// can resolve after the run's timeout fired; stop here rather
					// than publishing or prompting on behalf of a dead run.
					if (planAbort.signal.aborted || stopped()) return;
					if (!followUp && !steered()) {
						captured = preliminary;
						return;
					}
					publishPreliminary(preliminary);
				}
				setThinking(this.finalThinkingLevel);
				// Only a preliminary consumer actually received may turn the final answer into a delta.
				const fastAnswerDelivered = preliminary !== undefined && this.preliminaryCallback !== undefined;
				// The model gets its own first answer verbatim, with the [eN] ids it
				// wrote: the caller's copy was renumbered to [n], and the prompt tells
				// the model its [eN] ids are still valid.
				await this.searchProgress.step("verify", "Verifying the answer against the sources", () =>
					activeSession.prompt(
						this.buildRefinementPrompt(
							trimmedQuery,
							options,
							fastText,
							fastAnswerDelivered,
							this.interactive.steers,
						),
					),
				);
				// A steer typed after verification's last model request is still owed an
				// answer within this search: pi holds it queued, so verify once more.
				while (!stopped() && !planAbort.signal.aborted && this.interactive.queuedSteers.length > 0) {
					await activeSession.prompt(buildSteerContinuationPrompt(this.interactive.steers));
				}
				// Synchronous with the check above: a later steer is refused, not left unanswered.
				steering.acceptingSteers = false;
				if (stopped()) return;
				const finalText = finalAnswerText(transcript());
				if (finalText !== undefined) captured = await this.deriveAnswer(finalText);
			})().finally(() => {
				// Every early return decides synchronously after its last steer check.
				steering.acceptingSteers = false;
				// The answer is complete; judging its evidence for memory is housekeeping, not search.
				this.searchProgress.end();
			});

			if (captured === undefined) {
				// Local sessions resolve with a degraded response that carries the
				// run's retrieval trace instead of throwing away the whole run.
				const reason = lastAssistantText(session?.agent.state.messages ?? []);
				const modelError = lastModelRequestError(session?.agent.state.messages ?? []);
				const response: SearchDocumentsResponse = {
					sessionId,
					query: trimmedQuery,
					results: [],
					answer: buildMissingFinalAnswer(trimmedQuery, reason, this.retrievalTrace, modelError),
					searched: this.retrievalTrace.reduce((total, entry) => total + entry.resultCount, 0),
					warnings: [],
					diagnostics: [
						...this.collectComponentDiagnostics(),
						{
							code: "no-final-answer",
							severity: "warning",
							message:
								"The agent ended its run without a final answer message; returning a degraded response that carries the run's retrieval trace.",
						},
						...(modelError === undefined
							? []
							: [
									{
										code: "model-request-failed" as const,
										severity: "error" as const,
										message: `The model request failed: ${modelError}`,
									},
								]),
					],
					retrievalTrace: this.retrievalTrace,
				};
				this.runLogger.write({
					event: "search_completed",
					timestamp: new Date().toISOString(),
					sessionId,
					resultCount: 0,
					degraded: true,
				});
				return response;
			}
			const componentDiagnostics = this.collectComponentDiagnostics();
			const recorded = recordStructuredResultsSession(
				sessionId,
				trimmedQuery,
				captured,
				this.memory,
				componentDiagnostics,
				{ isolateMemory: isolatedRun },
			);
			// A config or organize report, or an extra turn that cited no evidence,
			// is not retrieved evidence.
			const memoryDiagnostic = isolatedRun
				? undefined
				: await this.recordJudgedEvidence(sessionId, trimmedQuery, captured);
			const response: SearchDocumentsResponse =
				memoryDiagnostic === undefined
					? recorded
					: { ...recorded, diagnostics: [...(recorded.diagnostics ?? []), memoryDiagnostic] };
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
			const cleanupActions: readonly (() => void)[] = [...unsubscribers, () => session?.dispose()];
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
			this.searchProgress.end();
			this.activeSession = undefined;
			this.activeRetrievalOptions = undefined;
			this.preliminaryCallback = undefined;
			this.activeRun = false;
			this.memoryContextCache = undefined;
		}
	}

	/**
	 * Stream bounded progress updates while retaining the stable
	 * {@link searchDocuments} promise API. Progress is the text an assistant
	 * message wrote before calling a tool (`stopReason: "toolUse"`); a message
	 * that ends its turn is an answer, delivered as `preliminary` or
	 * `complete`, never as progress. Callers can render it in a TUI, CLI, or
	 * parent agent and can stop by calling {@link abort}.
	 */
	async *searchDocumentsStream(
		query: string,
		options: RetrievalOptions = {},
	): AsyncGenerator<SearchDocumentsStreamEvent, void, void> {
		const queue: SearchDocumentsStreamEvent[] = [];
		let wake: (() => void) | undefined;
		let settled = false;
		let lastIntent: string | undefined;
		const push = (text: string): void => {
			queue.push({ type: "progress", sessionId: this.lastSessionId ?? "", query: query.trim(), text });
			wake?.();
			wake = undefined;
		};
		const unsubscribe = this.subscribe((event) => {
			// The model says what each tool call is for (intent tracing); the newest one is the progress.
			if (event.type === "message_update" || event.type === "message_end") {
				const intent = intentOfMessage(event.message);
				if (intent !== undefined && intent !== lastIntent) {
					lastIntent = intent;
					push(intent);
					return;
				}
			}
			if (event.type !== "message_end" || event.message.role !== "assistant") return;
			if (event.message.stopReason !== "toolUse") return;
			// A message whose tool call carries an intent has reported it above.
			if (intentOfMessage(event.message) !== undefined) return;
			const text = event.message.content
				.filter((block) => block.type === "text")
				.map((block) => block.text)
				.join("")
				.trim();
			if (text === "") return;
			push(text);
		});
		queue.push({
			type: "progress",
			sessionId: "",
			query: query.trim(),
			text: "Reviewing the query.",
		});
		this.preliminaryCallback = (response) => {
			queue.push({ type: "preliminary", response });
			wake?.();
			wake = undefined;
		};
		const run = this.searchDocuments(query, options)
			.then((response) => {
				queue.push({ type: "complete", response });
			})
			.catch((error) => {
				settled = true;
				wake?.();
				throw error;
			})
			.finally(() => {
				settled = true;
				wake?.();
			});
		try {
			while (!settled || queue.length > 0) {
				if (queue.length === 0) {
					await new Promise<void>((resolve) => {
						wake = resolve;
					});
					continue;
				}
				yield queue.shift() as SearchDocumentsStreamEvent;
			}
			await run;
		} finally {
			unsubscribe();
			await run.catch(() => undefined);
		}
	}

	/**
	 * Build the Pi agent-skill list for every configured datasource skill. All
	 * configured skills are model-visible; a connection is removed from the
	 * model surface by removing it from the config, not by a permission layer.
	 */
	private buildDatasourceAgentSkills(): DatasourceAgentSkill[] {
		const skills: DatasourceAgentSkill[] = [];
		for (const skill of this.datasourceSkills) {
			skills.push(toDatasourceAgentSkill(skill.skillManifest()));
		}
		return skills;
	}

	/**
	 * Per-connection tool specs for the generated `search_datasource_<id>`
	 * tools, one per configured datasource skill; duplicate ids collapse to the
	 * first registration.
	 */
	private singleDatasourceToolSpecs(): SingleDatasourceToolSpec[] {
		const seen = new Set<string>();
		const specs: SingleDatasourceToolSpec[] = [];
		for (const skill of this.datasourceSkills) {
			const descriptor = skill.describe();
			if (descriptor.datasourceId === undefined) continue;
			if (seen.has(descriptor.datasourceId)) continue;
			seen.add(descriptor.datasourceId);
			// Instance roots are two-segment sources like /kakao/personal; deeper
			// hierarchy entries would only bloat the tool description.
			const instanceScopes = skill
				.describeSources()
				.map((source) => source.source)
				.filter((source) => source.split("/").filter((segment) => segment.length > 0).length === 2)
				.slice(0, this.limits.toolDescriptionInstanceScopes);
			specs.push({
				datasourceId: descriptor.datasourceId,
				description: descriptor.description,
				instanceScopes,
			});
		}
		return specs;
	}

	/**
	 * Configured datasource descriptors for catalog/listing surfaces: every
	 * configured datasource is listed — including ones that expose no retrieval
	 * methods — and each entry carries only identity, capability tags, and
	 * source scope strings (never credentials, config paths, or raw instance
	 * metadata). Duplicate datasource ids collapse to the first registration.
	 */
	listDatasources(): DatasourceCatalogEntry[] {
		const seen = new Set<string>();
		const entries: DatasourceCatalogEntry[] = [];
		for (const skill of this.datasourceSkills) {
			const descriptor = skill.describe();
			if (descriptor.datasourceId === undefined) continue;
			if (seen.has(descriptor.datasourceId)) continue;
			seen.add(descriptor.datasourceId);
			entries.push({
				datasourceId: descriptor.datasourceId,
				name: descriptor.name,
				type: descriptor.type,
				description: descriptor.description,
				tags: [...descriptor.tags],
				capabilities: [...descriptor.capabilities],
				status: descriptor.status,
				sourceScopes: this.datasourceSourceScopes(skill),
			});
		}
		return entries;
	}

	/**
	 * Opaque source scope strings for one datasource. Sources containing a `#`
	 * fragment are invalid and skipped.
	 */
	private datasourceSourceScopes(skill: DatasourceSkill): string[] {
		const scopes = new Set<string>();
		for (const source of skill.describeSources()) {
			const scope = source.source;
			if (scope.includes("#")) continue;
			scopes.add(scope);
		}
		return [...scopes];
	}

	/**
	 * Resolve a configured datasource agent skill by model-visible name for the
	 * `load_datasource_skill` tool. Returns `undefined` for unknown names.
	 */
	loadDatasourceSkill(name: string): DatasourceAgentSkill | undefined {
		return this.datasourceAgentSkills.find((skill) => skill.name === name);
	}

	private async indexDatasources(
		signal?: AbortSignal,
		onProgress?: (progress: AutoRAGRefreshProgress) => void,
	): Promise<readonly DatasourceIndexResult[]> {
		const results: DatasourceIndexResult[] = [];
		const total = this.datasourceSkills.length;
		for (const [position, skill] of this.datasourceSkills.entries()) {
			signal?.throwIfAborted();
			onProgress?.({ phase: "datasources", processed: position, total });
			results.push(await indexWithRetry(skill, this.datasourceRetry, signal));
		}
		onProgress?.({ phase: "datasources", processed: total, total });
		return results;
	}

	/** Path-opaque component diagnostics for the search response. */
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
		for (const result of this.refreshState.datasources) {
			diagnostics.push(...mapDatasourceDiagnostics(result.diagnostics));
		}
		diagnostics.push(...this.routingDiagnostics);
		return diagnostics;
	}

	/**
	 * Jev check run after the fast answer: does the answer need correction,
	 * clarification, or further research? "No" ends the run with the fast
	 * answer as the final answer. Without Jev, or when the check fails, the run
	 * always continues into verification.
	 */
	private async shouldFollowUp(query: string, fastAnswer: AutoRAGResultsDetails): Promise<boolean> {
		if (this.jevJudge === undefined) return true;
		const decision = await needsFollowUp(this.jevJudge, query, fastAnswer.answer);
		if (decision.fallbackReason !== undefined) {
			this.routingDiagnostics.push({
				code: "follow-up-check-fallback",
				severity: "warning",
				message: `Jev follow-up check was unavailable; verifying the fast answer. ${decision.fallbackReason}`,
				source: "jev",
			});
			return true;
		}
		const probability = decision.probability?.toFixed(2) ?? "?";
		if (!decision.followUp) {
			this.routingDiagnostics.push({
				code: "follow-up-skipped",
				severity: "info",
				message: `Jev judged the fast answer final (P(follow-up)=${probability}); the verification phase was skipped.`,
				source: "jev",
			});
		}
		return decision.followUp;
	}

	/**
	 * Jev query pipeline, run before the fast answer. Jev picks the branch
	 * (local search, file organization, an extra task, AutoRAG configuration)
	 * and whether the question needs decomposition; a "yes" splits it into at
	 * most five search queries with the configured decomposition model
	 * (default: the session model). On the local branch, Jev also judges, per
	 * registered datasource, whether it must be searched (in parallel with
	 * decomposition). Without Jev, and on any routing failure, this is today's
	 * single local search for the original question with no datasource search.
	 * A non-local branch that misses the confidence floor is not a failure: it
	 * becomes a full local plan, decomposition and datasource check included.
	 */
	private async planQuery(
		query: string,
		resolved: {
			readonly model: Model<Api>;
			readonly apiKey?: string;
			readonly providerApiKeys?: Readonly<Record<string, string>>;
		},
		signal: AbortSignal,
	): Promise<{
		readonly route: QueryRoute;
		readonly queries: readonly string[];
		readonly datasources: readonly string[];
		/** True when Jev selected the public web, prefetched with the datasources. */
		readonly web: boolean;
		readonly selfConfigSkill?: PackagedAgentSkill;
		readonly organizeSkill?: PackagedAgentSkill;
	}> {
		const judge = this.jevJudge;
		if (judge === undefined) return { route: FALLBACK_QUERY_ROUTE, queries: [query], datasources: [], web: false };
		const decision = await this.searchProgress.step(
			"route",
			"Routing the question",
			() => routeQuery(judge, query, { selfConfig: this.selfConfig !== undefined }),
			(routed) => routed.route,
		);
		if (decision.fallbackReason !== undefined) {
			this.routingDiagnostics.push({
				code: "query-route-fallback",
				severity: "warning",
				message: `Jev query routing fell back to local search. ${decision.fallbackReason}`,
				source: "jev",
			});
			// The confidence floor keeps a real non-local verdict off the run but
			// the question is still answerable: run the full local plan. Only an
			// unavailable backend or an unusable verdict keeps the bare fallback.
			if (decision.floorApplied !== true) {
				return { route: FALLBACK_QUERY_ROUTE, queries: [query], datasources: [], web: false };
			}
		}
		let route = decision.route;
		let selfConfigSkill: PackagedAgentSkill | undefined;
		if (route === "config") {
			try {
				selfConfigSkill = loadPackagedSkill(SETUP_SKILL_NAME, this.selfConfig?.skillPath);
			} catch (error) {
				this.routingDiagnostics.push({
					code: "self-config-unavailable",
					severity: "warning",
					message: `Jev routed the question to AutoRAG configuration, but the setup skill could not be loaded; searching local sources instead. ${error instanceof Error ? error.message : String(error)}`,
					source: "self-config",
				});
				route = FALLBACK_QUERY_ROUTE;
			}
		}
		let organizeSkill: PackagedAgentSkill | undefined;
		if (route === "organize") {
			try {
				organizeSkill = loadPackagedSkill(ORGANIZE_SKILL_NAME, this.organizeSkillPath);
			} catch (error) {
				this.routingDiagnostics.push({
					code: "organize-skill-unavailable",
					severity: "warning",
					message: `Jev routed the question to file organization, but the organize skill could not be loaded; searching local sources instead. ${error instanceof Error ? error.message : String(error)}`,
					source: "organize",
				});
				route = FALLBACK_QUERY_ROUTE;
			}
		}
		const selectionPromise: Promise<SearchDatasourceChoice> =
			route === "local"
				? this.searchProgress.step(
						"datasource-select",
						"Choosing datasources to search",
						() => this.selectSearchDatasources(query),
						(chosen) => {
							const ids = [...chosen.datasources, ...(chosen.web ? [WEB_DATASOURCE_ID] : [])];
							return ids.length === 0 ? "none" : ids.join(", ");
						},
					)
				: Promise.resolve({ datasources: [], web: false });
		let queries: readonly string[] = [query];
		if (decision.decompose && route === "local") {
			const target = this.queryDecompositionModel ?? {
				model: resolved.model,
				...((resolved.apiKey ?? resolved.providerApiKeys?.[resolved.model.provider]) !== undefined
					? { apiKey: resolved.apiKey ?? resolved.providerApiKeys?.[resolved.model.provider] }
					: {}),
			};
			try {
				queries = await this.searchProgress.step(
					"decompose",
					"Splitting the question into search queries",
					() => decomposeQuery(createModelDecompositionCompleter(target, signal), query),
					(list) => `${list.length} ${list.length === 1 ? "query" : "queries"}`,
				);
			} catch (error) {
				this.routingDiagnostics.push({
					code: "query-decomposition-failed",
					severity: "warning",
					message: `Question decomposition failed; searching with the original question. ${error instanceof Error ? error.message : String(error)}`,
					source: "query-decomposition",
				});
			}
		}
		const selection = await selectionPromise;
		const probability = decision.routeProbability === undefined ? "" : ` (p=${decision.routeProbability.toFixed(2)})`;
		this.routingDiagnostics.push({
			code: "query-routed",
			severity: "info",
			message:
				`Jev routed the question to ${route}${probability}; ` +
				(route === "config"
					? "configuring AutoRAG with the full setup skill; no retrieval and no fast answer."
					: route === "organize"
						? "organizing files and folders with the organize skill loaded into the turn; every tool stays available, and no retrieval or verification runs."
						: route === "extra"
							? "carrying out the request with every tool; no retrieval and no verification."
							: `searching with ${queries.length} ${queries.length === 1 ? "query" : "queries"}: ${queries.map((entry) => JSON.stringify(entry)).join(", ")}.`),
			source: "jev",
		});
		return {
			route,
			queries,
			datasources: selection.datasources,
			web: selection.web,
			...(selfConfigSkill !== undefined ? { selfConfigSkill } : {}),
			...(organizeSkill !== undefined ? { organizeSkill } : {}),
		};
	}

	/**
	 * Jev datasource check for the local branch: one `noul` per configured
	 * datasource that has retrieval methods, plus the public web when its tools
	 * are enabled, with where similar past questions were answered (from
	 * retrieval memory) in the state. Returns the datasource ids and whether the
	 * web is to be searched with every query before the rerank; a failed check
	 * searches neither.
	 */
	private async selectSearchDatasources(query: string): Promise<SearchDatasourceChoice> {
		if (this.jevJudge === undefined) return { datasources: [], web: false };
		const searchable = new Set(
			this.methodRegistry
				.list()
				.map((method) => method.describe().datasourceId)
				.filter((datasourceId) => datasourceId !== undefined),
		);
		const catalog = this.listDatasources().filter((entry) => searchable.has(entry.datasourceId));
		const candidates: DatasourceCandidate[] = catalog.map(({ datasourceId, type, description }) => ({
			datasourceId,
			type,
			description,
		}));
		// Web tools off means the web is not a candidate at all: no question, no
		// web-attributed past hints, no web search.
		const webEnabled = this.webSearchOptions !== undefined;
		if (candidates.length === 0 && !webEnabled) return { datasources: [], web: false };
		const whereFound = this.evidenceAttribution(catalog);
		// A past result is shown only when its evidence is attributed to
		// something this run searches (a configured datasource, a configured
		// search path, or the web): its title is model-written text about that
		// evidence and leaves the machine. The memory context the run already
		// loaded is reused, so similar questions are searched once.
		const context = await this.loadMemoryContextFor(query);
		const pastSearches: PastSearchHint[] = [];
		for (const past of context.related) {
			const results: { title: string; foundIn: string[] }[] = [];
			for (const record of past.records) {
				const place = whereFound(record);
				if (place === undefined) continue;
				const existing = results.find((result) => result.title === record.title);
				if (existing === undefined) results.push({ title: record.title, foundIn: [place] });
				else if (!existing.foundIn.includes(place)) existing.foundIn.push(place);
			}
			if (results.length === 0) continue;
			pastSearches.push({ query: past.question, results: results.slice(0, SIMILAR_PAST_RESULT_LIMIT) });
			if (pastSearches.length === SIMILAR_PAST_SEARCH_LIMIT) break;
		}
		const selection = await selectDatasources(
			this.jevJudge,
			query,
			candidates,
			pastSearches,
			webEnabled ? { web: true } : {},
		);
		if (selection.fallbackReason !== undefined) {
			this.routingDiagnostics.push({
				code: "datasource-selection-fallback",
				severity: "warning",
				message: `Jev datasource check was unavailable; ${webEnabled ? "no datasource and no web search ran" : "no datasource was searched"} before the fast answer. ${selection.fallbackReason}`,
				source: "jev",
			});
			return { datasources: [], web: false };
		}
		// A configured datasource under the reserved id keeps it: the web
		// candidate was withheld, so the reserved id names that datasource.
		const webOffered = webEnabled && selection.webIdTaken !== true;
		if (selection.webIdTaken === true) {
			this.routingDiagnostics.push({
				code: "web-datasource-id-taken",
				severity: "warning",
				message: `The public web was not offered to Jev: a configured datasource already uses the reserved id "${WEB_DATASOURCE_ID}", so that question id would name two candidates. The configured datasource was judged as usual.`,
				source: "jev",
			});
		}
		const describe = (ids: readonly string[]): string =>
			ids.length === 0
				? "none"
				: ids.map((id) => `${id} (p=${selection.probabilities[id]?.toFixed(2) ?? "?"})`).join(", ");
		const judged = [...candidates.map((entry) => entry.datasourceId), ...(webOffered ? [WEB_DATASOURCE_ID] : [])];
		const skipped = judged.filter((id) => !selection.selected.includes(id));
		this.routingDiagnostics.push({
			code: "datasources-selected",
			severity: "info",
			message: `Jev selected datasources to search before the fast answer: ${describe(selection.selected)}; skipped: ${describe(skipped)}.`,
			source: "jev",
		});
		const datasources = webOffered ? selection.selected.filter((id) => id !== WEB_DATASOURCE_ID) : selection.selected;
		return { datasources, web: webOffered && selection.selected.includes(WEB_DATASOURCE_ID) };
	}

	/**
	 * Baseline local evidence for the fast answer. Jikji, MinSync, every
	 * Jev-selected datasource, and (when Jev selected it) one web search run for
	 * every search query in parallel (MinSync itself queues per workspace), the
	 * per-query hits are interleaved and deduplicated into one pool, and that
	 * pool is reranked against the original question.
	 */
	private async prefetchInitialRetrievalContext(
		query: string,
		searchQueries: readonly string[],
		options: RetrievalOptions,
		datasources: readonly string[] = [],
		web = false,
	): Promise<string> {
		// A steer typed during the prefetch stops it: the fast phase then starts with
		// whatever already returned plus the steer, instead of waiting on a slow source.
		const signal = this.interactive.steerInterrupt.signal;
		const retrieveOptions = { topK: this.limits.prefetch.minSyncTopK, scope: options.scope, signal };
		// Queries only read the prebuilt MinSync index; an unbuilt workspace makes
		// `minsync query` fail fast and this source contributes nothing.
		const formatSearchQueries = (list: readonly string[]): string =>
			`Search queries (decomposed from the original question):\n${list.map((entry, index) => `[${index + 1}] ${entry}`).join("\n")}`;
		const perQuery = await Promise.all(
			searchQueries.map((searchQuery) =>
				Promise.all([
					this.jikjiClient === undefined
						? Promise.resolve(undefined)
						: this.searchProgress
								.track(
									"jikji",
									"Jikji file discovery",
									this.findJikji(searchQuery, { topK: this.limits.prefetch.jikjiTopK, signal }),
									{
										count: (found) => found.answerPack?.answerPaths.length ?? 0,
									},
								)
								.catch(() => undefined),
					this.searchProgress
						.track(
							"minsync",
							"MinSync semantic search",
							this.minSyncMethod.retrieve(searchQuery, retrieveOptions),
							{
								count: (found) => found.length,
							},
						)
						.catch((error: unknown) => {
							// MinSync is required: its absence fails the run, anything else is one empty source.
							if (error instanceof MinSyncRequiredError) throw error;
							return [];
						}),
					Promise.all(
						datasources.map((datasourceId) =>
							this.searchProgress
								.track(
									`datasource:${datasourceId}`,
									`Datasource ${datasourceId}`,
									this.searchSingleDatasourceDocuments(datasourceId, searchQuery, {
										topK: this.limits.singleDatasourceTopK,
										scope: options.scope,
										signal,
									}),
									{ count: (outcome) => outcome.results.length },
								)
								.then((outcome) => outcome.results)
								.catch((): RetrievalResult[] => []),
						),
					),
					web
						? this.searchProgress.track("web", "Web search", this.prefetchWebSearch(searchQuery, signal), {
								count: (outcome) => outcome.results.length,
							})
						: Promise.resolve<PrefetchedWebSearch>({ results: [] }),
				]),
			),
		);
		const webFailures: string[] = [];
		for (const [index, [jikji, vector, perDatasource, webSearch]] of perQuery.entries()) {
			const searchQuery = searchQueries[index];
			if (searchQuery === undefined) continue;
			if (webSearch.error !== undefined) webFailures.push(`${JSON.stringify(searchQuery)}: ${webSearch.error}`);
			for (const path of jikji?.answerPack?.answerPaths ?? []) {
				this.evidenceOrigins.add({ query: searchQuery, method: JIKJI_FIND_TOOL_NAME, source: path, content: "" });
			}
			const labelled = [
				...vector.map((result) => ({ result, fallback: SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME })),
				...perDatasource.flatMap((results, datasourceIndex) =>
					results.map((result) => ({
						result,
						fallback: singleDatasourceToolName(datasources[datasourceIndex] ?? ""),
					})),
				),
				// Web hits are remembered under the web tool's method, so cited
				// evidence from a prefetched search is attributed to the web.
				...webSearch.results.map((result) => ({
					result,
					fallback: WEB_SEARCH_TOOL_NAME,
				})),
			];
			for (const { result, fallback } of labelled) {
				const method = typeof result.metadata.method === "string" ? result.metadata.method : fallback;
				this.evidenceOrigins.add({ query: searchQuery, method, source: result.source, content: result.content });
			}
		}
		if (webFailures.length > 0) {
			this.routingDiagnostics.push({
				code: "web-search-failed",
				severity: "warning",
				message:
					`The web search before the fast answer failed for ${webFailures.length} of ${searchQueries.length} ` +
					`${searchQueries.length === 1 ? "query" : "queries"}: ${webFailures.join("; ")}. ` +
					`The run continued without that web evidence.`,
				source: "web",
			});
		}
		const interleave = <T>(lists: readonly (readonly T[])[]): T[] => {
			const merged: T[] = [];
			const longest = Math.max(0, ...lists.map((list) => list.length));
			for (let rank = 0; rank < longest; rank++) {
				for (const list of lists) {
					const item = list[rank];
					if (item !== undefined) merged.push(item);
				}
			}
			return merged;
		};
		const jikjiFound = perQuery.some(([jikji]) => jikji?.answerPack !== undefined);
		const jikjiPaths = [...new Set(interleave(perQuery.map(([jikji]) => jikji?.answerPack?.answerPaths ?? [])))];
		const seenChunks = new Set<string>();
		const unseen = (result: RetrievalResult): boolean => {
			const key = `${result.source}\0${result.content}`;
			if (seenChunks.has(key)) return false;
			seenChunks.add(key);
			return true;
		};
		const minSyncResults = interleave(perQuery.map(([, vector]) => vector)).filter(unseen);
		const datasourceResults = interleave(perQuery.map(([, , perDatasource]) => interleave(perDatasource))).filter(
			unseen,
		);
		const webResults = interleave(perQuery.map(([, , , webSearch]) => webSearch.results)).filter(unseen);
		// Rerank the whole merged pre-fast-answer pool (every query's Jikji paths,
		// MinSync chunks, selected-datasource chunks, and web hits) against the
		// original question, so decomposed sub-query hits compete on relevance to
		// what the user asked. A configured reranker that fails is reported like a
		// model error and the baseline falls back to retrieval order cut to
		// `rerank.topN`; with reranking disabled the sections stay uncut.
		let reranked: RetrievalResult[] | undefined;
		let rerankFailed = false;
		try {
			reranked = await this.searchProgress.step(
				"rerank",
				"Ranking the candidates",
				() => this.rerankPrefetchPool(query, jikjiPaths, minSyncResults, datasourceResults, webResults, signal),
				(ranked) => (ranked === undefined ? undefined : `${ranked.length} kept`),
			);
		} catch (error) {
			rerankFailed = true;
			// A steer abort cuts the request short; that is not a reranker failure.
			if (!signal.aborted) this.reportRerankFailure(error);
		}
		if (reranked !== undefined) {
			const baseline = formatRerankedBaseline(
				reranked,
				reranked.map((result) => this.evidenceLedger.registerResult("baseline", result)),
			);
			return searchQueries.length > 1 ? `${formatSearchQueries(searchQueries)}\n\n${baseline}` : baseline;
		}

		// Evidence ids are issued once per candidate and never restart, so a
		// candidate id never means two different candidates.
		const sections: string[] = [];
		if (searchQueries.length > 1) sections.push(formatSearchQueries(searchQueries));
		const idFor = (result: RetrievalResult): string => this.evidenceLedger.registerResult("baseline", result);
		let shownPaths = jikjiPaths.slice(0, this.limits.prefetch.jikjiPathLimit);
		let shownMinSync = minSyncResults.slice(0, this.limits.prefetch.sectionLimit);
		let shownDatasource = datasourceResults.slice(0, this.limits.prefetch.sectionLimit);
		let shownWeb = webResults.slice(0, this.limits.prefetch.sectionLimit);
		if (rerankFailed && this.rerankTopN !== undefined) {
			// Retrieval order, not rerank order: fill the cut round-robin by rank
			// across the sources so one source cannot take every slot, and
			// each source keeps its own order.
			const sources = [shownPaths, shownMinSync, shownDatasource, shownWeb];
			const quota = sources.map(() => 0);
			for (let taken = 0, rank = 0; taken < this.rerankTopN; rank++) {
				let advanced = false;
				for (const [slot, source] of sources.entries()) {
					if (rank >= source.length || taken >= this.rerankTopN) continue;
					quota[slot] = rank + 1;
					taken++;
					advanced = true;
				}
				if (!advanced) break;
			}
			shownPaths = shownPaths.slice(0, quota[0]);
			shownMinSync = shownMinSync.slice(0, quota[1]);
			shownDatasource = shownDatasource.slice(0, quota[2]);
			shownWeb = shownWeb.slice(0, quota[3]);
		}
		if (jikjiFound && shownPaths.length > 0) {
			sections.push(
				`Jikji initial candidates (preserve order when agent_should_not_rerank=true):\n${shownPaths
					.map(
						(path) =>
							`[${this.evidenceLedger.register({ method: JIKJI_FIND_TOOL_NAME, source: path, content: path })}] ${path}`,
					)
					.join("\n")}`,
			);
		}
		if (shownMinSync.length > 0) {
			sections.push(
				`MinSync semantic initial candidates:\n${shownMinSync
					.map(
						(result) =>
							`[${idFor(result)}] ${result.source}${locationSuffix(result.metadata)}\n${result.content.replace(/\s+/gu, " ")}`,
					)
					.join("\n")}`,
			);
		}
		if (shownDatasource.length > 0) {
			sections.push(
				`Datasource initial candidates:\n${shownDatasource
					.map((result) => `[${idFor(result)}] ${result.source}\n${result.content.replace(/\s+/gu, " ")}`)
					.join("\n")}`,
			);
		}
		if (shownWeb.length > 0) {
			sections.push(
				`Web initial candidates:\n${shownWeb
					.map((result) => `[${idFor(result)}] ${result.source}\n${result.content.replace(/\s+/gu, " ")}`)
					.join("\n")}`,
			);
		}
		return sections.length === 0
			? "No initial retrieval candidates were available; use the configured tools and report degradation honestly."
			: sections.join("\n\n");
	}

	/**
	 * One web search for the baseline, on the same provider chain, ordering, and
	 * credentials as the `web_search` tool. Hits carry the URL as their source
	 * and the title plus snippet as their content, so they merge, rerank, and
	 * register like any other candidate. A failure contributes no evidence and
	 * is reported by the caller; a steer abort is the user stopping the search,
	 * not a failure.
	 */
	private async prefetchWebSearch(searchQuery: string, signal: AbortSignal): Promise<PrefetchedWebSearch> {
		const web = this.webSearchOptions;
		if (web === undefined) return { results: [] };
		const modelAuth = web.modelAuth?.();
		try {
			const search = await executeWebSearch(
				{ query: searchQuery, ...(web.provider !== undefined ? { provider: web.provider } : {}) },
				{
					signal,
					...(web.timeoutSeconds !== undefined ? { timeoutMs: web.timeoutSeconds * 1_000 } : {}),
					...(web.order !== undefined ? { order: web.order } : {}),
					...(web.exclude !== undefined ? { exclude: web.exclude } : {}),
					...(modelAuth !== undefined ? { modelAuth } : {}),
				},
			);
			if (search.details.error !== undefined) return { results: [], error: search.details.error };
			const sources = search.details.response.sources;
			return {
				results: sources.map((source, index) => ({
					id: `web:${index}:${source.url}`,
					source: source.url,
					content: source.snippet ? `${source.title} — ${source.snippet}` : source.title,
					score: 1 - index / Math.max(sources.length, 1),
					metadata: { method: WEB_SEARCH_TOOL_NAME },
				})),
			};
		} catch (error) {
			if (signal.aborted) return { results: [] };
			return { results: [], error: error instanceof Error ? error.message : String(error) };
		}
	}

	/**
	 * Rerank the pre-fast-answer baseline pool — Jikji answer paths, MinSync
	 * chunks, Jev-selected datasource chunks, and prefetched web hits — down to
	 * the configured `rerank.topN`. With decomposition the pool merges every
	 * search query's hits (interleaved), capped at the same per-source sizes as a
	 * single query so the rerank request does not grow with the query count.
	 * Returns `undefined` when reranking is disabled or the pool is empty. A
	 * configured reranker that is unavailable, throws, or returns nothing for a
	 * non-empty pool throws, so the caller can report it and fall back.
	 */
	private async rerankPrefetchPool(
		query: string,
		answerPaths: readonly string[],
		minSyncResults: readonly RetrievalResult[],
		datasourceResults: readonly RetrievalResult[],
		webResults: readonly RetrievalResult[],
		signal: AbortSignal,
	): Promise<RetrievalResult[] | undefined> {
		const reranker = this.reranker;
		if (reranker === undefined) return undefined;
		const candidates: RetrievalResult[] = answerPaths
			.slice(0, this.limits.prefetch.jikjiPathLimit)
			.map((path, index) => ({
				id: `jikji:${index}`,
				content: path,
				source: path,
				score: 1 - index / Math.max(answerPaths.length, 1),
				metadata: { method: "jikji" },
			}));
		candidates.push(...minSyncResults.slice(0, this.limits.prefetch.minSyncTopK));
		candidates.push(...datasourceResults.slice(0, this.limits.prefetch.sectionLimit));
		candidates.push(...webResults.slice(0, this.limits.prefetch.sectionLimit));
		if (candidates.length === 0) return undefined;
		const descriptor = reranker.describe();
		if (!descriptor.available) throw new Error(`reranker is unavailable: ${descriptor.reason ?? "unknown reason"}`);
		const reranked = await reranker.rerank(query, candidates, { topN: this.rerankTopN, signal });
		if (reranked.length === 0) throw new Error("reranker returned no results for a non-empty candidate pool");
		return reranked;
	}

	/**
	 * Surface a rerank failure the way a model error is surfaced: a diagnostic on
	 * the response and, in the TUI, a visible notice in the transcript. The run
	 * continues on the retrieval-order fallback.
	 */
	private reportRerankFailure(error: unknown): void {
		const name = this.reranker?.describe().name ?? "reranker";
		const reason = error instanceof Error ? error.message : String(error);
		const message = `Rerank failed (${name}); using retrieval order cut to the top ${this.rerankTopN} instead: ${reason}`;
		this.routingDiagnostics.push({ code: "rerank-failed", severity: "warning", message, source: name });
		void this.boundPiRuntime?.session
			.sendCustomMessage(
				{ customType: RERANK_FAILURE_MESSAGE_TYPE, content: [{ type: "text", text: message }], display: true },
				{ triggerTurn: false },
			)
			.catch(() => undefined);
	}

	/**
	 * Sentence for the phase prompts that forbid tool calls: `ask_user` is not a search tool, and a request
	 * that hinges on a choice only the user can make must be asked, not guessed. Empty when the session does
	 * not offer the tool.
	 */
	private askUserException(): string {
		return this.askUserOffered
			? `This rule does not cover ${ASK_USER_TOOL_NAME}: use it only when nothing you already have can settle something or the choice is the user's alone, and then ask through it instead of in plain text; never guess. `
			: "";
	}

	/**
	 * Fast-phase exception to the no-tools rule, naming only the web tools this
	 * session actually registered. Empty when web tools are off, so the prompt
	 * never mentions a tool the model does not have.
	 */
	private fastPhaseWebHint(): string {
		const names = new Set(this.tools.map((tool) => tool.name));
		if (!names.has(WEB_SEARCH_TOOL_NAME)) return "";
		const fetch = names.has(WEB_FETCH_TOOL_NAME) ? ` or ${WEB_FETCH_TOOL_NAME}` : "";
		return ` ${WEB_SEARCH_TOOL_NAME}${fetch} is allowed: when the question concerns public facts that may be recent or that changed since your training and the baseline lacks them, call it before answering.`;
	}

	/**
	 * Fast-phase prompt for two-phase searches. The baseline retrieval context
	 * is already gathered, so the model answers immediately with minimal thinking,
	 * as plain text, without further retrieval; the web tools are the only
	 * exception, for public facts the baseline cannot settle.
	 */
	buildFastAnswerPrompt(query: string, options: RetrievalOptions, baseline: string): string {
		const limit = typeof options.topK === "number" ? ` Cite at most ${options.topK} sources.` : "";
		const scope = options.scope ? ` Restrict search to virtual path scope ${options.scope}.` : "";
		return (
			`Answer this original query immediately: ${query}${limit}${scope}\n\n` +
			`Baseline retrieval evidence (already gathered for you):\n${baseline}\n\n` +
			`Produce the best complete, self-contained answer you can RIGHT NOW from this evidence. Do NOT call any local search, retrieval, datasource, or file-reading tool and do NOT wait for more evidence.${this.fastPhaseWebHint()} ${this.askUserException()}` +
			`If the query is answerable from general knowledge alone, answer directly.\n\n` +
			`Formatting and content rules for the answer:\n` +
			`- Unless the user explicitly requests a different format or length, provide the core answer to the user's question in at most 5 bullet points. If additional explanation is necessary, append it after the bullet points.\n` +
			`- ${ANSWER_LEAD_RULE}\n` +
			`- ${ANSWER_BULLET_RULE}\n` +
			`- ${ANSWER_EMPHASIS_RULE}\n` +
			`- ${ANSWER_LANGUAGE_RULE}\n` +
			`- Unless the user explicitly requests otherwise, answer the question directly. Do not include specific file paths, datasource descriptions, or retrieval mechanics in the answer text.\n` +
			`- ${ANSWER_DELIVERABLE_RULE}\n` +
			`- ${ANSWER_CITATION_RULE}\n` +
			`- ${ANSWER_IMAGE_EMBED_RULE}\n` +
			`- Do not report per-source negative findings (e.g. "no information found in Slack" or "checked Drive but found nothing").\n` +
			`- When evidence conflicts, treat the freshest (most recent) information as the correct source of truth.\n` +
			`- ${ANSWER_RELEVANCE_RULE}\n` +
			`- ${ANSWER_UNCONFIRMED_FAST_RULE}\n` +
			`- ${ANSWER_UNCERTAINTY_RULE}\n\n` +
			`Write the answer as your reply, then stop.`
		);
	}

	/**
	 * True when the bound interactive session already holds an assistant turn:
	 * the query continues a conversation instead of opening one. Read from the
	 * transcript, so a fresh session (`/new`) starts over with the full pipeline.
	 */
	private isConversationTurn(): boolean {
		if (this.boundPiRuntime === undefined) return false;
		return this.boundPiRuntime.session.messages.some((message) => message.role === "assistant");
	}

	/**
	 * Prompt for a request Jev routed to `organize`: act on files and folders
	 * (move, rename, convert, sort). No route changes the active tools any more,
	 * so the prompt names the tools the task uses and the ones it must not, and
	 * carries the loaded organize skill with the rules. No retrieval ran and no
	 * verification follows; the model's plain final message is the answer.
	 */
	buildOrganizePrompt(query: string, skill: PackagedAgentSkill): string {
		return (
			`The user asks you to organize files and folders: ${query}\n\n` +
			`${this.organizeToolScope()}\n\n` +
			`${formatSkillInvocation(skill)}\n\n` +
			`Finish with a short report as your final message, with no tool call: every change as old path → new path (one per line), what you skipped and why, and anything that needs the user. Write in the language of the request.`
		);
	}

	/**
	 * The tool-scope sentence of an `organize` turn: the tools the task uses and
	 * the content-retrieval and web tools it must not call. Every route keeps the
	 * whole tool set now, so the prompt is what scopes the turn; both lists name
	 * only tools this agent actually registered, so the prompt never points the
	 * model at a tool the session does not have.
	 */
	private organizeToolScope(): string {
		const registered = new Set(this.tools.map((tool) => tool.name));
		const discovery = [JIKJI_FIND_TOOL_NAME, EVERYTHING_SEARCH_TOOL_NAME, FSEARCH_SEARCH_TOOL_NAME].filter((name) =>
			registered.has(name),
		);
		const usable = [
			"bash",
			"read",
			"edit",
			"write",
			"ls",
			...[CONVERT_FILES_TOOL_NAME, LIST_FILE_CONVERSIONS_TOOL_NAME, SCAN_DUPLICATE_DOCUMENTS_TOOL_NAME].filter(
				(name) => registered.has(name),
			),
			...discovery,
		];
		const forbidden = this.tools
			.map((tool) => tool.name)
			.filter(
				(name) =>
					!discovery.includes(name) &&
					(this.searchToolNames.has(name) || name === WEB_SEARCH_TOOL_NAME || name === WEB_FETCH_TOOL_NAME),
			);
		return (
			`This is a file-organization task, not a document search: work with ${usable.join(", ")}` +
			`${forbidden.length === 0 ? "" : `, and do NOT call the retrieval, datasource, or web tools (${forbidden.join(", ")})`}.`
		);
	}

	/**
	 * Prompt for a request Jev routed to `extra`: no branch had a confident
	 * non-local verdict, and handling it needs no search of the user's records
	 * first. Nothing was retrieved and nothing follows this turn — no fast
	 * answer, follow-up check, or verification — so the single tool-capable
	 * turn either answers the question or carries the task out. No tool is
	 * forbidden: the model looks facts up itself when they are not its own.
	 */
	buildExtraPrompt(query: string): string {
		const registered = new Set(this.tools.map((tool) => tool.name));
		// Name the tools that can settle a question this turn cannot answer from
		// memory alone: the retrieval tools this agent registered, plus the web
		// tools when they are on. Never name a tool the model does not have.
		const lookups = [...registered].filter(
			(name) => this.searchToolNames.has(name) || name === WEB_SEARCH_TOOL_NAME || name === WEB_FETCH_TOOL_NAME,
		);
		const lookupHint = lookups.length === 0 ? "" : ` (${lookups.join(", ")})`;
		const askUser = this.askUserOffered
			? `ask through ${ASK_USER_TOOL_NAME} rather than guessing`
			: "say so in the reply rather than guessing";
		return (
			`Handle this request now, in this one turn, with the tools you have: ${query}\n\n` +
			`Nothing has been retrieved for you and no other turn follows: the message you end with is the final answer.\n\n` +
			`How to handle it:\n` +
			`- Answer from your own knowledge when that reliably settles the question: stable general knowledge, a definition, simple reasoning, or small talk.\n` +
			`- Look it up first when the answer depends on the user's own records (their files, chats, mail, notes, calendar, or history) or on facts that may be recent, specific, or changed${lookupHint}.\n` +
			`- Otherwise carry the task out with the tools.\n\n` +
			`Rules for your reply:\n` +
			`- ${ANSWER_LEAD_RULE}\n` +
			`- ${ANSWER_BULLET_RULE}\n` +
			`- ${ANSWER_EMPHASIS_RULE}\n` +
			`- ${ANSWER_LANGUAGE_RULE}\n` +
			`- ${ANSWER_RELEVANCE_RULE}\n` +
			`- ${ANSWER_UNCONFIRMED_FINAL_RULE}\n` +
			`- ${ANSWER_UNCERTAINTY_RULE}\n` +
			`- ${ANSWER_CITATION_RULE}\n` +
			`- When you did work instead of answering a question, reply with a short report as your final message, with no tool call: what you did, the result, and anything that still needs the user. The answer-format rules above do not apply to a report; write it in the language of the request.\n` +
			`- When something you need is not available to you and only the user can settle it, ${askUser}.\n\n` +
			`Write the answer or the report as your final message, then stop.`
		);
	}

	/**
	 * One tool-capable turn for an action route (`organize`, `extra`): the model
	 * carries the request out with the tools it has and reports through its plain
	 * final message. No retrieval, no fast answer, no verification — and no route
	 * changes the session's active tools, so the turn prompt is what scopes the
	 * work. Returns the report, or `undefined` when the user stopped the run or
	 * the turn ended without one.
	 */
	private async runActionTurn(
		session: AutoRAGSearchSession,
		signal: AbortSignal,
		step: { readonly id: string; readonly label: string; readonly prompt: string },
		transcript: () => readonly AgentMessage[],
	): Promise<string | undefined> {
		// pi's bash tool refuses to run from a missing cwd, and the request may
		// name paths under a workspace that has not been created yet.
		mkdirSync(this.workspaceProjectRoot, { recursive: true });
		await this.searchProgress.step(step.id, step.label, () => session.prompt(step.prompt));
		// A steer typed after the turn's last model request is still owed an answer.
		while (!this.interactive.userAborted && !signal.aborted && this.interactive.queuedSteers.length > 0) {
			await session.prompt(buildSteerContinuationPrompt(this.interactive.steers));
		}
		this.interactive.acceptingSteers = false;
		if (this.interactive.userAborted) return undefined;
		return finalAnswerText(transcript());
	}

	/** Discovery-tool hint naming only the registered discovery tools; empty when none. */
	private discoveryHint(sentence: (tools: string) => string): string {
		const names = this.tools
			.map((tool) => tool.name)
			.filter((name) => name === JIKJI_FIND_TOOL_NAME || name === EVERYTHING_SEARCH_TOOL_NAME);
		return names.length === 0 ? "" : sentence(names.join(" and "));
	}

	/**
	 * Verification-phase nudge toward the internet, for when the web was not
	 * selected before the fast answer or its evidence fell short: when local
	 * evidence is missing, thin, or possibly stale, the agent searches the web. Empty when web tools are off.
	 */
	private webFallbackHint(): string {
		const names = new Set(this.tools.map((tool) => tool.name));
		if (!names.has(WEB_SEARCH_TOOL_NAME)) return "";
		const fetch = names.has(WEB_FETCH_TOOL_NAME) ? ` (and ${WEB_FETCH_TOOL_NAME} to read a promising page)` : "";
		return `When local evidence is missing, thin, or possibly outdated — or the question concerns current events, recent releases, prices, or other public facts — use ${WEB_SEARCH_TOOL_NAME}${fetch} to confirm or correct claims and fill gaps; use URLs as result sources for web-derived evidence. `;
	}

	/**
	 * Verification-phase prompt for two-phase searches. The fast answer, when a
	 * consumer already received it, is embedded verbatim so the model can diff
	 * against it; the model then verifies with thinking on and ends with the
	 * final answer as a plain reply, returning only the delta.
	 */
	buildRefinementPrompt(
		query: string,
		options: RetrievalOptions,
		fastAnswer: string | undefined,
		fastAnswerDelivered: boolean,
		steers: readonly string[] = [],
	): string {
		const limit = typeof options.topK === "number" ? ` Cite at most ${options.topK} sources.` : "";
		const scope = options.scope ? ` Restrict search to virtual path scope ${options.scope}.` : "";
		const firstAnswer = formatFirstAnswerContext(fastAnswer, fastAnswerDelivered);
		const answerRules = fastAnswerDelivered
			? `Rules for your final answer (DELTA ONLY; the system prompt's answer guidelines still apply):\n` +
				`- The user already has the first answer above. Your final answer MUST contain only the delta against it: (a) corrections to anything in the first answer that is wrong, unsupported, or outdated, and (b) newly verified findings that were not present in the first answer. Mark each item as a correction or a new finding.\n` +
				`- NEVER restate first-answer facts that remain correct or repeat its bullet list. If verification changed nothing and found nothing new, the final answer is ONE short line and nothing more (if the first answer said the answer was not confirmed yet, the line says it still could not be confirmed): no mention of what was found instead (no "only X exists" contrast), no account of where or how you searched or what the documents contain.\n` +
				`- Do not include specific file paths, datasource descriptions, or retrieval mechanics in the answer text.\n` +
				`- A correction or new finding that relies on evidence the first answer cited must cite that evidence's id again.\n` +
				`- ${ANSWER_LANGUAGE_RULE}\n` +
				`- Leave out unrelated or merely adjacent findings.\n` +
				`- ${ANSWER_DELIVERABLE_DELTA_RULE}\n` +
				`- ${ANSWER_IMAGE_DELTA_RULE}`
			: `Formatting and content rules for the final answer (COMPLETE — no first answer reached the caller):\n` +
				`- Unless the user explicitly requests a different format or length, provide the core answer to the user's question in at most 5 bullet points. If additional explanation is necessary, append it after the bullet points.\n` +
				`- ${ANSWER_LEAD_RULE}\n` +
				`- ${ANSWER_BULLET_RULE}\n` +
				`- ${ANSWER_EMPHASIS_RULE}\n` +
				`- ${ANSWER_LANGUAGE_RULE}\n` +
				`- ${ANSWER_UNCERTAINTY_RULE}\n` +
				`- ${ANSWER_RELEVANCE_RULE}\n` +
				`- ${ANSWER_UNCONFIRMED_FINAL_RULE}\n` +
				`- Unless the user explicitly requests otherwise, answer the question directly. Do not include specific file paths, datasource descriptions, or retrieval mechanics in the answer text.\n` +
				`- ${ANSWER_DELIVERABLE_RULE}${fastAnswer !== undefined ? ` ${ANSWER_DELIVERABLE_DELTA_RULE}` : ""}\n` +
				`- ${ANSWER_CITATION_RULE}\n` +
				`- ${ANSWER_IMAGE_EMBED_RULE}\n` +
				`- Do not report per-source negative findings (e.g. "no information found in Slack").\n` +
				`- When evidence conflicts, treat the freshest (most recent) information as the correct source of truth.`;
		return (
			`Original query: ${query}${limit}${scope}\n\n` +
			`${firstAnswer}\n\n` +
			formatSteerContext(steers) +
			`Now verify it rigorously. ${this.discoveryHint((tools) => `Actively use ${tools} when discovering or exploring local files and folders. `)}Check important claims against source files with bash when needed, correct anything wrong or unsupported, fill gaps with retrieval tools, and resolve conflicts and freshness. ${this.webFallbackHint()}\n\n` +
			`${answerRules}\n\n` +
			`Do not use broad grep/find or recursive filesystem scans: only inspect a path or narrow neighborhood surfaced by retrieval, and only when evidence clearly points there. ` +
			`Avoid spinning repeated near-identical queries against the same datasource; once additional attempts stop surfacing new evidence, conclude from the evidence available. ` +
			`When finished, reply with the final answer and no tool call: that last reply is what the user receives, so keep progress notes out of it.`
		);
	}

	buildSearchPrompt(query: string, options: RetrievalOptions, initialRetrievalContext?: string): string {
		const limit = typeof options.topK === "number" ? ` Cite at most ${options.topK} sources.` : "";
		const scope = options.scope ? ` Restrict search to virtual path scope ${options.scope}.` : "";
		return (
			`Find and curate information for this original query: ${query}${limit}${scope}\n\n` +
			`Start by deciding whether this is answerable from general knowledge or memory. If it is a generic, stable question, answer it immediately without retrieval. ` +
			`Otherwise, baseline MinSync and Jikji retrieval is already running in parallel; do not give the final answer until its next message arrives.\n\n` +
			`Baseline retrieval context:\n${initialRetrievalContext ?? "Pending; continue only with a brief progress statement."}\n\n` +
			`Treat candidates as unverified evidence, verify important claims against source files when needed, and use additional tools when needed. ` +
			`Judge relevance, conflicts, freshness, and sufficiency in this agent loop.\n\n` +
			`Formatting and content rules for the answer:\n` +
			`- Unless the user explicitly requests a different format or length, provide the core answer to the user's question in at most 5 bullet points. If additional explanation is necessary, append it after the bullet points.\n` +
			`- ${ANSWER_LEAD_RULE}\n` +
			`- ${ANSWER_BULLET_RULE}\n` +
			`- ${ANSWER_EMPHASIS_RULE}\n` +
			`- ${ANSWER_LANGUAGE_RULE}\n` +
			`- Unless the user explicitly requests otherwise, answer the question directly. Do not include specific file paths, datasource descriptions, or retrieval mechanics in the answer text.\n` +
			`- ${ANSWER_DELIVERABLE_RULE}\n` +
			`- ${ANSWER_CITATION_RULE}\n` +
			`- ${ANSWER_IMAGE_EMBED_RULE}\n` +
			`- Do not report per-source negative findings (e.g. "no information found in Slack").\n` +
			`- When evidence conflicts, treat the freshest (most recent) information as the correct source of truth.\n` +
			`- ${ANSWER_RELEVANCE_RULE}\n` +
			`- ${ANSWER_UNCONFIRMED_FINAL_RULE}\n` +
			`- ${ANSWER_UNCERTAINTY_RULE}\n\n` +
			`${this.discoveryHint((tools) => `When exploring local files and folders, actively use ${tools} rather than exploratory bash commands. `)}Do not use broad grep/find or recursive filesystem scans: only inspect a path or narrow neighborhood surfaced by retrieval, and only when evidence clearly points there. ` +
			`Avoid spinning repeated near-identical queries against the same datasource; once additional attempts stop surfacing new evidence, conclude from the evidence available. ` +
			`When finished, reply with the final answer and no tool call: that last reply is what the user receives, so keep progress notes out of it.`
		);
	}

	async refresh(force = false, opts?: AutoRAGRefreshOptions): Promise<AutoRAGRefreshResult> {
		const signal = opts?.signal;
		signal?.throwIfAborted();
		const emitProgress = (event: AutoRAGRefreshProgress): void => {
			try {
				opts?.onProgress?.(event);
			} catch {
				// Progress is observational: a host callback bug must not fail the indexing run.
			}
		};
		const methods = opts?.methods;
		const allMethods = methods === undefined;
		const wants = (m: RefreshMethod): boolean => allMethods || (methods as readonly RefreshMethod[]).includes(m);
		// Parsed mirror is required when MinSync runs,
		// since they index over the parsed mirrors. Also run it when explicitly
		// requested or when all methods are selected.
		const needsParsed = allMethods || wants("parsed") || wants("minsync");
		const runId = randomUUID();
		const startedAt = new Date().toISOString();
		let progress: PersistedRefreshProgress = {
			version: 1,
			runId,
			pid: process.pid,
			state: "running",
			phase: "parsed",
			startedAt,
			updatedAt: startedAt,
		};
		this.refreshState = {
			...this.refreshState,
			inFlight: true,
			lastStartedAt: startedAt,
		};
		try {
			writeRefreshProgress(this.workspaceProjectRoot, progress);
			const summary = needsParsed
				? await this.syncParsedMirrors(force, {
						...(opts?.retrySkipped === true ? { retrySkipped: true } : {}),
						...(signal === undefined ? {} : { signal }),
						onProgress: (parsedProgress) => {
							progress = updateRefreshProgress(progress, {
								sourceFiles: { total: parsedProgress.total },
								parsedCounts: {
									scanned: parsedProgress.processed,
									written: parsedProgress.written,
									deleted: parsedProgress.deleted,
									skipped: parsedProgress.skipped,
								},
							});
							// One marker rewrite per mirror checkpoint (plus the first and last file), not per file: on Windows
							// every rewrite is a rename that can collide with another reader of the same marker.
							if (
								parsedProgress.processed % MIRROR_CHECKPOINT_EVERY === 0 ||
								parsedProgress.processed >= parsedProgress.total
							) {
								writeRefreshProgress(this.workspaceProjectRoot, progress);
							}
							emitProgress({
								phase: "parsed",
								processed: parsedProgress.processed,
								total: parsedProgress.total,
								...(parsedProgress.currentVirtualPath === undefined
									? {}
									: { currentVirtualPath: parsedProgress.currentVirtualPath }),
							});
						},
					})
				: await this.scanMirrorStaleness();
			signal?.throwIfAborted();
			progress = updateRefreshProgress(progress, {
				phase: wants("minsync")
					? "minsync"
					: wants("datasources")
						? "datasources"
						: wants("jikji")
							? "jikji"
							: "finalizing",
				sourceFiles: { total: summary.scanned },
				parsedCounts: {
					scanned: summary.scanned,
					written: summary.written,
					deleted: summary.deleted,
					skipped: summary.skipped,
				},
			});
			writeRefreshProgress(this.workspaceProjectRoot, progress);
			const minsync = wants("minsync") ? await this.syncMinSync(force, signal) : undefined;
			signal?.throwIfAborted();
			if (minsync !== undefined) {
				progress = updateRefreshProgress(progress, {
					phase: wants("datasources") ? "datasources" : wants("jikji") ? "jikji" : "finalizing",
					minsync: { synced: minsync.synced },
				});
				writeRefreshProgress(this.workspaceProjectRoot, progress);
			}
			const datasources = wants("datasources") ? await this.indexDatasources(signal, emitProgress) : [];
			if (wants("datasources")) {
				progress = updateRefreshProgress(progress, {
					phase: wants("jikji") || allMethods ? "jikji" : "finalizing",
				});
				writeRefreshProgress(this.workspaceProjectRoot, progress);
			}
			// A MinSync refresh also establishes Jikji's local discovery artifacts:
			// both indexes are first-class parts of the default local corpus.
			const jikji =
				allMethods || wants("jikji") || wants("minsync")
					? await this.executeJikjiPrepare({ ...opts?.jikjiPrepare, ...(signal === undefined ? {} : { signal }) })
					: undefined;
			signal?.throwIfAborted();
			// On Windows, the bundled Everything instance indexes the same local
			// roots by name so the agent can locate files instantly.
			let everything: AutoRAGEverythingRefreshResult | undefined;
			if (this.everythingClient !== undefined && (allMethods || wants("everything") || needsParsed)) {
				progress = updateRefreshProgress(progress, { phase: "everything" });
				writeRefreshProgress(this.workspaceProjectRoot, progress);
				signal?.throwIfAborted();
				const indexed = await this.everythingClient.index();
				everything = indexed.ok
					? { ok: true, indexedItems: indexed.indexedItems }
					: { ok: false, reason: indexed.message };
			}
			// On macOS/Linux, fsearch-cli indexes the same local roots by name
			// and keeps a per-workspace watch daemon serving live searches.
			let fsearch: AutoRAGFSearchRefreshResult | undefined;
			if (this.fsearchClient !== undefined && (allMethods || wants("fsearch") || needsParsed)) {
				progress = updateRefreshProgress(progress, { phase: "fsearch" });
				writeRefreshProgress(this.workspaceProjectRoot, progress);
				signal?.throwIfAborted();
				const indexed = await this.fsearchClient.index();
				fsearch = indexed.ok
					? { ok: true, indexedItems: indexed.indexedItems }
					: { ok: false, reason: indexed.reason, message: indexed.message };
			}
			progress = updateRefreshProgress(progress, { phase: "finalizing" });
			writeRefreshProgress(this.workspaceProjectRoot, progress);
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
				everything,
				fsearch,
				lastError: undefined,
			};
			if (needsParsed) {
				mkdirSync(dirname(refreshReadinessPath(this.workspaceProjectRoot)), { recursive: true });
				writeFileSync(
					refreshReadinessPath(this.workspaceProjectRoot),
					'{"version":1,"completed":true,"parsed":true}\n',
				);
			}
			progress = updateRefreshProgress(progress, {
				state: "success",
				finishedAt: new Date().toISOString(),
			});
			writeRefreshProgress(this.workspaceProjectRoot, progress);
			const minsyncDiagnostics = minSyncRefreshDiagnostics(minsync);
			const publicMinsync: AutoRAGMinSyncRefreshResult | undefined = minsync
				? {
						ok: minsync.ok,
						synced: minsync.synced,
						...(minsync.reason !== undefined ? { reason: sanitizeDiagnosticMessage(minsync.reason) } : {}),
						...(minsyncDiagnostics.length > 0 ? { diagnostics: minsyncDiagnostics } : {}),
						...(minsync.stagingExcluded !== undefined && minsync.stagingExcluded.length > 0
							? { stagingExcludedCount: minsync.stagingExcluded.length }
							: {}),
					}
				: undefined;
			const everythingDiagnostics = everythingRefreshDiagnostics(everything);
			const fsearchDiagnostics = fsearchRefreshDiagnostics(fsearch);
			return {
				...summary,
				diagnostics: [
					...this.startupDiagnostics,
					...summary.diagnostics,
					...minsyncDiagnostics,
					...stagingExcludedDiagnostics(minsync?.stagingExcluded),
					...fileSkippedDiagnostics(minsync?.filesFailed),
					...everythingDiagnostics,
					...fsearchDiagnostics,
				],
				minsync: publicMinsync,
				datasources,
				...(everything !== undefined ? { everything } : {}),
				...(fsearch !== undefined ? { fsearch } : {}),
			};
		} catch (error) {
			this.refreshState = {
				...this.refreshState,
				lastOutcome: "failed",
				lastError:
					error instanceof MirrorWorkspaceError
						? `Refresh failed: workspace not writable${error.code === undefined ? "" : ` (${error.code})`}`
						: error instanceof Error
							? `Refresh failed: ${error.name}`
							: "Refresh failed.",
			};
			progress = updateRefreshProgress(progress, {
				state: "failed",
				error: this.refreshState.lastError,
				finishedAt: new Date().toISOString(),
			});
			writeRefreshProgress(this.workspaceProjectRoot, progress);
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
	 * One warning when nothing can embed, so semantic search is not silently lexical-only. The text names the
	 * way out; AutoRAG does not start the daemon itself. Shown by `status`, `lite status`, and the MCP status tool.
	 */
	private async embeddingDiagnostics(): Promise<SearchDocumentDiagnostic[]> {
		const health = await this.embedder.health().catch(() => undefined);
		if (health === undefined || health.alive) return [];
		return [
			{
				code: "embedding-unavailable",
				severity: "warning",
				message:
					health.backend === "remote"
						? `The configured remote embedding endpoint is not answering (${health.detail ?? "no response"}); semantic search is unavailable until it is back.`
						: "No embedding backend is running, so semantic search is unavailable (keyword search still works). Start `minsync serve --idle-timeout 0` (docs/minsync-setup.md), press Ctrl+Shift+E in the TUI, or set minSync.embedder to a remote endpoint.",
				source: "embedding",
			},
		];
	}

	/**
	 * Path-opaque snapshot of corpus freshness and the last refresh outcome. Runs
	 * a cheap parse-free staleness scan (stat only); never parses in this path.
	 *
	 * Freshness is read from disk, not from this instance's history: a separate CLI
	 * process (for example `autorag status` or `autorag lite status`) reports the
	 * corpus as current when the last refresh left parsed mirrors behind and no
	 * source has changed since.
	 */
	async getRefreshStatus(): Promise<AutoRAGRefreshStatus> {
		const persistedProgress = readRefreshProgress(this.workspaceProjectRoot);
		const staleDiagnostics = await detectMirrorStaleness({
			...this.refreshLimits,
			root: this.workspaceProjectRoot,
			searchPaths: this.searchPaths,
			parserOptions: this.parserOptions,
			userExcludedSourcePaths: new Set(this.excludePaths),
		});
		const diagnostics: SearchDocumentDiagnostic[] = [
			...this.startupDiagnostics,
			...this.refreshState.mirrorDiagnostics.map(toSearchDiagnostic),
			...staleDiagnostics.map(toSearchDiagnostic),
			...(await this.embeddingDiagnostics()),
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
		for (const diag of minSyncRefreshDiagnostics(this.refreshState.minsync)) {
			if (!diagnostics.some((d) => d.code === diag.code && d.source === diag.source)) {
				diagnostics.push(diag);
			}
		}
		diagnostics.push(...everythingRefreshDiagnostics(this.refreshState.everything));
		diagnostics.push(...fsearchRefreshDiagnostics(this.refreshState.fsearch));
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
		const persistedRunning = persistedProgress?.state === "running";
		const ownerAlive = persistedProgress !== undefined && isRefreshOwnerAlive(persistedProgress);
		const interrupted = persistedRunning && !ownerAlive && !this.refreshState.inFlight;
		if (interrupted) {
			diagnostics.push({
				code: "refresh-interrupted",
				severity: "error",
				message: "A previous refresh stopped before it completed.",
				source: "refresh",
			});
		}
		const state: AutoRAGRefreshStatus["state"] =
			this.refreshState.inFlight || (persistedRunning && ownerAlive)
				? "indexing"
				: interrupted
					? "failed"
					: persistedProgress?.state === "success" || persistedProgress?.state === "failed"
						? persistedProgress.state
						: this.refreshState.lastOutcome === "never"
							? "idle"
							: this.refreshState.lastOutcome;
		const parsedMirrorReady =
			this.refreshState.lastOutcome === "success" ||
			persistedProgress?.state === "success" ||
			existsSync(refreshReadinessPath(this.workspaceProjectRoot));
		const effectiveProgress = this.refreshState.inFlight || persistedRunning ? persistedProgress : undefined;
		return {
			state,
			inFlight: this.refreshState.inFlight,
			lastStartedAt: this.refreshState.lastStartedAt ?? persistedProgress?.startedAt,
			lastFinishedAt: this.refreshState.lastFinishedAt ?? persistedProgress?.finishedAt,
			counts: this.refreshState.counts,
			progress:
				effectiveProgress === undefined
					? undefined
					: {
							phase: effectiveProgress.phase,
							sourceFiles: effectiveProgress.sourceFiles,
							parsedCounts: effectiveProgress.parsedCounts,
							minsync: effectiveProgress.minsync,
							ownerAlive,
						},
			stale: !parsedMirrorReady || staleDiagnostics.length > 0,
			diagnostics,
			components: this.refreshComponentStatus(),
			lastError:
				this.refreshState.lastError ??
				(persistedProgress?.state === "failed" ? persistedProgress.error : undefined),
		};
	}

	/**
	 * Synchronous per-component readiness snapshot (minsync/jikji/datasources/everything).
	 * `minsync` is "ready" only after a successful sync wrote its cursor;
	 * "configured" means the binary resolved but no index exists yet.
	 */
	refreshComponentStatus(): AutoRAGRefreshComponentStatus {
		const status: { minsync?: string; jikji?: string; datasources?: string; everything?: string; fsearch?: string } =
			{};
		status.minsync = this.minSyncMethod.isExplicitBinaryMissing()
			? "unavailable"
			: this.refreshState.minsync?.ok === false
				? "degraded"
				: this.minSyncMethod.isReady()
					? "ready"
					: "configured";
		if (this.jikjiClient !== undefined) {
			status.jikji = this.refreshState.jikjiDiagnostics.length > 0 ? "degraded" : "configured";
		}
		if (this.datasourceSkills.length > 0) {
			status.datasources = this.refreshState.datasources.some((result) => !result.ok) ? "degraded" : "configured";
		}
		if (this.everythingClient !== undefined) {
			const everything = this.refreshState.everything;
			status.everything = everything === undefined ? "configured" : everything.ok ? "ready" : "degraded";
		}
		if (this.fsearchClient !== undefined) {
			const fsearch = this.refreshState.fsearch;
			status.fsearch =
				fsearch === undefined
					? "configured"
					: fsearch.ok
						? "ready"
						: fsearch.reason === "binary-missing"
							? "unavailable"
							: "degraded";
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

	async syncParsedMirrors(
		force = false,
		options: Pick<ParsedMirrorSyncOptions, "signal" | "onProgress" | "retrySkipped"> = {},
	): Promise<ParsedMirrorSyncResult> {
		const duplicateFilter = await this.exactDuplicateExclusions();
		return syncParsedMirrors({
			...options,
			...this.refreshLimits,
			root: this.workspaceProjectRoot,
			searchPaths: this.searchPaths,
			force,
			parserOptions: this.parserOptions,
			excludeSourcePaths: duplicateFilter.excluded,
			userExcludedSourcePaths: new Set(this.excludePaths),
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
			...this.refreshLimits,
			root: this.workspaceProjectRoot,
			searchPaths: this.searchPaths,
			parserOptions: this.parserOptions,
			excludeSourcePaths: duplicateFilter.excluded,
			userExcludedSourcePaths: new Set(this.excludePaths),
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
		const userExcluded = new Set(this.excludePaths);
		for (const searchPath of this.searchPaths) {
			try {
				const scan = await scanWithDupey(searchPath, this.dupeyOptions || {});
				const selected = await selectExactDuplicateExclusions(searchPath, scan, (path) =>
					isPathExcluded(path, userExcluded),
				);
				for (const path of selected.excluded) excluded.add(path);
			} catch (error) {
				if (!(error instanceof DupeyCliError)) throw error;
				// Optional optimizer: missing/broken dupey must not make the corpus unavailable.
			}
		}
		return { excluded };
	}

	async syncMinSync(force = false, signal?: AbortSignal): Promise<MinSyncSyncResult> {
		const result = await this.minSyncMethod.sync(force, signal);
		this.refreshState = { ...this.refreshState, minsync: result };
		return result;
	}

	/** Provider for the Windows-only everything_search tool. */
	async searchEverything(request: EverythingSearchRequest): Promise<EverythingSearchResult> {
		if (this.everythingClient === undefined) {
			return { ok: false, reason: "unsupported-platform", message: "Everything is not enabled on this host." };
		}
		return this.everythingClient.search(request);
	}

	/**
	 * Exit this workspace's Everything instance. The instance otherwise stays
	 * running on purpose (live folder monitoring between CLI invocations), so
	 * call this only when the workspace is being torn down. No-op off Windows.
	 */
	async stopEverything(): Promise<void> {
		await this.everythingClient?.stop();
	}

	/** Provider for the macOS/Linux-only fsearch_search tool. */
	async searchFsearch(request: FSearchSearchRequest): Promise<FSearchSearchResult> {
		if (this.fsearchClient === undefined) {
			return { ok: false, reason: "unsupported-platform", message: "FSearch is not enabled on this host." };
		}
		return this.fsearchClient.search(request);
	}

	/**
	 * Terminate this workspace's fsearch-cli watch daemon. The daemon otherwise
	 * stays running on purpose (live index updates between CLI invocations), so
	 * call this only when the workspace is being torn down. No-op off
	 * macOS/Linux.
	 */
	async stopFsearch(): Promise<void> {
		await this.fsearchClient?.stop();
	}

	async prepareJikji(options?: JikjiPrepareOptions): Promise<readonly AutoRAGJikjiPrepareResult[] | undefined> {
		const results = await this.executeJikjiPrepare(options);
		return results?.map((result) => this.sanitizeJikjiPrepareResult(result));
	}

	/**
	 * Build or incrementally update every root's Jikji index. Called only by
	 * refresh/watch (and the explicit `prepareJikji()` API), never by a query:
	 * `jikji prepare` reuses unchanged documents. Roots are submitted together
	 * but `JikjiClient` runs them one at a time, because every root shares one
	 * central SQLite file that fails with "database is locked" under concurrent
	 * writers. Nested and duplicate roots are skipped: upstream rejects
	 * overlapping roots and the outer root's index already covers them. A failed
	 * or timed-out root is a degraded result (reported as a diagnostic), never a
	 * rejection, so the refresh still finishes. `jikji.parseTimeout` /
	 * `jikji.timeoutMs` / `jikji.prepareTimeoutMsPerFile` apply to every root.
	 */
	private async executeJikjiPrepare(
		options: JikjiPrepareOptions = {},
	): Promise<readonly JikjiPrepareResult[] | undefined> {
		const client = this.jikjiClient;
		if (client === undefined) return undefined;
		return Promise.all(this.jikjiRoots.map((sourcePath) => client.prepare(sourcePath, options)));
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
		opts?: { readonly topK?: number; readonly first?: boolean; readonly signal?: AbortSignal },
	): Promise<JikjiFindProviderResult> {
		if (this.jikjiClient === undefined) {
			return { answerPack: undefined, policy: undefined, diagnostics: [], roots: [], perRoot: [] };
		}
		const sourceRoots = planJikjiSourceRoots(this.jikjiRoots);
		const findOpts: JikjiFindOptions = {
			topK: opts?.topK,
			first: opts?.first,
			...(opts?.signal === undefined ? {} : { signal: opts.signal }),
		};
		const diagnostics: JikjiDiagnostic[] = [];
		const okPacks: { pack: JikjiAnswerPack; root: string }[] = [];
		const searchResults = await Promise.all(
			this.jikjiRoots.map(async (sourcePath) => {
				const result: JikjiFindResult = await this.jikjiClient!.find(sourcePath, query, findOpts);
				return { sourcePath, result };
			}),
		);
		for (const { sourcePath, result } of searchResults) {
			if (result.ok) {
				okPacks.push({ pack: result.answerPack, root: sourcePath });
			} else {
				const diag = jikjiFindDiagnostic(result);
				if (diag !== undefined) diagnostics.push(diag);
			}
		}
		if (okPacks.length === 0) {
			return { answerPack: undefined, policy: undefined, diagnostics, roots: this.jikjiRoots, perRoot: [] };
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
		return { answerPack: merged, policy, diagnostics, roots: this.jikjiRoots, perRoot };
	}

	/**
	 * Merge per-root answer packs into one. Concatenates answer_paths/candidates
	 * preserving per-root order; dedupes by normalized path. Does NOT cross-root
	 * rerank when any root has agentShouldNotRerank=true.
	 *
	 * Honours `excludePaths` at the retrieval boundary: Jikji indexes the source
	 * folders directly, bypassing the parsed mirror, so an excluded source still
	 * appears in the on-disk `.jikji_agent_map.md`. Dropping excluded paths here
	 * keeps them out of the agent-facing answer pack (the `jikji_find` tool and
	 * the baseline prefetch) while leaving the shared map artifact complete.
	 * Excluded paths remain reachable through direct file reads, as Jikji never
	 * blocks source verification.
	 */
	private mergeAnswerPacks(
		entries: readonly { pack: JikjiAnswerPack; root: string }[],
		sourceRoots: readonly JikjiSourceRoot[],
		policy: MergedJikjiPolicy,
	): JikjiAnswerPack {
		const excluded = new Set(this.excludePaths);
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
				if (norm !== undefined && !isPathExcluded(norm, excluded) && !seenPaths.has(norm)) {
					seenPaths.add(norm);
					answerPaths.push(norm);
				}
			}
			for (const rawPath of entry.pack.paths) {
				const norm = normalizeJikjiAnswerPath(rawPath, originRoots);
				if (norm !== undefined && !isPathExcluded(norm, excluded) && !allPaths.includes(norm)) {
					allPaths.push(norm);
				}
			}
			for (const cand of entry.pack.candidates) {
				const norm = normalizeJikjiAnswerPath(cand.path, originRoots);
				if (norm !== undefined && !isPathExcluded(norm, excluded) && !candidates.some((c) => c.path === norm)) {
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
				if (norm !== undefined && !isPathExcluded(norm, excluded) && !evidencePack.some((e) => e.path === norm)) {
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
		const filteredByMethod = filterDatasourceScope(byMethod, methods, options.scope);
		const merged = this.merger.merge(filteredByMethod, {
			topK: options.topK ?? this.limits.mergedEvidenceCeiling,
			dedup: true,
		});
		const results = await this.applyRerank(query, merged, diagnostics);
		return { results, diagnostics };
	}

	async searchAllDocuments(
		query: string,
		options: { readonly topK?: number; readonly scope?: string; readonly signal?: AbortSignal } = {},
	): Promise<{ results: RetrievalResult[]; diagnostics: RetrievalDiagnostic[] }> {
		return this.retrieveWithDiagnostics(query, {
			topK: options.topK,
			scope: options.scope,
			...(options.signal === undefined ? {} : { signal: options.signal }),
		});
	}

	/**
	 * Search one datasource connection only. Only the target connection's
	 * retrieval methods are registered with the retriever, so no other
	 * datasource CLI is spawned at all and only that connection's hits are
	 * returned. An unknown `datasourceId` yields an empty result set.
	 *
	 * This backs the generated `search_datasource_<id>` tools; every configured
	 * connection has one. Cross-datasource fan-out is
	 * {@link searchAllDocuments}, which spans every retrieval method.
	 */
	async searchSingleDatasourceDocuments(
		datasourceId: string,
		query: string,
		options: { readonly topK?: number; readonly scope?: string; readonly signal?: AbortSignal } = {},
	): Promise<{ results: RetrievalResult[]; diagnostics: RetrievalDiagnostic[] }> {
		const retrievalOptions: RetrievalOptions = {
			...this.activeRetrievalOptions,
			topK: options.topK,
			scope: options.scope,
			...(options.signal === undefined ? {} : { signal: options.signal }),
		};
		const methods = this.methodRegistry.list().filter((method) => {
			const descriptor = method.describe();
			return descriptor.datasourceId === datasourceId;
		});
		if (methods.length === 0) return { results: [], diagnostics: [] };
		const { results: byMethod, diagnostics } = await this.retriever.retrieveWithDiagnostics(
			methods,
			query,
			retrievalOptions,
		);
		const filteredByMethod = filterDatasourceScope(byMethod, methods, options.scope);
		const merged = this.merger.merge(filteredByMethod, {
			topK: options.topK ?? this.limits.singleDatasourceTopK,
			dedup: true,
		});
		// Single-datasource retrieval is intentionally NOT model-reranked: the
		// caller already narrowed to one connection, so the merged order is kept.
		return { results: merged, diagnostics };
	}

	/** The retrieval method registry (posix, MinSync, and datasource methods). */
	getMethodRegistry(): RetrievalMethodRegistry {
		return this.methodRegistry;
	}

	/**
	 * The standalone retrieval engine for this agent's method pipeline.
	 * Built on first access using the agent's registered methods and configured
	 * datasource catalog. Model-free — no agent state required.
	 */
	private retrievalEngine: RetrievalEngine | undefined;
	getRetrievalEngine(): RetrievalEngine {
		if (this.retrievalEngine === undefined) {
			this.retrievalEngine = new RetrievalEngine({
				defaultTopK: this.limits.mergedEvidenceCeiling,
				...(this.reranker !== undefined ? { reranker: this.reranker } : {}),
				...(this.rerankTopN !== undefined ? { rerankTopN: this.rerankTopN } : {}),
				datasourceIds: () => this.listDatasources().map((entry) => entry.datasourceId),
			});
			for (const method of this.methodRegistry.list()) {
				this.retrievalEngine.register(method);
			}
		}
		return this.retrievalEngine;
	}

	getSystemPrompt(): string {
		return this.innerAgent.state.systemPrompt;
	}

	/**
	 * Reorder merged evidence with the configured reranker. A configured-but-
	 * unavailable reranker, or a rerank failure, is reported as a diagnostic and
	 * the merged order is preserved — a reranker outage never hides evidence.
	 */
	private async applyRerank(
		query: string,
		results: RetrievalResult[],
		diagnostics: RetrievalDiagnostic[],
	): Promise<RetrievalResult[]> {
		if (this.reranker === undefined || results.length === 0) return results;
		const descriptor = this.reranker.describe();
		if (!descriptor.available) {
			diagnostics.push({
				code: "rerank-failed",
				severity: "warning",
				message: `Reranker "${descriptor.name}" is configured but unavailable; merged order preserved: ${descriptor.reason ?? "unknown reason"}`,
				source: descriptor.name,
				...(descriptor.reason !== undefined ? { reason: descriptor.reason } : {}),
			});
			return results;
		}
		try {
			return await this.reranker.rerank(query, results, { topN: this.rerankTopN });
		} catch (error) {
			const reason = error instanceof Error ? error.message : String(error);
			diagnostics.push({
				code: "rerank-failed",
				severity: "warning",
				message: `Reranker "${descriptor.name}" failed; merged order preserved: ${reason}`,
				source: descriptor.name,
				reason,
			});
			return results;
		}
	}

	private remoteFilteredRetrievalMethod<
		T extends {
			retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]>;
			describe(): { name: string };
		},
	>(method: T): T {
		return new Proxy(method, {
			get: (target, property, receiver) => {
				if (property !== "retrieve") return Reflect.get(target, property, receiver);
				return async (query: string, options: RetrievalOptions) => {
					const effective = { ...this.activeRetrievalOptions, ...options };
					return target.retrieve.call(target, query, effective);
				};
			},
		}) as T;
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

/**
 * Baseline block for the reranked pre-fast-answer pool. Each candidate carries
 * the evidence id the fast answer cites it by, so a source never has to be
 * retyped and never collides with a result citation number.
 */
function formatRerankedBaseline(results: readonly RetrievalResult[], evidenceIds: readonly string[]): string {
	const lines = results.map(
		(result, index) =>
			`[${evidenceIds[index]}] ${result.source}${locationSuffix(result.metadata)}\n${result.content.replace(/\s+/gu, " ")}`,
	);
	return `Reranked initial candidates (ordered by relevance to the query; cite evidence by its [eN] id):\n${lines.join("\n")}`;
}

/** The result of a retrieval tool a steer cut short, worded like pi's skipped-tool results. */
function skippedBySteer(toolName: string): AgentToolResult<{ readonly method: string; readonly skipped: "steer" }> {
	return {
		content: [
			{
				type: "text",
				text: `Skipped due to the user's steering message. ${toolName} was stopped before it returned; do not count it as completed work. Apply the steering message, then retry the search if it is still needed.`,
			},
		],
		details: { method: toolName, skipped: "steer" },
	};
}

/**
 * The model's view of a TUI steer. The user typed it while the search ran, to
 * correct a search that is going wrong, so it is framed as a correction of the
 * running search rather than a new question; the final reply stays the answer
 * to the original question.
 */
function buildSteerMessage(text: string): string {
	return (
		`The user interrupted the running search with this correction:\n\n${text}\n\n` +
		`Treat it as direction for the search in progress, not as a new question: adjust which sources, queries, or claims you pursue, then continue. ` +
		`When finished, reply with the final answer to the original question, updated for this correction, and no tool call.`
	);
}

/**
 * Frames the session's steers for the model. pi records, lists, and restores a
 * steer as the text the user typed; only the context sent to the model wraps
 * it in {@link buildSteerMessage}.
 */
function frameSteers(messages: AgentMessage[], steerTexts: ReadonlySet<string>): AgentMessage[] {
	if (steerTexts.size === 0) return messages;
	return messages.map((message) => {
		if (message.role !== "user" || typeof message.content === "string") return message;
		const [first, ...rest] = message.content;
		if (first?.type !== "text" || !steerTexts.has(first.text)) return message;
		return { ...message, content: [{ ...first, text: buildSteerMessage(first.text) }, ...rest] };
	});
}

/** Restates the run's steers for a verification phase that starts after them. */
function formatSteerContext(steers: readonly string[]): string {
	if (steers.length === 0) return "";
	return (
		`While the search ran, the user corrected it:\n${steers.map((steer) => `- ${steer}`).join("\n")}\n` +
		`The first answer predates these corrections; follow them over it.\n\n`
	);
}

/** One more verification turn for steers that arrived after the last phase's final model request. */
function buildSteerContinuationPrompt(steers: readonly string[]): string {
	return (
		`${formatSteerContext(steers)}Apply the latest correction: search or verify further if it needs that, ` +
		`then reply with the complete final answer to the original question and no tool call.`
	);
}

/**
 * The first answer rendered for the verification prompt. When a consumer
 * already received it, the model must diff against it and return only the
 * delta; otherwise the draft is internal context only and the final answer
 * must stay complete. The text is the model's own, still carrying its `[eN]`
 * ids, which stay valid for the whole run.
 */
function formatFirstAnswerContext(fastAnswer: string | undefined, delivered: boolean): string {
	if (fastAnswer === undefined) {
		return "(the fast phase produced no answer — write the complete answer)";
	}
	const header = delivered
		? "The user has ALREADY received this immediate first answer (its citations were renumbered for the user; the [eN] ids below are still valid for you):"
		: "An internal first-pass draft was produced but was NOT shown to the caller; write the complete answer:";
	return `${header}\n${fastAnswer}`;
}

/**
 * Degraded fallback answer for a run that ended without a final answer message:
 * states the search-range failure, carries the agent's own last note as the
 * reason, suggests next steps, and points at the attached retrieval trace.
 */
function buildMissingFinalAnswer(
	query: string,
	reason: string | undefined,
	trace: readonly SearchDocumentRetrievalTraceEntry[],
	modelError?: string,
): string {
	const lines = [
		`The search run for "${query}" ended without a final answer: the agent stopped before writing one, so no curated answer is available within the configured search range.`,
		...(modelError === undefined ? [] : [`The model request failed: ${modelError}`]),
		...(reason === undefined
			? modelError === undefined
				? ["The agent did not record why it stopped."]
				: []
			: [`The agent's last note before stopping: "${reason.length > 500 ? `${reason.slice(0, 500)}…` : reason}"`]),
		modelError === undefined
			? "Next steps: broaden the configured searchPaths, connect additional datasources via the datasources configuration, or retry with a narrower or different query."
			: "Next steps: fix the model provider error above (rate limit, quota, credentials, or model id) and retry the same query.",
		trace.length > 0
			? "Retrieval candidates gathered before the run ended are attached under `retrievalTrace` for inspection."
			: "No retrieval candidates were gathered before the run ended.",
	];
	return lines.join("\n\n");
}

/**
 * The provider error that ended the run, if the last assistant turn failed.
 * pi-agent-core records a failed request (HTTP 429, auth, unknown model) as an
 * assistant message with stopReason "error" and no text, so without this the
 * degraded answer only says the agent "did not record why it stopped".
 */
function lastModelRequestError(messages: readonly AgentMessage[]): string | undefined {
	for (let index = messages.length - 1; index >= 0; index--) {
		const message = messages[index];
		if (message.role !== "assistant") continue;
		if (message.stopReason !== "error") return undefined;
		return message.errorMessage?.trim() || "the provider returned an error without a message";
	}
	return undefined;
}

/**
 * The run's answer: the text of the last assistant message, when that message
 * ended its turn (`stop`) without calling a tool. A message that stopped on a
 * tool call is a progress note, one that errored, aborted, or hit the length
 * limit never finished its answer, so none of those count.
 */
function finalAnswerText(messages: readonly AgentMessage[]): string | undefined {
	for (let index = messages.length - 1; index >= 0; index--) {
		const message = messages[index];
		if (message.role !== "assistant") continue;
		if (message.stopReason !== "stop") return undefined;
		const text = message.content
			.filter((block) => block.type === "text")
			.map((block) => block.text)
			.join("")
			.trim();
		return text === "" ? undefined : text;
	}
	return undefined;
}

/** The last text the agent wrote, whatever stopped it: the reason a run ended without an answer. */
function lastAssistantText(messages: readonly AgentMessage[]): string | undefined {
	for (let index = messages.length - 1; index >= 0; index--) {
		const message = messages[index];
		if (message.role !== "assistant") continue;
		const text = message.content
			.filter((block) => block.type === "text")
			.map((block) => block.text)
			.join("")
			.trim();
		if (text !== "") return text;
	}
	return undefined;
}

/**
 * Documents the parsed mirror holds but the MinSync index cannot represent:
 * a file name with no canonical source-id form stays searchable only as a raw
 * file, so the gap is reported instead of being dropped silently.
 */
function stagingExcludedDiagnostics(excluded: readonly string[] | undefined): SearchDocumentDiagnostic[] {
	return (excluded ?? []).map((source) => ({
		code: "minsync-staging-excluded",
		severity: "warning",
		message: "Excluded from the MinSync index: this file name cannot be represented as a canonical source id.",
		source,
	}));
}

/** Files MinSync skipped on a partial sync (exit 6): searchable only from older chunks until the next sync retries them. */
function fileSkippedDiagnostics(failed: readonly MinSyncFileFailure[] | undefined): SearchDocumentDiagnostic[] {
	return (failed ?? []).map((failure) => ({
		code: "minsync-file-skipped",
		severity: "warning",
		message: `MinSync skipped this file at the ${failure.stage} stage and will retry it on the next sync: ${sanitizeDiagnosticMessage(failure.error)}`,
		source: failure.path,
	}));
}

function toSearchDiagnostic(diagnostic: ParsedMirrorDiagnostic): SearchDocumentDiagnostic {
	return {
		code: diagnostic.code,
		severity: diagnostic.severity,
		message: diagnostic.message,
		source: diagnostic.source,
	};
}

function sanitizeDiagnosticMessage(raw: string): string {
	// `\s+` must not overlap the leading `\n` or `.split` walks a long newline
	// run quadratically (CodeQL js/polynomial-redos): `[^\S\n]` is whitespace
	// that excludes the newline, so stack-frame indentation still matches.
	let out = raw.split(/\n[^\S\n]+at\s/)[0] ?? raw;
	out = out.replace(/(?:^|[^A-Za-z0-9])(\/(?:[^/\s]+\/)+[^/\s]+)/g, " <path>");
	out = out.replace(/[A-Za-z]:\\[^\s]+/g, "<path>");
	return out.replace(/\s{2,}/g, " ").trim();
}

/** A failed Everything index surfaces its underlying message verbatim as a refresh error. */
function everythingRefreshDiagnostics(
	everything: AutoRAGEverythingRefreshResult | undefined,
): SearchDocumentDiagnostic[] {
	if (everything === undefined || everything.ok) return [];
	return [
		{
			code: "everything-index-failed",
			severity: "error",
			message: `Everything file-name indexing failed: ${everything.reason ?? "unknown error"}`,
			source: "everything",
		},
	];
}

/**
 * A failed FSearch index surfaces its underlying message verbatim. A missing
 * fsearch-cli is a warning, not an error: FSearch is an optional user install
 * and searches degrade to the slow filesystem walk by design (#1763).
 */
function fsearchRefreshDiagnostics(fsearch: AutoRAGFSearchRefreshResult | undefined): SearchDocumentDiagnostic[] {
	if (fsearch === undefined || fsearch.ok) return [];
	if (fsearch.reason === "binary-missing") {
		return [
			{
				code: "fsearch-binary-missing",
				severity: "warning",
				message: `fsearch-cli is not installed; file-name search uses a slow filesystem walk: ${fsearch.message ?? "unknown error"}`,
				source: "fsearch",
			},
		];
	}
	return [
		{
			code: "fsearch-index-failed",
			severity: "error",
			message: `FSearch file-name indexing failed: ${fsearch.message ?? fsearch.reason ?? "unknown error"}`,
			source: "fsearch",
		},
	];
}

/**
 * Project a MinSync sync result onto refresh diagnostics. A structured MinSync
 * diagnostic wins; otherwise a failed sync is reported through its reason so a
 * degraded semantic index is never silent.
 */
function minSyncRefreshDiagnostics(minsync: MinSyncSyncResult | undefined): SearchDocumentDiagnostic[] {
	if (!minsync) return [];
	const diagnostics: SearchDocumentDiagnostic[] = [];
	if (minsync.diagnostic) diagnostics.push(toMinSyncDiagnostic(minsync.diagnostic, minsync.ok));
	else if (!minsync.ok && minsync.reason) diagnostics.push(toMinSyncReasonDiagnostic(minsync.reason));
	if (minsync.lexicalFallback) {
		diagnostics.push({
			code: "lexical-tokenizer-unavailable",
			severity: "warning",
			message: `The "${minsync.lexicalFallback.requested}" BM25 tokenizer needs the Kiwi native library and model, which were not found, so MinSync tokenizes on whitespace and a Korean word with a particle (인증서를) does not match its stem (인증서). Install Kiwi (set KIWI_LIBRARY_PATH and KIWI_MODEL_PATH, or install it under ~/.local/kiwi), then refresh again.`,
			source: "minsync",
		});
	}
	return diagnostics;
}

function toMinSyncDiagnostic(diag: MinSyncDiagnostic, ok: boolean): SearchDocumentDiagnostic {
	const code: SearchDocumentDiagnosticCode =
		diag.code === "embedder-unavailable"
			? "embedder-unavailable"
			: diag.code === "embedding-identity-mismatch"
				? "embedding-identity-mismatch"
				: diag.code === "minsync-lock-held"
					? "minsync-lock-held"
					: "minsync-sync-failed";
	return {
		code,
		severity: ok ? "info" : "error",
		message: sanitizeDiagnosticMessage(diag.message),
		source: "minsync",
	};
}

function toMinSyncReasonDiagnostic(reason: string): SearchDocumentDiagnostic {
	return {
		code: "minsync-sync-failed",
		severity: "error",
		message: `MinSync sync failed: ${sanitizeDiagnosticMessage(reason)}`,
		source: "minsync",
	};
}

/**
 * Canonicalize a user exclusion the same way collectFiles builds sourcePath:
 * realpath the parent chain (like pinSearchRoot does for roots) but keep the
 * final component literal, so excluding a symlinked file still matches.
 * Missing paths keep their resolved form — exclusion of a not-yet-existing
 * path is legitimate.
 */
function pinExcludedPath(path: string): string {
	const resolvedPath = resolve(path);
	try {
		return join(realpathSync(dirname(resolvedPath)), basename(resolvedPath));
	} catch {
		return resolvedPath;
	}
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

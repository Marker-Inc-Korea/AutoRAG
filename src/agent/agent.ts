import { randomUUID } from "node:crypto";
import { existsSync, watch as fsWatch, mkdirSync, realpathSync, statSync, writeFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import type { Agent, AgentEvent, AgentMessage, AgentTool } from "@earendil-works/pi-agent-core";
import type { Api, Model } from "@earendil-works/pi-ai";
import { clampThinkingLevel } from "@earendil-works/pi-ai/compat";
import type { ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { getAgentDir } from "@earendil-works/pi-coding-agent";
import { resolveAutoRAGHome } from "../config/home.ts";
import { DatasourceAccessContext, type DatasourceAccessContextOptions } from "../datasource/access-context.ts";
import { mapDatasourceDiagnostics } from "../datasource/diagnostics.ts";
import { DatasourceResultFilter } from "../datasource/result-filter.ts";
import type { DatasourceIndexResult, DatasourceSkill } from "../datasource/types.ts";
import { DupeyCliError, type DupeyCliOptions, scanWithDupey, selectExactDuplicateExclusions } from "../dupey/index.ts";
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
	type JikjiPrepareResult,
	type JikjiSourceRoot,
	normalizeJikjiAnswerPath,
	planJikjiSourceRoots,
} from "../jikji/index.ts";
import { DEFAULT_LANGUAGES, type LanguageTag } from "../language.ts";
import { loadManifests } from "../manifest/loader.ts";
import { createCheckMemoryTool } from "../memory/check-memory-tool.ts";
import type { ResultFeedback } from "../memory/memory.ts";
import { RetrievalMemory } from "../memory/memory.ts";
import { renderMemoryContext } from "../memory/renderer.ts";
import {
	type MinSyncDiagnostic,
	MinSyncHybridMethod,
	type MinSyncSyncResult,
	MinSyncVectorMethod,
	type MinSyncVectorMethodOptions,
} from "../minsync/index.ts";
import { PARSED_MIRROR_SUBDIR, refreshReadinessPath } from "../mirror/paths.ts";
import {
	detectMirrorStaleness,
	isPathExcluded,
	type ParsedMirrorDiagnostic,
	type ParsedMirrorSyncResult,
	syncParsedMirrors,
} from "../mirror/sync.ts";
import { AutoRAGRunLogger } from "../observability/run-log.ts";
import { scanOutboundPayload } from "../p2p/injection-classifier.ts";
import type { PolicyResolver } from "../p2p/policy-filter.ts";
import type { SimplexQueryState } from "../p2p/simplex-query-store.ts";
import { openPeerQueryTransport, type SimplexTransport } from "../p2p/simplex-transport.ts";
import type { PeerQueryResponse } from "../p2p/wire.ts";
import { type DefaultParserRegistryOptions, resolveParserOptions } from "../parser/index.ts";
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
import type { CuratedResult, RetrievalDiagnostic, RetrievalOptions, RetrievalResult } from "../retrieval/types.ts";
import { executeWebSearch } from "../web/search/index.ts";
import { type ModelNativeSearchAuth, modelNativeAuthFromAgentModel } from "../web/search/model-auth.ts";
import { ANSWER_CITATION_RULE, ANSWER_IMAGE_DELTA_RULE, ANSWER_IMAGE_EMBED_RULE } from "./answer-guidelines.ts";
import {
	createLoadDatasourceSkillTool,
	type DatasourceAgentSkill,
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
import { createEverythingSearchTool, EVERYTHING_SEARCH_TOOL_NAME } from "./everything-search-tool.ts";
import {
	type AutoRAGFastAnswerDetails,
	createEmitFastAnswerTool,
	EMIT_FAST_ANSWER_TOOL_NAME,
} from "./fast-answer-tool.ts";
import { createFSearchSearchTool, FSEARCH_SEARCH_TOOL_NAME } from "./fsearch-search-tool.ts";
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
import { createRecommendPeerTargetsTool, RECOMMEND_PEER_TARGETS_TOOL_NAME } from "./peer-target-tool.ts";
import {
	type AutoRAGPiInteractiveRuntime,
	type AutoRAGPiInteractiveRuntimeOptions,
	type AutoRAGPiSession,
	createAutoRAGPiInteractiveRuntime,
	createAutoRAGPiSession,
	PI_BUILTIN_TOOL_NAMES,
} from "./pi-session.ts";
import { createModelDecompositionCompleter, type DecompositionModel, decomposeQuery } from "./query-decomposition.ts";
import { createQueryPeerAgentTool, QUERY_PEER_AGENT_TOOL_NAME } from "./query-peer-tool.ts";
import { FALLBACK_QUERY_ROUTE, needsFollowUp, type QueryRoute, routeQuery } from "./query-routing.ts";
import {
	isRefreshOwnerAlive,
	type PersistedRefreshProgress,
	type RefreshProgressCounts,
	type RefreshProgressPhase,
	readRefreshProgress,
	updateRefreshProgress,
	writeRefreshProgress,
} from "./refresh-progress.ts";
import { createSearchAllDocumentsTool, SEARCH_ALL_DOCUMENTS_TOOL_NAME } from "./search-all-tool.ts";
import {
	createEmptySearchDocumentsResponse,
	createPreliminarySearchDocumentsResponse,
	recordNumberedFeedback,
	recordStructuredResultsSession,
	type SearchDocumentDiagnostic,
	type SearchDocumentDiagnosticCode,
	type SearchDocumentRetrievalTraceEntry,
	type SearchDocumentRetrievalTraceResult,
	type SearchDocumentsResponse,
	type SearchDocumentsStreamEvent,
} from "./search-documents.ts";
import { createSearchMinSyncDocumentsTool, SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME } from "./search-minsync-tool.ts";
import { createSingleDatasourceSearchTools, type SingleDatasourceToolSpec } from "./search-single-datasource-tool.ts";
import {
	buildSelfConfigPrompt,
	loadSetupSkill,
	rollbackIfBroken,
	type SelfConfigOptions,
	snapshotConfigFile,
} from "./self-config.ts";
import { buildSystemPrompt, type SystemPromptConfig } from "./system-prompt.ts";
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
 * tool budget and weak-signal memory. `bash` is a source-inspection tool,
 * not a search datasource, so it is deliberately excluded from this list.
 */
const SEARCH_TOOLS = [
	SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME,
	SEARCH_ALL_DOCUMENTS_TOOL_NAME,
	JIKJI_FIND_TOOL_NAME,
	EVERYTHING_SEARCH_TOOL_NAME,
	FSEARCH_SEARCH_TOOL_NAME,
] as const;

/** Only completed searches that returned evidence earn implicit positive feedback. */
function hasSearchEvidence(toolName: string, details: unknown, isError: boolean): boolean {
	if (isError || details === null || typeof details !== "object") return false;
	const outcome = details as Record<string, unknown>;
	if (outcome.available === false) return false;
	if (
		Array.isArray(outcome.diagnostics) &&
		outcome.diagnostics.some(
			(diagnostic) =>
				diagnostic !== null &&
				typeof diagnostic === "object" &&
				(diagnostic.severity === "error" ||
					diagnostic.code === "retrieval-method-failed" ||
					diagnostic.code === "minsync-unavailable" ||
					diagnostic.code === "jikji-find-failed" ||
					diagnostic.code === "jikji-unavailable"),
		)
	) {
		// Pipeline failures are warnings even when healthy methods return hits.
		// Without per-method attribution, the incomplete aggregate earns no credit.
		return false;
	}
	// Jikji reports answer paths, while the other search tools report resultCount.
	const countKey = toolName === JIKJI_FIND_TOOL_NAME ? "answerCount" : "resultCount";
	if (countKey in outcome) {
		const count = outcome[countKey];
		return typeof count === "number" && Number.isFinite(count) && count > 0;
	}
	// Legacy producers may omit counts. Never override an explicit zero/invalid
	// count, and require an actual source identity rather than an arbitrary item.
	return (
		(Array.isArray(outcome.sources) &&
			outcome.sources.some((source) => typeof source === "string" && source.trim().length > 0)) ||
		(Array.isArray(outcome.results) &&
			outcome.results.some(
				(result) =>
					result !== null &&
					typeof result === "object" &&
					typeof result.source === "string" &&
					result.source.trim().length > 0,
			))
	);
}

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

function positiveLimit(value: number | undefined, fallback: number, path: string): number {
	if (value === undefined) return fallback;
	if (!Number.isInteger(value) || value <= 0) throw new Error(`${path} must be a positive integer`);
	return value;
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

export interface AutoRAGRefreshOptions {
	/** Restrict refresh to specific methods. Defaults to all when undefined. */
	readonly methods?: readonly RefreshMethod[];
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

declare module "../retrieval/types.ts" {
	interface RetrievalOptions {
		/** Server-only P2P policy identity; never included in model tool schemas. */
		peerFingerprint?: string;
		/** Server-only P2P policy resolver. */
		resolvePolicy?: PolicyResolver;
		/** Server-only run registry populated from observed retrieval output. */
		observedSources?: Set<string>;
		/** Server-only datasource source globs derived from the loaded policy. */
		policyDatasourceScopes?: readonly string[];
	}
}

export type RemoteSessionRejectionCode = "injection-detected" | "outbound-leak-detected";

export class RemoteSessionRejectedError extends Error {
	readonly code: RemoteSessionRejectionCode;

	constructor(code: RemoteSessionRejectionCode) {
		super(`Remote session rejected: ${code}`);
		this.name = "RemoteSessionRejectedError";
		this.code = code;
	}
}

/** Thinking level applied per search phase. "off" requests no reasoning. */
export type AutoRAGThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";

/**
 * Per-phase thinking for AutoRAG's two-phase search: the fast phase delivers
 * an immediate first answer with {@link AutoRAGThinkingOptions.fast} thinking
 * (default "off"), then the verification phase re-checks and finalizes with
 * {@link AutoRAGThinkingOptions.final} thinking (default "high").
 */
export interface AutoRAGThinkingOptions {
	/** Thinking level for the immediate first answer. Default "off". */
	readonly fast?: AutoRAGThinkingLevel;
	/** Thinking level for the verification/finalization phase. Default "high". */
	readonly final?: AutoRAGThinkingLevel;
}

export interface PeerQueryOptions {
	readonly port?: number;
	readonly simplexDbPrefix?: string;
	readonly openTransport?: () => Promise<SimplexTransport>;
	readonly onResponse?: (state: SimplexQueryState, response: PeerQueryResponse) => void;
}

export interface AutoRAGAgentOptions {
	model?: Model<Api>;
	apiKey?: string;
	providerApiKeys?: Readonly<Record<string, string>>;
	searchPaths: string[];
	languages?: readonly LanguageTag[];
	manifestDir?: string;
	memoryPath?: string;
	workspacePath?: string;
	tools?: AgentTool[];
	minSync?: Omit<MinSyncVectorMethodOptions, "root"> | false;
	jikji?: JikjiOptions | false;
	/**
	 * Windows-only instant file/folder name search through the bundled
	 * voidtools Everything (portable, user-level, folder-only index over
	 * `searchPaths`). Enabled by default on Windows; ignored elsewhere.
	 * `false` disables it. The remaining fields are test seams.
	 */
	everything?: Omit<EverythingClientOptions, "root" | "folders"> | false;
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
	 * Anthropic/OpenAI/xAI web_search), the anonymous Perplexity ask
	 * endpoint, and Parallel's keyless MCP — then scraped engines with
	 * headless-browser escalation for bot challenges. A self-hosted
	 * SEARXNG_ENDPOINT is the only env-gated option. `false` disables both
	 * tools. The `fetch` sub-option tunes or disables `web_fetch` alone.
	 * Web tools are always omitted for remote P2P sessions.
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
	 * Jev (TypeSafe's judgment model: typed questions in, calibrated
	 * probabilities out). When set, the two-phase search asks Jev before the
	 * fast answer whether the question needs local search, web search, or a
	 * direct answer, and whether to decompose it; after the fast answer, whether
	 * verification is needed. It also exposes the `jev` tool. Always omitted for
	 * remote P2P sessions.
	 *
	 * The CLI config enables this by default on OpenRouter (`buildAgentOptions`
	 * fills it in). On this programmatic option, absent or `false` keeps Jev
	 * off, so library callers never make paid network calls they did not ask for.
	 */
	jev?: JevToolOptions | false;
	/**
	 * Agent self-configuration. When set (and Jev is on), Jev gains a `config`
	 * branch for questions about AutoRAG's own settings: the turn skips
	 * `emit_fast_answer`, receives the full `autorag-setup` skill, edits
	 * `configPath` with the pi tools, and reports through
	 * `emit_autorag_results`. Always omitted for remote P2P sessions.
	 */
	selfConfig?: SelfConfigOptions;
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
	datasourceSkills?: readonly DatasourceSkill[];
	datasourceAccess?: DatasourceAccessContextOptions;
	/** Non-fatal diagnostics from config/agent construction (e.g. skipped unknown datasources). */
	startupDiagnostics?: readonly SearchDocumentDiagnostic[];
	/** Maximum time a model/tool search may run before it is aborted. */
	searchTimeoutMs?: number;
	/** Maximum number of retrieval/tool executions allowed in one search. */
	maxSearchToolCalls?: number;
	/**
	 * Outbound SimpleX queries to trusted peer AutoRAG agents. Remote sessions
	 * never receive this tool.
	 */
	peerQuery?: PeerQueryOptions | false;
	/** Restrict the agent to retrieval and result-emission tools for remote runs. */
	remoteSession?: boolean;
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
}

/** Post-merge reranking options. Mirrors the CLI `RerankConfig` (secrets via env). */
export interface RerankAgentOptions {
	/** Provider id. @default "openrouter" */
	provider?: string;
	/** Wire model id. @default "voyageai/rerank-3-lite" */
	model?: string;
	/** API key. Prefer `apiKeyEnv`; this is a programmatic seam. */
	apiKey?: string;
	/** Environment variable holding the provider API key. */
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
	private lastSessionId: string | undefined;
	private readonly sessions = new Map<string, { query: string; registry: Map<number, CuratedResult> }>();
	private activeRun = false;
	private resultCapture: ((details: AutoRAGResultsDetails) => void) | undefined;
	private interactiveFastAnswerCallback: ((details: AutoRAGFastAnswerDetails) => void) | undefined;
	private modelNativeSearchAuth: ModelNativeSearchAuth | undefined;
	private retrievalTrace: SearchDocumentRetrievalTraceEntry[] = [];
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
	private readonly datasourceFilter = new DatasourceResultFilter();

	private readonly minSyncMethod: MinSyncVectorMethod | undefined;
	private readonly jikjiClient: JikjiClient | undefined;
	private readonly everythingClient: EverythingClient | undefined;
	private readonly fsearchClient: FSearchClient | undefined;
	private readonly datasourceSkills: readonly DatasourceSkill[];
	private readonly datasourceAccessOptions: DatasourceAccessContextOptions;
	private readonly startupDiagnostics: readonly SearchDocumentDiagnostic[];
	private readonly datasourceAgentSkills: readonly DatasourceAgentSkill[];
	private readonly parserOptions: DefaultParserRegistryOptions | undefined;
	private readonly dupeyOptions: DupeyCliOptions | false;
	/** pi extension registering the optional `jev` tool; undefined when disabled. */
	private readonly jevExtension: ExtensionFactory | undefined;
	/** Jev judge shared by the `jev` tool and the query router; undefined when disabled. */
	private readonly jevJudge: JevJudge | undefined;
	/** Self-configuration, enabled only with Jev on a local session. */
	private readonly selfConfig: SelfConfigOptions | undefined;
	private readonly queryDecompositionModel: DecompositionModel | undefined;
	/** Web search routing for the pipeline's web branch; undefined when web tools are off. */
	private readonly webSearchOptions: WebSearchToolOptions | undefined;
	/** Routing diagnostics for the in-flight search; reset per run. */
	private routingDiagnostics: SearchDocumentDiagnostic[] = [];
	private readonly excludeExactDuplicates: boolean;
	private readonly excludePaths: readonly string[];
	private readonly baseSystemPromptConfig: SystemPromptConfig;
	private readonly droppedCallerToolNames: readonly string[];
	private readonly searchTimeoutMs: number;
	private readonly maxSearchToolCalls: number;
	private readonly piAgentDir: string | undefined;
	private readonly piSessionDir: string | undefined;
	private readonly persistPiSessions: boolean;
	private readonly updateNotice: (() => Promise<string | undefined>) | undefined;
	private boundPiRuntime: AutoRAGPiInteractiveRuntime["runtime"] | undefined;
	/** True when this agent was constructed for an untrusted remote peer. */
	readonly remoteSession: boolean;
	private activeRetrievalOptions: RetrievalOptions | undefined;
	private searchToolCallCount = 0;
	private readonly limits: ResolvedRetrievalLimits;

	constructor(options: AutoRAGAgentOptions) {
		const { manifestDir, memoryPath } = options;
		this.configuredModel = options.model;
		this.remoteSession = options.remoteSession ?? false;
		this.searchTimeoutMs = options.searchTimeoutMs ?? (this.remoteSession ? 120_000 : 10 * 60 * 1000);
		this.maxSearchToolCalls = options.maxSearchToolCalls ?? 64;
		if (!Number.isFinite(this.searchTimeoutMs) || this.searchTimeoutMs <= 0) {
			throw new Error("searchTimeoutMs must be a positive finite number");
		}
		if (!Number.isInteger(this.maxSearchToolCalls) || this.maxSearchToolCalls <= 0) {
			throw new Error("maxSearchToolCalls must be a positive integer");
		}
		this.fastThinkingLevel = options.thinking?.fast ?? "off";
		this.finalThinkingLevel = options.thinking?.final ?? "high";
		this.apiKey = options.apiKey;
		this.providerApiKeys = options.providerApiKeys;
		this.piAgentDir = options.piAgentDir;
		this.piSessionDir = options.piSessionDir;
		this.persistPiSessions = options.persistPiSessions ?? true;
		this.updateNotice = options.updateNotice;
		const manifests = manifestDir ? loadManifests(manifestDir) : [];
		this.datasourceSkills = options.datasourceSkills ?? [];
		this.datasourceVirtualScopePrefixes = this.datasourceSkills.map((skill) =>
			normalizeVirtualPath(`/${skill.describe().name}`),
		);
		this.datasourceAccessOptions = options.datasourceAccess ?? {};
		this.startupDiagnostics = options.startupDiagnostics ?? [];
		this.datasourceAgentSkills = this.buildAuthorizedDatasourceSkills();
		this.configuredSearchPaths = options.searchPaths.map((searchPath) => resolve(searchPath));
		this.languages = options.languages ?? DEFAULT_LANGUAGES;
		this.searchPaths = options.searchPaths.map(pinSearchRoot);
		this.workspaceProjectRoot = options.workspacePath ?? process.cwd();
		this.retrievalScopeBindings = buildRetrievalScopeBindings(
			this.workspaceProjectRoot,
			this.searchPaths,
			this.configuredSearchPaths,
		);
		// The global language setting drives OCR engine selection inside the registry.
		this.parserOptions = resolveParserOptions(options.parserOptions, this.languages);
		this.dupeyOptions = options.dupey ?? {};
		this.excludeExactDuplicates = options.excludeExactDuplicates ?? true;
		this.excludePaths = (options.excludePaths ?? []).map(pinExcludedPath);
		this.limits = resolveRetrievalLimits(options.limits);
		this.reranker = createReranker(options.rerank === false || options.rerank === undefined ? false : options.rerank);
		this.rerankTopN =
			options.rerank === false || options.rerank === undefined
				? undefined
				: (options.rerank.topN ?? DEFAULT_RERANK_TOP_N);

		if (options.minSync !== false) {
			const minSyncOpts = options.minSync ?? { autoInstall: true };
			const minSyncDefaults = {
				...minSyncOpts,
				root: this.workspaceProjectRoot,
				defaultTopK: this.limits.minSyncTopK,
				scopedTopK: this.limits.minSyncScopedQueryTopK,
			};
			this.minSyncMethod = new MinSyncVectorMethod(minSyncDefaults);
			this.methodRegistry.register(this.minSyncMethod);
			this.methodRegistry.register(new MinSyncHybridMethod(minSyncDefaults));
		}
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
		this.runLogger = new AutoRAGRunLogger(join(dirname(memPath), "logs", "runs.jsonl"));

		const checkMemoryTool = createCheckMemoryTool(this.memory);
		// One tool per authorized datasource connection, so a question that
		// targets a single connection spawns only that connection's CLIs instead
		// of fanning out to every datasource. Generated from the same trusted
		// config + access context as the skill list, so disabled or denied
		// datasources never appear as tools. These are the only datasource
		// retrieval tools: cross-datasource fan-out lives in search_all_documents.
		const singleDatasourceTools = createSingleDatasourceSearchTools(this, this.singleDatasourceToolSpecs());
		this.searchToolNames = new Set([...SEARCH_TOOLS, ...singleDatasourceTools.map((tool) => tool.name)]);

		const searchMinSyncTool = createSearchMinSyncDocumentsTool(
			() => this.remoteFilteredRetrievalMethod(this.minSyncMethod),
			(scope) => this.resolveRetrievalScope(scope),
		);
		const searchAllTool = createSearchAllDocumentsTool(this);
		const loadDatasourceSkillTool = createLoadDatasourceSkillTool(this);
		const emitResultsTool = createEmitResultsTool((details) => this.resultCapture?.(details));
		const scanDuplicateDocumentsTool =
			this.dupeyOptions === false ? undefined : createScanDuplicateDocumentsTool(this);
		this.jevJudge =
			options.jev === undefined || options.jev === false || this.remoteSession
				? undefined
				: createJevJudge(options.jev);
		this.jevExtension = this.jevJudge === undefined ? undefined : createJevExtension(this.jevJudge);
		this.selfConfig = this.jevJudge === undefined || this.remoteSession ? undefined : options.selfConfig;
		this.queryDecompositionModel =
			options.queryDecomposition?.model === undefined
				? undefined
				: {
						model: options.queryDecomposition.model,
						...(options.queryDecomposition.apiKey !== undefined
							? { apiKey: options.queryDecomposition.apiKey }
							: {}),
					};

		const peerTargetTool = this.remoteSession ? undefined : createRecommendPeerTargetsTool(this.workspaceProjectRoot);
		const peerQuery = options.peerQuery;
		const queryPeerTool =
			this.remoteSession || peerQuery === undefined || peerQuery === false
				? undefined
				: createQueryPeerAgentTool({
						workspacePath: this.workspaceProjectRoot,
						onResponse: peerQuery?.onResponse,
						openTransport:
							peerQuery?.openTransport ??
							(() =>
								openPeerQueryTransport({
									dbPrefix:
										peerQuery?.simplexDbPrefix ??
										join(this.workspaceProjectRoot, ".autorag", "p2p", "simplex"),
									displayName: `autorag-${basename(this.workspaceProjectRoot) || "node"}`,
								})),
						autoStart: true,
						sessionId: () => this.lastSessionId,
					});

		const jikjiFindTool = this.jikjiClient !== undefined ? createJikjiFindTool(this) : undefined;
		// Remote peers never enumerate local file names.
		const everythingSearchTool =
			this.everythingClient !== undefined && !this.remoteSession ? createEverythingSearchTool(this) : undefined;
		const fsearchSearchTool =
			this.fsearchClient !== undefined && !this.remoteSession ? createFSearchSearchTool(this) : undefined;

		const webSearchOption = options.webSearch;
		const webToolsEnabled = webSearchOption !== false && !this.remoteSession;
		this.webSearchOptions = webToolsEnabled
			? { ...(webSearchOption ?? {}), modelAuth: () => this.modelNativeSearchAuth }
			: undefined;
		const webSearchTool =
			this.webSearchOptions !== undefined ? createWebSearchTool(this.webSearchOptions) : undefined;
		const webFetchTool =
			webToolsEnabled && webSearchOption?.fetch !== false
				? createWebFetchTool(webSearchOption?.fetch ?? {})
				: undefined;

		// Reserved AutoRAG tool names the agent always owns. Caller tools with
		// these names are dropped (reserved wins), never rejected.
		const reservedNames = new Set<string>([
			...PI_BUILTIN_TOOL_NAMES,
			"check_memory",
			...singleDatasourceTools.map((tool) => tool.name),
			LOAD_DATASOURCE_SKILL_TOOL_NAME,
			EMIT_AUTORAG_RESULTS_TOOL_NAME,
			EMIT_FAST_ANSWER_TOOL_NAME,
			SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME,
			SEARCH_ALL_DOCUMENTS_TOOL_NAME,
			JIKJI_FIND_TOOL_NAME,
			EVERYTHING_SEARCH_TOOL_NAME,
			FSEARCH_SEARCH_TOOL_NAME,
			SCAN_DUPLICATE_DOCUMENTS_TOOL_NAME,
			JEV_TOOL_NAME,
			RECOMMEND_PEER_TARGETS_TOOL_NAME,
			QUERY_PEER_AGENT_TOOL_NAME,
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
			emitResultsTool,
			...(scanDuplicateDocumentsTool !== undefined ? [scanDuplicateDocumentsTool] : []),
			...(jikjiFindTool !== undefined ? [jikjiFindTool] : []),
			...(everythingSearchTool !== undefined ? [everythingSearchTool] : []),
			...(fsearchSearchTool !== undefined ? [fsearchSearchTool] : []),
			...(peerTargetTool !== undefined ? [peerTargetTool] : []),
			...(queryPeerTool !== undefined ? [queryPeerTool as AgentTool] : []),
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
			memorySignalCount: this.memory.getSignalCount(),
			manifests,
			datasourceSkills: this.datasourceAgentSkills,
			jikjiIndexingEnabled: options.jikji !== false,
			retrievedContentGuard: false,
			remoteSession: this.remoteSession,
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
		extraTools: readonly AgentTool[] = [],
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
			customTools: [
				...this.tools.filter(
					(tool) => !PI_BUILTIN_TOOL_NAMES.includes(tool.name as (typeof PI_BUILTIN_TOOL_NAMES)[number]),
				),
				...extraTools,
			],
			remoteSession: this.remoteSession,
			contextTransform: (messages) => this.withMemoryContext(messages),
			...(this.jevExtension !== undefined
				? { extensionFactories: [this.jevExtension], extensionToolNames: [JEV_TOOL_NAME] }
				: {}),
		});
		const agent = piSession.session.agent;
		return {
			agent,
			piSession: piSession.session,
			prompt: async (prompt) => piSession.session.prompt(prompt, { source: "extension" }),
			abort: async () => piSession.session.abort(),
			dispose: () => piSession.session.dispose(),
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
		if (!this.searchToolNames.has(event.toolName)) return;
		this.searchToolCallCount += 1;
		if (this.searchToolCallCount >= this.maxSearchToolCalls) {
			void this.activeSession?.abort();
		}
		const details = event.result.details as
			| {
					method?: string;
					sources?: readonly string[];
					resultCount?: number;
					results?: readonly SearchDocumentRetrievalTraceResult[];
			  }
			| undefined;
		if (this.remoteSession && this.activeRetrievalOptions?.observedSources !== undefined) {
			for (const source of details?.sources ?? []) this.activeRetrievalOptions.observedSources.add(source);
		}
		if (Array.isArray(details?.results)) {
			const args = (event as { args?: { query?: unknown } }).args;
			this.retrievalTrace.push({
				tool: event.toolName,
				...(typeof args?.query === "string" ? { query: args.query } : {}),
				resultCount:
					typeof details?.resultCount === "number" ? details.resultCount : (details?.results?.length ?? 0),
				results: details?.results ?? [],
			});
		}
		if (hasSearchEvidence(event.toolName, details, event.isError)) {
			this.memory.recordWeakSignal(this.lastQuery, details?.method ?? event.toolName, "followup");
			this.memory.save();
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

	async createPiInteractiveRuntime(): Promise<AutoRAGPiInteractiveRuntime> {
		const resolved = this.configuredModel === undefined ? undefined : await this.resolveSessionModel();
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
					}),
				),
			contextTransform: (messages) => this.withMemoryContext(messages),
			customTools: [
				...this.tools.filter(
					(tool) => !PI_BUILTIN_TOOL_NAMES.includes(tool.name as (typeof PI_BUILTIN_TOOL_NAMES)[number]),
				),
				createEmitFastAnswerTool((details) => this.interactiveFastAnswerCallback?.(details)),
			],
			onQuery: (query, pi) => this.runInteractivePiQuery(query, pi),
			inactiveToolNames: [EMIT_FAST_ANSWER_TOOL_NAME],
			...(this.jevExtension !== undefined
				? { extensionFactories: [this.jevExtension], extensionToolNames: [JEV_TOOL_NAME] }
				: {}),
			...(this.updateNotice === undefined ? {} : { updateNotice: this.updateNotice }),
		});
		this.boundPiRuntime = runtime.runtime;
		runtime.runtime.session.setActiveToolsByName(
			runtime.runtime.session.getActiveToolNames().filter((name) => name !== EMIT_FAST_ANSWER_TOOL_NAME),
		);
		const dispose = runtime.dispose;
		return {
			...runtime,
			dispose: async () => {
				if (this.boundPiRuntime === runtime.runtime) this.boundPiRuntime = undefined;
				await dispose();
			},
		};
	}

	private async runInteractivePiQuery(
		query: string,
		pi: Parameters<NonNullable<AutoRAGPiInteractiveRuntimeOptions["onQuery"]>>[1],
	): Promise<void> {
		pi.setSessionName(query.slice(0, 80));
		for await (const event of this.searchDocumentsStream(query)) {
			const text = event.type === "progress" ? event.text : event.response.answer;
			// Display-only: these arrive while the search turn is streaming, and
			// pi's default delivery steers a custom message into that turn as a
			// user message, so the model would answer its own progress and the
			// query would never finish. triggerTurn:false shows them without
			// adding a turn.
			pi.sendMessage(
				{
					customType: `autorag.${event.type}`,
					content: [{ type: "text", text }],
					display: true,
					details: event,
				},
				{ triggerTurn: false },
			);
		}
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
			return createEmptySearchDocumentsResponse(sessionId, trimmedQuery, this.sessions, this.startupDiagnostics);
		}
		options = this.normalizeRetrievalOptions(options);
		if (this.remoteSession && options.observedSources !== undefined) options.observedSources.clear();
		this.activeRetrievalOptions = options;

		this.activeRun = true;
		this.searchToolCallCount = 0;
		this.retrievalTrace = [];
		this.routingDiagnostics = [];
		this.lastQuery = trimmedQuery;
		this.lastSessionId = sessionId;
		let captured: AutoRAGResultsDetails | undefined;
		let fastCaptured: AutoRAGFastAnswerDetails | undefined;
		/** True when this run took the Jev `config` branch; its report never feeds retrieval memory. */
		let selfConfigRun = false;
		/**
		 * Without Jev every fast answer goes on to verification, so it is
		 * published the moment emit_fast_answer runs. With Jev, publishing waits
		 * for the direct route and the follow-up check: an answer that turns out
		 * to be final reaches the caller once, as the complete response.
		 */
		const publishOnCapture = this.jevJudge === undefined;
		let published = false;
		const publishPreliminary = (details: AutoRAGFastAnswerDetails): void => {
			if (published) return;
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
		const emitPreliminary = (details: AutoRAGFastAnswerDetails): void => {
			if (fastCaptured !== undefined) return;
			fastCaptured = details;
			if (publishOnCapture) publishPreliminary(details);
		};
		let session: AutoRAGSearchSession | undefined;
		this.interactiveFastAnswerCallback = emitPreliminary;
		let unsubscribers: readonly (() => void)[] = [];
		this.resultCapture = (details) => {
			captured = details;
		};
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
				[createEmitFastAnswerTool(emitPreliminary)],
			);
			this.activeSession = session;
			unsubscribers = this.configureSearchSession(session);
			let timeout: NodeJS.Timeout | undefined;
			const planAbort = new AbortController();
			try {
				await Promise.race([
					(async () => {
						// Two-phase flow: fast thinking-off answer first, then a
						// thinking-on verification pass that finalizes the results.
						// With Jev enabled, Jev first picks local search, web search, or a
						// direct answer, and whether the question needs decomposition.
						const plan = await this.planQuery(trimmedQuery, resolved, planAbort.signal);
						const sessionAgent = session.piSession;
						const activeSession = session;
						const activateFastPhase = (): void => {
							if (sessionAgent !== undefined) {
								sessionAgent.setThinkingLevel(clampThinkingLevel(resolved.model, this.fastThinkingLevel));
								sessionAgent.setActiveToolsByName([
									...sessionAgent.getActiveToolNames().filter((name) => name !== EMIT_FAST_ANSWER_TOOL_NAME),
									EMIT_FAST_ANSWER_TOOL_NAME,
								]);
							} else {
								activeSession.agent.state.thinkingLevel = clampThinkingLevel(
									resolved.model,
									this.fastThinkingLevel,
								);
								activeSession.agent.state.tools = [
									...this.tools,
									{ name: EMIT_FAST_ANSWER_TOOL_NAME } as AgentTool,
								];
							}
						};
						if (plan.route === "config" && plan.selfConfigSkill !== undefined && this.selfConfig !== undefined) {
							// Self-configuration: no retrieval and no emit_fast_answer. The
							// model gets the full setup skill and edits the config itself,
							// then reports through emit_autorag_results.
							selfConfigRun = true;
							// pi's bash tool refuses to run from a missing cwd, and a freshly
							// initialised config has not created its workspace yet.
							mkdirSync(this.workspaceProjectRoot, { recursive: true });
							const previousTools = sessionAgent?.getActiveToolNames();
							if (sessionAgent !== undefined) {
								sessionAgent.setThinkingLevel(clampThinkingLevel(resolved.model, this.finalThinkingLevel));
								sessionAgent.setActiveToolsByName([...PI_BUILTIN_TOOL_NAMES, EMIT_AUTORAG_RESULTS_TOOL_NAME]);
							} else {
								activeSession.agent.state.thinkingLevel = clampThinkingLevel(
									resolved.model,
									this.finalThinkingLevel,
								);
								activeSession.agent.state.tools = this.tools.filter(
									(tool) => tool.name === EMIT_AUTORAG_RESULTS_TOOL_NAME,
								);
							}
							const configBefore = snapshotConfigFile(this.selfConfig.configPath);
							try {
								await session.prompt(
									buildSelfConfigPrompt({
										query: trimmedQuery,
										configPath: this.selfConfig.configPath,
										agentDir: this.piAgentDir ?? getAgentDir(),
										skill: plan.selfConfigSkill,
									}),
								);
								if (
									captured === undefined &&
									!planAbort.signal.aborted &&
									lastModelRequestError(session.piSession?.messages ?? session.agent.state.messages) ===
										undefined
								) {
									await session.prompt(buildFinalEmitReminder());
								}
								const rolledBack = await rollbackIfBroken(this.selfConfig, configBefore);
								if (rolledBack !== undefined) {
									this.routingDiagnostics.push({
										code: "self-config-rolled-back",
										severity: "warning",
										message: `The edited config did not validate, so the previous config was restored. ${rolledBack}`,
										source: "self-config",
									});
									const emitted = captured;
									if (emitted !== undefined) {
										captured = {
											...emitted,
											answer: `${emitted.answer}\n\nWarning: the edited config did not validate, so the previous config was restored (nothing was changed). Problem: ${rolledBack}`,
										};
									}
								}
							} finally {
								// A bound interactive session outlives this run: give it its search tools back.
								if (sessionAgent !== undefined && previousTools !== undefined) {
									sessionAgent.setActiveToolsByName(previousTools);
								}
							}
							return;
						}
						if (plan.route === "direct") {
							// Direct answers skip every retrieval step and the verification
							// phase: the fast answer is the final answer. Only Jev routes
							// here, so the preliminary was never published.
							activateFastPhase();
							await session.prompt(this.buildDirectAnswerPrompt(trimmedQuery));
							const answer =
								fastCaptured?.answer ??
								lastAssistantText(session.piSession?.messages ?? session.agent.state.messages);
							if (answer !== undefined) captured = { answer, results: [], mapping: [], warnings: [] };
							return;
						}
						const baseline =
							plan.route === "web"
								? await this.prefetchWebContext(plan.queries, planAbort.signal)
								: await this.prefetchInitialRetrievalContext(trimmedQuery, plan.queries, options);
						activateFastPhase();
						await session.prompt(this.buildFastAnswerPrompt(trimmedQuery, options, baseline));
						let preliminary = fastCaptured;
						if (preliminary === undefined) {
							const text = lastAssistantText(session.piSession?.messages ?? session.agent.state.messages);
							if (text !== undefined) preliminary = { answer: text, results: [], sources: [] };
						}
						if (preliminary !== undefined) emitPreliminary(preliminary);
						if (captured !== undefined) return;
						// With Jev enabled, a fast answer that needs no correction,
						// clarification, or further research ends the run here.
						if (preliminary !== undefined && !(await this.shouldFollowUp(trimmedQuery, preliminary))) {
							captured = fastAnswerAsFinal(preliminary);
							return;
						}
						if (preliminary !== undefined) publishPreliminary(preliminary);
						if (sessionAgent !== undefined) {
							sessionAgent.setThinkingLevel(clampThinkingLevel(resolved.model, this.finalThinkingLevel));
							sessionAgent.setActiveToolsByName(
								sessionAgent.getActiveToolNames().filter((name) => name !== EMIT_FAST_ANSWER_TOOL_NAME),
							);
						} else {
							session.agent.state.thinkingLevel = clampThinkingLevel(resolved.model, this.finalThinkingLevel);
							session.agent.state.tools = [...this.tools];
						}
						// Only a preliminary consumer actually received may turn the final answer into a delta.
						const fastAnswerDelivered = preliminary !== undefined && this.preliminaryCallback !== undefined;
						await session.prompt(
							this.buildRefinementPrompt(trimmedQuery, options, preliminary, fastAnswerDelivered, plan.route),
						);
						// Models sometimes end verification by writing the final answer as
						// prose instead of calling emit_autorag_results (seen on the web
						// route). One reminder turn lets them emit what they already have;
						// a second miss, a provider error, or an abort (tool-call limit,
						// timeout) falls through to the degraded response.
						if (
							captured === undefined &&
							!planAbort.signal.aborted &&
							this.searchToolCallCount < this.maxSearchToolCalls &&
							lastModelRequestError(session.piSession?.messages ?? session.agent.state.messages) === undefined
						) {
							await session.prompt(buildFinalEmitReminder());
						}
					})(),
					new Promise<never>((_, reject) => {
						timeout = setTimeout(() => {
							planAbort.abort();
							void Promise.resolve(session?.abort());
							reject(new Error(`search timed out after ${this.searchTimeoutMs}ms`));
						}, this.searchTimeoutMs);
					}),
				]);
			} finally {
				if (timeout !== undefined) clearTimeout(timeout);
			}

			let emittedNoVerifiedResults = false;
			if (captured === undefined) {
				if (this.remoteSession) {
					// Remote sessions fail soft: the peer must receive a structured
					// "no verified results" response, never an internal error.
					emittedNoVerifiedResults = true;
					captured = {
						answer: "No verified results were found for this query.",
						results: [],
						mapping: [],
						warnings: [],
					};
				} else {
					// Local sessions resolve with a degraded response that carries the
					// run's retrieval trace instead of throwing away the whole run.
					const reason = lastAssistantText(session?.agent.state.messages ?? []);
					const modelError = lastModelRequestError(session?.agent.state.messages ?? []);
					const response: SearchDocumentsResponse = {
						sessionId,
						query: trimmedQuery,
						results: [],
						answer: buildMissingFinalEmitAnswer(trimmedQuery, reason, this.retrievalTrace, modelError),
						searched: this.retrievalTrace.reduce((total, entry) => total + entry.resultCount, 0),
						warnings: [],
						diagnostics: [
							...this.collectComponentDiagnostics(),
							{
								code: "missing-final-emit",
								severity: "warning",
								message:
									"The agent ended its run without calling emit_autorag_results; returning a degraded response that carries the run's retrieval trace.",
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
			}
			if (this.remoteSession && options.observedSources !== undefined) {
				for (const entry of captured.mapping) options.observedSources.add(entry.source);
			}
			if (this.remoteSession) {
				const scan = scanOutboundPayload(
					[
						captured.answer,
						...captured.results.flatMap((result) => [
							result.summary,
							...result.evidence.map((evidence) => evidence.excerpt),
						]),
					],
					[this.workspaceProjectRoot, ...this.searchPaths].map((root) => resolve(root)),
				);
				if (!scan.ok) throw new RemoteSessionRejectedError(scan.code);
			}
			const componentDiagnostics = this.collectComponentDiagnostics();
			if (emittedNoVerifiedResults) {
				componentDiagnostics.push({
					code: "no-verified-results",
					severity: "info",
					message: "The agent completed without verified results; returning a structured empty response.",
					source: "agent",
				});
			}
			const response = recordStructuredResultsSession(
				sessionId,
				trimmedQuery,
				captured,
				this.sessions,
				this.memory,
				componentDiagnostics,
				{ isolateMemory: selfConfigRun },
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
			this.activeRetrievalOptions = undefined;
			this.preliminaryCallback = undefined;
			if (this.interactiveFastAnswerCallback === emitPreliminary) this.interactiveFastAnswerCallback = undefined;
			this.activeRun = false;
		}
	}

	/**
	 * Stream bounded progress updates while retaining the stable
	 * {@link searchDocuments} promise API. Progress is sourced from model text
	 * deltas, so callers can render it in a TUI, CLI, or parent agent and can
	 * stop by calling {@link abort}.
	 */
	async *searchDocumentsStream(
		query: string,
		options: RetrievalOptions = {},
	): AsyncGenerator<SearchDocumentsStreamEvent, void, void> {
		const queue: SearchDocumentsStreamEvent[] = [];
		let progressBuffer = "";
		let wake: (() => void) | undefined;
		let settled = false;
		const unsubscribe = this.subscribe((event) => {
			if (event.type !== "message_update" || event.assistantMessageEvent.type !== "text_delta") return;
			const text = event.assistantMessageEvent.delta;
			if (text.trim().length === 0) return;
			progressBuffer += text;
			if (!/[.!?。！？]\s*$/u.test(progressBuffer)) return;
			queue.push({
				type: "progress",
				sessionId: this.lastSessionId ?? "",
				query: query.trim(),
				text: progressBuffer,
			});
			progressBuffer = "";
			wake?.();
			wake = undefined;
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
				if (progressBuffer.trim() !== "") {
					queue.push({
						type: "progress",
						sessionId: this.lastSessionId ?? "",
						query: query.trim(),
						text: progressBuffer,
					});
					progressBuffer = "";
				}
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

	private datasourceAccessContext(options: RetrievalOptions = {}): DatasourceAccessContext {
		const effectiveOptions = this.remoteSession ? { ...this.activeRetrievalOptions, ...options } : options;
		const trustedTags = this.datasourceAccessOptions.allowedTags ?? [];
		const requestedTags = effectiveOptions.allowedTags;
		const allowedTags =
			requestedTags === undefined ? trustedTags : trustedTags.filter((tag) => requestedTags.includes(tag));
		return new DatasourceAccessContext({
			allowedTags,
			allowedScopes: this.datasourceAccessOptions.allowedScopes,
		});
	}

	/**
	 * Build the Pi agent-skill list for datasource skills authorized by the
	 * trusted, server-bound access context. Only authorized skills become
	 * model-visible; unauthorized skills are omitted entirely (default-deny).
	 */
	private buildAuthorizedDatasourceSkills(): DatasourceAgentSkill[] {
		const ctx = this.datasourceAccessContext();
		const skills: DatasourceAgentSkill[] = [];
		for (const skill of this.datasourceSkills) {
			if (!ctx.isAccessible(skill.describe())) continue;
			skills.push(toDatasourceAgentSkill(skill.skillManifest()));
		}
		return skills;
	}

	/**
	 * Per-connection tool specs for the generated `search_datasource_<id>`
	 * tools, built from the same trusted config and access context as
	 * {@link buildAuthorizedDatasourceSkills}. One spec per authorized
	 * datasource skill; duplicate ids collapse to the first registration.
	 */
	private singleDatasourceToolSpecs(): SingleDatasourceToolSpec[] {
		const ctx = this.datasourceAccessContext();
		const seen = new Set<string>();
		const specs: SingleDatasourceToolSpec[] = [];
		for (const skill of this.datasourceSkills) {
			const descriptor = skill.describe();
			if (descriptor.datasourceId === undefined) continue;
			if (!ctx.isAccessible(descriptor)) continue;
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
	 * Authorized configured datasource descriptors for catalog/listing surfaces.
	 *
	 * Built from the same trusted, server-bound access context as
	 * {@link buildAuthorizedDatasourceSkills}: only tag-authorized datasources
	 * are listed — including ones that expose no retrieval methods — and each
	 * entry carries only identity, capability tags, and authorized source scope
	 * strings (never credentials, config paths, or raw instance metadata).
	 * Duplicate datasource ids collapse to the first registration.
	 */
	listDatasources(): DatasourceCatalogEntry[] {
		const ctx = this.datasourceAccessContext();
		const seen = new Set<string>();
		const entries: DatasourceCatalogEntry[] = [];
		for (const skill of this.datasourceSkills) {
			const descriptor = skill.describe();
			if (descriptor.datasourceId === undefined) continue;
			if (!ctx.isAccessible(descriptor)) continue;
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
				sourceScopes: this.authorizedSourceScopes(skill, ctx),
			});
		}
		return entries;
	}

	/**
	 * Opaque source scope strings for one datasource that the trusted context
	 * authorizes. Datasources without the `scoped` capability expose their
	 * sources unfiltered (they are gated only at the tag level).
	 */
	private authorizedSourceScopes(skill: DatasourceSkill, ctx: DatasourceAccessContext): string[] {
		const scoped = skill.describe().capabilities.includes("scoped");
		const predicate = scoped ? ctx.allowedSourcesPredicate() : undefined;
		const scopes = new Set<string>();
		for (const source of skill.describeSources()) {
			const scope = source.source;
			if (scope.includes("#")) continue;
			if (predicate !== undefined && !predicate(scope)) continue;
			scopes.add(scope);
		}
		return [...scopes];
	}

	/**
	 * Resolve an authorized datasource agent skill by model-visible name for the
	 * `load_datasource_skill` tool. Returns `undefined` for unknown or
	 * unauthorized names — model/tool input can never widen authorization.
	 */
	loadDatasourceSkill(name: string): DatasourceAgentSkill | undefined {
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
		diagnostics.push(...this.routingDiagnostics);
		return diagnostics;
	}

	/**
	 * Jev check run after emit_fast_answer: does the answer need correction,
	 * clarification, or further research? "No" ends the run with the fast
	 * answer as the final answer. Without Jev, or when the check fails, the run
	 * always continues into verification.
	 */
	private async shouldFollowUp(query: string, fastAnswer: AutoRAGFastAnswerDetails): Promise<boolean> {
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
	 * (local search, web search, or a direct answer) and whether the question
	 * needs decomposition; a "yes" splits it into at most five search queries
	 * with the configured decomposition model (default: the session model).
	 * Without Jev, and on any routing failure, this is today's single local
	 * search for the original question.
	 */
	private async planQuery(
		query: string,
		resolved: {
			readonly model: Model<Api>;
			readonly apiKey?: string;
			readonly providerApiKeys?: Readonly<Record<string, string>>;
		},
		signal: AbortSignal,
	): Promise<{ readonly route: QueryRoute; readonly queries: readonly string[]; readonly selfConfigSkill?: string }> {
		if (this.jevJudge === undefined) return { route: FALLBACK_QUERY_ROUTE, queries: [query] };
		const decision = await routeQuery(this.jevJudge, query, { selfConfig: this.selfConfig !== undefined });
		if (decision.fallbackReason !== undefined) {
			this.routingDiagnostics.push({
				code: "query-route-fallback",
				severity: "warning",
				message: `Jev query routing was unavailable; searching local sources with the original question. ${decision.fallbackReason}`,
				source: "jev",
			});
			return { route: FALLBACK_QUERY_ROUTE, queries: [query] };
		}
		let route = decision.route;
		if (route === "web" && this.webSearchOptions === undefined) {
			this.routingDiagnostics.push({
				code: "query-route-fallback",
				severity: "info",
				message:
					"Jev routed the question to web search, but web tools are disabled; searching local sources instead.",
				source: "jev",
			});
			route = FALLBACK_QUERY_ROUTE;
		}
		let selfConfigSkill: string | undefined;
		if (route === "config") {
			try {
				selfConfigSkill = loadSetupSkill(this.selfConfig?.skillPath);
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
		let queries: readonly string[] = [query];
		if (decision.decompose && route !== "direct") {
			const target = this.queryDecompositionModel ?? {
				model: resolved.model,
				...((resolved.apiKey ?? resolved.providerApiKeys?.[resolved.model.provider]) !== undefined
					? { apiKey: resolved.apiKey ?? resolved.providerApiKeys?.[resolved.model.provider] }
					: {}),
			};
			try {
				queries = await decomposeQuery(createModelDecompositionCompleter(target, signal), query);
			} catch (error) {
				this.routingDiagnostics.push({
					code: "query-decomposition-failed",
					severity: "warning",
					message: `Question decomposition failed; searching with the original question. ${error instanceof Error ? error.message : String(error)}`,
					source: "query-decomposition",
				});
			}
		}
		const probability = decision.routeProbability === undefined ? "" : ` (p=${decision.routeProbability.toFixed(2)})`;
		this.routingDiagnostics.push({
			code: "query-routed",
			severity: "info",
			message:
				`Jev routed the question to ${route}${probability}; ` +
				(route === "direct"
					? "answering directly without retrieval."
					: route === "config"
						? "configuring AutoRAG with the full setup skill; no retrieval and no fast answer."
						: `searching with ${queries.length} ${queries.length === 1 ? "query" : "queries"}: ${queries.map((entry) => JSON.stringify(entry)).join(", ")}.`),
			source: "jev",
		});
		return { route, queries, ...(selfConfigSkill !== undefined ? { selfConfigSkill } : {}) };
	}

	/**
	 * Baseline local evidence for the fast answer. Jikji and MinSync run for
	 * every search query in parallel (MinSync itself queues per workspace), the
	 * per-query hits are interleaved and deduplicated into one pool, and that
	 * pool is reranked against the original question.
	 */
	private async prefetchInitialRetrievalContext(
		query: string,
		searchQueries: readonly string[],
		options: RetrievalOptions,
	): Promise<string> {
		const retrieveOptions = { topK: this.limits.prefetch.minSyncTopK, scope: options.scope };
		// Queries only read the prebuilt MinSync index; an unbuilt workspace makes
		// `minsync query` fail fast and this source contributes nothing.
		const formatSearchQueries = (list: readonly string[]): string =>
			`Search queries (decomposed from the original question):\n${list.map((entry, index) => `[${index + 1}] ${entry}`).join("\n")}`;
		const perQuery = await Promise.all(
			searchQueries.map((searchQuery) =>
				Promise.all([
					this.jikjiClient === undefined
						? Promise.resolve(undefined)
						: this.findJikji(searchQuery, { topK: this.limits.prefetch.jikjiTopK }).catch(() => undefined),
					this.minSyncMethod === undefined
						? Promise.resolve([])
						: this.minSyncMethod.retrieve(searchQuery, retrieveOptions).catch(() => []),
				]),
			),
		);
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
		const minSyncResults = interleave(perQuery.map(([, vector]) => vector)).filter((result) => {
			const key = `${result.source}\0${result.content}`;
			if (seenChunks.has(key)) return false;
			seenChunks.add(key);
			return true;
		});
		for (const result of minSyncResults) options.observedSources?.add(result.source);
		// Rerank the whole merged pre-fast-answer pool (every query's Jikji paths
		// and MinSync chunks) against the original question, so decomposed
		// sub-query hits compete on relevance to what the user asked. Falls back
		// to the unranked sections when reranking is disabled or unavailable.
		const reranked = await this.rerankPrefetchPool(query, jikjiPaths, minSyncResults);
		if (reranked !== undefined) {
			const baseline = formatRerankedBaseline(reranked);
			return searchQueries.length > 1 ? `${formatSearchQueries(searchQueries)}\n\n${baseline}` : baseline;
		}

		// One flat numbering across every section (issue #1788): candidate [n]
		// labels never restart, so no number means two different candidates.
		const sections: string[] = [];
		if (searchQueries.length > 1) sections.push(formatSearchQueries(searchQueries));
		let candidateNumber = 0;
		if (jikjiFound) {
			sections.push(
				`Jikji initial candidates (preserve order when agent_should_not_rerank=true):\n${jikjiPaths
					.slice(0, this.limits.prefetch.jikjiPathLimit)
					.map((path) => `[${++candidateNumber}] ${path}`)
					.join("\n")}`,
			);
		}
		if (minSyncResults.length > 0) {
			sections.push(
				`MinSync semantic initial candidates:\n${minSyncResults
					.slice(0, this.limits.prefetch.sectionLimit)
					.map((result) => `[${++candidateNumber}] ${result.source}\n${result.content.replace(/\s+/gu, " ")}`)
					.join("\n")}`,
			);
		}
		return sections.length === 0
			? "No initial retrieval candidates were available; use the configured tools and report degradation honestly."
			: sections.join("\n\n");
	}

	/** Baseline web evidence for the fast answer: one web search per query, all in parallel. */
	private async prefetchWebContext(queries: readonly string[], signal: AbortSignal): Promise<string> {
		const web = this.webSearchOptions ?? {};
		const modelAuth = web.modelAuth?.();
		const searches = await Promise.all(
			queries.map((query) =>
				executeWebSearch(
					{ query, ...(web.provider !== undefined ? { provider: web.provider } : {}) },
					{
						signal,
						...(web.timeoutSeconds !== undefined ? { timeoutMs: web.timeoutSeconds * 1_000 } : {}),
						...(web.order !== undefined ? { order: web.order } : {}),
						...(web.exclude !== undefined ? { exclude: web.exclude } : {}),
						...(modelAuth !== undefined ? { modelAuth } : {}),
					},
				).catch((error: unknown) => ({
					content: [],
					details: {
						response: { provider: "none" as const, sources: [] },
						error: error instanceof Error ? error.message : String(error),
					},
				})),
			),
		);
		const sections = searches.map((search, index) => {
			const label = `Web search results for ${JSON.stringify(queries[index])}`;
			return search.details.error !== undefined
				? `${label}: unavailable (${search.details.error})`
				: `${label}:\n${search.content.map((part) => part.text).join("\n")}`;
		});
		return searches.every((search) => search.details.error !== undefined)
			? `${sections.join("\n\n")}\n\nNo web evidence was available; use the configured tools and report degradation honestly.`
			: sections.join("\n\n");
	}

	/**
	 * Rerank the pre-fast-answer baseline pool — Jikji answer paths plus MinSync
	 * chunks — down to the configured `rerank.topN`. With decomposition the pool
	 * merges every search query's hits (interleaved), capped at the same
	 * per-source sizes as a single query so the rerank request does not grow
	 * with the query count. Returns `undefined` when reranking is
	 * disabled/unavailable or the pool is empty, so the caller keeps the
	 * unranked sections; a rerank failure never blocks the fast answer.
	 */
	private async rerankPrefetchPool(
		query: string,
		answerPaths: readonly string[],
		minSyncResults: readonly RetrievalResult[],
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
		if (candidates.length === 0) return undefined;
		if (!reranker.describe().available) return undefined;
		try {
			const reranked = await reranker.rerank(query, candidates, { topN: this.rerankTopN });
			return reranked.length > 0 ? reranked : undefined;
		} catch {
			return undefined;
		}
	}

	/**
	 * Fast-phase prompt for two-phase searches. The baseline retrieval context
	 * is already gathered, so the model answers immediately with thinking off
	 * via emit_fast_answer, without any further tool calls.
	 */
	buildFastAnswerPrompt(query: string, options: RetrievalOptions, baseline: string): string {
		const limit = typeof options.topK === "number" ? ` Return at most ${options.topK} knowledge units.` : "";
		const scope = options.scope ? ` Restrict search to virtual path scope ${options.scope}.` : "";
		return (
			`Answer this original query immediately: ${query}${limit}${scope}\n\n` +
			`Baseline retrieval evidence (already gathered for you):\n${baseline}\n\n` +
			`Produce the best complete, self-contained answer you can RIGHT NOW from this evidence. Do NOT call any search, retrieval, or file-reading tools and do NOT wait for more evidence. ` +
			`If the query is answerable from general knowledge alone, answer directly.\n\n` +
			`Formatting and content rules for the answer:\n` +
			`- Provide the core answer to the user's question in at most 5 bullet points. If additional explanation is necessary, append it after the bullet points.\n` +
			`- Answer the question directly. Do not include specific file paths, datasource descriptions, or retrieval mechanics in the answer text.\n` +
			`- Cite evidence with bracketed numbers only (e.g. [1], [2]); do not quote raw chunks or mention source paths directly in the answer.\n` +
			`- ${ANSWER_CITATION_RULE}\n` +
			`- ${ANSWER_IMAGE_EMBED_RULE}\n` +
			`- Do not report per-source negative findings (e.g. "no information found in Slack" or "checked Drive but found nothing").\n` +
			`- When evidence conflicts, treat the freshest (most recent) information as the correct source of truth.\n` +
			`- If information is incomplete or uncertain, acknowledge it briefly without lengthy explanations, stating that it is difficult to answer fully with the given information and searching continues. If there are partial clues or leads (even if not the exact answer), mention those clues concisely.\n\n` +
			`Call emit_fast_answer exactly once with the answer, its numbered knowledge units, and their real source paths, then stop.`
		);
	}

	/**
	 * Prompt for a question Jev routed to a direct answer: general knowledge or
	 * small talk. No retrieval ran and no verification phase follows, so the
	 * answer emitted here is final.
	 */
	buildDirectAnswerPrompt(query: string): string {
		return (
			`Answer this query directly from your own general knowledge: ${query}\n\n` +
			`It needs no search: it is general knowledge, simple reasoning, or conversation. Do NOT call any search, retrieval, web, or file-reading tools. ` +
			`Reply naturally and concisely; for small talk, just respond conversationally. Do not cite sources or mention retrieval.\n\n` +
			`Call emit_fast_answer exactly once with the answer and an empty results list, then stop.`
		);
	}

	/** Discovery-tool hint naming only the registered discovery tools; empty when none. */
	private discoveryHint(sentence: (tools: string) => string): string {
		const names = this.tools
			.map((tool) => tool.name)
			.filter((name) => name === JIKJI_FIND_TOOL_NAME || name === EVERYTHING_SEARCH_TOOL_NAME);
		return names.length === 0 ? "" : sentence(names.join(" and "));
	}

	/**
	 * Verification-phase prompt for two-phase searches. The fast answer, when a
	 * consumer already received it, is embedded verbatim so the model can diff
	 * against it; the model then verifies with thinking on and finalizes with
	 * emit_autorag_results exactly once, returning only the delta.
	 */
	buildRefinementPrompt(
		query: string,
		options: RetrievalOptions,
		fastAnswer: AutoRAGFastAnswerDetails | undefined,
		fastAnswerDelivered: boolean,
		route: QueryRoute = "local",
	): string {
		const limit = typeof options.topK === "number" ? ` Return at most ${options.topK} curated results.` : "";
		const scope = options.scope ? ` Restrict search to virtual path scope ${options.scope}.` : "";
		const firstAnswer = formatFirstAnswerContext(fastAnswer, fastAnswerDelivered);
		const answerRules = fastAnswerDelivered
			? `Formatting and content rules for the final \`answer\` (DELTA ONLY):\n` +
				`- The user already has the first answer above. \`answer\` MUST contain only the delta against it: (a) corrections to anything in the first answer that is wrong, unsupported, or outdated, and (b) newly verified findings that were not present in the first answer.\n` +
				`- NEVER restate or re-list first-answer facts that remain correct, and never repeat its bullet list.\n` +
				`- Mark each item clearly as a correction or as a new finding.\n` +
				`- If verification changed nothing and found nothing new, say so in one short line (the first answer is confirmed as-is) instead of restating it.\n` +
				`- Cite evidence with bracketed numbers only (e.g. [1], [2]); do not quote raw chunks or mention source paths directly in the answer.\n` +
				`- ${ANSWER_CITATION_RULE} A correction or new finding that relies on a first-answer unit must re-emit that evidence as a result of this call and cite its new number.\n` +
				`- ${ANSWER_IMAGE_EMBED_RULE} ${ANSWER_IMAGE_DELTA_RULE}\n` +
				`- Do not report per-source negative findings (e.g. "no information found in Slack").\n` +
				`- When evidence conflicts, treat the freshest (most recent) information as the correct source of truth.`
			: `Formatting and content rules for the final answer (COMPLETE — no first answer reached the caller):\n` +
				`- Provide the core answer to the user's question in at most 5 bullet points. If additional explanation is necessary, append it after the bullet points.\n` +
				`- Answer the question directly. Do not include specific file paths, datasource descriptions, or retrieval mechanics in the answer text.\n` +
				`- Cite evidence with bracketed numbers only (e.g. [1], [2]); do not quote raw chunks or mention source paths directly in the answer.\n` +
				`- ${ANSWER_CITATION_RULE}\n` +
				`- ${ANSWER_IMAGE_EMBED_RULE}\n` +
				`- Do not report per-source negative findings (e.g. "no information found in Slack").\n` +
				`- When evidence conflicts, treat the freshest (most recent) information as the correct source of truth.`;
		return (
			`Original query: ${query}${limit}${scope}\n\n` +
			`${firstAnswer}\n\n` +
			(route === "web"
				? `Now verify it rigorously on the internet: Jev routed this question to web search, so the answer lives in public web sources, not in local files. Use ${WEB_SEARCH_TOOL_NAME} (and ${WEB_FETCH_TOOL_NAME} to read a promising page) to confirm or correct each claim, fill gaps with focused web queries, and resolve conflicts and freshness. Use URLs as result sources. `
				: `Now verify it rigorously. ${this.discoveryHint((tools) => `Actively use ${tools} when discovering or exploring local files and folders. `)}Check important claims against source files with bash when needed, correct anything wrong or unsupported, fill gaps with retrieval tools, and resolve conflicts and freshness. `) +
			`Preserve real source paths and evidence excerpts in the result mapping.\n\n` +
			`${answerRules}\n\n` +
			`Do not use broad grep/find or recursive filesystem scans: only inspect a path or narrow neighborhood surfaced by retrieval, and only when evidence clearly points there. ` +
			`Avoid spinning repeated near-identical queries against the same datasource; once additional attempts stop surfacing new evidence, conclude from the evidence available. ` +
			`If more search is needed, first write a brief 1\u20132 line progress update stating the best current hypothesis and what you are checking next, then call retrieval tools. ` +
			`When finished, call ${EMIT_AUTORAG_RESULTS_TOOL_NAME} exactly once as your final action with the curated ` +
			`results and the internal number-to-source mapping.`
		);
	}

	buildSearchPrompt(query: string, options: RetrievalOptions, initialRetrievalContext?: string): string {
		const limit = typeof options.topK === "number" ? ` Return at most ${options.topK} curated results.` : "";
		const scope = options.scope ? ` Restrict search to virtual path scope ${options.scope}.` : "";
		return (
			`Find and curate information for this original query: ${query}${limit}${scope}\n\n` +
			`Start by deciding whether this is answerable from general knowledge or memory. If it is a generic, stable question, answer it immediately without retrieval and emit the structured result. ` +
			`Otherwise, baseline MinSync and Jikji retrieval is already running in parallel; do not emit final results until its next message arrives.\n\n` +
			`Baseline retrieval context:\n${initialRetrievalContext ?? "Pending; continue only with a brief progress statement."}\n\n` +
			`Treat candidates as unverified evidence, verify important claims against source files when needed, and use additional tools when needed. ` +
			`Judge relevance, conflicts, freshness, and sufficiency in this agent loop. Preserve real source paths and evidence excerpts in the result mapping.\n\n` +
			`Formatting and content rules for the answer:\n` +
			`- Provide the core answer to the user's question in at most 5 bullet points. If additional explanation is necessary, append it after the bullet points.\n` +
			`- Answer the question directly. Do not include specific file paths, datasource descriptions, or retrieval mechanics in the answer text.\n` +
			`- Cite evidence with bracketed numbers only (e.g. [1], [2]); do not quote raw chunks or mention source paths directly in the answer.\n` +
			`- ${ANSWER_CITATION_RULE}\n` +
			`- ${ANSWER_IMAGE_EMBED_RULE}\n` +
			`- Do not report per-source negative findings (e.g. "no information found in Slack").\n` +
			`- When evidence conflicts, treat the freshest (most recent) information as the correct source of truth.\n` +
			`- If information is incomplete or uncertain, acknowledge it briefly without lengthy explanations. If there are partial clues or leads, mention them concisely.\n\n` +
			`${this.discoveryHint((tools) => `When exploring local files and folders, actively use ${tools} rather than exploratory bash commands. `)}If more search is needed, first write a brief 1–2 line progress update stating the best current hypothesis and what you are checking next, then call retrieval tools. ` +
			`Never repeat a generic status message. Do not use broad grep/find or recursive filesystem scans: only inspect a path or narrow neighborhood surfaced by retrieval, and only when evidence clearly points there. ` +
			`Avoid spinning repeated near-identical queries against the same datasource; once additional attempts stop surfacing new evidence, conclude from the evidence available. ` +
			`When finished, call ${EMIT_AUTORAG_RESULTS_TOOL_NAME} exactly once as your final action with the curated ` +
			`results and the internal number-to-source mapping.`
		);
	}

	async refresh(force = false, opts?: AutoRAGRefreshOptions): Promise<AutoRAGRefreshResult> {
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
		writeRefreshProgress(this.workspaceProjectRoot, progress);
		this.refreshState = {
			...this.refreshState,
			inFlight: true,
			lastStartedAt: startedAt,
		};
		try {
			const summary = needsParsed ? await this.syncParsedMirrors(force) : await this.scanMirrorStaleness();
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
			const minsync = wants("minsync") ? await this.syncMinSync(force) : undefined;
			if (minsync !== undefined) {
				progress = updateRefreshProgress(progress, {
					phase: wants("datasources") ? "datasources" : wants("jikji") ? "jikji" : "finalizing",
					minsync: { synced: minsync.synced },
				});
				writeRefreshProgress(this.workspaceProjectRoot, progress);
			}
			const datasources = wants("datasources") ? await this.indexDatasources() : [];
			if (wants("datasources")) {
				progress = updateRefreshProgress(progress, {
					phase: wants("jikji") || allMethods ? "jikji" : "finalizing",
				});
				writeRefreshProgress(this.workspaceProjectRoot, progress);
			}
			// A MinSync refresh also establishes Jikji's local discovery artifacts:
			// both indexes are first-class parts of the default local corpus.
			const jikji = allMethods || wants("jikji") || wants("minsync") ? await this.executeJikjiPrepare() : undefined;
			// On Windows, the bundled Everything instance indexes the same local
			// roots by name so the agent can locate files instantly.
			let everything: AutoRAGEverythingRefreshResult | undefined;
			if (this.everythingClient !== undefined && (allMethods || wants("everything") || needsParsed)) {
				progress = updateRefreshProgress(progress, { phase: "everything" });
				writeRefreshProgress(this.workspaceProjectRoot, progress);
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
				lastError: error instanceof Error ? `Refresh failed: ${error.name}` : "Refresh failed.",
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
			root: this.workspaceProjectRoot,
			searchPaths: this.searchPaths,
			parserOptions: this.parserOptions,
			userExcludedSourcePaths: new Set(this.excludePaths),
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
		if (this.minSyncMethod !== undefined) {
			status.minsync = this.minSyncMethod.isExplicitBinaryMissing()
				? "unavailable"
				: this.refreshState.minsync?.ok === false
					? "degraded"
					: this.minSyncMethod.isReady()
						? "ready"
						: "configured";
		}
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

	async syncParsedMirrors(force = false): Promise<ParsedMirrorSyncResult> {
		const duplicateFilter = await this.exactDuplicateExclusions();
		return syncParsedMirrors({
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

	async syncMinSync(force = false): Promise<MinSyncSyncResult | undefined> {
		const result = await this.minSyncMethod?.sync(force);
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

	async prepareJikji(): Promise<readonly AutoRAGJikjiPrepareResult[] | undefined> {
		const results = await this.executeJikjiPrepare();
		return results?.map((result) => this.sanitizeJikjiPrepareResult(result));
	}

	/**
	 * Build or incrementally update every root's Jikji index. Called only by
	 * refresh/watch (and the explicit `prepareJikji()` API), never by a query:
	 * `jikji prepare` reuses unchanged documents, and roots run in parallel so
	 * one slow root does not serialize the rest.
	 */
	private async executeJikjiPrepare(): Promise<readonly JikjiPrepareResult[] | undefined> {
		const client = this.jikjiClient;
		if (client === undefined) return undefined;
		return Promise.all(this.searchPaths.map((sourcePath) => client.prepare(sourcePath)));
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
		const searchResults = await Promise.all(
			this.searchPaths.map(async (sourcePath) => {
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
		if (this.remoteSession) options = { ...this.activeRetrievalOptions, ...options };
		options = this.normalizeRetrievalOptions(options);
		const methods = this.methodRegistry.list();
		const { results: byMethod, diagnostics } = await this.retriever.retrieveWithDiagnostics(methods, query, options);
		const filteredByMethod = this.datasourceFilter.filter(
			byMethod,
			methods,
			this.datasourceAccessContext(options),
			options.scope,
			options.allowedScopes,
		);
		for (const results of filteredByMethod.values()) {
			for (const result of results) options.observedSources?.add(result.source);
		}
		if (this.minSyncMethod?.isBinaryMissing() && !diagnostics.some((d) => d.source === "minsync")) {
			diagnostics.push({
				code: "minsync-unavailable",
				severity: "warning",
				message: "MinSync semantic search is unavailable; results rely on other retrieval paths.",
				source: "minsync",
			});
		}
		const merged = this.rerankWithMemory(
			query,
			this.merger.merge(filteredByMethod, {
				topK: options.topK ?? this.limits.mergedEvidenceCeiling,
				dedup: true,
			}),
		);
		const results = await this.applyRerank(query, merged, diagnostics);
		return { results, diagnostics };
	}

	async searchAllDocuments(
		query: string,
		options: { readonly topK?: number; readonly scope?: string } = {},
	): Promise<{ results: RetrievalResult[]; diagnostics: RetrievalDiagnostic[] }> {
		return this.retrieveWithDiagnostics(query, { topK: options.topK, scope: options.scope });
	}

	/**
	 * Search one datasource connection only. Only the target connection's
	 * retrieval methods are registered with the retriever, so no other
	 * datasource CLI is spawned at all and only that connection's hits are
	 * returned. Access is still gated by the trusted datasource context: an
	 * unknown or unauthorized `datasourceId` yields an empty result set.
	 *
	 * This backs the generated `search_datasource_<id>` tools; every authorized
	 * connection has one. Cross-datasource fan-out is
	 * {@link searchAllDocuments}, which spans every retrieval method.
	 */
	async searchSingleDatasourceDocuments(
		datasourceId: string,
		query: string,
		options: { readonly topK?: number; readonly scope?: string } = {},
	): Promise<{ results: RetrievalResult[]; diagnostics: RetrievalDiagnostic[] }> {
		const retrievalOptions: RetrievalOptions = {
			...this.activeRetrievalOptions,
			topK: options.topK,
			scope: options.scope,
		};
		const ctx = this.datasourceAccessContext(retrievalOptions);
		const methods = this.methodRegistry.list().filter((method) => {
			const descriptor = method.describe();
			return descriptor.datasourceId === datasourceId && ctx.isAccessible(descriptor);
		});
		if (methods.length === 0) return { results: [], diagnostics: [] };
		const { results: byMethod, diagnostics } = await this.retriever.retrieveWithDiagnostics(
			methods,
			query,
			retrievalOptions,
		);
		const filteredByMethod = this.datasourceFilter.filter(byMethod, methods, ctx, options.scope);
		for (const results of filteredByMethod.values()) {
			for (const result of results) retrievalOptions.observedSources?.add(result.source);
		}
		const merged = this.rerankWithMemory(
			query,
			this.merger.merge(filteredByMethod, {
				topK: options.topK ?? this.limits.singleDatasourceTopK,
				dedup: true,
			}),
		);
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
	 * datasource access context. Model-free — no agent state required.
	 */
	private retrievalEngine: RetrievalEngine | undefined;
	getRetrievalEngine(): RetrievalEngine {
		if (this.retrievalEngine === undefined) {
			this.retrievalEngine = new RetrievalEngine({
				datasourceAccess: this.datasourceAccessOptions,
				defaultTopK: this.limits.mergedEvidenceCeiling,
				...(this.reranker !== undefined ? { reranker: this.reranker } : {}),
				...(this.rerankTopN !== undefined ? { rerankTopN: this.rerankTopN } : {}),
				isMinSyncBinaryMissing:
					this.minSyncMethod !== undefined ? () => this.minSyncMethod!.isBinaryMissing() : undefined,
				authorizedDatasourceIds: () => this.listDatasources().map((entry) => entry.datasourceId),
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

	private remoteFilteredRetrievalMethod<
		T extends {
			retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]>;
			describe(): { name: string };
		},
	>(method: T | undefined): T | undefined {
		if (!method) return method;
		return new Proxy(method, {
			get: (target, property, receiver) => {
				if (property !== "retrieve") return Reflect.get(target, property, receiver);
				return async (query: string, options: RetrievalOptions) => {
					const effective = { ...this.activeRetrievalOptions, ...options };
					const results = await target.retrieve.call(target, query, effective);
					for (const result of results) effective.observedSources?.add(result.source);
					return results;
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
 * Baseline block for the reranked pre-fast-answer pool. A single numbering
 * sequence replaces the per-section numbering so bracketed citations are
 * unambiguous in the fast-answer prompt.
 */
function formatRerankedBaseline(results: readonly RetrievalResult[]): string {
	const lines = results.map(
		(result, index) => `[${index + 1}] ${result.source}\n${result.content.replace(/\s+/gu, " ")}`,
	);
	return `Reranked initial candidates (ordered by relevance to the query):\n${lines.join("\n")}`;
}

/**
 * The fast answer as a final emit_autorag_results payload, used when Jev ends
 * the run after the fast phase. Every result the answer cites is kept: the
 * fast-answer `sources` mapping is optional and models routinely omit it, so
 * dropping source-less results would strip the answer's citations. A result
 * with no reported source keeps an empty mapping source (the response then
 * carries no `source` for it) rather than an invented path.
 */
function fastAnswerAsFinal(fastAnswer: AutoRAGFastAnswerDetails): AutoRAGResultsDetails {
	const sourceByNumber = new Map(fastAnswer.sources.map((entry) => [entry.number, entry.source]));
	return {
		answer: fastAnswer.answer,
		results: fastAnswer.results.map((result) => ({
			number: result.number,
			title: result.title,
			summary: result.summary,
			evidence: result.evidence,
			confidence: result.confidence ?? 0.5,
		})),
		mapping: fastAnswer.results.map((result) => ({
			number: result.number,
			source: sourceByNumber.get(result.number) ?? "",
			method: EMIT_FAST_ANSWER_TOOL_NAME,
			content: result.evidence.map((evidence) => evidence.excerpt).join("\n") || result.summary,
			evidenceRefs: [],
		})),
		warnings: [],
	};
}

/**
 * Render the fast-phase first answer for the verification prompt. When a
 * consumer already received it, the model must diff against it and return only
 * the delta; otherwise the draft is internal context only and the final answer
 * must stay complete.
 */
function formatFirstAnswerContext(fastAnswer: AutoRAGFastAnswerDetails | undefined, delivered: boolean): string {
	if (fastAnswer === undefined) {
		return "(the fast phase produced no answer — write the complete answer)";
	}
	const header = delivered
		? "The user has ALREADY received this immediate first answer:"
		: "An internal first-pass draft was produced but was NOT shown to the caller; write the complete answer:";
	const units =
		fastAnswer.results.length === 0
			? ""
			: `\n\nNumbered units of that first answer (its own numbering — NOT citation numbers for your final answer; cite only the results you emit):\n${fastAnswer.results
					.map((result) => `[${result.number}] ${result.title} — ${result.summary}`)
					.join("\n")}`;
	const sources =
		fastAnswer.sources.length === 0
			? ""
			: `\n\nSources of that first answer:\n${fastAnswer.sources
					.map((entry) => `[${entry.number}] -> ${entry.source}`)
					.join("\n")}`;
	return `${header}\n${fastAnswer.answer}${units}${sources}`;
}

/**
 * Degraded fallback answer for a run that ended without emit_autorag_results:
 * states the search-range failure, carries the agent's own last note as the
 * reason, suggests next steps, and points at the attached retrieval trace.
 */
function buildMissingFinalEmitAnswer(
	query: string,
	reason: string | undefined,
	trace: readonly SearchDocumentRetrievalTraceEntry[],
	modelError?: string,
): string {
	const lines = [
		`The search run for "${query}" ended without finalized results: the agent ended its run without calling emit_autorag_results, so no curated answer is available within the configured search range.`,
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
 * One-turn reminder sent when the verification phase stops without calling
 * emit_autorag_results. It asks only for the structured emit of the answer the
 * model already reached, never for more searching.
 */
function buildFinalEmitReminder(): string {
	return (
		`You ended without calling ${EMIT_AUTORAG_RESULTS_TOOL_NAME}, so the user has not received your verified answer. ` +
		`Do not search again. Call ${EMIT_AUTORAG_RESULTS_TOOL_NAME} now, exactly once, with the answer you just wrote, ` +
		`its numbered results, and the number-to-source mapping (use the real file paths or URLs you used). ` +
		`If verification found nothing usable, call it with an answer that says so and an empty results list.`
	);
}

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
	if (minsync.diagnostic) return [toMinSyncDiagnostic(minsync.diagnostic, minsync.ok)];
	if (!minsync.ok && minsync.reason) return [toMinSyncReasonDiagnostic(minsync.reason)];
	return [];
}

function toMinSyncDiagnostic(diag: MinSyncDiagnostic, ok: boolean): SearchDocumentDiagnostic {
	const code: SearchDocumentDiagnosticCode =
		diag.code === "embedder-unavailable"
			? "embedder-unavailable"
			: diag.code === "embedding-identity-mismatch"
				? "embedding-identity-mismatch"
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
		code: reason === "missing-binary" ? "minsync-unavailable" : "minsync-sync-failed",
		severity: reason === "missing-binary" ? "warning" : "error",
		message:
			reason === "missing-binary"
				? "MinSync binary is not available and auto-install was skipped."
				: `MinSync sync failed: ${sanitizeDiagnosticMessage(reason)}`,
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

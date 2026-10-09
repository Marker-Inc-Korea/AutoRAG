import { randomUUID } from "node:crypto";
import type {
	AutoRAGAgentOptions,
	AutoRAGJikjiPrepareResult,
	AutoRAGRefreshOptions,
	AutoRAGRefreshResult,
	AutoRAGRefreshStatus,
	AutoRAGWatchRefreshHandle,
	AutoRAGWatchRefreshOptions,
} from "./agent/agent.ts";
import { AutoRAGAgent } from "./agent/agent.ts";
import type { AutoRAGResultsDetails } from "./agent/emit-results-tool.ts";
import {
	recordStructuredResultsSession as persistStructuredResultsSession,
	type SearchDocumentsResponse,
} from "./agent/search-documents.ts";
import { buildAgentOptions, type CliConfig, type ResolveConfigInput, resolveConfig } from "./cli/config.ts";
import type { EverythingSearchRequest, EverythingSearchResult } from "./everything/index.ts";
import type { FSearchSearchRequest, FSearchSearchResult } from "./fsearch/index.ts";
import type { MemorySchemaV4 } from "./memory/memory.ts";
import { RetrievalMemory } from "./memory/memory.ts";
import type { MinSyncSyncResult } from "./minsync/types.ts";
import type { ParsedMirrorSyncResult } from "./mirror/sync.ts";
import type { RetrievalEngine } from "./retrieval/engine.ts";
import type { DatasourceCatalogEntry, RetrievalSelection } from "./retrieval/selection.ts";
import type {
	CuratedResult,
	RetrievalDiagnostic,
	RetrievalOptions,
	RetrievalResult,
	RetrievalUnsearchedSurface,
} from "./retrieval/types.ts";

/** Input accepted by {@link createAutoRAGLite}. Flags keep CLI precedence intact. */
export interface AutoRAGLiteOptions {
	readonly flags?: ResolveConfigInput["flags"];
	readonly env?: NodeJS.ProcessEnv;
	readonly cwd?: string;
	readonly readOnly?: boolean;
}

/** The model-free AutoRAG runtime facade. */
export class AutoRAGLite {
	readonly config: CliConfig;
	private readonly agent: AutoRAGAgent;
	private readonly retrievalEngine: RetrievalEngine;
	private readonly memory: RetrievalMemory;
	private readonly sessions = new Map<string, { query: string; registry: Map<number, CuratedResult> }>();

	constructor(config: CliConfig) {
		this.config = config;
		const agentOptions: Omit<AutoRAGAgentOptions, "model"> = buildAgentOptions(config);
		this.agent = new AutoRAGAgent(agentOptions);
		this.retrievalEngine = this.agent.getRetrievalEngine();
		this.memory = new RetrievalMemory({ storagePath: config.memoryPath });
		this.memory.load();
	}

	/** Refresh parsed mirrors and configured indexes without constructing a model. */
	refresh(force = false, options?: AutoRAGRefreshOptions): Promise<AutoRAGRefreshResult> {
		return this.agent.refresh(force, options);
	}

	/** Alias for callers that model indexing as a lifecycle operation. */
	index(force = false, options?: AutoRAGRefreshOptions): Promise<AutoRAGRefreshResult> {
		return this.refresh(force, options);
	}

	/** Return the path-opaque refresh and index health snapshot. */
	getRefreshStatus(): Promise<AutoRAGRefreshStatus> {
		return this.agent.getRefreshStatus();
	}

	/** Short status alias for headless lifecycle callers. */
	status(): Promise<AutoRAGRefreshStatus> {
		return this.getRefreshStatus();
	}

	/** Run only parsed-mirror indexing while preserving the agent's path pinning. */
	syncParsedMirrors(force = false): Promise<ParsedMirrorSyncResult> {
		return this.agent.syncParsedMirrors(force);
	}

	/** Run the configured MinSync indexer without constructing a model. */
	syncMinSync(): Promise<MinSyncSyncResult | undefined> {
		return this.agent.syncMinSync();
	}

	/** Run the configured Jikji preparer without constructing a model. */
	prepareJikji(): Promise<readonly AutoRAGJikjiPrepareResult[] | undefined> {
		return this.agent.prepareJikji();
	}

	/** Start an opt-in incremental filesystem refresh watcher. */
	startWatchRefresh(options?: AutoRAGWatchRefreshOptions): AutoRAGWatchRefreshHandle {
		return this.agent.startWatchRefresh(options);
	}

	/** Return the deterministic retrieval engine used by this configured runtime. */
	getRetrievalEngine(): RetrievalEngine {
		return this.retrievalEngine;
	}

	/**
	 * List the configured datasources for this runtime: identity, descriptive
	 * tags, and source scope strings only — no credentials or config metadata.
	 */
	listDatasources(): DatasourceCatalogEntry[] {
		return this.agent.listDatasources();
	}

	/**
	 * Search only the selected datasources/methods/local surfaces. Selection is
	 * applied before any backend is invoked, so an unselected datasource never
	 * runs. Unknown methods or unknown datasource ids throw
	 * {@link RetrievalSelectionError}.
	 */
	searchSelected(
		query: string,
		selection: RetrievalSelection = {},
		options?: RetrievalOptions,
	): Promise<{
		results: RetrievalResult[];
		diagnostics: RetrievalDiagnostic[];
		unsearched: RetrievalUnsearchedSurface[];
	}> {
		return this.retrievalEngine.retrieveSelected(query, selection, options);
	}

	/**
	 * Search the Windows-only Everything index through the model-free provider,
	 * delegating to the existing agent backend (unsupported off Windows).
	 */
	searchEverything(request: EverythingSearchRequest): Promise<EverythingSearchResult> {
		return this.agent.searchEverything(request);
	}

	/** Search the existing macOS/Linux FSearch client through the model-free provider. */
	searchFsearch(request: FSearchSearchRequest): Promise<FSearchSearchResult> {
		return this.agent.searchFsearch(request);
	}

	/** Retrieve merged results without entering the model-backed agent loop. */
	retrieve(
		query: string,
		options?: RetrievalOptions,
	): Promise<{
		results: RetrievalResult[];
		diagnostics: RetrievalDiagnostic[];
		unsearched: RetrievalUnsearchedSurface[];
	}> {
		return this.retrievalEngine.retrieve(query, options);
	}

	/** Retrieve per-method results without entering the model-backed agent loop. */
	retrieveByMethod(
		query: string,
		options?: RetrievalOptions,
	): Promise<{
		byMethod: Map<string, RetrievalResult[]>;
		diagnostics: RetrievalDiagnostic[];
		unsearched: RetrievalUnsearchedSurface[];
	}> {
		return this.retrievalEngine.retrieveByMethod(query, options);
	}

	/** Persist a typed structured report for later evidence and feedback commands. */
	recordStructuredResultsSession(
		sessionId: string,
		query: string,
		details: AutoRAGResultsDetails,
	): SearchDocumentsResponse {
		return persistStructuredResultsSession(sessionId, query, details, this.sessions, this.memory);
	}

	/** Generate a session id and persist a typed structured report. */
	recordReport(query: string, details: AutoRAGResultsDetails): SearchDocumentsResponse {
		return this.recordStructuredResultsSession(randomUUID(), query, details);
	}

	/** Return the opaque result registry associated with a persisted report. */
	getResultRegistry(sessionId: string): ReadonlyMap<number, CuratedResult> {
		return this.sessions.get(sessionId)?.registry ?? new Map();
	}

	/** Record numbered feedback and report whether numbers matched and changed memory. */
	recordFeedbackByNumbersDetailed(
		sessionId: string,
		usefulNumbers: readonly number[],
		notUsefulNumbers: readonly number[] = [],
	): { matched: boolean; applied: boolean } {
		const session = this.sessions.get(sessionId);
		const persistedResults = this.memory
			.getSchema()
			.curatedResults.filter((result) => result.sessionId === sessionId);
		const validNumbers = new Set(
			session === undefined ? persistedResults.map((result) => result.number) : [...session.registry.keys()],
		);
		const feedback = [
			...usefulNumbers.filter((number) => validNumbers.has(number)).map((number) => ({ number, useful: true })),
			...notUsefulNumbers.filter((number) => validNumbers.has(number)).map((number) => ({ number, useful: false })),
		];
		if (feedback.length === 0) return { matched: false, applied: false };
		const applied = this.memory.recordNumberedFeedback({
			sessionId,
			query: session?.query ?? persistedResults[0]?.query ?? "",
			feedback,
		});
		if (applied) this.memory.save();
		return { matched: true, applied };
	}

	/** Record numbered feedback against a report persisted by this facade. */
	recordFeedbackByNumbers(
		sessionId: string,
		usefulNumbers: readonly number[],
		notUsefulNumbers: readonly number[] = [],
	): boolean {
		return this.recordFeedbackByNumbersDetailed(sessionId, usefulNumbers, notUsefulNumbers).applied;
	}

	/** Return a detached snapshot of persisted evidence and feedback state. */
	getMemorySchema(): MemorySchemaV4 {
		return structuredClone(this.memory.getSchema());
	}
}

/** Resolve the existing CLI config and construct a model-free runtime. */
export function createAutoRAGLite(options: AutoRAGLiteOptions = {}): AutoRAGLite {
	const input: ResolveConfigInput = { flags: options.flags ?? {} };
	if (options.env !== undefined) input.env = options.env;
	if (options.cwd !== undefined) input.cwd = options.cwd;
	if (options.readOnly !== undefined) input.readOnly = options.readOnly;
	return new AutoRAGLite(resolveConfig(input));
}

export type {
	AgentModelAuth,
	AgentModelConfig,
	AgentModelResolutionSource,
	CliConfig,
	MinSyncMethodConfig,
	NormalizedIndexingConfig,
	RawIndexingMethods,
	ResolveAgentModelOptions,
	ResolveConfigInput,
	ResolvedAgentModel,
	ResolvedAgentModelDetailed,
	ResolvedAgentModelRole,
	ResolvedConfigPath,
} from "./cli/config.ts";
export {
	AUTORAG_HOME_ENV,
	buildAgentOptions,
	ConfigError,
	DEFAULT_CONFIG_FILENAME,
	LEGACY_CONFIG_FILENAME,
	normalizeEmbedder,
	normalizeIndexingConfig,
	normalizeLegacyConfigPaths,
	readRawConfigObject,
	resolveAgentModel,
	resolveAgentModelDetailed,
	resolveConfig,
	resolveConfigPath,
	resolveConfigReadOnly,
	resolveModel,
	writeConfigObject,
	writeDefaultConfig,
} from "./cli/config.ts";

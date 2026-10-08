import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import { basename, dirname, isAbsolute, join, resolve } from "node:path";
import type { Api, Model } from "@earendil-works/pi-ai";
import { findEnvKeys, getEnvApiKey } from "@earendil-works/pi-ai/compat";
import { getAgentDir, ModelRuntime, SettingsManager } from "@earendil-works/pi-coding-agent";
import type { AutoRAGAgentOptions, AutoRAGRetrievalLimits } from "../agent/agent.ts";
import type { JevBackendName } from "../agent/jev-extension.ts";
import {
	type LoadLocalAutoRAGModelOptions,
	type LocalAutoRAGModel,
	loadLocalAutoRAGModel,
} from "../agent/local-model.ts";
import type { DecompositionModel } from "../agent/query-decomposition.ts";
import type { SearchDocumentDiagnostic } from "../agent/search-documents.ts";
import { resolveAutoRAGHome } from "../config/home.ts";
import type { DatasourceAccessContextOptions } from "../datasource/access-context.ts";
import { buildDatasourceSkills, type DatasourcesConfig } from "../datasource/skills/factory.ts";
import { acquireFileLock, type FileLockHandle } from "../filesystem/file-lock.ts";
import { LanguageError, type LanguageTag, normalizeLanguages } from "../language.ts";
import type { EnsureMinSyncBinaryOptions, MinSyncEmbedderConfig } from "../minsync/index.ts";
import {
	DEFAULT_RERANK_API_KEY_ENV,
	DEFAULT_RERANK_MODEL,
	DEFAULT_RERANK_PROVIDER,
	DEFAULT_RERANK_TOP_N,
	SUPPORTED_RERANK_PROVIDERS,
} from "../retrieval/rerank.ts";
import { isSearchProviderId } from "../web/search/types.ts";

export const DEFAULT_CONFIG_FILENAME = "config.json";
export const LEGACY_CONFIG_FILENAME = "autorag.config.json";
export { AUTORAG_HOME_ENV, resolveAutoRAGHome } from "../config/home.ts";

const CONFIG_LOCK_RETRY_MS = 10;
const CONFIG_LOCK_TIMEOUT_MS = 10_000;
const CONFIG_LOCK_STALE_MS = 30_000;

export class ConfigError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "ConfigError";
	}
}

/**
 * Typed MinSync method config persisted in `config.json`. Missing `enabled`
 * means enabled (true); `autoInstall` defaults to true. `embedder` carries
 * the MinSync vector embedder settings validated by {@link normalizeEmbedder}.
 */
export interface MinSyncMethodConfig {
	enabled?: boolean;
	autoInstall?: boolean;
	workspacePath?: string;
	maxChunkSize?: number;
	installer?: Omit<EnsureMinSyncBinaryOptions, "root">;
	embedder?: MinSyncEmbedderConfig;
}

/** Indexing method config as it appears in a raw config file (before normalization). */
export interface RawIndexingMethods {
	minSync?: MinSyncMethodConfig | false;
}

/** Result of {@link normalizeIndexingConfig}: always fully populated. */
export interface NormalizedIndexingConfig {
	minSync: MinSyncMethodConfig;
}

/**
 * Web tools (`web_search` + `web_fetch`) config. Secrets never appear here:
 * the shipped chain is credential-free (model-native providers reuse the
 * agent's resolved model credentials; perplexity/parallel need nothing), and
 * the only env-gated option is a self-hosted SEARXNG_ENDPOINT.
 */
export interface WebSearchCliConfig {
	enabled?: boolean;
	provider?: string;
	order?: string[];
	exclude?: string[];
	/** Per-provider transport hard timeout in seconds (1-300). */
	timeoutSeconds?: number;
	fetch?:
		| {
				enabled?: boolean;
				/** Total page fetch/render timeout in seconds (1-300). */
				timeoutSeconds?: number;
		  }
		| false;
}

/**
 * Jev decision-tool config. Absent disables the tool; `enabled: false` disables
 * it explicitly. Secrets never appear here: the `jev-use` engine reads the
 * backend credential from its own environment variable (`TYPESAFE_API_KEY`,
 * `OPENROUTER_API_KEY`, or `AI_GATEWAY_API_KEY`).
 */
export interface JevCliConfig {
	enabled?: boolean;
	/** Force one backend; omit to let the first credential present win. */
	backend?: JevBackendName;
	/** Model id sent with every call, e.g. `jev-latest`. */
	model?: string;
	/** Escalate verdicts below this confidence (0-1). Default: per-source thresholds. */
	confidenceThreshold?: number;
}

/** Question-decomposition config. Secrets stay in env/pi auth like the main model. */
export interface QueryDecompositionConfig {
	/** Dedicated decomposition model; same shape and auth rules as the top-level `model`. */
	model?: AgentModelConfig;
}

/**
 * Post-merge reranking config. Routes merged evidence through a dedicated
 * rerank model. `provider` is `openrouter` today; `model` is the OpenRouter
 * wire id (default `voyageai/rerank-3-lite`). Secrets never appear here — only
 * the environment-variable name that holds the provider API key.
 */
export interface RerankConfig {
	/** `false` disables reranking. Missing means enabled when the block is present. */
	enabled?: boolean;
	/** Provider id. @default "openrouter" */
	provider?: string;
	/** Wire model id. @default "voyageai/rerank-3-lite" */
	model?: string;
	/** Environment variable holding the provider API key. @default "OPENROUTER_API_KEY" */
	apiKeyEnv?: string;
	/** Override the provider base URL (e.g. a gateway). */
	baseUrl?: string;
	/** Return only the top N merged results. @default 25 (DEFAULT_RERANK_TOP_N) */
	topN?: number;
	/** Per-request timeout in milliseconds. */
	timeoutMs?: number;
}

export interface P2pConfig {
	enabled?: boolean;
	port?: number;
	host?: string;
	/** SimpleX CLI database prefix (defaults to <workspace>/.autorag/p2p/simplex). */
	simplexDbPrefix?: string;
	maxBodyBytes?: number;
	maxFileBytes?: number;
	policy?: Record<string, unknown>;
	quotas?: {
		queriesPerHour?: number;
		burst?: number;
	};
	injectionClassifier?: boolean;
	piiNer?: boolean;
	searchTimeoutMs?: number;
	newFilesPublic?: boolean;
}

export interface AgentModelConfig {
	/** Provider identity used for auth lookup and Model.provider (e.g. openrouter, fireworks, ollama). */
	provider: string;
	/** Wire model id sent to the provider API. */
	id: string;
	/** Optional display name; defaults to id when omitted. */
	name?: string;
	/**
	 * API wire format. Required only when `baseUrl` is set and you need something
	 * other than the default `openai-completions`.
	 */
	api?: Api;
	/**
	 * Endpoint base URL. For a pi-ai catalog `provider/id`, it overrides only the
	 * catalog endpoint and keeps the catalog's reasoning, compat, and limits. For an
	 * id outside the catalog, AutoRAG builds a generic Model from this config.
	 * Omit for catalog/local models that use the catalog endpoint.
	 */
	baseUrl?: string;
	/**
	 * Environment variable name holding the API key (never the secret itself).
	 * Defaults to `${PROVIDER}_API_KEY` when `baseUrl` is set.
	 */
	apiKeyEnv?: string;
	reasoning?: boolean;
	input?: Array<"text" | "image">;
	contextWindow?: number;
	maxTokens?: number;
}

export interface CliConfig {
	/** Absolute path of the config file this config was resolved from (it may not exist yet). */
	configPath?: string;
	searchPaths: string[];
	workspacePath: string;
	memoryPath: string;
	languages?: string[];
	model?: AgentModelConfig;
	minSync?: MinSyncMethodConfig;
	jikji?: Record<string, unknown> | false;
	/** Windows-only bundled Everything file-name search. Default enabled on Windows; `false` disables. */
	everything?: { enabled?: boolean; timeoutMs?: number; startupTimeoutMs?: number; indexTimeoutMs?: number } | false;
	/** macOS/Linux fsearch-cli file-name search. Default enabled on macOS/Linux; `false` disables. */
	fsearch?:
		| {
				enabled?: boolean;
				binaryPath?: string;
				timeoutMs?: number;
				startupTimeoutMs?: number;
				indexTimeoutMs?: number;
				watch?: boolean;
		  }
		| false;
	webSearch?: WebSearchCliConfig;
	/**
	 * Jev query routing (local / web / direct), the decomposition check, the
	 * post-fast-answer follow-up check, and the `jev` tool. On by default with
	 * the OpenRouter backend; `false` or `enabled: false` disables it. Secrets
	 * never appear here: `jev-use` reads the backend key from its own
	 * environment variable.
	 */
	jev?: JevCliConfig | false;
	/**
	 * Question decomposition for the Jev query pipeline. `model` names the LLM
	 * that splits one question into at most five search queries; absent or `{}`
	 * uses {@link DEFAULT_QUERY_DECOMPOSITION_MODEL}, and `false` lets the
	 * agent's own model decompose.
	 */
	queryDecomposition?: QueryDecompositionConfig | false;
	/** Post-merge reranking. Absent ⇒ reranking disabled. `false` disables it. */
	rerank?: RerankConfig | false;
	parserOptions?: Record<string, unknown>;
	dupey?: {
		enabled?: boolean;
		binaryPath?: string;
		timeoutMs?: number;
	};
	excludeExactDuplicates?: boolean;
	/** Absolute or workspace-relative files/directories omitted from local indexing. */
	excludePaths?: string[];
	/** Hard caps on retrieval, baseline prefetch, and model-facing candidate lists. */
	limits?: AutoRAGRetrievalLimits;
	/** Trusted datasource skill configuration (skill name → config). */
	datasources?: DatasourcesConfig;
	/** Trusted datasource allow-tags/allow-scopes. Absent ⇒ default-deny. */
	datasourceAccess?: DatasourceAccessContextOptions;
	/** P2P sharing configuration. Disabled by default. */
	p2p?: P2pConfig;
}

export interface ResolveConfigInput {
	flags: Record<string, string | boolean | undefined>;
	env?: NodeJS.ProcessEnv;
	cwd?: string;
	readOnly?: boolean;
}

export interface ResolvedConfigPath {
	configPath: string;
	explicit: boolean;
	legacyPath?: string;
}

export function resolveConfigPath(input: ResolveConfigInput): ResolvedConfigPath {
	const flags = input.flags;
	const env = input.env ?? process.env;
	const cwd = input.cwd ?? process.cwd();

	const flagConfig = flags.config;
	if (typeof flagConfig === "string" && flagConfig.length > 0) {
		return { configPath: flagConfig, explicit: true };
	}
	const envConfig = env.AUTORAG_CONFIG;
	if (typeof envConfig === "string" && envConfig.length > 0) {
		return { configPath: envConfig, explicit: true };
	}
	return {
		configPath: join(resolveAutoRAGHome(env), DEFAULT_CONFIG_FILENAME),
		explicit: false,
		legacyPath: join(cwd, LEGACY_CONFIG_FILENAME),
	};
}

function readConfigFile(configPath: string, explicit: boolean): Partial<CliConfig> | undefined {
	let exists: boolean;
	try {
		exists = existsSync(configPath);
	} catch {
		exists = false;
	}
	if (!exists) {
		if (explicit) {
			throw new ConfigError(`Config file not found: ${configPath}`);
		}
		return undefined;
	}
	let text: string;
	try {
		text = readFileSync(configPath, "utf8");
	} catch (err) {
		throw new ConfigError(`Failed to read config file: ${(err as Error).message}`);
	}
	let parsed: unknown;
	try {
		parsed = JSON.parse(text);
	} catch (err) {
		throw new ConfigError(`Failed to parse config file: ${(err as Error).message}`);
	}
	if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
		throw new ConfigError("Config file must be a JSON object");
	}
	return parsed as Partial<CliConfig>;
}

function isEexistError(error: unknown): boolean {
	return typeof error === "object" && error !== null && "code" in error && error.code === "EEXIST";
}

function isEnoentError(error: unknown): boolean {
	return typeof error === "object" && error !== null && "code" in error && error.code === "ENOENT";
}

function removeFileIfPresent(path: string): void {
	try {
		unlinkSync(path);
	} catch (error) {
		if (!isEnoentError(error)) throw error;
	}
}
function acquireConfigWriteLock(configPath: string): FileLockHandle {
	return acquireFileLock(`${configPath}.lock`, {
		timeoutMs: CONFIG_LOCK_TIMEOUT_MS,
		staleMs: CONFIG_LOCK_STALE_MS,
		retryMs: CONFIG_LOCK_RETRY_MS,
		timeoutError: () => new ConfigError(`Timed out waiting to write config file: ${configPath}`),
	});
}

function replaceFileAtomically(
	path: string,
	contents: string | NodeJS.ArrayBufferView,
	assertCommitAllowed?: () => void,
): void {
	const temporaryPath = join(dirname(path), `.${basename(path)}.${process.pid}.${randomUUID()}.tmp`);
	try {
		writeFileSync(temporaryPath, contents, { encoding: "utf8", flag: "wx", flush: true, mode: 0o600 });
		assertCommitAllowed?.();
		renameSync(temporaryPath, path);
	} finally {
		removeFileIfPresent(temporaryPath);
	}
}

function migrateLegacyConfig(configPath: string, legacyPath: string): Partial<CliConfig> | undefined {
	if (existsSync(configPath) || !existsSync(legacyPath)) return undefined;
	const legacy = readConfigFile(legacyPath, true);
	const legacyBytes = readFileSync(legacyPath);
	const migrated = normalizeLegacyConfigPaths(legacy ?? {}, dirname(legacyPath));
	const migratedPathsUnchanged =
		legacy?.workspacePath === migrated.workspacePath &&
		legacy?.memoryPath === migrated.memoryPath &&
		JSON.stringify(legacy?.searchPaths) === JSON.stringify(migrated.searchPaths);
	const migratedBytes = migratedPathsUnchanged ? legacyBytes : `${JSON.stringify(migrated, null, 2)}\n`;
	mkdirSync(dirname(configPath), { recursive: true });
	const lock = acquireConfigWriteLock(configPath);
	try {
		const winner = readConfigFile(configPath, false);
		if (winner !== undefined) return winner;
		try {
			replaceFileAtomically(configPath, migratedBytes, lock.assertOwned);
		} catch (error) {
			if (isEexistError(error)) {
				const concurrentWinner = readConfigFile(configPath, false);
				if (concurrentWinner !== undefined) return concurrentWinner;
			}
			throw error;
		}
	} finally {
		lock.release();
	}
	return migrated;
}
/**
 * Read-only config loading for health: never migrate, write, or lock. When the
 * implicit home config is missing but a legacy cwd config exists, read the
 * legacy file and normalize its paths in memory only.
 */
function resolveConfigFileReadOnly(
	configPath: string,
	explicit: boolean,
	legacyPath: string | undefined,
): Partial<CliConfig> {
	const home = readConfigFile(configPath, explicit);
	if (home !== undefined) return home;
	if (explicit || legacyPath === undefined) return {};
	const legacy = readConfigFile(legacyPath, false);
	if (legacy === undefined) return {};
	return normalizeLegacyConfigPaths(legacy, dirname(legacyPath));
}

function resolveSearchPaths(searchPaths: readonly string[], origin: string): string[] {
	return searchPaths.map((searchPath) => resolvePersistedPath(searchPath, origin));
}

function resolvePersistedPath(path: string, origin: string): string {
	return isAbsolute(path) ? path : resolve(origin, path);
}

/** Normalize inherited legacy paths against the legacy workspace, not the caller's cwd. */
export function normalizeLegacyConfigPaths(partial: Partial<CliConfig>, origin: string): Partial<CliConfig> {
	const workspacePath = resolvePersistedPath(partial.workspacePath ?? ".", origin);
	return {
		...partial,
		workspacePath,
		...(partial.searchPaths === undefined
			? {}
			: { searchPaths: resolveSearchPaths(partial.searchPaths, workspacePath) }),
		...(partial.memoryPath === undefined
			? {}
			: { memoryPath: resolvePersistedPath(partial.memoryPath, workspacePath) }),
	};
}

function parseCsv(value: string | undefined): string[] | undefined {
	if (typeof value !== "string" || value.length === 0) return undefined;
	const parts = value
		.split(",")
		.map((part) => part.trim())
		.filter((part) => part.length > 0);
	return parts.length > 0 ? parts : undefined;
}

function flagString(flags: Record<string, string | boolean | undefined>, key: string): string | undefined {
	const value = flags[key];
	if (typeof value === "string" && value.length > 0) return value;
	return undefined;
}

function envString(env: NodeJS.ProcessEnv, key: string): string | undefined {
	const value = env[key];
	if (typeof value === "string" && value.length > 0) return value;
	return undefined;
}

function normalizeConfigLanguages(raw: unknown): LanguageTag[] {
	try {
		return normalizeLanguages(raw);
	} catch (error) {
		if (error instanceof LanguageError) throw new ConfigError(error.message);
		throw error;
	}
}

const AGENT_MODEL_APIS = new Set<Api>([
	"openai-completions",
	"openai-responses",
	"anthropic-messages",
	"openai-codex-responses",
	"azure-openai-responses",
]);

const AGENT_MODEL_INPUTS = new Set(["text", "image"]);

function modelReference(value: unknown, path: string): AgentModelConfig | undefined {
	if (value === undefined) return undefined;
	if (typeof value !== "object" || value === null || Array.isArray(value)) {
		throw new ConfigError(`${path} must be an object with provider and id`);
	}
	const record = value as Record<string, unknown>;
	const allowed = new Set([
		"provider",
		"id",
		"name",
		"api",
		"baseUrl",
		"apiKeyEnv",
		"reasoning",
		"input",
		"contextWindow",
		"maxTokens",
	]);
	for (const key of Object.keys(record)) {
		if (!allowed.has(key)) {
			throw new ConfigError(`${path}.${key} is not a recognized agent model field`);
		}
	}
	if (typeof record.provider !== "string" || record.provider.trim() === "") {
		throw new ConfigError(`${path}.provider must be a non-empty string`);
	}
	if (typeof record.id !== "string" || record.id.trim() === "") {
		throw new ConfigError(`${path}.id must be a non-empty string`);
	}
	const out: AgentModelConfig = {
		provider: record.provider.trim(),
		id: record.id.trim(),
	};
	if (record.name !== undefined) {
		if (typeof record.name !== "string" || record.name.trim() === "") {
			throw new ConfigError(`${path}.name must be a non-empty string`);
		}
		out.name = record.name.trim();
	}
	if (record.api !== undefined) {
		if (typeof record.api !== "string" || !AGENT_MODEL_APIS.has(record.api as Api)) {
			throw new ConfigError(`${path}.api must be one of: ${[...AGENT_MODEL_APIS].join(", ")}`);
		}
		out.api = record.api as Api;
	}
	if (record.baseUrl !== undefined) {
		if (typeof record.baseUrl !== "string" || record.baseUrl.trim() === "") {
			throw new ConfigError(`${path}.baseUrl must be a non-empty string`);
		}
		out.baseUrl = record.baseUrl.trim();
	}
	if (record.apiKeyEnv !== undefined) {
		if (typeof record.apiKeyEnv !== "string" || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(record.apiKeyEnv)) {
			throw new ConfigError(`${path}.apiKeyEnv must match /^[A-Za-z_][A-Za-z0-9_]*$/`);
		}
		out.apiKeyEnv = record.apiKeyEnv;
	}
	if (record.reasoning !== undefined) {
		if (typeof record.reasoning !== "boolean") {
			throw new ConfigError(`${path}.reasoning must be a boolean`);
		}
		out.reasoning = record.reasoning;
	}
	if (record.input !== undefined) {
		if (!Array.isArray(record.input) || record.input.length === 0) {
			throw new ConfigError(`${path}.input must be a non-empty array of "text" | "image"`);
		}
		const input: Array<"text" | "image"> = [];
		for (const item of record.input) {
			if (typeof item !== "string" || !AGENT_MODEL_INPUTS.has(item)) {
				throw new ConfigError(`${path}.input must contain only "text" and/or "image"`);
			}
			input.push(item as "text" | "image");
		}
		out.input = input;
	}
	if (record.contextWindow !== undefined) {
		if (
			typeof record.contextWindow !== "number" ||
			!Number.isInteger(record.contextWindow) ||
			record.contextWindow <= 0
		) {
			throw new ConfigError(`${path}.contextWindow must be a positive integer`);
		}
		out.contextWindow = record.contextWindow;
	}
	if (record.maxTokens !== undefined) {
		if (typeof record.maxTokens !== "number" || !Number.isInteger(record.maxTokens) || record.maxTokens <= 0) {
			throw new ConfigError(`${path}.maxTokens must be a positive integer`);
		}
		out.maxTokens = record.maxTokens;
	}
	return out;
}

function applyRoleFlagOverrides(
	fileRef: AgentModelConfig | undefined,
	provider: string | undefined,
	id: string | undefined,
	path: string,
): AgentModelConfig | undefined {
	if (provider === undefined && id === undefined) return fileRef;
	if (provider === undefined || id === undefined) {
		throw new ConfigError(`${path} requires both provider and id`);
	}
	if (fileRef !== undefined && fileRef.provider === provider && fileRef.id === id) {
		return fileRef;
	}
	// Flag overrides replace the role with a catalog-style provider/id pair.
	return { provider, id };
}

function pickString(
	flags: Record<string, string | boolean | undefined>,
	env: NodeJS.ProcessEnv,
	flagKey: string,
	envKey: string,
	fileValue: string | undefined,
): string | undefined {
	return flagString(flags, flagKey) ?? envString(env, envKey) ?? fileValue;
}

const API_KEY_ENV_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/;
const POSITIVE_INT_FIELDS = ["dimension", "timeoutMs", "batchSize", "maxRetries", "maxConcurrent"] as const;
const EMBEDDER_ALLOWLIST = new Set<string>([
	"id",
	"baseUrl",
	"apiKeyEnv",
	"dimension",
	"queryPrefix",
	"passagePrefix",
	"timeoutMs",
	"batchSize",
	"maxRetries",
	"maxConcurrent",
	"profile",
]);

/**
 * Normalize and validate a raw MinSync embedder config object.
 * Rejects unknown fields (e.g. `extraArgs`), invalid `apiKeyEnv` names, and
 * non-positive numeric fields. Returns a cleaned {@link MinSyncEmbedderConfig}.
 */
export function normalizeEmbedder(raw: unknown, path: string): MinSyncEmbedderConfig {
	if (raw === undefined || raw === null) return {};
	if (typeof raw !== "object" || Array.isArray(raw)) {
		throw new ConfigError(`${path} must be an object`);
	}
	const record = raw as Record<string, unknown>;
	for (const key of Object.keys(record)) {
		if (!EMBEDDER_ALLOWLIST.has(key)) {
			throw new ConfigError(`${path}.${key} is not a recognized embedder field`);
		}
	}
	const out: {
		id?: string;
		baseUrl?: string;
		apiKeyEnv?: string;
		dimension?: number;
		queryPrefix?: string;
		passagePrefix?: string;
		timeoutMs?: number;
		batchSize?: number;
		maxRetries?: number;
		maxConcurrent?: number;
		profile?: MinSyncEmbedderConfig["profile"];
	} = {};
	if (record.id !== undefined) {
		if (typeof record.id !== "string" || record.id.trim() === "") {
			throw new ConfigError(`${path}.id must be a non-empty string`);
		}
		out.id = record.id;
	}
	if (record.baseUrl !== undefined) {
		if (typeof record.baseUrl !== "string" || record.baseUrl.trim() === "") {
			throw new ConfigError(`${path}.baseUrl must be a non-empty string`);
		}
		out.baseUrl = record.baseUrl;
	}
	if (record.apiKeyEnv !== undefined) {
		if (typeof record.apiKeyEnv !== "string" || !API_KEY_ENV_PATTERN.test(record.apiKeyEnv)) {
			throw new ConfigError(`${path}.apiKeyEnv must match /^[A-Za-z_][A-Za-z0-9_]*$/`);
		}
		out.apiKeyEnv = record.apiKeyEnv;
	}
	if (record.queryPrefix !== undefined) {
		if (typeof record.queryPrefix !== "string") {
			throw new ConfigError(`${path}.queryPrefix must be a string`);
		}
		out.queryPrefix = record.queryPrefix;
	}
	if (record.passagePrefix !== undefined) {
		if (typeof record.passagePrefix !== "string") {
			throw new ConfigError(`${path}.passagePrefix must be a string`);
		}
		out.passagePrefix = record.passagePrefix;
	}
	if (record.profile !== undefined) {
		if (record.profile !== "qwen3-embedding-0.6b" && record.profile !== "embeddinggemma-300m") {
			throw new ConfigError(`${path}.profile must be a supported runtime profile`);
		}
		out.profile = record.profile;
	}
	for (const field of POSITIVE_INT_FIELDS) {
		const value = record[field];
		if (value === undefined) continue;
		if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
			throw new ConfigError(`${path}.${field} must be a positive integer`);
		}
		out[field] = value;
	}
	return out;
}
const MINSYNC_ALLOWLIST = new Set<string>([
	"enabled",
	"autoInstall",
	"workspacePath",
	"maxChunkSize",
	"installer",
	"embedder",
]);

/**
 * Fields that older configs may still carry. MinSync is now resolved from PATH
 * and the workspace cache, so a persisted `binaryPath` is ignored rather than
 * rejected; failing here would break every command for existing installs.
 */
const MINSYNC_LEGACY_IGNORED = new Set<string>(["binaryPath"]);

const P2P_ALLOWLIST = new Set([
	"enabled",
	"port",
	"host",
	"simplexDbPrefix",
	"maxBodyBytes",
	"maxFileBytes",
	"policy",
	"quotas",
	"injectionClassifier",
	"piiNer",
	"searchTimeoutMs",
	"newFilesPublic",
]);

const P2P_QUOTA_ALLOWLIST = new Set(["queriesPerHour", "burst"]);

function normalizeP2pConfig(raw: unknown): P2pConfig {
	const out: P2pConfig = {};
	if (raw === undefined || raw === null) {
		out.enabled = false;
		out.host = "127.0.0.1";
		out.port = 7583;
		out.injectionClassifier = true;
		out.piiNer = false;
		out.searchTimeoutMs = 120000;
		return out;
	}
	if (typeof raw !== "object" || Array.isArray(raw)) {
		throw new ConfigError("Config field 'p2p' must be an object");
	}
	const record = raw as Record<string, unknown>;
	for (const key of Object.keys(record)) {
		if (!P2P_ALLOWLIST.has(key)) {
			throw new ConfigError(`p2p.${key} is not a recognized field`);
		}
	}
	if (record.enabled !== undefined) {
		if (typeof record.enabled !== "boolean") {
			throw new ConfigError("p2p.enabled must be a boolean");
		}
		out.enabled = record.enabled;
	}
	out.enabled ??= false;

	if (record.host !== undefined) {
		if (typeof record.host !== "string" || record.host.trim().length === 0) {
			throw new ConfigError("p2p.host must be a non-empty string");
		}
		out.host = record.host.trim();
	}
	out.host ??= "127.0.0.1";

	if (record.simplexDbPrefix !== undefined) {
		if (typeof record.simplexDbPrefix !== "string" || record.simplexDbPrefix.trim().length === 0) {
			throw new ConfigError("p2p.simplexDbPrefix must be a non-empty string");
		}
		out.simplexDbPrefix = record.simplexDbPrefix;
	}

	if (record.port !== undefined) {
		if (typeof record.port !== "number" || !Number.isInteger(record.port) || record.port < 1 || record.port > 65535) {
			throw new ConfigError("p2p.port must be an integer between 1 and 65535");
		}
		out.port = record.port;
	}
	out.port ??= 5225;

	if (record.maxBodyBytes !== undefined) {
		if (
			typeof record.maxBodyBytes !== "number" ||
			!Number.isInteger(record.maxBodyBytes) ||
			record.maxBodyBytes <= 0
		) {
			throw new ConfigError("p2p.maxBodyBytes must be a positive integer");
		}
		out.maxBodyBytes = record.maxBodyBytes;
	}

	if (record.maxFileBytes !== undefined) {
		if (
			typeof record.maxFileBytes !== "number" ||
			!Number.isInteger(record.maxFileBytes) ||
			record.maxFileBytes <= 0
		) {
			throw new ConfigError("p2p.maxFileBytes must be a positive integer");
		}
		out.maxFileBytes = record.maxFileBytes;
	}

	if (record.policy !== undefined) {
		if (typeof record.policy !== "object" || record.policy === null || Array.isArray(record.policy)) {
			throw new ConfigError("p2p.policy must be an object");
		}
		out.policy = record.policy as Record<string, unknown>;
	}

	if (record.quotas !== undefined) {
		if (typeof record.quotas !== "object" || record.quotas === null || Array.isArray(record.quotas)) {
			throw new ConfigError("p2p.quotas must be an object");
		}
		const quotasRecord = record.quotas as Record<string, unknown>;
		for (const key of Object.keys(quotasRecord)) {
			if (!P2P_QUOTA_ALLOWLIST.has(key)) {
				throw new ConfigError(`p2p.quotas.${key} is not a recognized field`);
			}
		}
		const quotas: { queriesPerHour?: number; burst?: number } = {};
		if (quotasRecord.queriesPerHour !== undefined) {
			if (
				typeof quotasRecord.queriesPerHour !== "number" ||
				!Number.isInteger(quotasRecord.queriesPerHour) ||
				quotasRecord.queriesPerHour <= 0
			) {
				throw new ConfigError("p2p.quotas.queriesPerHour must be a positive integer");
			}
			quotas.queriesPerHour = quotasRecord.queriesPerHour;
		}
		if (quotasRecord.burst !== undefined) {
			if (
				typeof quotasRecord.burst !== "number" ||
				!Number.isInteger(quotasRecord.burst) ||
				quotasRecord.burst <= 0
			) {
				throw new ConfigError("p2p.quotas.burst must be a positive integer");
			}
			quotas.burst = quotasRecord.burst;
		}
		out.quotas = quotas;
	}

	if (record.injectionClassifier !== undefined) {
		if (typeof record.injectionClassifier !== "boolean") {
			throw new ConfigError("p2p.injectionClassifier must be a boolean");
		}
		out.injectionClassifier = record.injectionClassifier;
	}
	out.injectionClassifier ??= true;

	if (record.piiNer !== undefined) {
		if (typeof record.piiNer !== "boolean") {
			throw new ConfigError("p2p.piiNer must be a boolean");
		}
		out.piiNer = record.piiNer;
	}
	out.piiNer ??= false;

	if (record.newFilesPublic !== undefined) {
		if (typeof record.newFilesPublic !== "boolean") {
			throw new ConfigError("p2p.newFilesPublic must be a boolean");
		}
		out.newFilesPublic = record.newFilesPublic;
	}

	if (record.searchTimeoutMs !== undefined) {
		if (
			typeof record.searchTimeoutMs !== "number" ||
			!Number.isInteger(record.searchTimeoutMs) ||
			record.searchTimeoutMs < 5000 ||
			record.searchTimeoutMs > 600000
		) {
			throw new ConfigError("p2p.searchTimeoutMs must be an integer between 5000 and 600000");
		}
		out.searchTimeoutMs = record.searchTimeoutMs;
	}
	out.searchTimeoutMs ??= 120000;

	return out;
}

function normalizeMinSyncMethod(raw: MinSyncMethodConfig | false | undefined): MinSyncMethodConfig {
	if (raw === false) return { enabled: false };
	if (raw === undefined || raw === null) return { enabled: true, autoInstall: true };
	if (typeof raw !== "object" || Array.isArray(raw)) {
		throw new ConfigError("minSync must be an object or false");
	}
	const record = raw as Record<string, unknown>;
	for (const key of Object.keys(record)) {
		if (MINSYNC_LEGACY_IGNORED.has(key)) continue;
		if (!MINSYNC_ALLOWLIST.has(key)) {
			throw new ConfigError(`minSync.${key} is not a recognized field`);
		}
	}
	const enabled = record.enabled !== false;
	const out: MinSyncMethodConfig = { enabled, autoInstall: record.autoInstall !== false };
	if (typeof record.workspacePath === "string" && record.workspacePath.length > 0) {
		out.workspacePath = record.workspacePath;
	}
	if (record.maxChunkSize !== undefined) {
		if (
			typeof record.maxChunkSize !== "number" ||
			!Number.isInteger(record.maxChunkSize) ||
			record.maxChunkSize <= 0
		) {
			throw new ConfigError("minSync.maxChunkSize must be a positive integer");
		}
		out.maxChunkSize = record.maxChunkSize;
	}
	if (record.installer !== undefined && record.installer !== null) {
		if (typeof record.installer !== "object" || Array.isArray(record.installer)) {
			throw new ConfigError("minSync.installer must be an object");
		}
		out.installer = record.installer as Omit<EnsureMinSyncBinaryOptions, "root">;
	}
	if (record.embedder !== undefined && record.embedder !== null) {
		out.embedder = normalizeEmbedder(record.embedder, "minSync.embedder");
	}
	return out;
}

/**
 * Normalize raw indexing method config into a fully-populated shape.
 *
 * - `undefined` / missing key => `{ enabled: true, autoInstall: true }`
 * - `false` => `{ enabled: false }` (disabled marker)
 * - object merges with `enabled: true` default and is validated
 *
 * Unknown fields, invalid embedder settings, and bad numeric values throw
 * {@link ConfigError}.
 */
export function normalizeIndexingConfig(raw: RawIndexingMethods): NormalizedIndexingConfig {
	return {
		minSync: normalizeMinSyncMethod(raw.minSync),
	};
}

const LIMITS_ALLOWLIST: Record<string, true> = {
	mergedEvidenceCeiling: true,
	singleDatasourceTopK: true,
	minSyncTopK: true,
	minSyncScopedQueryTopK: true,
	toolDescriptionInstanceScopes: true,
	prefetch: true,
};

const LIMITS_PREFETCH_ALLOWLIST: Record<string, true> = {
	jikjiTopK: true,
	minSyncTopK: true,
	jikjiPathLimit: true,
	sectionLimit: true,
};

function positiveLimitField(record: Record<string, unknown>, key: string, path: string): number | undefined {
	const value = record[key];
	if (value === undefined) return undefined;
	if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
		throw new ConfigError(`${path}.${key} must be a positive integer`);
	}
	return value;
}

/** Validate the `limits` section and map it onto the agent option. */
export function normalizeLimitsConfig(raw: unknown): AutoRAGRetrievalLimits {
	if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
		throw new ConfigError("Config field 'limits' must be an object");
	}
	const record = raw as Record<string, unknown>;
	for (const key of Object.keys(record)) {
		if (!Object.hasOwn(LIMITS_ALLOWLIST, key)) throw new ConfigError(`limits.${key} is not a recognized field`);
	}
	const mergedEvidenceCeiling = positiveLimitField(record, "mergedEvidenceCeiling", "limits");
	const singleDatasourceTopK = positiveLimitField(record, "singleDatasourceTopK", "limits");
	const minSyncTopK = positiveLimitField(record, "minSyncTopK", "limits");
	const minSyncScopedQueryTopK = positiveLimitField(record, "minSyncScopedQueryTopK", "limits");
	const toolDescriptionInstanceScopes = positiveLimitField(record, "toolDescriptionInstanceScopes", "limits");
	let prefetch: AutoRAGRetrievalLimits["prefetch"];
	const rawPrefetch = record.prefetch;
	if (rawPrefetch !== undefined) {
		if (typeof rawPrefetch !== "object" || rawPrefetch === null || Array.isArray(rawPrefetch)) {
			throw new ConfigError("limits.prefetch must be an object");
		}
		const prefetchRecord = rawPrefetch as Record<string, unknown>;
		for (const key of Object.keys(prefetchRecord)) {
			if (!Object.hasOwn(LIMITS_PREFETCH_ALLOWLIST, key)) {
				throw new ConfigError(`limits.prefetch.${key} is not a recognized field`);
			}
		}
		const jikjiTopK = positiveLimitField(prefetchRecord, "jikjiTopK", "limits.prefetch");
		const prefetchMinSyncTopK = positiveLimitField(prefetchRecord, "minSyncTopK", "limits.prefetch");
		const jikjiPathLimit = positiveLimitField(prefetchRecord, "jikjiPathLimit", "limits.prefetch");
		const sectionLimit = positiveLimitField(prefetchRecord, "sectionLimit", "limits.prefetch");
		prefetch = {
			...(jikjiTopK !== undefined ? { jikjiTopK } : {}),
			...(prefetchMinSyncTopK !== undefined ? { minSyncTopK: prefetchMinSyncTopK } : {}),
			...(jikjiPathLimit !== undefined ? { jikjiPathLimit } : {}),
			...(sectionLimit !== undefined ? { sectionLimit } : {}),
		};
	}
	return {
		...(mergedEvidenceCeiling !== undefined ? { mergedEvidenceCeiling } : {}),
		...(singleDatasourceTopK !== undefined ? { singleDatasourceTopK } : {}),
		...(minSyncTopK !== undefined ? { minSyncTopK } : {}),
		...(minSyncScopedQueryTopK !== undefined ? { minSyncScopedQueryTopK } : {}),
		...(toolDescriptionInstanceScopes !== undefined ? { toolDescriptionInstanceScopes } : {}),
		...(prefetch !== undefined ? { prefetch } : {}),
	};
}

export function resolveConfig(input: ResolveConfigInput): CliConfig {
	const flags = input.flags;
	const env = input.env ?? process.env;
	const cwd = input.cwd ?? process.cwd();

	const { configPath, explicit, legacyPath } = resolveConfigPath(input);
	const readOnly = input.readOnly === true;
	const file = readOnly
		? resolveConfigFileReadOnly(configPath, explicit, legacyPath)
		: !explicit && legacyPath
			? (migrateLegacyConfig(configPath, legacyPath) ?? readConfigFile(configPath, explicit) ?? {})
			: (readConfigFile(configPath, explicit) ?? {});

	const defaultSearchPaths = ["."];
	const defaultWorkspacePath = cwd;
	const defaultMemoryPath = join(resolveAutoRAGHome(env), "memory.json");

	const flagSearchPaths = parseCsv(flagString(flags, "search-paths"));
	const envSearchPaths = parseCsv(envString(env, "AUTORAG_SEARCH_PATHS"));
	const configOrigin = dirname(resolve(configPath));
	const fileWorkspacePath =
		typeof file.workspacePath === "string" ? resolvePersistedPath(file.workspacePath, configOrigin) : undefined;
	const fileSearchPaths = file.searchPaths
		? resolveSearchPaths(file.searchPaths, fileWorkspacePath ?? configOrigin)
		: undefined;
	if (
		file.excludePaths !== undefined &&
		(!Array.isArray(file.excludePaths) ||
			file.excludePaths.some((path) => typeof path !== "string" || path.length === 0))
	) {
		throw new ConfigError("Config field 'excludePaths' must be a non-empty string array");
	}
	const fileExcludePaths = file.excludePaths
		? resolveSearchPaths(file.excludePaths, fileWorkspacePath ?? configOrigin)
		: undefined;
	const searchPaths = flagSearchPaths ?? envSearchPaths ?? fileSearchPaths ?? defaultSearchPaths;

	const flagWorkspacePath = flagString(flags, "workspace");
	const envWorkspacePath = envString(env, "AUTORAG_WORKSPACE");
	const workspacePath = flagWorkspacePath ?? envWorkspacePath ?? fileWorkspacePath ?? defaultWorkspacePath;

	const flagMemoryPath = flagString(flags, "memory-path");
	const envMemoryPath = envString(env, "AUTORAG_MEMORY_PATH");
	// Persisted relative memory paths are workspace-relative so home/global configs remain stable across cwd changes.
	const fileMemoryPath =
		typeof file.memoryPath === "string" ? resolvePersistedPath(file.memoryPath, workspacePath) : undefined;
	const memoryPath = flagMemoryPath ?? envMemoryPath ?? fileMemoryPath ?? defaultMemoryPath;
	const flagLanguages = typeof flags.languages === "string" ? flags.languages : undefined;
	const envLanguages = env.AUTORAG_LANGUAGES;
	const languages =
		flagLanguages !== undefined
			? normalizeConfigLanguages(flagLanguages)
			: envLanguages !== undefined
				? normalizeConfigLanguages(envLanguages)
				: normalizeConfigLanguages(file.languages);

	const fileModel = modelReference(file.model, "model");
	const flagModelProvider = pickString(flags, env, "model-provider", "AUTORAG_MODEL_PROVIDER", undefined);
	const flagModelId = pickString(flags, env, "model-id", "AUTORAG_MODEL_ID", undefined);
	const model = applyRoleFlagOverrides(fileModel, flagModelProvider, flagModelId, "model");

	const config: CliConfig = {
		configPath: resolve(configPath),
		searchPaths,
		workspacePath,
		memoryPath,
		languages,
	};
	if (model) config.model = model;
	if (fileExcludePaths !== undefined) config.excludePaths = fileExcludePaths;
	if (file.limits !== undefined) config.limits = normalizeLimitsConfig(file.limits);
	const normalized = normalizeIndexingConfig({
		minSync: file.minSync as MinSyncMethodConfig | false | undefined,
	});
	config.minSync = normalized.minSync;
	config.jikji = file.jikji === false ? false : (file.jikji ?? {});
	if (file.everything !== undefined) {
		if (
			file.everything !== false &&
			(typeof file.everything !== "object" || file.everything === null || Array.isArray(file.everything))
		) {
			throw new ConfigError("Config field 'everything' must be false or an object");
		}
		config.everything = file.everything as CliConfig["everything"];
	}
	if (file.fsearch !== undefined) {
		if (
			file.fsearch !== false &&
			(typeof file.fsearch !== "object" || file.fsearch === null || Array.isArray(file.fsearch))
		) {
			throw new ConfigError("Config field 'fsearch' must be false or an object");
		}
		config.fsearch = file.fsearch as CliConfig["fsearch"];
	}
	if (file.jev !== undefined) config.jev = file.jev === false ? false : normalizeJevConfig(file.jev);
	if (file.queryDecomposition !== undefined) {
		config.queryDecomposition = normalizeQueryDecompositionConfig(file.queryDecomposition);
	}
	if (file.parserOptions) config.parserOptions = file.parserOptions;
	if (file.dupey !== undefined) {
		if (typeof file.dupey !== "object" || file.dupey === null || Array.isArray(file.dupey)) {
			throw new ConfigError("Config field 'dupey' must be an object");
		}
		config.dupey = file.dupey as CliConfig["dupey"];
	}
	if (file.excludeExactDuplicates !== undefined && typeof file.excludeExactDuplicates !== "boolean") {
		throw new ConfigError("Config field 'excludeExactDuplicates' must be a boolean");
	}
	config.excludeExactDuplicates = file.excludeExactDuplicates ?? true;
	if (file.datasources !== undefined) {
		if (typeof file.datasources !== "object" || file.datasources === null || Array.isArray(file.datasources)) {
			throw new ConfigError("Config field 'datasources' must be an object mapping skill names to their config");
		}
		config.datasources = file.datasources as DatasourcesConfig;
	}
	if (file.datasourceAccess !== undefined) {
		if (
			typeof file.datasourceAccess !== "object" ||
			file.datasourceAccess === null ||
			Array.isArray(file.datasourceAccess)
		) {
			throw new ConfigError("Config field 'datasourceAccess' must be an object with allowedTags/allowedScopes");
		}
		config.datasourceAccess = file.datasourceAccess as DatasourceAccessContextOptions;
	}
	config.p2p = normalizeP2pConfig(file.p2p);
	if (file.rerank !== undefined) config.rerank = normalizeRerankConfig(file.rerank, "rerank");
	return config;
}

/**
 * Resolve config without writing, migrating, or locking. Health and other
 * non-destructive preflights use this so legacy cwd configs are read as search
 * would resolve them but never copied into `~/.autorag/config.json`.
 */
export function resolveConfigReadOnly(input: ResolveConfigInput): CliConfig {
	return resolveConfig({ ...input, readOnly: true });
}

const JEV_CONFIG_FIELDS: Record<string, true> = {
	enabled: true,
	backend: true,
	model: true,
	confidenceThreshold: true,
};

/** Backends the `jev-use` engine can resolve from the environment. */
const JEV_BACKEND_NAMES: readonly JevBackendName[] = ["typesafe", "openrouter", "vercel"];

/** Validate and normalize the `jev` config section. */
export function normalizeJevConfig(raw: unknown): JevCliConfig {
	if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
		throw new ConfigError("Config field 'jev' must be an object or false");
	}
	const record = raw as Record<string, unknown>;
	for (const key of Object.keys(record)) {
		if (JEV_CONFIG_FIELDS[key] !== true) {
			throw new ConfigError(`jev.${key} is not a recognized field`);
		}
	}
	const out: JevCliConfig = {};
	if (record.enabled !== undefined) {
		if (typeof record.enabled !== "boolean") throw new ConfigError("jev.enabled must be a boolean");
		out.enabled = record.enabled;
	}
	if (record.backend !== undefined) {
		const backend = record.backend;
		if (typeof backend !== "string" || !JEV_BACKEND_NAMES.some((known) => known === backend)) {
			throw new ConfigError(`jev.backend must be one of: ${JEV_BACKEND_NAMES.join(", ")}`);
		}
		// Narrowed by the membership check above.
		out.backend = backend as JevBackendName;
	}
	if (record.model !== undefined) {
		if (typeof record.model !== "string" || record.model.trim().length === 0) {
			throw new ConfigError("jev.model must be a non-empty string");
		}
		out.model = record.model.trim();
	}
	if (record.confidenceThreshold !== undefined) {
		if (
			typeof record.confidenceThreshold !== "number" ||
			!Number.isFinite(record.confidenceThreshold) ||
			record.confidenceThreshold < 0 ||
			record.confidenceThreshold > 1
		) {
			throw new ConfigError("jev.confidenceThreshold must be a number between 0 and 1");
		}
		out.confidenceThreshold = record.confidenceThreshold;
	}
	return out;
}

/** Jev backend used when the config names none (absent section or `{}`). */
export const DEFAULT_JEV_BACKEND: JevBackendName = "openrouter";

/**
 * Question-decomposition model used when the config names none. Picked from a
 * live OpenRouter bench (6 questions incl. Korean, 2 runs each): 12/12 valid,
 * covering, language-preserving decompositions at ~0.9s p50, about 3.5x
 * cheaper per call than google/gemini-2.5-flash-lite at equal quality.
 */
export const DEFAULT_QUERY_DECOMPOSITION_MODEL: AgentModelConfig = { provider: "openrouter", id: "qwen/qwen3.7-flash" };

/** Validate the `queryDecomposition` config section (`false` = decompose with the session model). */
export function normalizeQueryDecompositionConfig(raw: unknown): QueryDecompositionConfig | false {
	if (raw === false) return false;
	if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
		throw new ConfigError("Config field 'queryDecomposition' must be an object or false");
	}
	const record = raw as Record<string, unknown>;
	for (const key of Object.keys(record)) {
		if (key !== "model") throw new ConfigError(`queryDecomposition.${key} is not a recognized field`);
	}
	const model = modelReference(record.model, "queryDecomposition.model");
	return model === undefined ? {} : { model };
}

/**
 * Map the `jev` config section onto the agent option. Jev is on by default:
 * an absent section or `{}` enables it on {@link DEFAULT_JEV_BACKEND}; `false`
 * and `enabled: false` are the opt-out.
 */
function buildJevAgentOption(raw: JevCliConfig | false | undefined): AutoRAGAgentOptions["jev"] {
	if (raw === false) return false;
	const normalized = normalizeJevConfig(raw ?? {});
	if (normalized.enabled === false) return false;
	const { enabled: _omitJevEnabled, ...fields } = normalized;
	return { backend: DEFAULT_JEV_BACKEND, ...fields };
}

/** Validate and map the webSearch config section onto the agent option. */
function buildWebSearchAgentOption(
	raw: WebSearchCliConfig | undefined,
): (AutoRAGAgentOptions["webSearch"] & object) | false | undefined {
	if (raw === undefined) return undefined;
	if (raw.enabled === false) return false;
	if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
		throw new ConfigError("webSearch must be an object");
	}
	const out: Record<string, unknown> = {};
	if (raw.provider !== undefined) {
		if (typeof raw.provider !== "string" || !isSearchProviderId(raw.provider)) {
			throw new ConfigError(`webSearch.provider is not a recognized provider id: ${String(raw.provider)}`);
		}
		out.provider = raw.provider;
	}
	for (const key of ["order", "exclude"] as const) {
		const list = raw[key];
		if (list === undefined) continue;
		if (!Array.isArray(list) || list.some((id) => typeof id !== "string" || !isSearchProviderId(id))) {
			throw new ConfigError(`webSearch.${key} must be an array of recognized provider ids`);
		}
		out[key] = list;
	}
	if (raw.timeoutSeconds !== undefined) {
		if (
			typeof raw.timeoutSeconds !== "number" ||
			!Number.isInteger(raw.timeoutSeconds) ||
			raw.timeoutSeconds < 1 ||
			raw.timeoutSeconds > 300
		) {
			throw new ConfigError("webSearch.timeoutSeconds must be an integer between 1 and 300");
		}
		out.timeoutSeconds = raw.timeoutSeconds;
	}
	if (raw.fetch !== undefined) {
		if (raw.fetch === false) {
			out.fetch = false;
		} else {
			if (typeof raw.fetch !== "object" || raw.fetch === null || Array.isArray(raw.fetch)) {
				throw new ConfigError("webSearch.fetch must be an object or false");
			}
			if (raw.fetch.enabled === false) {
				out.fetch = false;
			} else {
				const fetchOut: Record<string, unknown> = {};
				if (raw.fetch.timeoutSeconds !== undefined) {
					if (
						typeof raw.fetch.timeoutSeconds !== "number" ||
						!Number.isInteger(raw.fetch.timeoutSeconds) ||
						raw.fetch.timeoutSeconds < 1 ||
						raw.fetch.timeoutSeconds > 300
					) {
						throw new ConfigError("webSearch.fetch.timeoutSeconds must be an integer between 1 and 300");
					}
					fetchOut.timeoutSeconds = raw.fetch.timeoutSeconds;
				}
				out.fetch = fetchOut;
			}
		}
	}
	return out as AutoRAGAgentOptions["webSearch"] & object;
}

const RERANK_ALLOWLIST = new Set<string>(["enabled", "provider", "model", "apiKeyEnv", "baseUrl", "topN", "timeoutMs"]);

/** Normalize and validate the `rerank` config section, filling in defaults. */
export function normalizeRerankConfig(raw: unknown, path: string): RerankConfig | false {
	if (raw === false) return false;
	const out: RerankConfig = {
		provider: DEFAULT_RERANK_PROVIDER,
		model: DEFAULT_RERANK_MODEL,
		apiKeyEnv: DEFAULT_RERANK_API_KEY_ENV,
		topN: DEFAULT_RERANK_TOP_N,
	};
	if (raw === undefined || raw === null) return out;
	if (typeof raw !== "object" || Array.isArray(raw)) {
		throw new ConfigError(`${path} must be an object or false`);
	}
	const record = raw as Record<string, unknown>;
	for (const key of Object.keys(record)) {
		if (!RERANK_ALLOWLIST.has(key)) throw new ConfigError(`${path}.${key} is not a recognized field`);
	}
	if (record.enabled !== undefined) {
		if (typeof record.enabled !== "boolean") throw new ConfigError(`${path}.enabled must be a boolean`);
		out.enabled = record.enabled;
	}
	if (record.provider !== undefined) {
		if (typeof record.provider !== "string" || record.provider.trim() === "") {
			throw new ConfigError(`${path}.provider must be a non-empty string`);
		}
		out.provider = record.provider.trim();
	}
	if (!SUPPORTED_RERANK_PROVIDERS.includes(out.provider ?? "")) {
		throw new ConfigError(`${path}.provider must be one of: ${SUPPORTED_RERANK_PROVIDERS.join(", ")}`);
	}
	if (record.model !== undefined) {
		if (typeof record.model !== "string" || record.model.trim() === "") {
			throw new ConfigError(`${path}.model must be a non-empty string`);
		}
		out.model = record.model.trim();
	}
	if (record.apiKeyEnv !== undefined) {
		if (typeof record.apiKeyEnv !== "string" || !API_KEY_ENV_PATTERN.test(record.apiKeyEnv)) {
			throw new ConfigError(`${path}.apiKeyEnv must match ${API_KEY_ENV_PATTERN}`);
		}
		out.apiKeyEnv = record.apiKeyEnv;
	}
	if (record.baseUrl !== undefined) {
		if (typeof record.baseUrl !== "string" || record.baseUrl.trim() === "") {
			throw new ConfigError(`${path}.baseUrl must be a non-empty string`);
		}
		out.baseUrl = record.baseUrl.trim();
	}
	for (const field of ["topN", "timeoutMs"] as const) {
		const value = record[field];
		if (value === undefined) continue;
		if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
			throw new ConfigError(`${path}.${field} must be a positive integer`);
		}
		out[field] = value;
	}
	return out;
}

export function buildAgentOptions(config: CliConfig): Omit<AutoRAGAgentOptions, "model"> {
	const opts: Record<string, unknown> = {
		searchPaths: config.searchPaths,
		languages: config.languages,
	};
	if (config.workspacePath) opts.workspacePath = config.workspacePath;
	if (config.memoryPath) opts.memoryPath = config.memoryPath;
	if (config.minSync && config.minSync.enabled !== false) {
		const { enabled: _omitMinSyncEnabled, ...minSyncFields } = config.minSync;
		opts.minSync = minSyncFields;
	} else {
		opts.minSync = false;
	}
	opts.jikji = config.jikji === false ? false : (config.jikji ?? {});
	if (config.everything === false || config.everything?.enabled === false) {
		opts.everything = false;
	} else if (config.everything !== undefined) {
		const { enabled: _omitEverythingEnabled, ...everythingFields } = config.everything;
		opts.everything = everythingFields;
	}
	if (config.fsearch === false || config.fsearch?.enabled === false) {
		opts.fsearch = false;
	} else if (config.fsearch !== undefined) {
		const { enabled: _omitFSearchEnabled, ...fsearchFields } = config.fsearch;
		opts.fsearch = fsearchFields;
	}
	opts.webSearch = buildWebSearchAgentOption(config.webSearch);
	opts.jev = buildJevAgentOption(config.jev);
	// The Jev `config` branch edits the very file this process resolved; it only
	// takes effect when Jev is on, so the option is always safe to pass.
	if (config.configPath !== undefined) opts.selfConfig = { configPath: config.configPath };
	if (config.rerank !== undefined) {
		opts.rerank =
			config.rerank === false || config.rerank.enabled === false
				? false
				: {
						...(config.rerank.provider !== undefined ? { provider: config.rerank.provider } : {}),
						...(config.rerank.model !== undefined ? { model: config.rerank.model } : {}),
						...(config.rerank.apiKeyEnv !== undefined ? { apiKeyEnv: config.rerank.apiKeyEnv } : {}),
						...(config.rerank.baseUrl !== undefined ? { baseUrl: config.rerank.baseUrl } : {}),
						...(config.rerank.topN !== undefined ? { topN: config.rerank.topN } : {}),
						...(config.rerank.timeoutMs !== undefined ? { timeoutMs: config.rerank.timeoutMs } : {}),
					};
	}
	if (config.parserOptions) opts.parserOptions = config.parserOptions;
	if (config.dupey?.enabled === false) {
		opts.dupey = false;
	} else {
		opts.dupey = {
			...(config.dupey?.binaryPath ? { executable: config.dupey.binaryPath } : {}),
			...(config.dupey?.timeoutMs ? { timeoutMs: config.dupey.timeoutMs } : {}),
		};
	}
	opts.excludeExactDuplicates = config.excludeExactDuplicates ?? true;
	if (config.excludePaths !== undefined) opts.excludePaths = config.excludePaths;
	if (config.limits !== undefined) opts.limits = config.limits;
	if (config.datasources !== undefined) {
		const { skills, unknown } = buildDatasourceSkills(config.datasources, config.workspacePath);
		if (skills.length > 0) opts.datasourceSkills = skills;
		if (unknown.length > 0) {
			const safeNames = unknown.map((name) => name.replace(/[^A-Za-z0-9._-]/g, "?").slice(0, 80));
			const diagnostic: SearchDocumentDiagnostic = {
				code: "unknown-datasource-skill",
				severity: "warning",
				message: `Unknown datasource skill(s) in config were skipped: ${safeNames.join(", ")}`,
				source: "datasources",
			};
			opts.startupDiagnostics = [diagnostic];
		}
	}
	if (config.datasourceAccess !== undefined) opts.datasourceAccess = config.datasourceAccess;
	if (config.p2p?.enabled === true) {
		opts.peerQuery = {
			...(config.p2p.port !== undefined ? { port: config.p2p.port } : {}),
			...(config.p2p.simplexDbPrefix !== undefined ? { simplexDbPrefix: config.p2p.simplexDbPrefix } : {}),
		};
	} else {
		opts.peerQuery = false;
	}
	return opts as Omit<AutoRAGAgentOptions, "model">;
}

/** True when the role config declares an explicit OpenAI-compatible endpoint. */
function isConfiguredEndpoint(
	reference: AgentModelConfig | undefined,
): reference is AgentModelConfig & { baseUrl: string } {
	return typeof reference?.baseUrl === "string" && reference.baseUrl.trim().length > 0;
}

function configuredApiKeyEnv(reference: AgentModelConfig): string {
	return reference.apiKeyEnv ?? providerApiKeyEnvName(reference.provider);
}

/**
 * Build a pi-ai Model from a config-declared OpenAI-compatible endpoint.
 * Any provider works: OpenRouter, Fireworks, Ollama, LiteLLM, corporate proxies, etc.
 */
function buildModelFromConfiguredEndpoint(reference: AgentModelConfig & { baseUrl: string }): Model<Api> {
	const api = reference.api ?? "openai-completions";
	return {
		id: reference.id,
		name: reference.name ?? reference.id,
		api,
		provider: reference.provider,
		baseUrl: reference.baseUrl,
		reasoning: reference.reasoning === true,
		input: reference.input ?? ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: reference.contextWindow ?? 128_000,
		maxTokens: reference.maxTokens ?? 16_384,
	};
}

/**
 * Merge a config model reference over the pi runtime catalog entry (built-in
 * provider catalog plus `models.json`, custom, and extension providers). The
 * catalog entry is the base; only fields the config declares override it, so a
 * configured endpoint keeps the catalog's reasoning, compat, thinking map, and
 * limits.
 */
function mergeCatalogModel(catalog: Model<Api>, reference: AgentModelConfig): Model<Api> {
	return {
		...catalog,
		...(reference.name !== undefined ? { name: reference.name } : {}),
		...(reference.api !== undefined ? { api: reference.api } : {}),
		...(isConfiguredEndpoint(reference) ? { baseUrl: reference.baseUrl } : {}),
		...(reference.reasoning !== undefined ? { reasoning: reference.reasoning } : {}),
		...(reference.input !== undefined ? { input: reference.input } : {}),
		...(reference.contextWindow !== undefined ? { contextWindow: reference.contextWindow } : {}),
		...(reference.maxTokens !== undefined ? { maxTokens: reference.maxTokens } : {}),
	};
}

/** Resolve a config model reference against the pi runtime catalog. */
function resolveRuntimeCatalogModel(runtime: ModelRuntime, reference: AgentModelConfig): Model<Api> | undefined {
	const catalog = runtime.getModel(reference.provider, reference.id) as Model<Api> | undefined;
	if (catalog === undefined) return undefined;
	return mergeCatalogModel(catalog, reference);
}

const UNKNOWN_MODEL_HINT =
	"Add baseUrl (and optional api/apiKeyEnv) for an OpenAI-compatible endpoint outside the pi-ai catalog, or use a pi-ai catalog model id.";

function resolveRegisteredModel(runtime: ModelRuntime, reference: AgentModelConfig): Model<Api> {
	const catalog = resolveRuntimeCatalogModel(runtime, reference);
	if (catalog !== undefined) return catalog;
	if (isConfiguredEndpoint(reference)) return buildModelFromConfiguredEndpoint(reference);
	throw new ConfigError(`Unknown configured model: ${reference.provider}/${reference.id}. ${UNKNOWN_MODEL_HINT}`);
}

function resolveBuiltInModel(runtime: ModelRuntime, reference: AgentModelConfig | undefined): Model<Api> | undefined {
	if (reference === undefined) return undefined;
	const catalog = resolveRuntimeCatalogModel(runtime, reference);
	if (catalog !== undefined) return catalog;
	if (isConfiguredEndpoint(reference)) return buildModelFromConfiguredEndpoint(reference);
	// Known catalog provider with an unknown model id is a hard config error.
	// Unknown providers fall through so a local runtime (e.g. codex proxy) can supply them.
	if (runtime.getProvider(reference.provider) !== undefined) {
		throw new ConfigError(`Unknown configured model: ${reference.provider}/${reference.id}. ${UNKNOWN_MODEL_HINT}`);
	}
	return undefined;
}

/**
 * Extra resolution inputs beyond the local codex-runtime options. `agentDir`
 * points at the pi agent home (`~/.pi/agent` by default) that owns
 * `auth.json`, `models.json`, and `settings.json`.
 */
export interface ResolveAgentModelOptions extends LoadLocalAutoRAGModelOptions {
	readonly agentDir?: string;
	readonly cwd?: string;
	/** Pre-built pi model runtime, for callers that already own one (and tests). */
	readonly runtime?: ModelRuntime;
}

const modelRuntimeCache = new Map<string, Promise<ModelRuntime>>();

function resolveAgentDir(options: ResolveAgentModelOptions): string {
	return options.agentDir ?? getAgentDir();
}

/**
 * Load the pi model runtime for an agent home. The runtime composes the
 * built-in catalog with `models.json`, custom, and extension providers and
 * reads stored credentials (`auth.json`, including OAuth). Network catalog
 * refresh is disabled so model resolution stays fast and offline-safe.
 */
function getModelRuntime(agentDir: string): Promise<ModelRuntime> {
	const cached = modelRuntimeCache.get(agentDir);
	if (cached !== undefined) return cached;
	const created = ModelRuntime.create({
		authPath: join(agentDir, "auth.json"),
		modelsPath: join(agentDir, "models.json"),
		allowModelNetwork: false,
	});
	modelRuntimeCache.set(agentDir, created);
	created.catch(() => {
		modelRuntimeCache.delete(agentDir);
	});
	return created;
}

export async function resolveModel(config: CliConfig, options: ResolveAgentModelOptions = {}): Promise<Model<Api>> {
	if (!config.model) {
		throw new ConfigError(
			'No model configured. Provide --model-provider and --model-id on the command line, or set the "model" key (with provider and id) in the config file.',
		);
	}
	const runtime = options.runtime ?? (await getModelRuntime(resolveAgentDir(options)));
	return resolveRegisteredModel(runtime, config.model);
}

export interface ResolvedAgentModel {
	readonly model: Model<Api>;
	readonly apiKey?: string;
	readonly providerApiKeys?: Readonly<Record<string, string>>;
}

export type AgentModelResolutionSource =
	| "config"
	| "flags"
	| "env"
	| "local_runtime"
	| "catalog"
	| "configured_alias"
	| "mixed";

export interface AgentModelAuth {
	readonly present: boolean;
	readonly source: "env" | "local_runtime" | "pi_auth" | "catalog" | "none" | "unknown";
	readonly envName?: string;
}

export interface ResolvedAgentModelRole {
	readonly provider: string;
	readonly modelId: string;
	readonly displayName: string;
	readonly api: Api;
	readonly baseUrl: string | undefined;
	readonly contextWindow: number | undefined;
	readonly maxTokens: number | undefined;
	readonly capabilities: { readonly input: readonly string[]; readonly reasoning: boolean };
	readonly auth: AgentModelAuth;
	readonly resolutionSource: AgentModelResolutionSource;
}

export interface ResolvedAgentModelDetailed {
	readonly model: Model<Api>;
	readonly apiKey?: string;
	readonly providerApiKeys?: Readonly<Record<string, string>>;
	readonly role: ResolvedAgentModelRole;
}

interface ResolvedModelAuth {
	readonly apiKey?: string;
	readonly providerApiKeys?: Readonly<Record<string, string>>;
	readonly present: boolean;
	readonly source: AgentModelAuth["source"];
	readonly envName?: string;
}

interface AgentModelCore {
	readonly model: Model<Api>;
	readonly modelRef: AgentModelConfig | undefined;
	readonly fromLocal: boolean;
	readonly configuredEndpoint: boolean;
	readonly catalog: boolean;
	readonly local: LocalAutoRAGModel | undefined;
	readonly env: NodeJS.ProcessEnv;
	readonly auth: ResolvedModelAuth;
}

function localFallbackOptions(
	options: ResolveAgentModelOptions,
	modelId: string | undefined,
): LoadLocalAutoRAGModelOptions {
	return {
		...(options.configPath !== undefined ? { configPath: options.configPath } : {}),
		...(options.env !== undefined ? { env: options.env } : {}),
		...(modelId !== undefined ? { modelId } : {}),
	};
}

function localRuntimeAuth(local: LocalAutoRAGModel): ResolvedModelAuth {
	return {
		apiKey: local.apiKey,
		providerApiKeys: { [local.provider]: local.apiKey },
		present: true,
		source: "local_runtime",
	};
}

/**
 * Resolve the model selected through pi settings (`defaultProvider`/
 * `defaultModel`) when AutoRAG has no explicit model. Returns undefined when
 * settings name no model, the runtime does not know it, or the provider has no
 * configured credential — callers then fall back to the local codex runtime.
 */
async function resolvePiDefaultModel(
	runtime: ModelRuntime,
	options: ResolveAgentModelOptions,
): Promise<Model<Api> | undefined> {
	let provider: string | undefined;
	let id: string | undefined;
	try {
		const settings = SettingsManager.create(options.cwd ?? process.cwd(), resolveAgentDir(options));
		provider = settings.getDefaultProvider();
		id = settings.getDefaultModel();
	} catch {
		return undefined;
	}
	if (provider === undefined || id === undefined) return undefined;
	const model = runtime.getModel(provider, id) as Model<Api> | undefined;
	if (model === undefined) return undefined;
	try {
		if ((await runtime.checkAuth(provider)) === undefined) return undefined;
	} catch {
		return undefined;
	}
	return model;
}

function resolveConfiguredEndpointAuth(
	reference: AgentModelConfig & { baseUrl: string },
	model: Model<Api>,
	env: NodeJS.ProcessEnv,
): ResolvedModelAuth {
	const envName = configuredApiKeyEnv(reference);
	const value = env[envName];
	const providerApiKeys = typeof value === "string" && value.length > 0 ? { [model.provider]: value } : undefined;
	const fromProcessEnv = process.env[envName];
	const present = providerApiKeys !== undefined || (typeof fromProcessEnv === "string" && fromProcessEnv.length > 0);
	return {
		...(providerApiKeys !== undefined ? { apiKey: providerApiKeys[model.provider], providerApiKeys } : {}),
		present,
		source: present ? "env" : "none",
		envName,
	};
}

/**
 * Resolve credentials for a catalog/custom model through the pi runtime: the
 * stored `auth.json` credential (including refreshed OAuth) wins, then the
 * provider's environment variables, then an explicit none.
 */
async function resolveCatalogAuth(
	runtime: ModelRuntime,
	model: Model<Api>,
	env: NodeJS.ProcessEnv,
): Promise<ResolvedModelAuth> {
	try {
		const result = await runtime.getAuth(model);
		if (result !== undefined) {
			const apiKey = result.auth.apiKey;
			if (typeof apiKey === "string" && apiKey.length > 0) {
				return { apiKey, providerApiKeys: { [model.provider]: apiKey }, present: true, source: "pi_auth" };
			}
			// Headers-only auth (for example OAuth): pi resolves it inside the session.
			return { present: true, source: "pi_auth" };
		}
	} catch {
		// Fall through to environment-based reporting; a broken stored credential
		// must not make model resolution itself fail.
	}
	const envKeys = findEnvKeys(model.provider);
	if (envKeys !== undefined && envKeys.length > 0) {
		for (const key of envKeys) {
			const fromOptionEnv = env[key];
			const fromProcessEnv = process.env[key];
			if (typeof fromOptionEnv === "string" && fromOptionEnv.length > 0) {
				return { present: true, source: "catalog", envName: key };
			}
			if (typeof fromProcessEnv === "string" && fromProcessEnv.length > 0) {
				return { present: true, source: "catalog", envName: key };
			}
		}
		return { present: false, source: "none", envName: envKeys[0] };
	}
	if (getEnvApiKey(model.provider) !== undefined) {
		return { present: true, source: "catalog" };
	}
	return { present: false, source: "none", envName: providerApiKeyEnvName(model.provider) };
}

async function resolveAgentModelCore(
	config: CliConfig,
	options: ResolveAgentModelOptions = {},
): Promise<AgentModelCore> {
	const env = options.env ?? process.env;
	const runtime = options.runtime ?? (await getModelRuntime(resolveAgentDir(options)));
	const modelRef = config.model;

	// 1. Explicit OpenAI-compatible endpoint in config wins over pi credentials.
	if (modelRef !== undefined && isConfiguredEndpoint(modelRef)) {
		const catalogModel = resolveRuntimeCatalogModel(runtime, modelRef);
		const model = catalogModel ?? buildModelFromConfiguredEndpoint(modelRef);
		return {
			model,
			modelRef,
			fromLocal: false,
			configuredEndpoint: true,
			catalog: catalogModel !== undefined,
			local: undefined,
			env,
			auth: resolveConfiguredEndpointAuth(modelRef, model, env),
		};
	}

	// 2. Config model id resolves against the pi runtime catalog.
	const registered = resolveBuiltInModel(runtime, modelRef);
	if (registered !== undefined) {
		return {
			model: registered,
			modelRef,
			fromLocal: false,
			configuredEndpoint: false,
			catalog: true,
			local: undefined,
			env,
			auth: await resolveCatalogAuth(runtime, registered, env),
		};
	}

	// 3. No model configured: prefer a pi-selected usable model, then the local default runtime.
	if (modelRef === undefined) {
		const piDefault = await resolvePiDefaultModel(runtime, options);
		if (piDefault !== undefined) {
			return {
				model: piDefault,
				modelRef,
				fromLocal: false,
				configuredEndpoint: false,
				catalog: true,
				local: undefined,
				env,
				auth: await resolveCatalogAuth(runtime, piDefault, env),
			};
		}
		const local = loadLocalAutoRAGModel(localFallbackOptions(options, undefined));
		return {
			model: local.model as Model<Api>,
			modelRef,
			fromLocal: true,
			configuredEndpoint: false,
			catalog: false,
			local,
			env,
			auth: localRuntimeAuth(local),
		};
	}

	// 4. Config names a provider outside the catalog: a local runtime may supply it.
	const local = loadLocalAutoRAGModel(localFallbackOptions(options, modelRef.id));
	const fromLocal = modelRef.provider === local.provider;
	const model = fromLocal ? (local.model as Model<Api>) : resolveRegisteredModel(runtime, modelRef);
	return {
		model,
		modelRef,
		fromLocal,
		configuredEndpoint: false,
		catalog: !fromLocal,
		local: fromLocal ? local : undefined,
		env,
		auth: fromLocal ? localRuntimeAuth(local) : await resolveCatalogAuth(runtime, model, env),
	};
}

export async function resolveAgentModel(
	config: CliConfig,
	options: ResolveAgentModelOptions = {},
): Promise<ResolvedAgentModel> {
	const core = await resolveAgentModelCore(config, options);
	return {
		model: core.model,
		...(core.auth.apiKey !== undefined ? { apiKey: core.auth.apiKey } : {}),
		...(core.auth.providerApiKeys !== undefined ? { providerApiKeys: core.auth.providerApiKeys } : {}),
	};
}

/**
 * Resolve the question-decomposition model and its credential through the same
 * chain as the agent model. An absent section or `{}` uses
 * {@link DEFAULT_QUERY_DECOMPOSITION_MODEL}; `queryDecomposition: false`
 * returns undefined so the agent's own model decomposes.
 */
export async function resolveQueryDecompositionModel(
	config: CliConfig,
	options: ResolveAgentModelOptions = {},
): Promise<DecompositionModel | undefined> {
	if (config.queryDecomposition === false) return undefined;
	const reference = config.queryDecomposition?.model ?? DEFAULT_QUERY_DECOMPOSITION_MODEL;
	const resolved = await resolveAgentModel({ ...config, model: reference }, options);
	const apiKey = resolved.apiKey ?? resolved.providerApiKeys?.[resolved.model.provider];
	return { model: resolved.model, ...(apiKey !== undefined ? { apiKey } : {}) };
}

function providerApiKeyEnvName(provider: string): string {
	return `${provider.replace(/[^A-Za-z0-9_]/g, "_").toUpperCase()}_API_KEY`;
}

function resolveRoleSource(
	ref: AgentModelConfig | undefined,
	fromLocal: boolean,
	configuredEndpoint: boolean,
	catalog: boolean,
): AgentModelResolutionSource {
	if (configuredEndpoint) return "config";
	if (fromLocal) return "mixed";
	if (catalog) return "catalog";
	if (ref === undefined) return "local_runtime";
	return "config";
}

function buildResolvedRole(
	model: Model<Api>,
	auth: AgentModelAuth,
	resolutionSource: AgentModelResolutionSource,
): ResolvedAgentModelRole {
	return {
		provider: model.provider,
		modelId: model.id,
		displayName: model.name,
		api: model.api,
		baseUrl: model.baseUrl,
		contextWindow: model.contextWindow,
		maxTokens: model.maxTokens,
		capabilities: { input: model.input ?? [], reasoning: model.reasoning === true },
		auth,
		resolutionSource,
	};
}

export async function resolveAgentModelDetailed(
	config: CliConfig,
	options: ResolveAgentModelOptions = {},
): Promise<ResolvedAgentModelDetailed> {
	const core = await resolveAgentModelCore(config, options);
	const auth: AgentModelAuth = {
		present: core.auth.present,
		source: core.auth.source,
		...(core.auth.envName !== undefined ? { envName: core.auth.envName } : {}),
	};
	const source = resolveRoleSource(core.modelRef, core.fromLocal, core.configuredEndpoint, core.catalog);
	return {
		model: core.model,
		...(core.auth.apiKey !== undefined ? { apiKey: core.auth.apiKey } : {}),
		...(core.auth.providerApiKeys !== undefined ? { providerApiKeys: core.auth.providerApiKeys } : {}),
		role: buildResolvedRole(core.model, auth, source),
	};
}

export function readRawConfigObject(path: string): Record<string, unknown> {
	const parsed = readConfigFile(path, true);
	if (parsed === undefined) throw new ConfigError(`Config file not found: ${path}`);
	return { ...(parsed as Record<string, unknown>) };
}

/** Atomically replace a config file, preserving the caller's JSON object. */
export function writeConfigObject(path: string, config: unknown): void {
	if (config === null || typeof config !== "object" || Array.isArray(config)) {
		throw new ConfigError("Config file must be a JSON object");
	}
	mkdirSync(dirname(path), { recursive: true });
	const contents = `${JSON.stringify(config, null, 2)}\n`;
	const lock = acquireConfigWriteLock(path);
	try {
		replaceFileAtomically(path, contents, lock.assertOwned);
	} finally {
		lock.release();
	}
}

export function writeDefaultConfig(
	path: string,
	partial: Partial<CliConfig>,
	opts: {
		force?: boolean;
		atomicCreate?: boolean;
		cwd?: string;
		env?: NodeJS.ProcessEnv;
		/**
		 * Whether the target path was selected explicitly by the caller
		 * (`--config` / `AUTORAG_CONFIG`). When `false`, `force` refuses to
		 * replace an existing file: an implicit home config may only be
		 * replaced through an explicit config path. `undefined` (callers that
		 * do not distinguish) keeps the historical force semantics.
		 */
		explicit?: boolean;
	} = {},
): void {
	const cwd = resolve(opts.cwd ?? process.cwd());
	const workspacePath = resolvePersistedPath(partial.workspacePath ?? ".", cwd);
	const memoryPath =
		partial.memoryPath === undefined
			? join(resolveAutoRAGHome(opts.env), "memory.json")
			: resolvePersistedPath(partial.memoryPath, workspacePath);
	const full: CliConfig = {
		searchPaths: resolveSearchPaths(partial.searchPaths ?? ["."], cwd),
		workspacePath,
		memoryPath,
		languages: normalizeConfigLanguages(partial.languages),
	};
	if (partial.model) full.model = partial.model;
	// Indexing method defaults: enabled when not explicitly provided.
	// Never inject embedder id defaults; preserve partial embedder config as-is.
	const normalizedMethods = normalizeIndexingConfig({
		minSync: partial.minSync as MinSyncMethodConfig | false | undefined,
	});
	full.minSync = normalizedMethods.minSync;
	// Jikji find-first discovery is enabled by default for new configs; the CLI
	// auto-installs the jikji binary on first use when cargo is available.
	full.jikji = partial.jikji ?? {};
	full.dupey = partial.dupey ?? { enabled: true };
	full.excludeExactDuplicates = partial.excludeExactDuplicates ?? true;
	if (partial.excludePaths !== undefined) full.excludePaths = resolveSearchPaths(partial.excludePaths, workspacePath);
	if (partial.limits !== undefined) full.limits = normalizeLimitsConfig(partial.limits);
	if (partial.parserOptions) full.parserOptions = partial.parserOptions;
	full.rerank = partial.rerank ?? {
		provider: DEFAULT_RERANK_PROVIDER,
		model: DEFAULT_RERANK_MODEL,
		apiKeyEnv: DEFAULT_RERANK_API_KEY_ENV,
		topN: DEFAULT_RERANK_TOP_N,
	};
	if (partial.p2p !== undefined) full.p2p = normalizeP2pConfig(partial.p2p);
	else full.p2p = { enabled: false };
	// Jev routing and question decomposition are on by default; new configs
	// spell the defaults out so they are visible and editable.
	full.jev = partial.jev ?? { backend: DEFAULT_JEV_BACKEND };
	full.queryDecomposition = partial.queryDecomposition ?? { model: { ...DEFAULT_QUERY_DECOMPOSITION_MODEL } };
	mkdirSync(dirname(path), { recursive: true });
	const contents = `${JSON.stringify(full, null, 2)}\n`;
	const lock = acquireConfigWriteLock(path);
	try {
		const exists = existsSync(path);
		if (!opts.force && exists) {
			throw new ConfigError(`Config file already exists: ${path}`);
		}
		// `--force` must not silently replace an implicit home config: the
		// caller has to name the config explicitly (--config / AUTORAG_CONFIG).
		// The check runs inside the write lock so a concurrent first-time
		// writer still wins over a stale pre-lock existsSync.
		if (opts.force && opts.explicit === false && exists) {
			throw new ConfigError(
				`Refusing to overwrite existing config ${path} without an explicit config path. ` +
					"Pass --config <path> or set AUTORAG_CONFIG to select the config, or re-run without --force.",
			);
		}
		if (opts.force || opts.atomicCreate) replaceFileAtomically(path, contents, lock.assertOwned);
		else {
			lock.assertOwned();
			writeFileSync(path, contents, { encoding: "utf8", flag: "wx", flush: true, mode: 0o600 });
		}
	} catch (error) {
		if (!opts.force && isEexistError(error)) {
			throw new ConfigError(`Config file already exists: ${path}`);
		}
		throw error;
	} finally {
		lock.release();
	}
}

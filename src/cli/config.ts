import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import { basename, dirname, isAbsolute, join, resolve } from "node:path";
import type { Api, Model } from "@earendil-works/pi-ai";
import { findEnvKeys, getEnvApiKey, getModel, getProviders } from "@earendil-works/pi-ai/compat";
import type { AutoRAGAgentOptions } from "../agent/agent.ts";
import { resolveAutoRAGHome } from "../config/home.ts";
import type { DatasourceAccessContextOptions } from "../datasource/access-context.ts";
import { buildDatasourceSkills, type DatasourcesConfig } from "../datasource/skills/factory.ts";
import { acquireFileLock, type FileLockHandle } from "../filesystem/file-lock.ts";
import type { EnsureMinSyncBinaryOptions, MinSyncEmbedderConfig } from "../minsync/index.ts";
import type { BM25Engine, BM25FallbackMode } from "../retrieval/methods/bm25.ts";
import {
	type LoadLocalAutoRAGModelsOptions,
	type LocalAutoRAGModels,
	loadLocalAutoRAGModels,
} from "../subagents/local-models.ts";

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
 * Typed BM25 method config persisted in `config.json`. Missing `enabled` means
 * enabled (true). `false` as a top-level value is also accepted and disables.
 */
export interface Bm25MethodConfig {
	enabled?: boolean;
	indexPath?: string;
	fallback?: BM25FallbackMode;
	forceEngine?: Exclude<BM25Engine, "none">;
	importBinding?: () => Promise<typeof import("@pngwasi/node-tantivy-binding")>;
}

/**
 * Typed MinSync method config persisted in `config.json`. Missing `enabled`
 * means enabled (true); `autoInstall` defaults to false. `embedder` carries
 * the MinSync vector embedder settings validated by {@link normalizeEmbedder}.
 */
export interface MinSyncMethodConfig {
	enabled?: boolean;
	autoInstall?: boolean;
	binaryPath?: string;
	workspacePath?: string;
	installer?: Omit<EnsureMinSyncBinaryOptions, "root">;
	embedder?: MinSyncEmbedderConfig;
}

/** Indexing method config as it appears in a raw config file (before normalization). */
export interface RawIndexingMethods {
	bm25?: Bm25MethodConfig | false;
	minSync?: MinSyncMethodConfig | false;
}

/** Result of {@link normalizeIndexingConfig}: always fully populated. */
export interface NormalizedIndexingConfig {
	bm25: Bm25MethodConfig;
	minSync: MinSyncMethodConfig;
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
	 * Endpoint base URL. When set, AutoRAG builds a Model from this config
	 * instead of requiring a pi-ai catalog entry. Omit for catalog/local models.
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
	searchPaths: string[];
	workspacePath: string;
	memoryPath: string;
	/** @deprecated Prefer agents.orchestrator. Kept for legacy single-model configs. */
	model?: AgentModelConfig;
	agents?: {
		orchestrator?: AgentModelConfig;
		explorer?: AgentModelConfig;
	};
	minSync?: MinSyncMethodConfig;
	bm25?: Bm25MethodConfig;
	jikji?: Record<string, unknown>;
	parserOptions?: Record<string, unknown>;
	/** Trusted datasource skill configuration (skill name → config). */
	datasources?: DatasourcesConfig;
	/** Trusted datasource allow-tags/allow-scopes. Absent ⇒ default-deny. */
	datasourceAccess?: DatasourceAccessContextOptions;
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
	const migratedBytes =
		legacy?.workspacePath === migrated.workspacePath &&
		legacy?.memoryPath === migrated.memoryPath &&
		JSON.stringify(legacy?.searchPaths) === JSON.stringify(migrated.searchPaths)
			? legacyBytes
			: `${JSON.stringify(migrated, null, 2)}\n`;
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
const BM25_ALLOWLIST = new Set<string>(["enabled", "indexPath", "fallback", "forceEngine", "importBinding"]);
const MINSYNC_ALLOWLIST = new Set<string>([
	"enabled",
	"autoInstall",
	"binaryPath",
	"workspacePath",
	"installer",
	"embedder",
]);

function normalizeBm25Method(raw: Bm25MethodConfig | false | undefined): Bm25MethodConfig {
	if (raw === false) return { enabled: false };
	if (raw === undefined || raw === null) return { enabled: true };
	if (typeof raw !== "object" || Array.isArray(raw)) {
		throw new ConfigError("bm25 must be an object or false");
	}
	const record = raw as Record<string, unknown>;
	for (const key of Object.keys(record)) {
		if (!BM25_ALLOWLIST.has(key)) {
			throw new ConfigError(`bm25.${key} is not a recognized field`);
		}
	}
	const enabled = record.enabled !== false;
	const out: Bm25MethodConfig = { enabled };
	if (typeof record.indexPath === "string" && record.indexPath.length > 0) out.indexPath = record.indexPath;
	if (record.fallback === "typescript" || record.fallback === "disabled") out.fallback = record.fallback;
	if (record.forceEngine === "tantivy" || record.forceEngine === "typescript-fallback") {
		out.forceEngine = record.forceEngine;
	}
	if (typeof record.importBinding === "function") {
		out.importBinding = record.importBinding as Bm25MethodConfig["importBinding"];
	}
	return out;
}

function normalizeMinSyncMethod(raw: MinSyncMethodConfig | false | undefined): MinSyncMethodConfig {
	if (raw === false) return { enabled: false };
	if (raw === undefined || raw === null) return { enabled: true, autoInstall: false };
	if (typeof raw !== "object" || Array.isArray(raw)) {
		throw new ConfigError("minSync must be an object or false");
	}
	const record = raw as Record<string, unknown>;
	for (const key of Object.keys(record)) {
		if (!MINSYNC_ALLOWLIST.has(key)) {
			throw new ConfigError(`minSync.${key} is not a recognized field`);
		}
	}
	const enabled = record.enabled !== false;
	const out: MinSyncMethodConfig = { enabled, autoInstall: record.autoInstall === true };
	if (typeof record.binaryPath === "string" && record.binaryPath.length > 0) out.binaryPath = record.binaryPath;
	if (typeof record.workspacePath === "string" && record.workspacePath.length > 0) {
		out.workspacePath = record.workspacePath;
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
 * - `undefined` / missing key => `{ enabled: true }` (minSync `autoInstall: false`)
 * - `false` => `{ enabled: false }` (disabled marker)
 * - object merges with `enabled: true` default and is validated
 *
 * Unknown fields, invalid embedder settings, and bad numeric values throw
 * {@link ConfigError}.
 */
export function normalizeIndexingConfig(raw: RawIndexingMethods): NormalizedIndexingConfig {
	return {
		bm25: normalizeBm25Method(raw.bm25),
		minSync: normalizeMinSyncMethod(raw.minSync),
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

	const fileOrchestrator = modelReference(
		(file.agents as { orchestrator?: unknown } | undefined)?.orchestrator ?? file.model,
		"agents.orchestrator",
	);
	const fileExplorer = modelReference(
		(file.agents as { explorer?: unknown } | undefined)?.explorer,
		"agents.explorer",
	);
	const flagOrchestratorProvider =
		pickString(flags, env, "orchestrator-model-provider", "AUTORAG_ORCHESTRATOR_MODEL_PROVIDER", undefined) ??
		pickString(flags, env, "model-provider", "AUTORAG_MODEL_PROVIDER", undefined);
	const flagOrchestratorId =
		pickString(flags, env, "orchestrator-model-id", "AUTORAG_ORCHESTRATOR_MODEL_ID", undefined) ??
		pickString(flags, env, "model-id", "AUTORAG_MODEL_ID", undefined);
	const flagExplorerProvider = pickString(
		flags,
		env,
		"explorer-model-provider",
		"AUTORAG_EXPLORER_MODEL_PROVIDER",
		undefined,
	);
	const flagExplorerId = pickString(flags, env, "explorer-model-id", "AUTORAG_EXPLORER_MODEL_ID", undefined);

	const orchestrator = applyRoleFlagOverrides(
		fileOrchestrator,
		flagOrchestratorProvider,
		flagOrchestratorId,
		"agents.orchestrator",
	);
	const explorer = applyRoleFlagOverrides(fileExplorer, flagExplorerProvider, flagExplorerId, "agents.explorer");

	const config: CliConfig = {
		searchPaths,
		workspacePath,
		memoryPath,
	};
	if (orchestrator) {
		// Legacy single-model field remains provider+id only.
		config.model = { provider: orchestrator.provider, id: orchestrator.id };
	}
	if (orchestrator || explorer) {
		config.agents = {
			...(orchestrator ? { orchestrator } : {}),
			...(explorer ? { explorer } : {}),
		};
	}
	const normalized = normalizeIndexingConfig({
		bm25: file.bm25 as Bm25MethodConfig | false | undefined,
		minSync: file.minSync as MinSyncMethodConfig | false | undefined,
	});
	config.bm25 = normalized.bm25;
	config.minSync = normalized.minSync;
	if (file.jikji) config.jikji = file.jikji;
	if (file.parserOptions) config.parserOptions = file.parserOptions;
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

export function buildAgentOptions(config: CliConfig): Omit<AutoRAGAgentOptions, "model"> {
	const opts: Record<string, unknown> = {
		searchPaths: config.searchPaths,
	};
	if (config.workspacePath) opts.workspacePath = config.workspacePath;
	if (config.memoryPath) opts.memoryPath = config.memoryPath;
	if (config.minSync && config.minSync.enabled !== false) {
		const { enabled: _omitMinSyncEnabled, ...minSyncFields } = config.minSync;
		opts.minSync = minSyncFields;
	} else {
		opts.minSync = false;
	}
	if (config.bm25 && config.bm25.enabled !== false) {
		const { enabled: _omitBm25Enabled, ...bm25Fields } = config.bm25;
		opts.bm25 = bm25Fields;
	} else {
		opts.bm25 = false;
	}
	if (config.jikji) opts.jikji = config.jikji;
	if (config.parserOptions) opts.parserOptions = config.parserOptions;
	if (config.datasources !== undefined) {
		const { skills, unknown } = buildDatasourceSkills(config.datasources, config.workspacePath);
		if (unknown.length > 0) {
			throw new ConfigError(`Unknown datasource skill(s) in config: ${unknown.join(", ")}`);
		}
		if (skills.length > 0) opts.datasourceSkills = skills;
	}
	if (config.datasourceAccess !== undefined) opts.datasourceAccess = config.datasourceAccess;
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

function resolveCatalogModel(reference: AgentModelConfig): Model<Api> | undefined {
	if (!(getProviders() as readonly string[]).includes(reference.provider)) return undefined;
	return getModel(reference.provider as never, reference.id as never) as Model<Api> | undefined;
}

function resolveRegisteredModel(reference: AgentModelConfig, role?: "orchestrator" | "explorer"): Model<Api> {
	if (isConfiguredEndpoint(reference)) {
		return buildModelFromConfiguredEndpoint(reference);
	}
	const catalog = resolveCatalogModel(reference);
	if (catalog !== undefined) return catalog;
	const roleLabel = role === undefined ? "" : `${role} `;
	throw new ConfigError(
		`Unknown configured ${roleLabel}model: ${reference.provider}/${reference.id}. ` +
			`Add baseUrl (and optional api/apiKeyEnv) for OpenAI-compatible endpoints, or use a pi-ai catalog model id.`,
	);
}

function resolveBuiltInModel(
	reference: AgentModelConfig | undefined,
	role: "orchestrator" | "explorer",
): Model<Api> | undefined {
	if (reference === undefined) return undefined;
	if (isConfiguredEndpoint(reference)) {
		return buildModelFromConfiguredEndpoint(reference);
	}
	const catalog = resolveCatalogModel(reference);
	if (catalog !== undefined) return catalog;
	// Known catalog provider with an unknown model id is a hard config error.
	// Unknown providers fall through so a local runtime (e.g. codex proxy) can supply them.
	if ((getProviders() as readonly string[]).includes(reference.provider)) {
		throw new ConfigError(
			`Unknown configured ${role} model: ${reference.provider}/${reference.id}. ` +
				`Add baseUrl (and optional api/apiKeyEnv) for OpenAI-compatible endpoints, or use a pi-ai catalog model id.`,
		);
	}
	return undefined;
}

export function resolveModel(config: CliConfig): Model<Api> {
	if (!config.model) {
		throw new ConfigError(
			'No model configured. Provide --model-provider and --model-id on the command line, or set the "model" key (with provider and id) in the config file.',
		);
	}
	return resolveRegisteredModel(config.model);
}

export interface ResolvedAgentModel {
	readonly model: Model<Api>;
	readonly explorerModel: Model<Api>;
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
	readonly source: "env" | "local_runtime" | "catalog" | "none" | "unknown";
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
	readonly explorerModel: Model<Api>;
	readonly apiKey?: string;
	readonly providerApiKeys?: Readonly<Record<string, string>>;
	readonly roles: { readonly orchestrator: ResolvedAgentModelRole; readonly explorer: ResolvedAgentModelRole };
}

interface AgentModelCore {
	readonly model: Model<Api>;
	readonly explorerModel: Model<Api>;
	readonly apiKey?: string;
	readonly providerApiKeys?: Readonly<Record<string, string>>;
	readonly orchestratorRef: AgentModelConfig | undefined;
	readonly explorerRef: AgentModelConfig | undefined;
	readonly orchestratorModel: Model<Api>;
	readonly explorerModelResolved: Model<Api>;
	readonly orchestratorFromLocal: boolean;
	readonly explorerFromLocal: boolean;
	readonly orchestratorConfiguredEndpoint: boolean;
	readonly explorerConfiguredEndpoint: boolean;
	readonly orchestratorCatalog: boolean;
	readonly explorerCatalog: boolean;
	readonly local: LocalAutoRAGModels | undefined;
	readonly usesConfiguredEndpoint: boolean;
	readonly env: NodeJS.ProcessEnv;
}

function resolveAgentModelCore(config: CliConfig, localOptions: LoadLocalAutoRAGModelsOptions = {}): AgentModelCore {
	const orchestratorRef = config.agents?.orchestrator ?? config.model;
	const explorerRef = config.agents?.explorer;
	const registeredOrchestrator = resolveBuiltInModel(orchestratorRef, "orchestrator");
	const registeredExplorer = resolveBuiltInModel(explorerRef, "explorer");
	const needsLocal =
		orchestratorRef === undefined ||
		explorerRef === undefined ||
		registeredOrchestrator === undefined ||
		registeredExplorer === undefined;
	const local = needsLocal
		? loadLocalAutoRAGModels({
				...localOptions,
				orchestratorModelId: orchestratorRef?.id,
				explorerModelId: explorerRef?.id,
			})
		: undefined;
	const model =
		registeredOrchestrator ??
		(orchestratorRef === undefined || orchestratorRef.provider === local?.provider
			? (local?.orchestrator as Model<Api>)
			: resolveRegisteredModel(orchestratorRef, "orchestrator"));
	const explorerModel =
		registeredExplorer ??
		(explorerRef === undefined || explorerRef.provider === local?.provider
			? (local?.explorer as Model<Api>)
			: resolveRegisteredModel(explorerRef, "explorer"));

	const orchestratorConfiguredEndpoint = isConfiguredEndpoint(orchestratorRef);
	const explorerConfiguredEndpoint = isConfiguredEndpoint(explorerRef);
	const usesConfiguredEndpoint = orchestratorConfiguredEndpoint || explorerConfiguredEndpoint;

	const env = localOptions.env ?? process.env;
	const orchestratorFromLocal =
		registeredOrchestrator === undefined &&
		(orchestratorRef === undefined || orchestratorRef.provider === local?.provider);
	const explorerFromLocal =
		registeredExplorer === undefined && (explorerRef === undefined || explorerRef.provider === local?.provider);
	const orchestratorCatalog = registeredOrchestrator !== undefined && !orchestratorConfiguredEndpoint;
	const explorerCatalog = registeredExplorer !== undefined && !explorerConfiguredEndpoint;

	if (local === undefined && !usesConfiguredEndpoint) {
		return {
			model,
			explorerModel,
			orchestratorRef,
			explorerRef,
			orchestratorModel: model,
			explorerModelResolved: explorerModel,
			orchestratorFromLocal,
			explorerFromLocal,
			orchestratorConfiguredEndpoint,
			explorerConfiguredEndpoint,
			orchestratorCatalog,
			explorerCatalog,
			local,
			usesConfiguredEndpoint,
			env,
		};
	}

	const providerApiKeys: Record<string, string> = {};
	if (local !== undefined) providerApiKeys[local.provider] = local.apiKey;
	for (const ref of [orchestratorRef, explorerRef]) {
		if (!isConfiguredEndpoint(ref)) continue;
		const envName = configuredApiKeyEnv(ref);
		const value = env[envName];
		if (typeof value === "string" && value.length > 0) {
			providerApiKeys[ref.provider] = value;
		}
	}
	const orchestratorKey = providerApiKeys[model.provider];
	const apiKey =
		local !== undefined && (orchestratorRef === undefined || orchestratorRef.provider === local.provider)
			? local.apiKey
			: orchestratorKey !== undefined && usesConfiguredEndpoint
				? orchestratorKey
				: undefined;
	return {
		model,
		explorerModel,
		...(apiKey !== undefined ? { apiKey } : {}),
		...(Object.keys(providerApiKeys).length > 0 ? { providerApiKeys } : {}),
		orchestratorRef,
		explorerRef,
		orchestratorModel: model,
		explorerModelResolved: explorerModel,
		orchestratorFromLocal,
		explorerFromLocal,
		orchestratorConfiguredEndpoint,
		explorerConfiguredEndpoint,
		orchestratorCatalog,
		explorerCatalog,
		local,
		usesConfiguredEndpoint,
		env,
	};
}

export function resolveAgentModel(
	config: CliConfig,
	localOptions: LoadLocalAutoRAGModelsOptions = {},
): ResolvedAgentModel {
	const core = resolveAgentModelCore(config, localOptions);
	return {
		model: core.model,
		explorerModel: core.explorerModel,
		...(core.apiKey !== undefined ? { apiKey: core.apiKey } : {}),
		...(core.providerApiKeys !== undefined ? { providerApiKeys: core.providerApiKeys } : {}),
	};
}

function providerApiKeyEnvName(provider: string): string {
	return `${provider.replace(/[^A-Za-z0-9_]/g, "_").toUpperCase()}_API_KEY`;
}

function resolveRoleAuth(
	model: Model<Api>,
	fromLocal: boolean,
	local: LocalAutoRAGModels | undefined,
	providerApiKeys: Readonly<Record<string, string>> | undefined,
	configuredEndpoint: boolean,
	apiKeyEnv: string | undefined,
	env: NodeJS.ProcessEnv,
): AgentModelAuth {
	if (fromLocal && local !== undefined && model.provider === local.provider) {
		return { present: true, source: "local_runtime" };
	}
	if (configuredEndpoint) {
		const envName = apiKeyEnv ?? providerApiKeyEnvName(model.provider);
		if (providerApiKeys?.[model.provider] !== undefined) {
			return { present: true, source: "env", envName };
		}
		const fromEnv = env[envName] ?? process.env[envName];
		if (typeof fromEnv === "string" && fromEnv.length > 0) {
			return { present: true, source: "env", envName };
		}
		return { present: false, source: "none", envName };
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
	const catalogKey = getEnvApiKey(model.provider);
	if (catalogKey !== undefined) {
		return { present: true, source: "catalog" };
	}
	return { present: false, source: "none", envName: providerApiKeyEnvName(model.provider) };
}

function resolveRoleSource(
	ref: AgentModelConfig | undefined,
	fromLocal: boolean,
	configuredEndpoint: boolean,
	catalog: boolean,
): AgentModelResolutionSource {
	if (ref === undefined) return "local_runtime";
	if (configuredEndpoint) return "config";
	if (fromLocal) return "mixed";
	if (catalog) return "catalog";
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

export function resolveAgentModelDetailed(
	config: CliConfig,
	localOptions: LoadLocalAutoRAGModelsOptions = {},
): ResolvedAgentModelDetailed {
	const core = resolveAgentModelCore(config, localOptions);
	const orchestratorAuth = resolveRoleAuth(
		core.orchestratorModel,
		core.orchestratorFromLocal,
		core.local,
		core.providerApiKeys,
		core.orchestratorConfiguredEndpoint,
		core.orchestratorRef !== undefined ? configuredApiKeyEnv(core.orchestratorRef) : undefined,
		core.env,
	);
	const explorerAuth = resolveRoleAuth(
		core.explorerModelResolved,
		core.explorerFromLocal,
		core.local,
		core.providerApiKeys,
		core.explorerConfiguredEndpoint,
		core.explorerRef !== undefined ? configuredApiKeyEnv(core.explorerRef) : undefined,
		core.env,
	);
	const orchestratorSource = resolveRoleSource(
		core.orchestratorRef,
		core.orchestratorFromLocal,
		core.orchestratorConfiguredEndpoint,
		core.orchestratorCatalog,
	);
	const explorerSource = resolveRoleSource(
		core.explorerRef,
		core.explorerFromLocal,
		core.explorerConfiguredEndpoint,
		core.explorerCatalog,
	);
	return {
		model: core.model,
		explorerModel: core.explorerModel,
		...(core.apiKey !== undefined ? { apiKey: core.apiKey } : {}),
		...(core.providerApiKeys !== undefined ? { providerApiKeys: core.providerApiKeys } : {}),
		roles: {
			orchestrator: buildResolvedRole(core.orchestratorModel, orchestratorAuth, orchestratorSource),
			explorer: buildResolvedRole(core.explorerModelResolved, explorerAuth, explorerSource),
		},
	};
}

export function writeDefaultConfig(
	path: string,
	partial: Partial<CliConfig>,
	opts: { force?: boolean; atomicCreate?: boolean; cwd?: string; env?: NodeJS.ProcessEnv } = {},
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
	};
	const orchestrator = partial.agents?.orchestrator ?? partial.model;
	const explorer = partial.agents?.explorer;
	if (orchestrator !== undefined || explorer !== undefined) {
		full.agents = {
			...(orchestrator !== undefined ? { orchestrator } : {}),
			...(explorer !== undefined ? { explorer } : {}),
		};
	}
	if (partial.model) full.model = partial.model;
	// Indexing method defaults: enabled when not explicitly provided.
	// Never inject embedder id defaults; preserve partial embedder config as-is.
	const normalizedMethods = normalizeIndexingConfig({
		bm25: partial.bm25 as Bm25MethodConfig | false | undefined,
		minSync: partial.minSync as MinSyncMethodConfig | false | undefined,
	});
	full.bm25 = normalizedMethods.bm25;
	full.minSync = normalizedMethods.minSync;
	// Jikji find-first discovery is enabled by default for new configs; the CLI
	// auto-installs the jikji binary on first use when cargo is available.
	full.jikji = partial.jikji ?? {};
	if (partial.parserOptions) full.parserOptions = partial.parserOptions;
	mkdirSync(dirname(path), { recursive: true });
	const contents = `${JSON.stringify(full, null, 2)}\n`;
	const lock = acquireConfigWriteLock(path);
	try {
		if (!opts.force && existsSync(path)) {
			throw new ConfigError(`Config file already exists: ${path}`);
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

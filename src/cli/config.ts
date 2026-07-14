import { randomUUID } from "node:crypto";
import type { EnsureMinSyncBinaryOptions, MinSyncEmbedderConfig } from "../minsync/index.ts";
import type { BM25Engine, BM25FallbackMode } from "../retrieval/methods/bm25.ts";
import { existsSync, mkdirSync, readFileSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import { basename, dirname, isAbsolute, join, resolve } from "node:path";
import { type Api, getModel, getProviders, type Model } from "@earendil-works/pi-ai";
import type { AutoRAGAgentOptions } from "../agent/agent.ts";
import { resolveAutoRAGHome } from "../config/home.ts";
import { acquireFileLock, type FileLockHandle } from "../filesystem/file-lock.ts";
import { type LoadLocalAutoRAGModelsOptions, loadLocalAutoRAGModels } from "../subagents/local-models.ts";

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

export interface CliConfig {
	searchPaths: string[];
	workspacePath: string;
	memoryPath: string;
	model?: { provider: string; id: string };
	agents?: {
		orchestrator?: { provider: string; id: string };
		explorer?: { provider: string; id: string };
	};
	minSync?: MinSyncMethodConfig;
	bm25?: Bm25MethodConfig;
	jikji?: Record<string, unknown>;
	parserOptions?: Record<string, unknown>;
}

export interface ResolveConfigInput {
	flags: Record<string, string | boolean | undefined>;
	env?: NodeJS.ProcessEnv;
	cwd?: string;
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

function modelReference(value: unknown, path: string): { provider: string; id: string } | undefined {
	if (value === undefined) return undefined;
	if (typeof value !== "object" || value === null || Array.isArray(value)) {
		throw new ConfigError(`${path} must be an object with provider and id`);
	}
	const record = value as Record<string, unknown>;
	if (typeof record.provider !== "string" || record.provider.trim() === "") {
		throw new ConfigError(`${path}.provider must be a non-empty string`);
	}
	if (typeof record.id !== "string" || record.id.trim() === "") {
		throw new ConfigError(`${path}.id must be a non-empty string`);
	}
	return { provider: record.provider, id: record.id };
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
	const out: { id?: string; baseUrl?: string; apiKeyEnv?: string; dimension?: number; queryPrefix?: string; passagePrefix?: string; timeoutMs?: number; batchSize?: number; maxRetries?: number; maxConcurrent?: number } = {};
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
	const enabled = record.enabled === false ? false : true;
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
	const enabled = record.enabled === false ? false : true;
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
	const migrated = !explicit && legacyPath ? migrateLegacyConfig(configPath, legacyPath) : undefined;
	const file = migrated ?? readConfigFile(configPath, explicit) ?? {};

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

	const fileModel = file.model;
	const fileAgents = file.agents;
	const fileOrchestrator = modelReference(fileAgents?.orchestrator ?? fileModel, "agents.orchestrator");
	const fileExplorer = modelReference(fileAgents?.explorer, "agents.explorer");
	const provider =
		pickString(flags, env, "orchestrator-model-provider", "AUTORAG_ORCHESTRATOR_MODEL_PROVIDER", undefined) ??
		pickString(flags, env, "model-provider", "AUTORAG_MODEL_PROVIDER", fileOrchestrator?.provider);
	const id =
		pickString(flags, env, "orchestrator-model-id", "AUTORAG_ORCHESTRATOR_MODEL_ID", undefined) ??
		pickString(flags, env, "model-id", "AUTORAG_MODEL_ID", fileOrchestrator?.id);
	const explorerProvider = pickString(
		flags,
		env,
		"explorer-model-provider",
		"AUTORAG_EXPLORER_MODEL_PROVIDER",
		fileExplorer?.provider,
	);
	const explorerId = pickString(flags, env, "explorer-model-id", "AUTORAG_EXPLORER_MODEL_ID", fileExplorer?.id);

	const config: CliConfig = {
		searchPaths,
		workspacePath,
		memoryPath,
	};
	if (provider && id) {
		config.model = { provider, id };
	}
	if (provider || id || explorerProvider || explorerId) {
		if ((provider && !id) || (!provider && id)) {
			throw new ConfigError("agents.orchestrator requires both provider and id");
		}
		if ((explorerProvider && !explorerId) || (!explorerProvider && explorerId)) {
			throw new ConfigError("agents.explorer requires both provider and id");
		}
		config.agents = {
			...(provider && id ? { orchestrator: { provider, id } } : {}),
			...(explorerProvider && explorerId ? { explorer: { provider: explorerProvider, id: explorerId } } : {}),
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
	return config;
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
	return opts as Omit<AutoRAGAgentOptions, "model">;
}

/** OpenAI-compatible models declared in ~/.gjc/agent/models.yml but not always in pi-ai catalogs. */
const CONFIGURED_OPENAI_COMPAT_MODELS: ReadonlyArray<Model<Api>> = [
	{
		id: "x-ai/grok-4.5",
		name: "Grok 4.5 (OpenRouter)",
		api: "openai-completions",
		provider: "openrouter",
		baseUrl: "https://openrouter.ai/api/v1",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 2.0, output: 6.0, cacheRead: 0.5, cacheWrite: 0 },
		contextWindow: 256_000,
		maxTokens: 32_768,
	},
	{
		// short alias used by ~/.gjc/agent/models.yml
		id: "grok-4.5",
		name: "Grok 4.5 (OpenRouter)",
		api: "openai-completions",
		provider: "openrouter",
		baseUrl: "https://openrouter.ai/api/v1",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 2.0, output: 6.0, cacheRead: 0.5, cacheWrite: 0 },
		contextWindow: 256_000,
		maxTokens: 32_768,
	},
	{
		id: "accounts/fireworks/routers/glm-5p2-fast",
		name: "GLM-5.2 Fast (Fireworks)",
		api: "openai-completions",
		provider: "fireworks",
		baseUrl: "https://api.fireworks.ai/inference/v1",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 202_752,
		maxTokens: 131_072,
	},
	{
		// short alias used by ~/.gjc/agent/models.yml
		id: "glm-5.2-fast",
		name: "GLM-5.2 Fast (Fireworks)",
		api: "openai-completions",
		provider: "fireworks",
		baseUrl: "https://api.fireworks.ai/inference/v1",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 202_752,
		maxTokens: 131_072,
	},
] as const as unknown as ReadonlyArray<Model<Api>>;

function resolveConfiguredOpenAICompatibleModel(
	reference: { provider: string; id: string },
): Model<Api> | undefined {
	const hit = CONFIGURED_OPENAI_COMPAT_MODELS.find(
		(model) => model.provider === reference.provider && model.id === reference.id,
	);
	if (hit === undefined) return undefined;
	// Map short aliases to the wire model ids expected by the remote API.
	if (reference.provider === "openrouter" && reference.id === "grok-4.5") {
		return { ...hit, id: "x-ai/grok-4.5" };
	}
	if (reference.provider === "fireworks" && reference.id === "glm-5.2-fast") {
		return { ...hit, id: "accounts/fireworks/routers/glm-5p2-fast" };
	}
	return { ...hit };
}

function resolveRegisteredModel(
	reference: { provider: string; id: string },
	role?: "orchestrator" | "explorer",
): Model<Api> {
	const model = getModel(reference.provider as never, reference.id as never) as Model<Api> | undefined;
	if (model !== undefined) return model;
	const configured = resolveConfiguredOpenAICompatibleModel(reference);
	if (configured !== undefined) return configured;
	const roleLabel = role === undefined ? "" : `${role} `;
	throw new ConfigError(`Unknown configured ${roleLabel}model: ${reference.provider}/${reference.id}`);
}

function resolveBuiltInModel(
	reference: { provider: string; id: string } | undefined,
	role: "orchestrator" | "explorer",
): Model<Api> | undefined {
	if (reference === undefined || !(getProviders() as readonly string[]).includes(reference.provider)) return undefined;
	return resolveRegisteredModel(reference, role);
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

export function resolveAgentModel(
	config: CliConfig,
	localOptions: LoadLocalAutoRAGModelsOptions = {},
): ResolvedAgentModel {
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

	// Built-in catalog models keep credentials environment-backed via pi-ai.
	// Explicit providerApiKeys are only needed when:
	//  - local Codex/proxy models supply a runtime key, or
	//  - at least one role uses a non-catalog OpenAI-compatible model we inject.
	const usesConfiguredOpenAICompatible =
		(orchestratorRef !== undefined &&
			resolveConfiguredOpenAICompatibleModel(orchestratorRef) !== undefined &&
			getModel(orchestratorRef.provider as never, orchestratorRef.id as never) === undefined) ||
		(explorerRef !== undefined &&
			resolveConfiguredOpenAICompatibleModel(explorerRef) !== undefined &&
			getModel(explorerRef.provider as never, explorerRef.id as never) === undefined);

	if (local === undefined && !usesConfiguredOpenAICompatible) {
		return { model, explorerModel };
	}

	const providerApiKeys: Record<string, string> = {};
	if (local !== undefined) providerApiKeys[local.provider] = local.apiKey;
	if (usesConfiguredOpenAICompatible) {
		const env = localOptions.env ?? process.env;
		for (const provider of new Set([model.provider, explorerModel.provider])) {
			const envName = `${provider.replace(/[^A-Za-z0-9_]/g, "_").toUpperCase()}_API_KEY`;
			const value = env[envName];
			if (typeof value === "string" && value.length > 0) providerApiKeys[provider] = value;
		}
	}
	const orchestratorKey = providerApiKeys[model.provider];
	return {
		model,
		explorerModel,
		...(local !== undefined && (orchestratorRef === undefined || orchestratorRef.provider === local.provider)
			? { apiKey: local.apiKey }
			: orchestratorKey !== undefined && usesConfiguredOpenAICompatible
				? { apiKey: orchestratorKey }
				: {}),
		...(Object.keys(providerApiKeys).length > 0 ? { providerApiKeys } : {}),
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
	if (partial.jikji) full.jikji = partial.jikji;
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

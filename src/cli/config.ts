import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { getModel } from "@earendil-works/pi-ai";
import type { AutoRAGAgentOptions } from "../agent/agent.ts";

export const DEFAULT_CONFIG_FILENAME = "autorag.config.json";

export class ConfigError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "ConfigError";
	}
}

export interface CliConfig {
	searchPaths: string[];
	workspacePath: string;
	memoryPath: string;
	model?: { provider: string; id: string };
	minSync?: Record<string, unknown>;
	bm25?: Record<string, unknown>;
	jikji?: Record<string, unknown>;
	parserOptions?: Record<string, unknown>;
}

export interface ResolveConfigInput {
	flags: Record<string, string | boolean | undefined>;
	env?: NodeJS.ProcessEnv;
	cwd?: string;
}

interface ResolvedConfigPath {
	configPath: string;
	explicit: boolean;
}

function resolveConfigPath(input: ResolveConfigInput): ResolvedConfigPath {
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
	return { configPath: join(cwd, DEFAULT_CONFIG_FILENAME), explicit: false };
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

function pickString(
	flags: Record<string, string | boolean | undefined>,
	env: NodeJS.ProcessEnv,
	flagKey: string,
	envKey: string,
	fileValue: string | undefined,
): string | undefined {
	return flagString(flags, flagKey) ?? envString(env, envKey) ?? fileValue;
}

export function resolveConfig(input: ResolveConfigInput): CliConfig {
	const flags = input.flags;
	const env = input.env ?? process.env;
	const cwd = input.cwd ?? process.cwd();

	const { configPath, explicit } = resolveConfigPath(input);
	const file = readConfigFile(configPath, explicit) ?? {};

	const defaultSearchPaths = ["."];
	const defaultWorkspacePath = cwd;
	const defaultMemoryPath = join(cwd, ".autorag", "memory.json");

	const flagSearchPaths = parseCsv(flagString(flags, "search-paths"));
	const envSearchPaths = parseCsv(envString(env, "AUTORAG_SEARCH_PATHS"));
	const searchPaths = flagSearchPaths ?? envSearchPaths ?? file.searchPaths ?? defaultSearchPaths;

	const workspacePath =
		flagString(flags, "workspace") ??
		envString(env, "AUTORAG_WORKSPACE") ??
		file.workspacePath ??
		defaultWorkspacePath;

	const memoryPath =
		flagString(flags, "memory-path") ?? envString(env, "AUTORAG_MEMORY_PATH") ?? file.memoryPath ?? defaultMemoryPath;

	const fileModel = file.model;
	const provider = pickString(flags, env, "model-provider", "AUTORAG_MODEL_PROVIDER", fileModel?.provider);
	const id = pickString(flags, env, "model-id", "AUTORAG_MODEL_ID", fileModel?.id);

	const config: CliConfig = {
		searchPaths,
		workspacePath,
		memoryPath,
	};
	if (provider && id) {
		config.model = { provider, id };
	}
	if (file.minSync) config.minSync = file.minSync;
	if (file.bm25) config.bm25 = file.bm25;
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
	if (config.minSync) opts.minSync = config.minSync;
	if (config.bm25) opts.bm25 = config.bm25;
	if (config.jikji) opts.jikji = config.jikji;
	if (config.parserOptions) opts.parserOptions = config.parserOptions;
	return opts as Omit<AutoRAGAgentOptions, "model">;
}

export function resolveModel(config: CliConfig): ReturnType<typeof getModel> {
	if (!config.model) {
		throw new ConfigError(
			'No model configured. Provide --model-provider and --model-id on the command line, or set the "model" key (with provider and id) in the config file.',
		);
	}
	return getModel(config.model.provider as never, config.model.id as never);
}

export function writeDefaultConfig(path: string, partial: Partial<CliConfig>, opts?: { force?: boolean }): void {
	if (!opts?.force && existsSync(path)) {
		throw new ConfigError(`Config file already exists: ${path}`);
	}
	const cwd = process.cwd();
	const full: CliConfig = {
		searchPaths: partial.searchPaths ?? ["."],
		workspacePath: partial.workspacePath ?? cwd,
		memoryPath: partial.memoryPath ?? join(cwd, ".autorag", "memory.json"),
	};
	if (partial.model) full.model = partial.model;
	if (partial.minSync) full.minSync = partial.minSync;
	if (partial.bm25) full.bm25 = partial.bm25;
	if (partial.jikji) full.jikji = partial.jikji;
	if (partial.parserOptions) full.parserOptions = partial.parserOptions;
	writeFileSync(path, `${JSON.stringify(full, null, 2)}\n`, "utf8");
}

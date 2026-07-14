import { existsSync, readFileSync } from "node:fs";
import { dirname } from "node:path";
import {
	type CliConfig,
	ConfigError,
	normalizeLegacyConfigPaths,
	resolveConfigPath,
	writeDefaultConfig,
} from "../config.ts";
import { renderError } from "../output.ts";
import type { CommandContext } from "./types.ts";

/**
 * `autorag init` writes the default config to `~/.autorag/config.json`.
 * Role-specific model flags are folded into the generated config. A legacy
 * `autorag.config.json` in the current working directory may be migrated
 * non-destructively. An existing home config is not overwritten without
 * `--force`.
 */
export async function runInit(ctx: CommandContext): Promise<number> {
	const flags = ctx.flags;
	const partial: Partial<CliConfig> = {};

	const searchPathsFlag = typeof flags["search-paths"] === "string" ? flags["search-paths"] : undefined;
	if (searchPathsFlag) {
		partial.searchPaths = searchPathsFlag
			.split(",")
			.map((entry) => entry.trim())
			.filter((entry) => entry.length > 0);
	}
	if (typeof flags.workspace === "string") partial.workspacePath = flags.workspace;
	if (typeof flags["memory-path"] === "string") partial.memoryPath = flags["memory-path"];

	const orchestratorProvider =
		typeof flags["orchestrator-model-provider"] === "string"
			? flags["orchestrator-model-provider"]
			: typeof flags["model-provider"] === "string"
				? flags["model-provider"]
				: undefined;
	const orchestratorId =
		typeof flags["orchestrator-model-id"] === "string"
			? flags["orchestrator-model-id"]
			: typeof flags["model-id"] === "string"
				? flags["model-id"]
				: undefined;
	const explorerProvider =
		typeof flags["explorer-model-provider"] === "string" ? flags["explorer-model-provider"] : undefined;
	const explorerId = typeof flags["explorer-model-id"] === "string" ? flags["explorer-model-id"] : undefined;
	if ((orchestratorProvider && !orchestratorId) || (!orchestratorProvider && orchestratorId)) {
		ctx.stderr(renderError(new ConfigError("orchestrator model requires both provider and id"), { json: ctx.json }));
		return 2;
	}
	if ((explorerProvider && !explorerId) || (!explorerProvider && explorerId)) {
		ctx.stderr(renderError(new ConfigError("explorer model requires both provider and id"), { json: ctx.json }));
		return 2;
	}
	if (orchestratorProvider || explorerProvider) {
		partial.agents = {
			...(orchestratorProvider && orchestratorId
				? { orchestrator: { provider: orchestratorProvider, id: orchestratorId } }
				: {}),
			...(explorerProvider && explorerId ? { explorer: { provider: explorerProvider, id: explorerId } } : {}),
		};
	}

	// Embedder flags → minSync.embedder. Only non-secret fields: id, baseUrl,
	// apiKeyEnv (the env-var *name*, never the key value), dimension, prefixes,
	// timeout, batch size. All optional; only set fields the user provided.
	try {
		const embedder = parseEmbedderFlags(flags);
		if (embedder !== undefined) {
			partial.minSync = {
				...(partial.minSync ?? {}),
				enabled: true,
				autoInstall: false,
				embedder,
			};
		}
	} catch (error) {
		if (error instanceof ConfigError) {
			ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
			return 2;
		}
		throw error;
	}

	const resolvedPath = resolveConfigPath({ flags, cwd: ctx.cwd });
	const migratingLegacy =
		!resolvedPath.explicit &&
		resolvedPath.legacyPath !== undefined &&
		existsSync(resolvedPath.legacyPath) &&
		!existsSync(resolvedPath.configPath);
	if (migratingLegacy) {
		try {
			const legacyPath = resolvedPath.legacyPath as string;
			const legacy = JSON.parse(readFileSync(legacyPath, "utf8")) as Partial<CliConfig>;
			const normalizedLegacy = normalizeLegacyConfigPaths(legacy, dirname(legacyPath));
			partial.searchPaths ??= normalizedLegacy.searchPaths;
			partial.workspacePath ??= normalizedLegacy.workspacePath;
			partial.memoryPath ??= normalizedLegacy.memoryPath;
			partial.model ??= legacy.model;
			if (legacy.agents !== undefined || partial.agents !== undefined) {
				partial.agents = {
					...(legacy.agents ?? {}),
					...(partial.agents ?? {}),
				};
			}
			partial.minSync ??= legacy.minSync;
			partial.bm25 ??= legacy.bm25;
			partial.jikji ??= legacy.jikji;
			partial.parserOptions ??= legacy.parserOptions;
		} catch (error) {
			ctx.stderr(
				renderError(new ConfigError(`Failed to migrate legacy config: ${(error as Error).message}`), {
					json: ctx.json,
				}),
			);
			return 2;
		}
	}

	try {
		writeDefaultConfig(resolvedPath.configPath, partial, {
			force: flags.force === true,
			atomicCreate: migratingLegacy,
			cwd: ctx.cwd,
		});
	} catch (error) {
		if (error instanceof ConfigError) {
			ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
			return 2;
		}
		throw error;
	}

	const envelope = { ok: true, wrote: [resolvedPath.configPath] };
	if (ctx.json) {
		ctx.stdout(JSON.stringify(envelope));
	} else {
		ctx.stdout(`Wrote ${resolvedPath.configPath}.`);
	}
	return 0;
}

/**
 * Parse `--embedder-*` flags into a `MinSyncEmbedderConfig`-shaped object.
 * Only non-secret fields are accepted: id, baseUrl, apiKeyEnv (env-var name,
 * never the key value), dimension, queryPrefix, passagePrefix, timeoutMs,
 * batchSize. Returns `undefined` when no embedder flags are present.
 */
function parseEmbedderFlags(flags: Record<string, string | boolean | undefined>): Record<string, unknown> | undefined {
	const flagStr = (key: string): string | undefined =>
		typeof flags[key] === "string" ? (flags[key] as string) : undefined;
	const flagInt = (key: string): number | undefined => {
		const raw = flagStr(key);
		if (raw === undefined) return undefined;
		const n = Number(raw);
		if (!Number.isInteger(n) || n <= 0) {
			throw new ConfigError(`--${key} must be a positive integer`);
		}
		return n;
	};

	const embedder: Record<string, unknown> = {};
	const id = flagStr("embedder-id");
	if (id !== undefined) embedder.id = id;
	const baseUrl = flagStr("embedder-base-url");
	if (baseUrl !== undefined) embedder.baseUrl = baseUrl;
	const apiKeyEnv = flagStr("embedder-api-key-env");
	if (apiKeyEnv !== undefined) embedder.apiKeyEnv = apiKeyEnv;
	const dimension = flagInt("embedder-dimension");
	if (dimension !== undefined) embedder.dimension = dimension;
	const queryPrefix = flagStr("embedder-query-prefix");
	if (queryPrefix !== undefined) embedder.queryPrefix = queryPrefix;
	const passagePrefix = flagStr("embedder-passage-prefix");
	if (passagePrefix !== undefined) embedder.passagePrefix = passagePrefix;
	const timeoutMs = flagInt("embedder-timeout-ms");
	if (timeoutMs !== undefined) embedder.timeoutMs = timeoutMs;
	const batchSize = flagInt("embedder-batch-size");
	if (batchSize !== undefined) embedder.batchSize = batchSize;

	return Object.keys(embedder).length > 0 ? embedder : undefined;
}

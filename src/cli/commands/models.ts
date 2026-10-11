import { join } from "node:path";
import { getAgentDir, ModelRuntime } from "@earendil-works/pi-coding-agent";
import { registerAutoRAGProvider } from "../../cloud/provider.ts";
import { importModel, prefetchModel, verifyModel } from "../../embedding-runtime/index.ts";
import type { ProfileId } from "../../embedding-runtime/types.ts";
import { renderError } from "../output.ts";
import type { CommandContext } from "./types.ts";

/**
 * One chat model discovered through the pi model runtime (built-in catalog plus
 * `models.json`, custom, and extension providers). `available` reports whether
 * the provider has complete auth configuration, including stored `auth.json`
 * credentials (api-key or OAuth).
 */
export interface ModelCatalogEntry {
	readonly provider: string;
	readonly id: string;
	readonly name: string;
	readonly api: string;
	readonly available: boolean;
}

export interface ModelCatalogOptions {
	readonly provider?: string;
	/** Only list models whose provider has complete auth configuration. */
	readonly available?: boolean;
	readonly agentDir?: string;
}

export interface ModelsCommandDeps {
	prefetchModel?: (profileId?: ProfileId) => Promise<{ profileId: ProfileId; path: string }>;
	importModel?: (
		profileId: ProfileId | undefined,
		sourcePath: string,
	) => Promise<{ profileId: ProfileId; path: string }>;
	verifyModel?: (profileId?: ProfileId) => Promise<{ profileId: ProfileId; path: string; hash: string }>;
	listModels?: (options: ModelCatalogOptions) => Promise<readonly ModelCatalogEntry[]>;
}

/**
 * Discover chat models through the pi model runtime rather than the static
 * pi-ai catalog, so `models.json`, custom/extension providers, and stored
 * credentials are honored. Network catalog refresh is disabled.
 */
async function listModelCatalog(options: ModelCatalogOptions): Promise<readonly ModelCatalogEntry[]> {
	const agentDir = options.agentDir ?? getAgentDir();
	const runtime = await ModelRuntime.create({
		authPath: join(agentDir, "auth.json"),
		modelsPath: join(agentDir, "models.json"),
		allowModelNetwork: false,
	});
	// Register the hosted AutoRAG plan so `--provider autorag` sees its
	// persisted catalog snapshot; the registration refresh is cache-only.
	await registerAutoRAGProvider(runtime);
	const key = (provider: string, id: string): string => `${provider}/${id}`;
	let availableKeys = new Set<string>();
	try {
		availableKeys = new Set(
			(await runtime.getAvailable(options.provider)).map((model) => key(model.provider, model.id)),
		);
	} catch {
		// Availability is best-effort; listing still works without it.
	}
	const models = [...runtime.getModels(options.provider)].filter(
		(model) => options.available !== true || availableKeys.has(key(model.provider, model.id)),
	);
	return models
		.map((model) => ({
			provider: model.provider,
			id: model.id,
			name: model.name,
			api: model.api,
			available: availableKeys.has(key(model.provider, model.id)),
		}))
		.sort((a, b) => a.provider.localeCompare(b.provider) || a.id.localeCompare(b.id));
}

function profile(ctx: CommandContext): ProfileId | undefined {
	const value = ctx.flags.profile;
	return typeof value === "string" ? (value as ProfileId) : undefined;
}

function flagString(flags: CommandContext["flags"], name: string): string | undefined {
	const value = flags[name];
	return typeof value === "string" && value.length > 0 ? value : undefined;
}

function output(ctx: CommandContext, result: unknown, action: string): void {
	ctx.stdout(
		ctx.json || ctx.flags.format === "json"
			? JSON.stringify({ ok: true, action, ...((result ?? {}) as object) }, null, 2)
			: `models: ${action}`,
	);
}

async function runList(ctx: CommandContext, deps: ModelsCommandDeps): Promise<number> {
	const provider = flagString(ctx.flags, "provider");
	const available = ctx.flags.available === true;
	const models = await (deps.listModels ?? listModelCatalog)({
		...(provider !== undefined ? { provider } : {}),
		...(available ? { available: true } : {}),
	});
	if (ctx.json || ctx.flags.format === "json") {
		ctx.stdout(
			JSON.stringify(
				{
					ok: true,
					action: "list",
					...(provider !== undefined ? { provider } : {}),
					count: models.length,
					models,
				},
				null,
				2,
			),
		);
		return 0;
	}
	ctx.stdout(`models: ${models.length} model(s)`);
	for (const model of models) {
		ctx.stdout(`${model.provider}/${model.id}  ${model.name}${model.available ? " (auth)" : ""}`);
	}
	return 0;
}

export async function runModels(ctx: CommandContext, deps: ModelsCommandDeps = {}): Promise<number> {
	const sub = ctx.positionals[0];
	try {
		if (sub === "list" || sub === "ls") {
			return await runList(ctx, deps);
		}
		if (sub === "prefetch") {
			const result = await (deps.prefetchModel ?? prefetchModel)(profile(ctx));
			output(ctx, result, "prefetch");
			return 0;
		}
		if (sub === "import") {
			const source = ctx.positionals[1];
			if (!source) throw new UsageError("Usage: autorag models import <path> [--profile <id>]");
			const result = await (deps.importModel ?? importModel)(profile(ctx), source);
			output(ctx, result, "import");
			return 0;
		}
		if (sub === "verify") {
			const result = await (deps.verifyModel ?? verifyModel)(profile(ctx));
			output(ctx, result, "verify");
			return 0;
		}
		throw new UsageError(
			"Usage: autorag models list [--provider <id>] [--available] | prefetch | import <path> | verify [--profile <id>]",
		);
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json || ctx.flags.format === "json", debug: ctx.debug }));
		return error instanceof UsageError ? 2 : 1;
	}
}
class UsageError extends Error {}

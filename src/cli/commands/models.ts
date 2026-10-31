import { join } from "node:path";
import { getAgentDir, ModelRuntime } from "@earendil-works/pi-coding-agent";
import { PROVIDER_ID as DAZZI_PROVIDER_ID } from "../../cloud/config.ts";
import { isDazziSignedIn } from "../../cloud/credential.ts";
import { DAZZI_DEFAULT_JEV_MODEL } from "../../cloud/jev-backend.ts";
import { registerDazziProvider } from "../../cloud/provider.ts";
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
	/** List decision models (the `jev.backend`/`jev.model` candidates) instead of chat models. */
	readonly decision?: boolean;
	readonly agentDir?: string;
}

export interface ModelsCommandDeps {
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
	// Register the hosted Dazzi Models plan so `--provider dazzi` sees its
	// persisted catalog snapshot; the registration refresh is cache-only.
	await registerDazziProvider(runtime);
	const key = (provider: string, id: string): string => `${provider}/${id}`;
	const type = options.decision === true ? "classifier" : "chat";
	let availableKeys = new Set<string>();
	try {
		availableKeys = new Set(
			(await runtime.getAvailableOfType(type, options.provider)).map((model) => key(model.provider, model.id)),
		);
	} catch {
		// Availability is best-effort; listing still works without it.
	}
	const listed: { provider: string; id: string; name: string; api: string }[] = [
		...runtime.getModelsOfType(type, options.provider),
	];
	// The hosted plan's Jev answers on its own endpoint, not through a pi classifier model.
	if (options.decision === true && (options.provider === undefined || options.provider === DAZZI_PROVIDER_ID)) {
		listed.push({ provider: DAZZI_PROVIDER_ID, id: DAZZI_DEFAULT_JEV_MODEL, name: "Jev (Dazzi Models)", api: "jev" });
		if (isDazziSignedIn(agentDir)) availableKeys.add(key(DAZZI_PROVIDER_ID, DAZZI_DEFAULT_JEV_MODEL));
	}
	const models = listed.filter(
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

function flagString(flags: CommandContext["flags"], name: string): string | undefined {
	const value = flags[name];
	return typeof value === "string" && value.length > 0 ? value : undefined;
}

async function runList(ctx: CommandContext, deps: ModelsCommandDeps): Promise<number> {
	const provider = flagString(ctx.flags, "provider");
	const available = ctx.flags.available === true;
	const decision = ctx.flags.decision === true;
	const models = await (deps.listModels ?? listModelCatalog)({
		...(provider !== undefined ? { provider } : {}),
		...(available ? { available: true } : {}),
		...(decision ? { decision: true } : {}),
	});
	if (ctx.json || ctx.flags.format === "json") {
		ctx.stdout(
			JSON.stringify(
				{
					ok: true,
					action: "list",
					...(provider !== undefined ? { provider } : {}),
					...(decision ? { type: "decision" } : {}),
					count: models.length,
					models,
				},
				null,
				2,
			),
		);
		return 0;
	}
	ctx.stdout(`models: ${models.length} ${decision ? "decision " : ""}model(s)`);
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
		throw new UsageError("Usage: autorag models list [--provider <id>] [--available] [--decision]");
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json || ctx.flags.format === "json", debug: ctx.debug }));
		return error instanceof UsageError ? 2 : 1;
	}
}
class UsageError extends Error {}

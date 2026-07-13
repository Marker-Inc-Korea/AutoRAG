import { AutoRAGAgent, type AutoRAGAgentOptions } from "../../agent/agent.ts";
import type { SearchDocumentsResponse } from "../../agent/search-documents.ts";
import {
	buildAgentOptions,
	type CliConfig,
	type ResolvedAgentModel,
	resolveAgentModel,
	resolveConfig,
} from "../config.ts";
import { renderError, renderSearch } from "../output.ts";
import type { CommandContext } from "./types.ts";

/**
 * Search-command dependencies. Tests inject `agentFactory` to bypass real
 * model construction and AutoRAGAgent instantiation, returning a stub whose
 * `searchDocuments` resolves a canned {@link SearchDocumentsResponse}.
 */
export interface SearchDeps {
	agentFactory?: (opts: AutoRAGAgentOptions) => Pick<AutoRAGAgent, "searchDocuments">;
	modelResolver?: (config: CliConfig) => ResolvedAgentModel;
}

interface SearchOptions {
	topK?: number;
	scope?: string;
	allowedTags?: string[];
}

function parseIntOptional(value: string | boolean | undefined): number | undefined {
	if (typeof value !== "string" || value.trim() === "") return undefined;
	const parsed = Number(value);
	if (!Number.isFinite(parsed)) return undefined;
	return Math.trunc(parsed);
}

function parseCsvStrings(value: string | boolean | undefined): string[] | undefined {
	if (typeof value !== "string" || value.trim() === "") return undefined;
	const parts = value
		.split(",")
		.map((part) => part.trim())
		.filter((part) => part !== "");
	return parts.length > 0 ? parts : undefined;
}

function buildSearchOptions(flags: CommandContext["flags"]): SearchOptions {
	const options: SearchOptions = {};
	const topK = parseIntOptional(flags["top-k"]);
	if (topK !== undefined) options.topK = topK;
	if (typeof flags.scope === "string" && flags.scope.trim() !== "") options.scope = flags.scope;
	const tags = parseCsvStrings(flags.tags);
	if (tags !== undefined) options.allowedTags = tags;
	return options;
}

/**
 * Run the `autorag search` command. Returns exit code 0 on success, 2 for
 * usage/config errors (empty query, missing model), 1 for runtime errors.
 */
export async function runSearch(ctx: CommandContext, deps: SearchDeps = {}): Promise<number> {
	const query = ctx.positionals.join(" ").trim();
	if (query.length === 0) {
		ctx.stderr(
			renderError(new Error("Usage: autorag search <query> [--top-k N] [--scope SCOPE] [--tags tag1,tag2]"), {
				json: ctx.json,
				debug: ctx.debug,
			}),
		);
		return 2;
	}

	let config: CliConfig;
	try {
		config = resolveConfig({ flags: ctx.flags, cwd: ctx.cwd });
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return 2;
	}

	let agent: Pick<AutoRAGAgent, "searchDocuments">;
	if (deps.agentFactory && deps.modelResolver === undefined) {
		agent = deps.agentFactory({ ...buildAgentOptions(config) });
	} else {
		let resolvedModel: ResolvedAgentModel;
		try {
			resolvedModel = (deps.modelResolver ?? resolveAgentModel)(config);
		} catch (error) {
			ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
			return 2;
		}
		const agentOptions: AutoRAGAgentOptions = {
			...buildAgentOptions(config),
			model: resolvedModel.model,
			...(resolvedModel.apiKey !== undefined ? { apiKey: resolvedModel.apiKey } : {}),
		};
		agent = deps.agentFactory ? deps.agentFactory(agentOptions) : new AutoRAGAgent(agentOptions);
	}

	const options = buildSearchOptions(ctx.flags);
	try {
		const resp = await agent.searchDocuments(query, options);
		ctx.stdout(renderSearch(resp, { json: ctx.json, debug: ctx.debug }));
		return 0;
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return 1;
	}
}

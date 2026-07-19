import { AutoRAGAgent, type AutoRAGAgentOptions } from "../../agent/agent.ts";
import type { SearchDocumentsResponse } from "../../agent/search-documents.ts";
import {
	buildAgentOptions,
	type CliConfig,
	ConfigError,
	type ResolvedAgentModel,
	resolveAgentModel,
	resolveConfig,
} from "../config.ts";
import { renderError, renderSearch } from "../output.ts";
import type { CommandContext } from "./types.ts";

// ---------------------------------------------------------------------------
// Search health hints (Issue #49)
// ---------------------------------------------------------------------------

/**
 * A recoverable health hint surfaced when `autorag search` fails for a reason
 * that `autorag health` can diagnose. The classifier only produces hints for
 * model/provider/subagent failures — never for retrieval, index, datasource,
 * or empty-query errors.
 */
export interface SearchHealthHint {
	command: "autorag health";
	reason: "model_resolution" | "subagent_failed" | "auth_missing" | "provider_unreachable" | "timeout";
	message: string;
}

const HEALTH_HINT_MESSAGE = "Run autorag health to diagnose model/provider and explorer subagent setup.";

/** Substring patterns that map a runtime error to a {@link SearchHealthHint} reason. */
const SUBAGENT_PATTERNS: readonly string[] = [
	"pi-subagents",
	"autorag-explorer",
	"concurrent pi sessions",
	"concurrent autorag session busy",
];

const AUTH_PATTERNS: readonly string[] = [
	"api key",
	"apikey",
	"401",
	"403",
	"unauthorized",
	"forbidden",
	"authentication",
];

const PROVIDER_PATTERNS: readonly string[] = [
	"enotfound",
	"econnrefused",
	"etimedout",
	"provider unreachable",
	"unreachable",
	"network",
	"socket hang up",
];

/**
 * Classify a search error into an optional {@link SearchHealthHint}.
 * Returns `undefined` for errors outside the allowlist (empty query,
 * retrieval/index/datasource failures, generic errors).
 */
export function classifySearchHealthHint(error: unknown): SearchHealthHint | undefined {
	const message = (error instanceof Error ? error.message : String(error)).toLowerCase();
	const name = error instanceof Error ? error.name : "";

	// 1. ConfigError / unknown model / no model configured → model_resolution.
	if (error instanceof ConfigError || name === "ConfigError") {
		return { command: "autorag health", reason: "model_resolution", message: HEALTH_HINT_MESSAGE };
	}
	if (message.includes("unknown configured") || message.includes("no model configured")) {
		return { command: "autorag health", reason: "model_resolution", message: HEALTH_HINT_MESSAGE };
	}

	// 2. Subagent-related failures → subagent_failed.
	for (const pattern of SUBAGENT_PATTERNS) {
		if (message.includes(pattern)) {
			return { command: "autorag health", reason: "subagent_failed", message: HEALTH_HINT_MESSAGE };
		}
	}

	// 3. Auth / API key / 401 / 403 → auth_missing.
	for (const pattern of AUTH_PATTERNS) {
		if (message.includes(pattern)) {
			return { command: "autorag health", reason: "auth_missing", message: HEALTH_HINT_MESSAGE };
		}
	}

	// 4. Network / provider unreachable → provider_unreachable.
	if (typeof error === "object" && error !== null && "code" in error) {
		const code = (error as { code?: unknown }).code;
		if (code === "ENOTFOUND" || code === "ECONNREFUSED" || code === "ETIMEDOUT") {
			return { command: "autorag health", reason: "provider_unreachable", message: HEALTH_HINT_MESSAGE };
		}
	}
	for (const pattern of PROVIDER_PATTERNS) {
		if (message.includes(pattern)) {
			return { command: "autorag health", reason: "provider_unreachable", message: HEALTH_HINT_MESSAGE };
		}
	}

	// 5. AbortError / timeout → timeout.
	if (
		name === "AbortError" ||
		message.includes("abort") ||
		message.includes("timeout") ||
		message.includes("timed out")
	) {
		return { command: "autorag health", reason: "timeout", message: HEALTH_HINT_MESSAGE };
	}

	return undefined;
}

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
			const hint = classifySearchHealthHint(error);
			ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug, hint }));
			return 2;
		}
		const agentOptions: AutoRAGAgentOptions = {
			...buildAgentOptions(config),
			model: resolvedModel.model,
			explorerModel: resolvedModel.explorerModel,
			...(resolvedModel.apiKey !== undefined ? { apiKey: resolvedModel.apiKey } : {}),
			...(resolvedModel.providerApiKeys !== undefined ? { providerApiKeys: resolvedModel.providerApiKeys } : {}),
		};
		agent = deps.agentFactory ? deps.agentFactory(agentOptions) : new AutoRAGAgent(agentOptions);
	}

	const options = buildSearchOptions(ctx.flags);
	try {
		const resp = await agent.searchDocuments(query, options);
		ctx.stdout(renderSearch(resp, { json: ctx.json, debug: ctx.debug }));
		return 0;
	} catch (error) {
		const hint = classifySearchHealthHint(error);
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug, hint }));
		return 1;
	}
}

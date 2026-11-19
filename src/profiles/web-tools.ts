/**
 * Opt-in `web_search` / `web_fetch` tools for a helper profile.
 *
 * The retrieval toolkit never hands out the agent's web surfaces; a profile
 * that genuinely needs to look something up opts in here. Both tools share one
 * fresh {@link EvidenceLedger}, so their `[eN]` ids resolve within the profile
 * session that called them — never the user's retrieval memory, and nothing is
 * persisted. Constructing the tools performs no network work; `createWebSearchTool`
 * and `createWebFetchTool` only build the provider/render callbacks.
 */
import type { ToolDefinition } from "@earendil-works/pi-coding-agent";
import { EvidenceLedger } from "../agent/evidence-ledger.ts";
import { toToolDefinition } from "../agent/pi-session.ts";
import { createWebFetchTool, type WebFetchToolOptions } from "../agent/web-fetch-tool.ts";
import { createWebSearchTool, type WebSearchToolOptions } from "../agent/web-search-tool.ts";

/** Which of AutoRAG's web tools a profile session is offered. */
export interface WebToolsOptions {
	/** `false` leaves web_search out. Otherwise the same options `AutoRAGAgent` takes for `webSearch`. */
	readonly search?: WebSearchToolOptions | false;
	/** `false` leaves web_fetch out. Otherwise the same options `AutoRAGAgent` takes for `webFetch`. */
	readonly fetch?: WebFetchToolOptions | false;
}

/** The `web_search` and `web_fetch` tools, in that order, sharing one fresh evidence ledger. */
export function createWebTools(options: WebToolsOptions = {}): readonly ToolDefinition[] {
	const ledger = new EvidenceLedger();
	return [
		...(options.search === false ? [] : [toToolDefinition(createWebSearchTool(options.search ?? {}, ledger))]),
		...(options.fetch === false ? [] : [toToolDefinition(createWebFetchTool(options.fetch ?? {}, ledger))]),
	];
}

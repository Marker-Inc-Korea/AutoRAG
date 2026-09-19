/**
 * Anthropic model-native web search provider.
 *
 * Calls the Messages API with the built-in `web_search_20250305` tool using
 * the model credential AutoRAG already resolves (injected agent credential or
 * ANTHROPIC_API_KEY) — no search-specific key or signup.
 *
 * Ported from oh-my-pi (can1357/oh-my-pi, MIT)
 * `web/search/providers/anthropic.ts`. Dropped: oh-my-pi's AuthStorage/OAuth
 * broker, Claude-Code billing fingerprints, and catalog-based sampling
 * restrictions — AutoRAG uses the plain API-key request shape via
 * `../model-auth.ts`.
 */
import { envCredential } from "../credentials.ts";
import { resolveModelNativeCredential } from "../model-auth.ts";
import { formatQuery, parseSearchQuery, type QuerySyntax, type StructuredQuery } from "../query.ts";
import { type SearchCitation, SearchProviderError, type SearchResponse, type SearchSource } from "../types.ts";
import type { SearchAvailabilityContext, SearchParams } from "./base.ts";
import { SearchProvider } from "./base.ts";
import { classifyProviderHttpError, readLimitedText, withHardTimeout } from "./utils.ts";

const DEFAULT_BASE_URL = "https://api.anthropic.com";
const DEFAULT_MODEL = "claude-haiku-4-5";
const DEFAULT_MAX_TOKENS = 4096;
const MAX_RESPONSE_BYTES = 4 * 1024 * 1024;
const MAX_ERROR_BYTES = 8 * 1024;
const WEB_SEARCH_TOOL_TYPE = "web_search_20250305";

/**
 * Claude's search backend understands common Google-style operators, so most
 * directives are re-emitted as query text. `site:` maps onto the tool's
 * native allowed/blocked domains instead.
 */
const ANTHROPIC_QUERY_SYNTAX: QuerySyntax = {
	phrases: true,
	negation: true,
	or: true,
	inUrl: true,
	inTitle: true,
	filetype: true,
	dateRange: true,
};

interface AnthropicQueryPlan {
	query: string;
	allowedDomains?: string[];
	blockedDomains?: string[];
}

/** Map `site:`/`-site:` directives onto native domain filters; re-emit the rest as query syntax. */
function planQuery(rawQuery: string, parsed: StructuredQuery): AnthropicQueryPlan {
	if (!parsed.hasDirectives) return { query: rawQuery };
	const hosts = (sites: readonly string[]) => {
		const unique = new Set<string>();
		for (const site of sites) {
			const slash = site.indexOf("/");
			const host = slash === -1 ? site : site.slice(0, slash);
			if (host.length > 0) unique.add(host);
		}
		return [...unique];
	};
	const allowed = hosts(parsed.sites);
	const blocked = allowed.length === 0 ? hosts(parsed.excludedSites) : [];
	return {
		query: formatQuery(parsed, ANTHROPIC_QUERY_SYNTAX),
		...(allowed.length > 0 ? { allowedDomains: allowed } : {}),
		...(blocked.length > 0 ? { blockedDomains: blocked } : {}),
	};
}

/** Parse a human-readable page age ("2 days ago") into seconds. */
function parsePageAge(pageAge: string | null | undefined): number | undefined {
	if (!pageAge) return undefined;
	const match = pageAge.match(/^(\d+)\s*(s|sec|second|m|min|minute|h|hour|d|day|w|week|mo|month|y|year)s?\s*(ago)?$/i);
	if (!match) return undefined;
	const multipliers: Record<string, number> = {
		s: 1,
		sec: 1,
		second: 1,
		m: 60,
		min: 60,
		minute: 60,
		h: 3600,
		hour: 3600,
		d: 86400,
		day: 86400,
		w: 604800,
		week: 604800,
		mo: 2592000,
		month: 2592000,
		y: 31536000,
		year: 31536000,
	};
	return Number.parseInt(match[1] ?? "0", 10) * (multipliers[(match[2] ?? "d").toLowerCase()] ?? 86400);
}

interface AnthropicContentBlock {
	type: string;
	name?: string;
	text?: string;
	input?: { query?: string };
	content?: Array<{ type: string; title?: string; url?: string; page_age?: string }>;
	citations?: Array<{ url?: string; title?: string; cited_text?: string }>;
}

interface AnthropicApiResponse {
	id?: string;
	model?: string;
	content?: AnthropicContentBlock[];
	usage?: { input_tokens?: number; output_tokens?: number; server_tool_use?: { web_search_requests?: number } };
}

function parseResponse(response: AnthropicApiResponse): Omit<SearchResponse, "provider"> {
	const answerParts: string[] = [];
	const searchQueries: string[] = [];
	const sources: SearchSource[] = [];
	const citations: SearchCitation[] = [];

	for (const block of response.content ?? []) {
		if (block.type === "server_tool_use" && block.name === "web_search" && block.input?.query) {
			searchQueries.push(block.input.query);
		} else if (block.type === "web_search_tool_result" && Array.isArray(block.content)) {
			for (const result of block.content) {
				if (
					result.type === "web_search_result" &&
					typeof result.url === "string" &&
					typeof result.title === "string"
				) {
					sources.push({
						title: result.title,
						url: result.url,
						...(result.page_age ? { publishedDate: result.page_age } : {}),
						...(parsePageAge(result.page_age) !== undefined ? { ageSeconds: parsePageAge(result.page_age) } : {}),
					});
				}
			}
		} else if (block.type === "text" && block.text) {
			answerParts.push(block.text);
			for (const citation of block.citations ?? []) {
				if (typeof citation.url === "string" && typeof citation.title === "string") {
					citations.push({
						url: citation.url,
						title: citation.title,
						...(citation.cited_text !== undefined ? { citedText: citation.cited_text } : {}),
					});
				}
			}
		}
	}

	return {
		...(answerParts.length > 0 ? { answer: answerParts.join("\n\n") } : {}),
		sources,
		...(citations.length > 0 ? { citations } : {}),
		...(searchQueries.length > 0 ? { searchQueries } : {}),
		...(response.usage
			? {
					usage: {
						...(response.usage.input_tokens !== undefined ? { inputTokens: response.usage.input_tokens } : {}),
						...(response.usage.output_tokens !== undefined ? { outputTokens: response.usage.output_tokens } : {}),
						...(response.usage.server_tool_use?.web_search_requests !== undefined
							? { searchRequests: response.usage.server_tool_use.web_search_requests }
							: {}),
					},
				}
			: {}),
		...(response.model !== undefined ? { model: response.model } : {}),
		...(response.id !== undefined ? { requestId: response.id } : {}),
	};
}

export class AnthropicProvider extends SearchProvider {
	readonly id = "anthropic" as const;
	readonly label = "Anthropic";

	isAvailable(context?: SearchAvailabilityContext): boolean {
		return resolveModelNativeCredential("anthropic", context?.modelAuth) !== undefined;
	}

	async search(params: SearchParams): Promise<SearchResponse> {
		const credential = resolveModelNativeCredential("anthropic", params.modelAuth);
		if (!credential) {
			throw new SearchProviderError(
				"anthropic",
				"Anthropic credentials not found. Configure an Anthropic model for the agent or set ANTHROPIC_API_KEY.",
				401,
			);
		}
		const model =
			envCredential("ANTHROPIC_SEARCH_MODEL") ??
			(credential.source === "injected" ? credential.modelId : undefined) ??
			DEFAULT_MODEL;
		const parsed = params.parsedQuery ?? parseSearchQuery(params.query);
		const plan = planQuery(params.query, parsed);
		const baseUrl = (credential.baseUrl ?? DEFAULT_BASE_URL).replace(/\/+$/, "");

		const fetchImpl = params.fetch ?? fetch;
		const response = await fetchImpl(`${baseUrl}/v1/messages?beta=true`, {
			method: "POST",
			headers: {
				"content-type": "application/json",
				"x-api-key": credential.apiKey,
				"anthropic-version": "2023-06-01",
				"anthropic-beta": "web-search-2025-03-05",
			},
			body: JSON.stringify({
				model,
				max_tokens: params.maxOutputTokens ?? DEFAULT_MAX_TOKENS,
				...(params.systemPrompt ? { system: params.systemPrompt } : {}),
				messages: [{ role: "user", content: plan.query }],
				tools: [
					{
						type: WEB_SEARCH_TOOL_TYPE,
						name: "web_search",
						...(plan.allowedDomains ? { allowed_domains: plan.allowedDomains } : {}),
						...(plan.blockedDomains ? { blocked_domains: plan.blockedDomains } : {}),
					},
				],
			}),
			signal: withHardTimeout(params.signal, params.timeoutMs),
		});

		if (!response.ok) {
			const errorText = await readLimitedText(response, "anthropic", MAX_ERROR_BYTES, true);
			const classified = classifyProviderHttpError("anthropic", response.status, errorText);
			if (classified) throw classified;
			throw new SearchProviderError(
				"anthropic",
				`Anthropic API error (${response.status}): ${errorText}`,
				response.status,
			);
		}

		const raw = await readLimitedText(response, "anthropic", MAX_RESPONSE_BYTES);
		let data: AnthropicApiResponse;
		try {
			data = JSON.parse(raw) as AnthropicApiResponse;
		} catch {
			throw new SearchProviderError("anthropic", "Anthropic API returned invalid JSON", 500);
		}

		const parsedResponse = parseResponse(data);
		const numResults = params.numSearchResults ?? params.limit;
		const sources = numResults ? parsedResponse.sources.slice(0, numResults) : parsedResponse.sources;
		return {
			...parsedResponse,
			provider: "anthropic",
			sources,
			authMode: credential.source === "injected" ? "agent-model" : "api_key",
		};
	}
}

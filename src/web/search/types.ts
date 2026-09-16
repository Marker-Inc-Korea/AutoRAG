/**
 * Web Search Types
 *
 * Unified types for web search responses across supported providers.
 * Ported from oh-my-pi (can1357/oh-my-pi, MIT) `web/search/types.ts`,
 * trimmed to the providers AutoRAG ships: simple env-key REST providers and
 * the credential-free engines. OAuth/model-native providers from oh-my-pi
 * (perplexity, gemini, anthropic, codex, xai, zai, tinyfish, synthetic,
 * ollama, parallel, firecrawl) are intentionally absent: they depend on
 * oh-my-pi's auth broker rather than plain environment variables.
 */

export const SEARCH_PROVIDER_OPTIONS = [
	{
		value: "auto",
		label: "Auto",
		description: "Automatically uses the first configured web-search provider",
	},
	{ value: "brave", label: "Brave", description: "Requires BRAVE_API_KEY" },
	{ value: "tavily", label: "Tavily", description: "Requires TAVILY_API_KEY" },
	{ value: "exa", label: "Exa", description: "Requires EXA_API_KEY" },
	{ value: "jina", label: "Jina", description: "Requires JINA_API_KEY" },
	{ value: "kagi", label: "Kagi", description: "Requires KAGI_API_KEY" },
	{
		value: "kimi",
		label: "Kimi",
		description: "Kimi Code search (requires KIMI_SEARCH_API_KEY or MOONSHOT_SEARCH_API_KEY)",
	},
	{ value: "searxng", label: "SearXNG", description: "Requires SEARXNG_ENDPOINT" },
	{
		value: "startpage",
		label: "Startpage",
		description: "Credential-free scrape of Startpage (Google-backed) results; may be bot-challenged",
	},
	{
		value: "duckduckgo",
		label: "DuckDuckGo",
		description: "Credential-free best-effort fallback; may be bot-challenged on datacenter/shared-egress IPs",
	},
	{
		value: "ecosia",
		label: "Ecosia",
		description: "Credential-free scrape of Ecosia (Google-backed) results",
	},
	{
		value: "google",
		label: "Google",
		description: "Credential-free fallback; slower and may be bot-challenged",
	},
	{
		value: "mojeek",
		label: "Mojeek",
		description: "Credential-free scrape of Mojeek's independent index",
	},
	{
		value: "public",
		label: "Public Web",
		description: "Queries every credential-free engine in parallel and consolidates deduplicated results",
	},
] as const;

/** Default hard timeout for each web-search provider transport. */
export const DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS = 60;

/** Maximum configurable hard timeout for each web-search provider transport. */
export const MAX_WEB_SEARCH_TIMEOUT_SECONDS = 300;

/** Supported web search providers (every option except `auto`). */
export type SearchProviderId = Exclude<(typeof SEARCH_PROVIDER_OPTIONS)[number]["value"], "auto">;

/**
 * Auto-resolution priority order. Derived from {@link SEARCH_PROVIDER_OPTIONS}
 * (minus `auto`) so any dropdown/setting and `resolveProviderChain()` share
 * one source of truth and never drift apart: keyed providers first (used
 * only when their env credential exists), then credential-free engines, with
 * the public fan-out last (explicit selection only).
 */
export const SEARCH_PROVIDER_ORDER: readonly SearchProviderId[] = SEARCH_PROVIDER_OPTIONS.flatMap((option) =>
	option.value === "auto" ? [] : [option.value],
);

/** Concrete provider choices (no `auto` sentinel) — for list-valued settings like order/exclude. */
export const SEARCH_PROVIDER_CHOICES = SEARCH_PROVIDER_OPTIONS.filter((option) => option.value !== "auto");

export const SEARCH_PROVIDER_PREFERENCES = ["auto", ...SEARCH_PROVIDER_ORDER] as const;

/** Display labels, derived from {@link SEARCH_PROVIDER_OPTIONS}. */
export const SEARCH_PROVIDER_LABELS = Object.fromEntries(
	SEARCH_PROVIDER_OPTIONS.flatMap((option) =>
		option.value === "auto" ? [] : [[option.value, option.label] as const],
	),
) as Record<SearchProviderId, string>;

export function isSearchProviderId(value: string): value is SearchProviderId {
	return SEARCH_PROVIDER_ORDER.includes(value as SearchProviderId);
}

export function isSearchProviderPreference(value: string): value is SearchProviderId | "auto" {
	return SEARCH_PROVIDER_PREFERENCES.includes(value as SearchProviderId | "auto");
}

/** Source returned by search (all providers) */
export interface SearchSource {
	title: string;
	url: string;
	snippet?: string;
	/** ISO date string or relative ("2d ago") */
	publishedDate?: string;
	/** Age in seconds for consistent formatting */
	ageSeconds?: number;
	author?: string;
}

/** Citation with text reference (LLM-mediated providers) */
export interface SearchCitation {
	url: string;
	title: string;
	citedText?: string;
}

/** Usage metrics */
export interface SearchUsage {
	inputTokens?: number;
	outputTokens?: number;
	/** Anthropic: number of web search requests made */
	searchRequests?: number;
	/** Perplexity: combined token count */
	totalTokens?: number;
}

/** Unified response across providers */
export interface SearchResponse {
	provider: SearchProviderId | "none";
	/** Synthesized answer text (LLM-mediated providers) */
	answer?: string;
	/** Search result sources */
	sources: SearchSource[];
	/** Text citations with context */
	citations?: SearchCitation[];
	/** Intermediate search queries */
	searchQueries?: string[];
	/** Follow-up question suggestions (provider-dependent) */
	relatedQuestions?: string[];
	/** Token usage metrics */
	usage?: SearchUsage;
	/** Model used */
	model?: string;
	/** Request ID for debugging */
	requestId?: string;
	/** Authentication mode used by the provider (e.g. oauth, api-key) */
	authMode?: string;
}

/** Provider-specific error with optional HTTP status */
export class SearchProviderError extends Error {
	constructor(
		public readonly provider: SearchProviderId,
		message: string,
		public readonly status?: number,
	) {
		super(message);
		this.name = "SearchProviderError";
	}
}

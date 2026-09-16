/**
 * Unified Web Search execution.
 *
 * Walks the configured provider chain with automatic fallback (quota,
 * auth, bot-challenge, and transport failures advance to the next provider),
 * post-filters results against structured query constraints, and formats the
 * winning response for LLM consumption.
 *
 * Ported from oh-my-pi (can1357/oh-my-pi, MIT) `web/search/index.ts`,
 * adapted to AutoRAG: credentials come from environment variables (see
 * `credentials.ts`), ordering/exclusion come from `provider.ts` setters
 * instead of a settings singleton, and there is no TUI rendering layer.
 */
import { formatForLLM, hasRenderableSearchContent, WEB_SEARCH_SYSTEM_PROMPT } from "./format.ts";
import {
	formatSearchProviderFailure,
	formatSearchProviderFailures,
	getSearchProvider,
	getSearchProviderLabel,
	resolveProviderCandidates,
	type SearchProviderCandidate,
	type SearchProviderContract,
} from "./provider.ts";
import { applyQueryConstraints, parseSearchQuery } from "./query.ts";
import {
	DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS,
	MAX_WEB_SEARCH_TIMEOUT_SECONDS,
	SearchProviderError,
	type SearchProviderId,
	type SearchResponse,
} from "./types.ts";

export interface WebSearchQueryParams {
	query: string;
	recency?: "day" | "week" | "month" | "year";
	limit?: number;
	max_tokens?: number;
	temperature?: number;
	num_search_results?: number;
	provider?: SearchProviderId | "auto";
}

export interface WebSearchResultDetails {
	response: SearchResponse;
	error?: string;
}

export interface WebSearchExecuteResult {
	content: Array<{ type: "text"; text: string }>;
	details: WebSearchResultDetails;
}

export interface WebSearchExecuteOptions {
	signal?: AbortSignal;
	/** Per-provider transport hard timeout in milliseconds (default 60s, capped at 300s). */
	timeoutMs?: number;
	/** Transport injection for tests/proxies; forwarded to providers that accept it. */
	fetch?: (input: string | URL | Request, init?: RequestInit) => Promise<Response>;
}

function resolveTimeoutMs(timeoutMs: number | undefined): number {
	if (timeoutMs === undefined || !Number.isFinite(timeoutMs) || timeoutMs <= 0) {
		return DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS * 1_000;
	}
	return Math.min(timeoutMs, MAX_WEB_SEARCH_TIMEOUT_SECONDS * 1_000);
}

/** Execute a web search through the provider fallback chain. */
export async function executeWebSearch(
	params: WebSearchQueryParams,
	options: WebSearchExecuteOptions = {},
): Promise<WebSearchExecuteResult> {
	const { signal } = options;
	const explicitProvider = params.provider;
	let candidates: SearchProviderCandidate[];
	if (explicitProvider && explicitProvider !== "auto") {
		candidates = [{ id: explicitProvider, explicit: true }];
	} else {
		// `auto` and the default both walk the configured chain;
		// exclusions still apply.
		candidates = resolveProviderCandidates();
	}

	const parsedQuery = parseSearchQuery(params.query);
	const timeoutMs = resolveTimeoutMs(options.timeoutMs);

	const failures: Array<{ provider: Pick<SearchProviderContract, "id" | "label">; error: unknown }> = [];
	let availableProviderCount = 0;
	let lastProvider: Pick<SearchProviderContract, "id" | "label"> | undefined;
	for (const candidate of candidates) {
		let provider: SearchProviderContract | undefined;
		const providerMeta = { id: candidate.id, label: getSearchProviderLabel(candidate.id) };
		lastProvider = providerMeta;
		try {
			provider = await getSearchProvider(candidate.id);
			// Plain-object providers (test fakes, host embeddings) may skip the
			// `isExplicitlyAvailable` override; it defaults to `isAvailable`.
			const available = candidate.explicit
				? (await provider.isExplicitlyAvailable?.()) ?? (await provider.isAvailable())
				: await provider.isAvailable();
			if (!available && !candidate.explicit) continue;
			if (!available && candidate.explicit) {
				throw new SearchProviderError(
					provider.id,
					`${provider.label} web search is unavailable. Configure its credentials or select the automatic provider chain.`,
				);
			}
			availableProviderCount++;
			lastProvider = provider;

			const response = await provider.search({
				query: params.query,
				parsedQuery,
				limit: params.limit,
				recency: params.recency,
				systemPrompt: WEB_SEARCH_SYSTEM_PROMPT,
				maxOutputTokens: params.max_tokens,
				numSearchResults: params.num_search_results,
				temperature: params.temperature,
				signal,
				timeoutMs,
				fetch: options.fetch,
			});

			// Lenient constraint pass over whatever the provider returned: enforce
			// site:/inurl:/intitle:/filetype:/date directives the provider could
			// not (or only partially) honor natively, relaxing any dimension that
			// would wipe out every result. Citations/answer text stay untouched.
			let finalResponse = response;
			const constraintNotes: string[] = [];
			if (parsedQuery.hasConstraints && response.sources.length > 0) {
				const filtered = applyQueryConstraints(response.sources, parsedQuery);
				if (filtered.sources.length !== response.sources.length) {
					finalResponse = { ...response, sources: filtered.sources };
				}
				for (const label of filtered.dropped) {
					constraintNotes.push(`no results matched \`${label}\`; the constraint was relaxed`);
				}
			}

			if (!hasRenderableSearchContent(finalResponse)) {
				throw new SearchProviderError(provider.id, `${provider.label} returned no renderable search content.`, 204);
			}

			const text = formatForLLM(finalResponse, constraintNotes);

			return {
				content: [{ type: "text" as const, text }],
				details: { response: finalResponse },
			};
		} catch (error) {
			// Surface user-initiated cancellation immediately so the caller sees
			// a clean abort instead of a generic "all providers failed" message.
			if (signal?.aborted) {
				throw new DOMException("The operation was aborted.", "AbortError");
			}
			failures.push({ provider: provider ?? providerMeta, error });
		}
	}

	if (availableProviderCount === 0 && failures.length === 0) {
		const message = "No web search provider configured.";
		return {
			content: [{ type: "text" as const, text: `Error: ${message}` }],
			details: { response: { provider: "none", sources: [] }, error: message },
		};
	}

	const lastFailure = failures[failures.length - 1];
	const baseMessage = lastFailure
		? formatSearchProviderFailure(lastFailure.error, lastFailure.provider)
		: `Unknown error from ${lastProvider?.label ?? "web search provider"}`;
	const message =
		failures.length > 1 ? `All web search providers failed: ${formatSearchProviderFailures(failures)}` : baseMessage;

	return {
		content: [{ type: "text" as const, text: `Error: ${message}` }],
		details: {
			response: { provider: lastFailure?.provider.id ?? lastProvider?.id ?? "none", sources: [] },
			error: message,
		},
	};
}

export { getSearchProvider, setExcludedSearchProviders, setSearchProviderOrder } from "./provider.ts";
export type { SearchProviderId, SearchResponse } from "./types.ts";
export { isSearchProviderId, isSearchProviderPreference } from "./types.ts";

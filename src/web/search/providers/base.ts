/**
 * Shared web search provider contract.
 *
 * Ported from oh-my-pi (can1357/oh-my-pi, MIT) `web/search/providers/base.ts`
 * with one adaptation: oh-my-pi resolves credentials through its AuthStorage
 * broker; AutoRAG providers read their documented environment variables
 * directly (see `../credentials.ts`), so `isAvailable` takes no arguments.
 */
import type { ModelNativeSearchAuth } from "../model-auth.ts";
import type { StructuredQuery } from "../query.ts";
import type { SearchProviderId, SearchResponse } from "../types.ts";

/** Minimal fetch signature providers accept for transport injection (tests, proxies). */
export type FetchImpl = (input: string | URL | Request, init?: RequestInit) => Promise<Response>;

/**
 * Shared web search parameters passed to providers.
 */
export interface SearchParams {
	query: string;
	/**
	 * Structured view of `query`, parsed once by the search pipeline:
	 * Google-style directives (`site:`, `before:`/`after:`, `inurl:`,
	 * `intitle:`, `filetype:`, quoted phrases, `OR` groups, `-exclusions`)
	 * extracted into fields.
	 *
	 * Providers SHOULD map constraints onto native API parameters
	 * (domain/date filters) or engine query syntax (`formatQuery`) where the
	 * upstream supports them, and lean lenient otherwise: the pipeline
	 * post-filters every response with `applyQueryConstraints`, which
	 * relaxes any constraint that would eliminate all results — so a
	 * best-effort search always beats an empty one. When absent (direct
	 * provider calls), parse with `parseSearchQuery(params.query)`.
	 */
	parsedQuery?: StructuredQuery;
	limit?: number;
	/**
	 * Temporal filter narrowing results to the specified time window.
	 *
	 * Providers MUST interpret this as a pure time filter. Providers MUST NOT
	 * use recency as an implicit signal to change topic scope, content domain,
	 * or ranking strategy. Providers that do not support temporal filtering
	 * MUST ignore this field silently; they MUST NOT approximate it by
	 * rewriting the query or altering any other request parameter.
	 */
	recency?: "day" | "week" | "month" | "year";
	systemPrompt?: string;
	signal?: AbortSignal;
	/** Hard timeout for this provider's search transport, in milliseconds. */
	timeoutMs?: number;
	fetch?: FetchImpl;
	maxOutputTokens?: number;
	numSearchResults?: number;
	temperature?: number;
	/**
	 * Agent model credential for the model-native providers, carried per
	 * request so concurrent agents never share one process-wide key.
	 */
	modelAuth?: ModelNativeSearchAuth;
	/** Providers this request must not use, including inside the `public` fan-out. */
	excludedProviders?: readonly SearchProviderId[];
}

/** Per-request context consulted by provider availability checks. */
export interface SearchAvailabilityContext {
	readonly modelAuth?: ModelNativeSearchAuth;
}

/**
 * Structural provider contract accepted by the registry. Production
 * providers extend {@link SearchProvider} (which supplies the
 * `isExplicitlyAvailable` default); test fakes and host embeddings may be
 * plain objects matching this shape.
 */
export interface SearchProviderContract {
	readonly id: SearchProviderId;
	readonly label: string;
	isAvailable(context?: SearchAvailabilityContext): Promise<boolean> | boolean;
	isExplicitlyAvailable?(context?: SearchAvailabilityContext): Promise<boolean> | boolean;
	search(params: SearchParams): Promise<SearchResponse>;
}

/** Base class for web search providers. */
export abstract class SearchProvider implements SearchProviderContract {
	abstract readonly id: SearchProviderId;
	abstract readonly label: string;

	/**
	 * Indicates whether this provider has the credentials/config it needs to
	 * service a request right now. AutoRAG providers consult their documented
	 * environment variables; credential-free providers always return true.
	 *
	 * Drives auto-chain admission: providers that return `false` are skipped
	 * when the chain walks the order. Explicit selection uses
	 * {@link isExplicitlyAvailable} instead.
	 */
	abstract isAvailable(context?: SearchAvailabilityContext): Promise<boolean> | boolean;

	/**
	 * Returns `true` when this provider should run when the user explicitly
	 * selects it, even if {@link isAvailable} would reject it for the auto
	 * chain. Providers that ship an unauthenticated fallback override this so
	 * explicit selection still routes through the fallback rather than
	 * silently falling back to another provider.
	 *
	 * Defaults to mirroring {@link isAvailable}.
	 */
	isExplicitlyAvailable(context?: SearchAvailabilityContext): Promise<boolean> | boolean {
		return this.isAvailable(context);
	}

	/**
	 * Execute a search.
	 */
	abstract search(params: SearchParams): Promise<SearchResponse>;
}

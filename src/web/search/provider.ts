// Lazy registry of web search providers.
//
// Each provider is loaded on first use; importing this module loads zero
// provider implementations. Provider modules are heavy (each pulls in
// fetch/parse/format helpers) and only one — at most — is needed per session,
// so eager construction was wasted work at startup.
//
// Ported from oh-my-pi (can1357/oh-my-pi, MIT) `web/search/provider.ts`,
// trimmed to the providers AutoRAG ships and extended with an explicit
// instance-registration seam (host embeddings + tests).

import type { SearchProvider, SearchProviderContract } from "./providers/base.ts";
import { SEARCH_PROVIDER_LABELS, SEARCH_PROVIDER_ORDER, SearchProviderError, type SearchProviderId } from "./types.ts";

export type { FetchImpl, SearchParams } from "./providers/base.ts";
export { SearchProvider } from "./providers/base.ts";
export type { SearchProviderContract } from "./providers/base.ts";
export { SEARCH_PROVIDER_ORDER } from "./types.ts";

interface ProviderMeta {
	id: SearchProviderId;
	label: string;
	load: () => Promise<SearchProviderContract>;
}

/** Lazy factories. Each `load()` dynamic-imports its provider module on first call. */
const PROVIDER_META: Record<SearchProviderId, ProviderMeta> = {
	brave: {
		id: "brave",
		label: SEARCH_PROVIDER_LABELS.brave,
		load: async () => new (await import("./providers/brave.ts")).BraveProvider(),
	},
	tavily: {
		id: "tavily",
		label: SEARCH_PROVIDER_LABELS.tavily,
		load: async () => new (await import("./providers/tavily.ts")).TavilyProvider(),
	},
	exa: {
		id: "exa",
		label: SEARCH_PROVIDER_LABELS.exa,
		load: async () => new (await import("./providers/exa.ts")).ExaProvider(),
	},
	jina: {
		id: "jina",
		label: SEARCH_PROVIDER_LABELS.jina,
		load: async () => new (await import("./providers/jina.ts")).JinaProvider(),
	},
	kagi: {
		id: "kagi",
		label: SEARCH_PROVIDER_LABELS.kagi,
		load: async () => new (await import("./providers/kagi.ts")).KagiProvider(),
	},
	kimi: {
		id: "kimi",
		label: SEARCH_PROVIDER_LABELS.kimi,
		load: async () => new (await import("./providers/kimi.ts")).KimiProvider(),
	},
	searxng: {
		id: "searxng",
		label: SEARCH_PROVIDER_LABELS.searxng,
		load: async () => new (await import("./providers/searxng.ts")).SearXNGProvider(),
	},
	startpage: {
		id: "startpage",
		label: SEARCH_PROVIDER_LABELS.startpage,
		load: async () => new (await import("./providers/startpage.ts")).StartpageProvider(),
	},
	duckduckgo: {
		id: "duckduckgo",
		label: SEARCH_PROVIDER_LABELS.duckduckgo,
		load: async () => new (await import("./providers/duckduckgo.ts")).DuckDuckGoProvider(),
	},
	ecosia: {
		id: "ecosia",
		label: SEARCH_PROVIDER_LABELS.ecosia,
		load: async () => new (await import("./providers/ecosia.ts")).EcosiaProvider(),
	},
	google: {
		id: "google",
		label: SEARCH_PROVIDER_LABELS.google,
		load: async () => new (await import("./providers/google.ts")).GoogleProvider(),
	},
	mojeek: {
		id: "mojeek",
		label: SEARCH_PROVIDER_LABELS.mojeek,
		load: async () => new (await import("./providers/mojeek.ts")).MojeekProvider(),
	},
	public: {
		id: "public",
		label: SEARCH_PROVIDER_LABELS.public,
		load: async () => new (await import("./providers/public.ts")).PublicWebProvider(),
	},
};

const instanceCache = new Map<SearchProviderId, SearchProviderContract>();

/** Cheap, sync metadata accessor — never triggers a provider load. */
export function getSearchProviderLabel(id: SearchProviderId): string {
	return PROVIDER_META[id]?.label ?? id;
}

/** Format one provider failure for the user-facing fallback summary. */
export function formatSearchProviderFailure(error: unknown, provider: Pick<SearchProvider, "id" | "label">): string {
	if (error instanceof SearchProviderError) {
		if (error.status === 401 || error.status === 403) {
			return `${getSearchProviderLabel(error.provider)} authorization failed (${error.status}). Check API key or base URL.`;
		}
		return error.message;
	}
	if (error instanceof Error) return error.message;
	return `Unknown error from ${provider.label}`;
}

/** Format the ordered provider fallback failures for terminal/tool output. */
export function formatSearchProviderFailures(
	failures: readonly { provider: Pick<SearchProvider, "id" | "label">; error: unknown }[],
): string {
	return failures.map((f) => `${f.provider.id}: ${formatSearchProviderFailure(f.error, f.provider)}`).join("; ");
}

/**
 * Resolve and cache a provider instance. First call for a given id loads the
 * underlying module; subsequent calls return the cached singleton.
 * Instances registered through {@link registerSearchProvider} take
 * precedence over lazy module loads (host embedding, tests).
 */
export async function getSearchProvider(id: SearchProviderId): Promise<SearchProviderContract> {
	const cached = instanceCache.get(id);
	if (cached) return cached;
	const meta = PROVIDER_META[id];
	if (!meta) {
		throw new Error(`Unknown search provider: ${id}`);
	}
	const provider = await meta.load();
	instanceCache.set(id, provider);
	return provider;
}

/**
 * Register (or replace) a provider instance, bypassing the lazy module load.
 * Test seam and host-embedding hook; production provider modules register
 * nothing themselves.
 */
export function registerSearchProvider(provider: SearchProviderContract): void {
	instanceCache.set(provider.id, provider);
}

/** Drop every registered/cached provider instance (tests). */
export function clearRegisteredSearchProviders(): void {
	instanceCache.clear();
}

/** Provider fallback order set via configuration (default: built-in order). */
let orderedProvIds: readonly SearchProviderId[] = SEARCH_PROVIDER_ORDER;
/** Providers the user explicitly listed in configuration. */
let explicitProvIds = new Set<SearchProviderId>();

/**
 * Prioritize configured providers while retaining every unlisted provider in
 * its built-in relative order. Invalid IDs are ignored defensively. Listed
 * providers are treated as explicit selections: they resolve through
 * `isExplicitlyAvailable`.
 */
export function setSearchProviderOrder(providers: readonly SearchProviderId[]): void {
	const prioritized = new Set(providers.filter((id) => SEARCH_PROVIDER_ORDER.includes(id)));
	explicitProvIds = prioritized;
	orderedProvIds =
		prioritized.size === 0
			? SEARCH_PROVIDER_ORDER
			: [...prioritized, ...SEARCH_PROVIDER_ORDER.filter((id) => !prioritized.has(id))];
}

/** Providers excluded from web search resolution via configuration. */
let excludedProvIds = new Set<SearchProviderId>();

/** Set providers that web search should never use, including fallbacks. */
export function setExcludedSearchProviders(providers: readonly SearchProviderId[]): void {
	excludedProvIds = new Set(providers);
}

/** `true` when configuration excludes `id` from web search (auto chain and the Public Web fan-out). */
export function isSearchProviderExcluded(id: SearchProviderId): boolean {
	return excludedProvIds.has(id);
}

export interface SearchProviderCandidate {
	id: SearchProviderId;
	explicit: boolean;
}

/**
 * Return provider candidates in fallback order without loading their modules.
 * `forcedProvider` (a per-request `provider` argument) is terminal-first and
 * bypasses exclusion; configured-order entries carry `explicit: true`.
 */
export function resolveProviderCandidates(forcedProvider?: SearchProviderId): SearchProviderCandidate[] {
	const candidates: SearchProviderCandidate[] = [];

	if (forcedProvider !== undefined && !isSearchProviderExcluded(forcedProvider)) {
		candidates.push({ id: forcedProvider, explicit: true });
	}

	for (const id of orderedProvIds) {
		if (id === forcedProvider || isSearchProviderExcluded(id)) continue;
		candidates.push({ id, explicit: explicitProvIds.has(id) });
	}

	return candidates;
}

/**
 * Resolve the complete available provider chain.
 *
 * This compatibility helper loads every candidate. Search execution should use
 * {@link resolveProviderCandidates} so fallback modules load only when reached.
 */
export async function resolveProviderChain(forcedProvider?: SearchProviderId): Promise<SearchProviderContract[]> {
	const providers: SearchProviderContract[] = [];

	for (const candidate of resolveProviderCandidates(forcedProvider)) {
		const provider = await getSearchProvider(candidate.id);
		const available = candidate.explicit
			? (await provider.isExplicitlyAvailable?.()) ?? (await provider.isAvailable())
			: await provider.isAvailable();
		if (available) providers.push(provider);
	}

	return providers;
}

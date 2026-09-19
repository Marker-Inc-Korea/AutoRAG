/**
 * Public Web aggregate provider — fans out to every credential-free engine
 * in parallel and consolidates deduplicated results by cross-engine
 * consensus.
 *
 * Ported from oh-my-pi (can1357/oh-my-pi, MIT)
 * `packages/coding-agent/src/web/search/providers/public.ts`.
 * Dropped: `Bun.sleep` → `setTimeout`-based delay; `AuthStorage` parameter
 * from `isAvailable`/`isExplicitlyAvailable`. The `PublicWebDeadlines` test
 * seam is kept. Engine instances come from `getSearchProvider` (registered
 * fakes in tests, lazy module loads in production).
 */
import { hasRenderableSearchContent } from "../format.ts";
import { formatSearchProviderFailures, getSearchProvider, isSearchProviderExcluded } from "../provider.ts";
import { SearchProviderError, type SearchProviderId, type SearchResponse, type SearchSource } from "../types.ts";
import { clampNumResults } from "../utils.ts";
import type { SearchParams } from "./base.ts";
import { SearchProvider } from "./base.ts";
import { withHardTimeout } from "./utils.ts";

const PUBLIC_ENGINE_IDS = [
	"startpage",
	"google",
	"duckduckgo",
	"ecosia",
	"mojeek",
] as const satisfies readonly SearchProviderId[];

const DEFAULT_NUM_RESULTS = 15;
const MAX_NUM_RESULTS = 30;
const SOFT_DEADLINE_MS = 5_000;
const HARD_DEADLINE_MS = 30_000;

export interface PublicWebDeadlines {
	softMs?: number;
	hardMs?: number;
}

interface MergedSource {
	source: SearchSource;
	engines: number;
	bestRank: number;
	order: number;
}

function dedupKey(rawUrl: string): string {
	try {
		const url = new URL(rawUrl);
		const host = url.hostname.toLowerCase().replace(/^www\./, "");
		let path = url.pathname;
		if (path.length > 1 && path.endsWith("/")) path = path.slice(0, -1);
		return `${host}${path}${url.search}`;
	} catch {
		return rawUrl;
	}
}

function mergeSources(merged: Map<string, MergedSource>, sources: readonly SearchSource[]): void {
	for (const [rank, source] of sources.entries()) {
		const key = dedupKey(source.url);
		const existing = merged.get(key);
		if (!existing) {
			merged.set(key, { source: { ...source }, engines: 1, bestRank: rank, order: merged.size });
			continue;
		}
		existing.engines += 1;
		if (rank < existing.bestRank) {
			existing.bestRank = rank;
			existing.source.title = source.title;
			existing.source.url = source.url;
		}
		if (source.snippet && source.snippet.length > (existing.source.snippet?.length ?? 0)) {
			existing.source.snippet = source.snippet;
		}
		if (source.publishedDate !== undefined) existing.source.publishedDate ??= source.publishedDate;
		if (source.ageSeconds !== undefined) existing.source.ageSeconds ??= source.ageSeconds;
	}
}

function sleep(ms: number, signal: AbortSignal | undefined): Promise<void> {
	if (ms <= 0) return Promise.resolve();
	let resolveFn: () => void;
	let rejectFn: (reason: unknown) => void;
	const promise = new Promise<void>((resolve, reject) => {
		resolveFn = resolve;
		rejectFn = reject;
	});
	const timer = setTimeout(() => {
		signal?.removeEventListener("abort", onAbort);
		resolveFn();
	}, ms);
	const onAbort = () => {
		clearTimeout(timer);
		rejectFn(new DOMException("The operation was aborted.", "AbortError"));
	};
	signal?.addEventListener("abort", onAbort, { once: true });
	if (signal?.aborted) onAbort();
	return promise;
}

export async function searchPublicWeb(
	params: SearchParams,
	deadlines: PublicWebDeadlines = {},
): Promise<SearchResponse> {
	const softMs = deadlines.softMs ?? SOFT_DEADLINE_MS;
	const hardMs = deadlines.hardMs ?? HARD_DEADLINE_MS;
	const numResults = clampNumResults(params.numSearchResults ?? params.limit, DEFAULT_NUM_RESULTS, MAX_NUM_RESULTS);
	const engineIds = PUBLIC_ENGINE_IDS.filter((id) => !isSearchProviderExcluded(id, params.excludedProviders));
	if (engineIds.length === 0) {
		throw new SearchProviderError("public", "Every credential-free engine is excluded by settings.", 400);
	}

	const straggler = new AbortController();
	const signal = AbortSignal.any([withHardTimeout(params.signal, params.timeoutMs), straggler.signal]);

	const responses: (SearchResponse | undefined)[] = new Array(engineIds.length);
	const failures: { provider: { id: SearchProviderId; label: string }; error: unknown }[] = [];
	let resolveFirstSuccess: () => void = () => {};
	const firstSuccess = new Promise<void>((resolve) => {
		resolveFirstSuccess = resolve;
	});
	// A scraped engine answers HTTP 200 with zero parsed results all the time.
	// Only a response that actually carries results ends the wait; an empty one
	// must not abort the engines that are still working.
	const isUseful = (response: SearchResponse | undefined): boolean =>
		response !== undefined && hasRenderableSearchContent(response);
	const all = Promise.all(
		engineIds.map(async (id, index) => {
			try {
				const provider = await getSearchProvider(id);
				const response = await provider.search({ ...params, signal });
				responses[index] = response;
				if (hasRenderableSearchContent(response)) resolveFirstSuccess();
			} catch (error) {
				failures.push({ provider: { id, label: id }, error });
			}
		}),
	);

	await Promise.race([all, sleep(softMs, signal)]);
	if (!responses.some(isUseful) && failures.length < engineIds.length) {
		await Promise.race([all, firstSuccess, sleep(Math.max(0, hardMs - softMs), signal)]);
	}
	straggler.abort();

	const merged = new Map<string, MergedSource>();
	for (const response of responses) {
		if (response) mergeSources(merged, response.sources);
	}

	if (merged.size === 0 && failures.length === engineIds.length) {
		throw new SearchProviderError(
			"public",
			`All public engines failed: ${formatSearchProviderFailures(failures)}`,
			503,
		);
	}

	const sources = [...merged.values()]
		.sort((a, b) => b.engines - a.engines || a.bestRank - b.bestRank || a.order - b.order)
		.slice(0, numResults)
		.map((entry) => entry.source);

	return { provider: "public", sources };
}

export class PublicWebProvider extends SearchProvider {
	readonly id = "public" as const;
	readonly label = "Public Web";

	isAvailable(): boolean {
		return false;
	}

	override isExplicitlyAvailable(): boolean {
		return true;
	}

	search(params: SearchParams): Promise<SearchResponse> {
		return searchPublicWeb(params);
	}
}

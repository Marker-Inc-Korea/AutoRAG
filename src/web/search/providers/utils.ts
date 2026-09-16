/**
 * Shared helpers for web search providers.
 *
 * Ported from oh-my-pi (can1357/oh-my-pi, MIT) `web/search/providers/utils.ts`.
 */
import {
	DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS,
	SEARCH_PROVIDER_LABELS,
	SearchProviderError,
	type SearchProviderId,
	type SearchSource,
} from "../types.ts";
import { dateToAgeSeconds } from "../utils.ts";

/**
 * The 60-second default tolerates legitimate slow LLM-mediated responses
 * while bounding stalls when the runtime's `AbortSignal` fails to propagate.
 */
export const SEARCH_HARD_TIMEOUT_MS = DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS * 1_000;

/**
 * Compose a caller-supplied {@link AbortSignal} with a hard timeout so an
 * outbound `fetch()` is guaranteed to settle within `ms` even when the
 * runtime fails to propagate cancellation to the underlying transport.
 */
export function withHardTimeout(signal: AbortSignal | undefined, ms: number = SEARCH_HARD_TIMEOUT_MS): AbortSignal {
	const timeout = AbortSignal.timeout(ms);
	return signal ? AbortSignal.any([signal, timeout]) : timeout;
}

/**
 * Map a provider's raw source list to the unified SearchSource shape,
 * clamped to the requested result count and annotated with ageSeconds.
 */
export function toSearchSources(
	sources: ReadonlyArray<{
		title: string;
		url: string;
		snippet?: string;
		publishedDate?: string;
	}>,
	numResults: number,
): SearchSource[] {
	return sources.slice(0, numResults).map((source) => ({
		title: source.title,
		url: source.url,
		snippet: source.snippet,
		publishedDate: source.publishedDate,
		ageSeconds: dateToAgeSeconds(source.publishedDate),
	}));
}

/**
 * Quota/auth signals across providers. Map those into compact,
 * provider-tagged messages so the orchestrator can chain-advance cleanly and
 * the final summary stays legible when every provider rejects the request.
 *
 * Returns `null` when the response does not match a known quota/auth signal,
 * leaving the caller to throw its provider-specific fallback error.
 */
const CREDIT_BODY_PATTERN = /credits?\s*(?:exhausted|exceeded)|quota|insufficient/i;

export function classifyProviderHttpError(
	provider: SearchProviderId,
	status: number,
	body: string,
): SearchProviderError | null {
	if (CREDIT_BODY_PATTERN.test(body)) {
		return new SearchProviderError(provider, `${provider}: credits exhausted`, status);
	}
	if (status === 402) {
		return new SearchProviderError(provider, `${provider}: 402 credits exhausted`, status);
	}
	if (status === 401) {
		return new SearchProviderError(provider, `${provider}: 401 unauthorized`, status);
	}
	if (status === 403) {
		return new SearchProviderError(provider, `${provider}: 403 forbidden`, status);
	}
	return null;
}

/**
 * Collapse runs of whitespace in a loosely-typed provider field, returning
 * `undefined` for missing/non-string/blank values. Shared so tab/newline
 * folding cannot drift between providers.
 */
export function normalizeSearchText(value: unknown): string | undefined {
	if (typeof value !== "string") return undefined;
	const text = value.replace(/\s+/g, " ").trim();
	return text.length > 0 ? text : undefined;
}

/**
 * Read a provider response body up to a byte cap, truncating or throwing when
 * the limit is exceeded. Shared so streaming-cap fixes land in one place.
 */
export async function readLimitedText(
	response: Response,
	provider: SearchProviderId,
	maxBytes: number,
	truncate = false,
): Promise<string> {
	if (!response.body) return "";
	const reader = response.body.getReader();
	let buffer = new Uint8Array(Math.min(maxBytes, 64 * 1024));
	let bytes = 0;

	try {
		for (; ;) {
			const { done, value } = await reader.read();
			if (done) break;
			const accepted = Math.min(value.byteLength, maxBytes - bytes);
			const nextBytes = bytes + accepted;
			if (nextBytes > buffer.byteLength) {
				const grown = new Uint8Array(Math.min(maxBytes, Math.max(nextBytes, buffer.byteLength * 2)));
				grown.set(buffer.subarray(0, bytes));
				buffer = grown;
			}
			buffer.set(value.subarray(0, accepted), bytes);
			bytes = nextBytes;
			if (accepted < value.byteLength) {
				await reader.cancel().catch(() => undefined);
				if (!truncate)
					throw new SearchProviderError(
						provider,
						`${SEARCH_PROVIDER_LABELS[provider]} API response exceeded ${Math.round(maxBytes / (1024 * 1024))} MiB`,
						500,
					);
				break;
			}
		}
	} finally {
		reader.releaseLock();
	}

	return new TextDecoder().decode(buffer.subarray(0, bytes));
}

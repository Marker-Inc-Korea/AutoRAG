/**
 * Perplexity web search provider — anonymous ask endpoint.
 *
 * Perplexity's consumer `perplexity_ask` endpoint accepts anonymous
 * browser-style requests, so this provider is always available with zero
 * credentials: the strongest always-on route in the credential-free chain.
 *
 * Ported from oh-my-pi (can1357/oh-my-pi, MIT)
 * `web/search/providers/perplexity.ts`. Dropped: oh-my-pi's OAuth session /
 * cookie / API-key paths (they need oh-my-pi's auth broker or a user-issued
 * key) — AutoRAG ships the anonymous route only.
 */
import { formatQuery, parseSearchQuery, type QuerySyntax } from "../query.ts";
import { SearchProviderError, type SearchResponse, type SearchSource } from "../types.ts";
import { dateToAgeSeconds } from "../utils.ts";
import type { FetchImpl, SearchParams } from "./base.ts";
import { SearchProvider } from "./base.ts";
import { classifyProviderHttpError, readLimitedText, withHardTimeout } from "./utils.ts";

const PERPLEXITY_ASK_URL = "https://www.perplexity.ai/rest/sse/perplexity_ask";
const API_VERSION = "2.18";
const MAX_ERROR_BYTES = 8 * 1024;
const ANONYMOUS_USER_AGENT =
	"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";

/** The ask endpoint's backend understands common Google-style operators. */
const PERPLEXITY_QUERY_SYNTAX: QuerySyntax = {
	phrases: true,
	negation: true,
	or: true,
	inUrl: true,
	inTitle: true,
	filetype: true,
	site: true,
	dateRange: true,
};

interface PerplexityStreamSource {
	name?: string;
	title?: string;
	url?: string;
	snippet?: string;
	timestamp?: string;
	date?: string;
}

interface PerplexityStreamBlock {
	intended_usage?: string;
	markdown_block?: { chunks?: string[]; chunk_starting_offset?: number; answer?: string };
	web_result_block?: { web_results?: PerplexityStreamSource[] };
}

interface PerplexityStreamEvent {
	status?: string;
	final?: boolean;
	text?: string;
	blocks?: PerplexityStreamBlock[];
	sources_list?: PerplexityStreamSource[];
	error_code?: string;
	error_message?: string;
	display_model?: string;
	user_selected_model?: string;
	uuid?: string;
}

/** Read an SSE response body as parsed JSON events (`data:` lines, `[DONE]` terminator). */
async function* readSseJson(body: ReadableStream<Uint8Array>, signal?: AbortSignal): AsyncGenerator<unknown> {
	const reader = body.getReader();
	const decoder = new TextDecoder();
	let buffer = "";
	try {
		for (;;) {
			if (signal?.aborted) return;
			const { done, value } = await reader.read();
			if (done) break;
			buffer += decoder.decode(value, { stream: true });
			let boundary = buffer.indexOf("\n\n");
			while (boundary !== -1) {
				const rawEvent = buffer.slice(0, boundary);
				buffer = buffer.slice(boundary + 2);
				boundary = buffer.indexOf("\n\n");
				for (const line of rawEvent.split("\n")) {
					if (!line.startsWith("data:")) continue;
					const data = line.slice(5).trim();
					if (data === "[DONE]" || data.length === 0) continue;
					try {
						yield JSON.parse(data) as unknown;
					} catch {
						// Tolerate partial JSON lines from chunk boundaries.
					}
				}
			}
		}
	} finally {
		reader.releaseLock();
	}
}

/** Merge incremental block snapshots keyed by intended_usage (markdown chunks merge by offset). */
function mergeBlocks(
	existing: readonly PerplexityStreamBlock[],
	incoming: readonly PerplexityStreamBlock[],
): PerplexityStreamBlock[] {
	const byUsage = new Map<string, PerplexityStreamBlock>();
	for (const block of existing) {
		if (block.intended_usage) byUsage.set(block.intended_usage, block);
	}
	for (const block of incoming) {
		if (!block.intended_usage) continue;
		const prior = byUsage.get(block.intended_usage);
		if (!prior) {
			byUsage.set(block.intended_usage, block);
			continue;
		}
		const merged: PerplexityStreamBlock = { ...prior, ...block };
		const priorMarkdown = prior.markdown_block;
		const incomingMarkdown = block.markdown_block;
		if (priorMarkdown && incomingMarkdown) {
			const mergedMarkdown = { ...priorMarkdown, ...incomingMarkdown };
			if (incomingMarkdown.chunks?.length) {
				const offset = incomingMarkdown.chunk_starting_offset ?? 0;
				const priorChunks = priorMarkdown.chunks ?? [];
				mergedMarkdown.chunks =
					offset === 0
						? [...incomingMarkdown.chunks]
						: [...priorChunks.slice(0, offset), ...incomingMarkdown.chunks];
			}
			merged.markdown_block = mergedMarkdown;
		}
		byUsage.set(block.intended_usage, merged);
	}
	return [...byUsage.values()];
}

function eventSources(event: PerplexityStreamEvent): SearchSource[] {
	const webResults =
		event.blocks?.find((block) => block.intended_usage === "web_results")?.web_result_block?.web_results ?? [];
	const raw = webResults.length > 0 ? webResults : (event.sources_list ?? []);
	const sources: SearchSource[] = [];
	for (const result of raw) {
		if (typeof result.url !== "string" || result.url.length === 0) continue;
		const publishedDate = result.timestamp ?? result.date;
		sources.push({
			title: result.name ?? result.title ?? result.url,
			url: result.url,
			...(result.snippet !== undefined ? { snippet: result.snippet } : {}),
			...(publishedDate !== undefined ? { publishedDate } : {}),
			...(dateToAgeSeconds(publishedDate) !== undefined ? { ageSeconds: dateToAgeSeconds(publishedDate) } : {}),
		});
	}
	return sources;
}

function eventAnswer(event: PerplexityStreamEvent): string {
	const markdownBlock = event.blocks?.find(
		(block) => block.intended_usage?.includes("markdown") || block.intended_usage === "ask_text",
	)?.markdown_block;
	if (markdownBlock) {
		if (markdownBlock.chunks?.length) return markdownBlock.chunks.join("");
		if (typeof markdownBlock.answer === "string") return markdownBlock.answer;
	}
	return "";
}

export class PerplexityProvider extends SearchProvider {
	readonly id = "perplexity" as const;
	readonly label = "Perplexity";

	/** The anonymous ask endpoint needs no credential of any kind. */
	isAvailable(): boolean {
		return true;
	}

	async search(params: SearchParams): Promise<SearchResponse> {
		const parsed = params.parsedQuery ?? parseSearchQuery(params.query);
		const query = parsed.hasDirectives ? formatQuery(parsed, PERPLEXITY_QUERY_SYNTAX) : params.query;
		const requestId = crypto.randomUUID();

		const fetchImpl: FetchImpl = params.fetch ?? fetch;
		const requestInit: RequestInit = {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				Accept: "text/event-stream",
				Origin: "https://www.perplexity.ai",
				Referer: "https://www.perplexity.ai/",
				"User-Agent": ANONYMOUS_USER_AGENT,
				"X-Request-ID": requestId,
			},
			body: JSON.stringify({
				query_str: query,
				params: {
					query_str: query,
					search_focus: "internet",
					mode: "copilot",
					model_preference: "experimental",
					sources: ["web"],
					attachments: [],
					frontend_uuid: crypto.randomUUID(),
					frontend_context_uuid: crypto.randomUUID(),
					version: API_VERSION,
					language: "en-US",
					timezone: Intl.DateTimeFormat().resolvedOptions().timeZone ?? "UTC",
					search_recency_filter: params.recency ?? null,
					is_incognito: true,
					use_schematized_api: true,
					// Force retrieval: the ask backend may otherwise answer from
					// memory, which is not a search result.
					skip_search_enabled: false,
					always_search_override: true,
					prompt_source: "user",
					source: "default",
					local_search_enabled: false,
					should_ask_for_mcp_tool_confirmation: false,
					supports_tool_approval_modal: false,
					force_enable_browser_agent: false,
					is_local_browser_available: false,
					is_local_browser_allowed: false,
					send_back_text_in_streaming_api: true,
				},
			}),
			signal: withHardTimeout(params.signal, params.timeoutMs),
		};

		// The consumer ask endpoint intermittently drops the socket before
		// sending an HTTP response; retry the transport exactly once. Once an
		// HTTP response exists the outcome is final (no retrying real 4xx/5xx).
		let response: Response;
		try {
			response = await fetchImpl(PERPLEXITY_ASK_URL, requestInit);
		} catch (error) {
			if (params.signal?.aborted) throw error;
			response = await fetchImpl(PERPLEXITY_ASK_URL, requestInit);
		}

		if (!response.ok) {
			const errorText = await readLimitedText(response, "perplexity", MAX_ERROR_BYTES, true);
			const classified = classifyProviderHttpError("perplexity", response.status, errorText);
			if (classified) throw classified;
			throw new SearchProviderError(
				"perplexity",
				`Perplexity ask API error (${response.status}): ${errorText}`,
				response.status,
			);
		}
		if (!response.body) {
			throw new SearchProviderError("perplexity", "Perplexity ask API returned no response body", 500);
		}

		let merged: PerplexityStreamEvent = { blocks: [] };
		let answer = "";
		let model: string | undefined;
		let finalRequestId: string | undefined;
		const sourcesByUrl = new Map<string, SearchSource>();

		for await (const rawEvent of readSseJson(response.body, params.signal)) {
			const event = rawEvent as PerplexityStreamEvent;
			if (event.error_code) {
				throw new SearchProviderError(
					"perplexity",
					`Perplexity ask stream error: ${event.error_message ?? event.error_code}`,
					400,
				);
			}
			merged = {
				...merged,
				...event,
				blocks: event.blocks?.length ? mergeBlocks(merged.blocks ?? [], event.blocks) : (merged.blocks ?? []),
				sources_list: merged.sources_list ?? event.sources_list,
			};
			const eventAnswerText = eventAnswer(merged);
			if (eventAnswerText.length > 0) answer = eventAnswerText;
			for (const source of eventSources(merged)) {
				sourcesByUrl.set(source.url.replace(/\/$/, "").toLowerCase(), source);
			}
			const reportedModel = [merged.user_selected_model, merged.display_model].find(
				(candidate) => candidate && candidate !== "turbo",
			);
			if (reportedModel) model = reportedModel;
			if (merged.uuid) finalRequestId = merged.uuid;
			if (merged.final || merged.status === "COMPLETED") break;
		}

		const sources = [...sourcesByUrl.values()];
		const numResults = params.numSearchResults ?? params.limit;
		return {
			provider: "perplexity",
			...(answer.length > 0 ? { answer } : {}),
			sources: numResults ? sources.slice(0, numResults) : sources,
			...(model !== undefined ? { model } : {}),
			requestId: finalRequestId ?? requestId,
			authMode: "anonymous",
		};
	}
}

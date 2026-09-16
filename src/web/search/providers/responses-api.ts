/**
 * Shared OpenAI-Responses-API web search transport.
 *
 * Both `codex` (api.openai.com) and `xai` (api.x.ai) speak the Responses API
 * shape with a hosted `web_search` tool, so their transports share one
 * implementation parameterized by provider id, base URL, and default model.
 *
 * Ported from oh-my-pi (can1357/oh-my-pi, MIT) `web/search/providers/codex.ts`
 * and `xai.ts`. Dropped: oh-my-pi's ChatGPT-backend SSE stream (it speaks an
 * undocumented request shape tied to ChatGPT OAuth) — AutoRAG uses the
 * documented non-streaming Responses API with a plain API key via
 * `../model-auth.ts`.
 */
import { type ModelNativeSearchProviderId, resolveModelNativeCredential } from "../model-auth.ts";
import { formatQuery, GOOGLE_QUERY_SYNTAX, parseSearchQuery } from "../query.ts";
import { SearchProviderError, type SearchResponse, type SearchSource } from "../types.ts";
import type { SearchParams } from "./base.ts";
import { classifyProviderHttpError, readLimitedText, withHardTimeout } from "./utils.ts";

const MAX_RESPONSE_BYTES = 4 * 1024 * 1024;
const MAX_ERROR_BYTES = 8 * 1024;

interface ResponsesApiSource {
	url?: string;
	title?: string;
}

interface ResponsesApiAnnotation {
	type?: string;
	url?: string;
	title?: string;
	start_index?: number;
	end_index?: number;
}

interface ResponsesApiItem {
	type?: string;
	action?: { type?: string; query?: string; sources?: ResponsesApiSource[] };
	sources?: ResponsesApiSource[];
	content?: Array<{ type?: string; text?: string; annotations?: ResponsesApiAnnotation[] }>;
}

interface ResponsesApiResponse {
	id?: string;
	model?: string;
	output?: ResponsesApiItem[];
	usage?: { input_tokens?: number; output_tokens?: number; total_tokens?: number };
}

/** Strip prose punctuation and unmatched closing delimiters from extracted URLs. */
function normalizeExtractedUrl(candidate: string): string | undefined {
	let url = candidate.trim();
	const countChar = (target: string) => url.split(target).length - 1;
	while (url.length > 0) {
		const last = url.at(-1);
		if (!last) break;
		if (/[.,!?;:'"]/u.test(last)) {
			url = url.slice(0, -1);
			continue;
		}
		if (last === ")" && countChar(")") > countChar("(")) {
			url = url.slice(0, -1);
			continue;
		}
		if (last === "]" && countChar("]") > countChar("[")) {
			url = url.slice(0, -1);
			continue;
		}
		break;
	}
	if (!/^https?:\/\//.test(url)) return undefined;
	try {
		return new URL(url).toString();
	} catch {
		return undefined;
	}
}

function addSource(sources: SearchSource[], source: SearchSource): void {
	const normalized = normalizeExtractedUrl(source.url);
	if (!normalized) return;
	if (sources.some((candidate) => candidate.url === normalized)) return;
	sources.push({ ...source, url: normalized });
}

/** Extract citation sources from markdown links and bare URLs when annotations are absent. */
function extractTextSources(text: string): SearchSource[] {
	const sources: SearchSource[] = [];
	for (const match of text.matchAll(/\[([^\]]*)\]\((https?:\/\/[^)]+)\)/g)) {
		const url = normalizeExtractedUrl(match[2] ?? "");
		if (url) addSource(sources, { title: (match[1] ?? "").trim() || url, url });
	}
	for (const match of text.matchAll(/https?:\/\/\S+/g)) {
		const url = normalizeExtractedUrl(match[0] ?? "");
		if (url) addSource(sources, { title: url, url });
	}
	return sources;
}

function parseOutput(data: ResponsesApiResponse): { answer: string; sources: SearchSource[]; searchQueries: string[] } {
	const answerParts: string[] = [];
	const sources: SearchSource[] = [];
	const searchQueries: string[] = [];

	for (const item of data.output ?? []) {
		if (item.type === "web_search_call") {
			if (item.action?.query) searchQueries.push(item.action.query);
			for (const source of [...(item.action?.sources ?? []), ...(item.sources ?? [])]) {
				if (source.url) addSource(sources, { title: source.title ?? source.url, url: source.url });
			}
			continue;
		}
		if (item.type === "message") {
			for (const part of item.content ?? []) {
				if (part.type !== "output_text" || !part.text) continue;
				answerParts.push(part.text);
				for (const annotation of part.annotations ?? []) {
					if (annotation.type === "url_citation" && annotation.url) {
						addSource(sources, { title: annotation.title ?? annotation.url, url: annotation.url });
					}
				}
			}
		}
	}

	const answer = answerParts.join("\n\n").trim();
	if (sources.length === 0 && answer.length > 0) {
		for (const source of extractTextSources(answer)) addSource(sources, source);
	}
	return { answer, sources, searchQueries };
}

export interface ResponsesApiSearchOptions {
	/** Search provider id ("codex" | "xai") for credential resolution and errors. */
	readonly providerId: Extract<ModelNativeSearchProviderId, "codex" | "xai">;
	readonly defaultBaseUrl: string;
	readonly defaultModel: string;
	readonly modelEnvVar: string;
	readonly missingCredentialMessage: string;
}

/** Execute a Responses-API web search for the given provider configuration. */
export async function searchResponsesApi(
	options: ResponsesApiSearchOptions,
	params: SearchParams,
): Promise<SearchResponse> {
	const credential = resolveModelNativeCredential(options.providerId);
	if (!credential) {
		throw new SearchProviderError(options.providerId, options.missingCredentialMessage, 401);
	}
	const model =
		process.env[options.modelEnvVar]?.trim() ||
		(credential.source === "injected" ? credential.modelId : undefined) ||
		options.defaultModel;
	const parsed = params.parsedQuery ?? parseSearchQuery(params.query);
	const query = parsed.hasDirectives ? formatQuery(parsed, GOOGLE_QUERY_SYNTAX) : params.query;
	const baseUrl = (credential.baseUrl ?? options.defaultBaseUrl).replace(/\/+$/, "");

	const fetchImpl = params.fetch ?? fetch;
	const response = await fetchImpl(`${baseUrl}/responses`, {
		method: "POST",
		headers: {
			"content-type": "application/json",
			authorization: `Bearer ${credential.apiKey}`,
		},
		body: JSON.stringify({
			model,
			input: query,
			...(params.systemPrompt ? { instructions: params.systemPrompt } : {}),
			tools: [{ type: "web_search" }],
			include: ["web_search_call.action.sources"],
		}),
		signal: withHardTimeout(params.signal, params.timeoutMs),
	});

	if (!response.ok) {
		const errorText = await readLimitedText(response, options.providerId, MAX_ERROR_BYTES, true);
		const classified = classifyProviderHttpError(options.providerId, response.status, errorText);
		if (classified) throw classified;
		throw new SearchProviderError(
			options.providerId,
			`${options.providerId} API error (${response.status}): ${errorText}`,
			response.status,
		);
	}

	const raw = await readLimitedText(response, options.providerId, MAX_RESPONSE_BYTES);
	let data: ResponsesApiResponse;
	try {
		data = JSON.parse(raw) as ResponsesApiResponse;
	} catch {
		throw new SearchProviderError(options.providerId, `${options.providerId} API returned invalid JSON`, 500);
	}

	const { answer, sources, searchQueries } = parseOutput(data);
	const numResults = params.numSearchResults ?? params.limit;
	return {
		provider: options.providerId,
		...(answer.length > 0 ? { answer } : {}),
		sources: numResults ? sources.slice(0, numResults) : sources,
		...(searchQueries.length > 0 ? { searchQueries } : {}),
		...(data.usage
			? {
					usage: {
						...(data.usage.input_tokens !== undefined ? { inputTokens: data.usage.input_tokens } : {}),
						...(data.usage.output_tokens !== undefined ? { outputTokens: data.usage.output_tokens } : {}),
						...(data.usage.total_tokens !== undefined ? { totalTokens: data.usage.total_tokens } : {}),
					},
				}
			: {}),
		...(data.model !== undefined ? { model: data.model } : { model }),
		...(data.id !== undefined ? { requestId: data.id } : {}),
		authMode: credential.source === "injected" ? "agent-model" : "api_key",
	};
}

/**
 * Gemini model-native web search provider.
 *
 * Calls the Gemini developer API's `generateContent` with Google Search
 * grounding enabled, using the model credential AutoRAG already resolves
 * (injected agent credential or GEMINI_API_KEY/GOOGLE_API_KEY) — no
 * search-specific key or signup.
 *
 * Ported from oh-my-pi (can1357/oh-my-pi, MIT)
 * `web/search/providers/gemini.ts`. Dropped: oh-my-pi's OAuth Cloud Code
 * Assist path, antigravity endpoints, and retry budget — AutoRAG uses the
 * plain developer API key shape via `../model-auth.ts`.
 */
import { envCredential } from "../credentials.ts";
import { resolveModelNativeCredential } from "../model-auth.ts";
import { formatQuery, GOOGLE_QUERY_SYNTAX, parseSearchQuery } from "../query.ts";
import { type SearchCitation, SearchProviderError, type SearchResponse, type SearchSource } from "../types.ts";
import type { SearchParams } from "./base.ts";
import { SearchProvider } from "./base.ts";
import { classifyProviderHttpError, readLimitedText, withHardTimeout } from "./utils.ts";

const DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com";
const API_VERSION = "v1beta";
const DEFAULT_MODEL = "gemini-2.5-flash";
const MAX_RESPONSE_BYTES = 4 * 1024 * 1024;
const MAX_ERROR_BYTES = 8 * 1024;

interface GeminiGroundingChunk {
	web?: { uri?: string; title?: string };
}

interface GeminiGroundingSupport {
	segment?: { text?: string };
	groundingChunkIndices?: number[];
}

interface GeminiCandidate {
	content?: { parts?: Array<{ text?: string }> };
	groundingMetadata?: {
		groundingChunks?: GeminiGroundingChunk[];
		groundingSupports?: GeminiGroundingSupport[];
		webSearchQueries?: string[];
	};
}

interface GeminiApiResponse {
	candidates?: GeminiCandidate[];
	usageMetadata?: { promptTokenCount?: number; candidatesTokenCount?: number; totalTokenCount?: number };
}

function parseResponse(data: GeminiApiResponse): Omit<SearchResponse, "provider"> {
	const candidate = data.candidates?.[0];
	const answerParts: string[] = [];
	for (const part of candidate?.content?.parts ?? []) {
		if (part.text) answerParts.push(part.text);
	}

	const sources: SearchSource[] = [];
	const seenUrls = new Set<string>();
	const citations: SearchCitation[] = [];
	const searchQueries: string[] = [];
	const metadata = candidate?.groundingMetadata;
	for (const chunk of metadata?.groundingChunks ?? []) {
		const uri = chunk.web?.uri;
		if (!uri || seenUrls.has(uri)) continue;
		seenUrls.add(uri);
		sources.push({ title: chunk.web?.title ?? uri, url: uri });
	}
	for (const support of metadata?.groundingSupports ?? []) {
		for (const index of support.groundingChunkIndices ?? []) {
			const chunk = metadata?.groundingChunks?.[index];
			if (chunk?.web?.uri) {
				citations.push({
					url: chunk.web.uri,
					title: chunk.web.title ?? chunk.web.uri,
					...(support.segment?.text !== undefined ? { citedText: support.segment.text } : {}),
				});
			}
		}
	}
	for (const query of metadata?.webSearchQueries ?? []) {
		if (!searchQueries.includes(query)) searchQueries.push(query);
	}

	return {
		...(answerParts.length > 0 ? { answer: answerParts.join("") } : {}),
		sources,
		...(citations.length > 0 ? { citations } : {}),
		...(searchQueries.length > 0 ? { searchQueries } : {}),
		...(data.usageMetadata
			? {
					usage: {
						...(data.usageMetadata.promptTokenCount !== undefined
							? { inputTokens: data.usageMetadata.promptTokenCount }
							: {}),
						...(data.usageMetadata.candidatesTokenCount !== undefined
							? { outputTokens: data.usageMetadata.candidatesTokenCount }
							: {}),
						...(data.usageMetadata.totalTokenCount !== undefined
							? { totalTokens: data.usageMetadata.totalTokenCount }
							: {}),
					},
				}
			: {}),
	};
}

export class GeminiProvider extends SearchProvider {
	readonly id = "gemini" as const;
	readonly label = "Gemini";

	isAvailable(): boolean {
		return resolveModelNativeCredential("gemini") !== undefined;
	}

	async search(params: SearchParams): Promise<SearchResponse> {
		const credential = resolveModelNativeCredential("gemini");
		if (!credential) {
			throw new SearchProviderError(
				"gemini",
				"Gemini credentials not found. Configure a Google model for the agent or set GEMINI_API_KEY/GOOGLE_API_KEY.",
				401,
			);
		}
		const model =
			envCredential("GEMINI_SEARCH_MODEL") ??
			(credential.source === "injected" ? credential.modelId : undefined) ??
			DEFAULT_MODEL;
		const parsed = params.parsedQuery ?? parseSearchQuery(params.query);
		const query = parsed.hasDirectives ? formatQuery(parsed, GOOGLE_QUERY_SYNTAX) : params.query;
		const baseUrl = (credential.baseUrl ?? DEFAULT_BASE_URL).replace(/\/+$/, "");

		const fetchImpl = params.fetch ?? fetch;
		const response = await fetchImpl(`${baseUrl}/${API_VERSION}/models/${model}:generateContent`, {
			method: "POST",
			headers: {
				"content-type": "application/json",
				"x-goog-api-key": credential.apiKey,
			},
			body: JSON.stringify({
				contents: [{ role: "user", parts: [{ text: query }] }],
				tools: [{ google_search: {} }],
				...(params.systemPrompt ? { systemInstruction: { parts: [{ text: params.systemPrompt }] } } : {}),
			}),
			signal: withHardTimeout(params.signal, params.timeoutMs),
		});

		if (!response.ok) {
			const errorText = await readLimitedText(response, "gemini", MAX_ERROR_BYTES, true);
			const classified = classifyProviderHttpError("gemini", response.status, errorText);
			if (classified) throw classified;
			throw new SearchProviderError(
				"gemini",
				`Gemini API error (${response.status}): ${errorText}`,
				response.status,
			);
		}

		const raw = await readLimitedText(response, "gemini", MAX_RESPONSE_BYTES);
		let data: GeminiApiResponse;
		try {
			data = JSON.parse(raw) as GeminiApiResponse;
		} catch {
			throw new SearchProviderError("gemini", "Gemini API returned invalid JSON", 500);
		}

		const parsedResponse = parseResponse(data);
		const numResults = params.numSearchResults ?? params.limit;
		const sources = numResults ? parsedResponse.sources.slice(0, numResults) : parsedResponse.sources;
		return {
			...parsedResponse,
			provider: "gemini",
			sources,
			model,
			authMode: credential.source === "injected" ? "agent-model" : "api_key",
		};
	}
}

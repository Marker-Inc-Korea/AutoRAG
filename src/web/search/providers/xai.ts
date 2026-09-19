/**
 * xAI model-native web search provider.
 *
 * Calls xAI's Responses-compatible API with the hosted `web_search` tool
 * using the model credential AutoRAG already resolves (injected agent
 * credential or XAI_API_KEY) — no search-specific key or signup. Transport
 * lives in `responses-api.ts`, shared with OpenAI.
 *
 * Ported from oh-my-pi (can1357/oh-my-pi, MIT) `web/search/providers/xai.ts`.
 * Dropped: oh-my-pi's OAuth path and collections/agent-tools support —
 * AutoRAG ships the plain API-key web_search path only.
 */
import { resolveModelNativeCredential } from "../model-auth.ts";
import type { SearchResponse } from "../types.ts";
import type { SearchAvailabilityContext, SearchParams } from "./base.ts";
import { SearchProvider } from "./base.ts";
import { searchResponsesApi } from "./responses-api.ts";

export class XaiProvider extends SearchProvider {
	readonly id = "xai" as const;
	readonly label = "xAI";

	isAvailable(context?: SearchAvailabilityContext): boolean {
		return resolveModelNativeCredential("xai", context?.modelAuth) !== undefined;
	}

	search(params: SearchParams): Promise<SearchResponse> {
		return searchResponsesApi(
			{
				providerId: "xai",
				defaultBaseUrl: "https://api.x.ai/v1",
				defaultModel: "grok-4.5",
				modelEnvVar: "XAI_SEARCH_MODEL",
				missingCredentialMessage:
					"xAI credentials not found. Configure an xAI model for the agent or set XAI_API_KEY.",
			},
			params,
		);
	}
}

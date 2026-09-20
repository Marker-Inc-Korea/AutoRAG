/**
 * OpenAI model-native web search provider.
 *
 * Calls the Responses API's hosted `web_search` tool using the model
 * credential AutoRAG already resolves (injected agent credential or
 * OPENAI_API_KEY) — no search-specific key or signup. Transport lives in
 * `responses-api.ts`, shared with xAI.
 *
 * Ported from oh-my-pi (can1357/oh-my-pi, MIT) `web/search/providers/codex.ts`.
 * Dropped: oh-my-pi's ChatGPT-OAuth backend and model-candidate retry chain —
 * AutoRAG ships the plain API-key Responses path only.
 */
import { resolveModelNativeCredential } from "../model-auth.ts";
import type { SearchResponse } from "../types.ts";
import type { SearchAvailabilityContext, SearchParams } from "./base.ts";
import { SearchProvider } from "./base.ts";
import { searchResponsesApi } from "./responses-api.ts";

export class CodexProvider extends SearchProvider {
	readonly id = "codex" as const;
	readonly label = "OpenAI";

	isAvailable(context?: SearchAvailabilityContext): boolean {
		return resolveModelNativeCredential("codex", context?.modelAuth) !== undefined;
	}

	search(params: SearchParams): Promise<SearchResponse> {
		return searchResponsesApi(
			{
				providerId: "codex",
				defaultBaseUrl: "https://api.openai.com/v1",
				defaultModel: "gpt-5",
				modelEnvVar: "CODEX_SEARCH_MODEL",
				missingCredentialMessage:
					"OpenAI credentials not found. Configure an OpenAI model for the agent or set OPENAI_API_KEY.",
			},
			params,
		);
	}
}

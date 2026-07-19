import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { Model } from "@earendil-works/pi-ai";
import { parse } from "smol-toml";

const ORCHESTRATOR_MODEL_ID = "gpt-5.6-sol";
const EXPLORER_MODEL_ID = "gpt-5.6-luna";

interface ProviderConfig {
	readonly base_url?: unknown;
	readonly wire_api?: unknown;
	readonly env_key?: unknown;
}

export interface LocalAutoRAGModels {
	readonly provider: string;
	readonly apiKey: string;
	readonly orchestrator: Model<"openai-responses">;
	readonly explorer: Model<"openai-responses">;
}

export interface LoadLocalAutoRAGModelsOptions {
	readonly configPath?: string;
	readonly env?: Readonly<Record<string, string | undefined>>;
	readonly orchestratorModelId?: string;
	readonly explorerModelId?: string;
}

function createModel(
	provider: string,
	baseUrl: string,
	id: string,
	defaultId: string,
	defaultName: string,
): Model<"openai-responses"> {
	return {
		id,
		name: id === defaultId ? defaultName : id,
		api: "openai-responses",
		provider,
		baseUrl,
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 400_000,
		maxTokens: 128_000,
	};
}

export function loadLocalAutoRAGModels(options: LoadLocalAutoRAGModelsOptions = {}): LocalAutoRAGModels {
	const configPath = options.configPath ?? join(homedir(), ".codex", "config.toml");
	const config = parse(readFileSync(configPath, "utf8")) as Record<string, unknown>;
	const provider = config.model_provider;
	if (typeof provider !== "string" || provider.length === 0) {
		throw new Error(`AutoRAG requires model_provider in ${configPath}`);
	}
	const providers = config.model_providers;
	const providerConfig =
		providers && typeof providers === "object"
			? ((providers as Record<string, unknown>)[provider] as ProviderConfig | undefined)
			: undefined;
	if (!providerConfig || typeof providerConfig.base_url !== "string") {
		throw new Error(`AutoRAG requires model_providers.${provider}.base_url in ${configPath}`);
	}
	if (providerConfig.wire_api !== "responses") {
		throw new Error(
			`AutoRAG requires a Responses-compatible provider; ${provider} uses ${String(providerConfig.wire_api)}`,
		);
	}
	if (typeof providerConfig.env_key !== "string" || providerConfig.env_key.length === 0) {
		throw new Error(`AutoRAG requires model_providers.${provider}.env_key in ${configPath}`);
	}
	const apiKey = (options.env ?? process.env)[providerConfig.env_key];
	if (!apiKey) throw new Error(`AutoRAG model credential is missing from ${providerConfig.env_key}`);

	return {
		provider,
		apiKey,
		orchestrator: createModel(
			provider,
			providerConfig.base_url,
			options.orchestratorModelId ?? ORCHESTRATOR_MODEL_ID,
			ORCHESTRATOR_MODEL_ID,
			"GPT-5.6 Sol",
		),
		explorer: createModel(
			provider,
			providerConfig.base_url,
			options.explorerModelId ?? EXPLORER_MODEL_ID,
			EXPLORER_MODEL_ID,
			"GPT-5.6 Luna",
		),
	};
}

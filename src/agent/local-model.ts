import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { Model } from "@earendil-works/pi-ai";
import { parse } from "smol-toml";

const DEFAULT_MODEL_ID = "gpt-5.6-sol";

interface ProviderConfig {
	readonly base_url?: unknown;
	readonly wire_api?: unknown;
	readonly env_key?: unknown;
}

export interface LocalAutoRAGModel {
	readonly provider: string;
	readonly apiKey: string;
	readonly model: Model<"openai-responses">;
}

export interface LoadLocalAutoRAGModelOptions {
	readonly configPath?: string;
	readonly env?: Readonly<Record<string, string | undefined>>;
	readonly modelId?: string;
}

export function loadLocalAutoRAGModel(options: LoadLocalAutoRAGModelOptions = {}): LocalAutoRAGModel {
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

	const id = options.modelId ?? DEFAULT_MODEL_ID;
	return {
		provider,
		apiKey,
		model: {
			id,
			name: id === DEFAULT_MODEL_ID ? "GPT-5.6 Sol" : id,
			api: "openai-responses",
			provider,
			baseUrl: providerConfig.base_url,
			reasoning: true,
			input: ["text", "image"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 400_000,
			maxTokens: 128_000,
		},
	};
}

import { join } from "node:path";
import { type Api, InMemoryCredentialStore, type Model } from "@earendil-works/pi-ai";
import { getEnvApiKey } from "@earendil-works/pi-ai/compat";
import { getAgentDir, ModelRuntime } from "@earendil-works/pi-coding-agent";
import { registerDazziProvider } from "../cloud/provider.ts";
import type { LanguageTag } from "../language.ts";
import type { VisualMirrorSyncOptions } from "../mirror/visual-mirror.ts";
import { tesseractCachePathFor } from "../parser/defaults.ts";
import { createDefaultPageRenderer, type PageRenderer } from "../parser/page-render.ts";
import {
	createOcrVisualPageExtractor,
	createVlmVisualPageExtractor,
	type VisualExtractorKind,
	type VisualPageExtractor,
} from "../parser/visual-extractor.ts";
import { createVisualPageGate, type VisualPageGate } from "../parser/visual-gate.ts";
import type { JevToolOptions } from "./jev-extension.ts";

export interface VisualParseGateAgentConfig {
	readonly backend?: string;
	readonly model?: string;
	readonly threshold?: number;
}

export interface VisualParseExtractorModelRef {
	readonly provider: string;
	readonly id: string;
	readonly name?: string;
	readonly api?: Api;
	readonly baseUrl?: string;
	readonly apiKeyEnv?: string;
	readonly reasoning?: boolean;
	readonly input?: Array<"text" | "image">;
	readonly contextWindow?: number;
	readonly maxTokens?: number;
}

export interface VisualParseResolvedModel {
	readonly model: Model<Api>;
	readonly apiKey?: string;
}

export interface VisualParseExtractorAgentConfig {
	readonly kind: VisualExtractorKind;
	readonly model?: VisualParseExtractorModelRef | VisualParseResolvedModel;
	readonly languages?: readonly LanguageTag[];
	readonly timeoutMs?: number;
	readonly cachePath?: string;
}

export interface VisualParseAgentOptions {
	/** Opt-in switch. Absent means enabled when a visualParse block is supplied; `false` disables it. */
	readonly enabled?: boolean;
	readonly gate?: VisualParseGateAgentConfig | VisualPageGate;
	readonly extractor?: VisualParseExtractorAgentConfig | VisualPageExtractor;
	readonly renderer?: PageRenderer;
	readonly maxPagesPerFile?: number;
	readonly maxWidthPx?: number;
}

function isVisualPageGate(value: VisualParseGateAgentConfig | VisualPageGate | undefined): value is VisualPageGate {
	return value !== undefined && typeof (value as VisualPageGate).judge === "function";
}

function isVisualPageExtractor(
	value: VisualParseExtractorAgentConfig | VisualPageExtractor | undefined,
): value is VisualPageExtractor {
	return value !== undefined && typeof (value as VisualPageExtractor).extract === "function";
}

function isResolvedModel(
	value: VisualParseExtractorModelRef | VisualParseResolvedModel | undefined,
): value is VisualParseResolvedModel {
	return value !== undefined && "model" in value && typeof value.model === "object" && value.model !== null;
}

function envKeyNameForProvider(provider: string): string {
	return `${provider.replace(/[^A-Za-z0-9_]/gu, "_").toUpperCase()}_API_KEY`;
}

async function resolveVlmModelRef(
	ref: VisualParseExtractorModelRef,
	agentDir: string,
	env: NodeJS.ProcessEnv = process.env,
): Promise<VisualParseResolvedModel> {
	const environmentOnly = env.AUTORAG_AUTH_MODE === "environment";
	const runtime = await ModelRuntime.create({
		...(environmentOnly
			? { credentials: new InMemoryCredentialStore(), modelsPath: null, refreshOnCreate: false }
			: { authPath: join(agentDir, "auth.json"), modelsPath: join(agentDir, "models.json") }),
		allowModelNetwork: false,
	});
	await registerDazziProvider(runtime, env);

	const catalog = runtime.getModel(ref.provider, ref.id) as Model<Api> | undefined;
	let model: Model<Api>;
	if (catalog !== undefined) {
		model = {
			...catalog,
			...(ref.name !== undefined ? { name: ref.name } : {}),
			...(ref.api !== undefined ? { api: ref.api } : {}),
			...(ref.baseUrl !== undefined ? { baseUrl: ref.baseUrl } : {}),
			...(ref.input !== undefined ? { input: ref.input } : {}),
			...(ref.contextWindow !== undefined ? { contextWindow: ref.contextWindow } : {}),
			...(ref.maxTokens !== undefined ? { maxTokens: ref.maxTokens } : {}),
		};
	} else {
		if (ref.baseUrl === undefined) {
			throw new Error(
				`Unknown visualParse.extractor.model "${ref.provider}/${ref.id}": not in pi model catalog and no baseUrl set.`,
			);
		}
		model = {
			id: ref.id,
			name: ref.name ?? ref.id,
			api: ref.api ?? "openai-completions",
			provider: ref.provider,
			baseUrl: ref.baseUrl,
			reasoning: ref.reasoning ?? false,
			input: ref.input ?? ["text", "image"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: ref.contextWindow ?? 128_000,
			maxTokens: ref.maxTokens ?? 8_192,
		};
	}

	const envMap = Object.fromEntries(
		Object.entries(env).filter((entry): entry is [string, string] => entry[1] !== undefined),
	);
	let apiKey: string | undefined;
	if (ref.apiKeyEnv !== undefined && env[ref.apiKeyEnv]) {
		apiKey = env[ref.apiKeyEnv];
	} else if (ref.baseUrl !== undefined && env[envKeyNameForProvider(ref.provider)]) {
		apiKey = env[envKeyNameForProvider(ref.provider)];
	} else {
		apiKey = getEnvApiKey(ref.provider, envMap);
		if (apiKey === undefined) {
			try {
				const auth = await runtime.getAuth(ref.provider);
				if (auth !== undefined && "key" in auth && typeof auth.key === "string") apiKey = auth.key;
				else if (auth !== undefined && "access" in auth && typeof auth.access === "string") apiKey = auth.access;
			} catch {
				// Leave apiKey undefined; the provider call will report auth errors cleanly.
			}
		}
	}

	return { model, ...(apiKey !== undefined ? { apiKey } : {}) };
}

/**
 * Resolve `visualParse` agent options into concrete `VisualMirrorSyncOptions` (gate + extractor + renderer).
 * Returns `undefined` when visual parsing is disabled or unconfigured.
 */
export async function resolveVisualParseSyncOptions(input: {
	readonly visualParse: VisualParseAgentOptions | false | undefined;
	readonly workspaceRoot: string;
	readonly languages: readonly LanguageTag[];
	readonly inheritedJev?: JevToolOptions;
	readonly agentDir?: string;
	readonly env?: NodeJS.ProcessEnv;
}): Promise<VisualMirrorSyncOptions | undefined> {
	const { visualParse } = input;
	if (visualParse === undefined || visualParse === false || visualParse.enabled === false) {
		return undefined;
	}
	const agentDir = input.agentDir ?? getAgentDir();
	const inheritedBackend = typeof input.inheritedJev?.backend === "string" ? input.inheritedJev.backend : undefined;
	const inheritedJev =
		inheritedBackend !== undefined || input.inheritedJev?.model !== undefined
			? {
					...(inheritedBackend !== undefined ? { backend: inheritedBackend } : {}),
					...(input.inheritedJev?.model !== undefined ? { model: input.inheritedJev.model } : {}),
				}
			: undefined;

	const gate: VisualPageGate = isVisualPageGate(visualParse.gate)
		? visualParse.gate
		: createVisualPageGate({
				...(visualParse.gate?.backend !== undefined ? { backend: visualParse.gate.backend } : {}),
				...(visualParse.gate?.model !== undefined ? { model: visualParse.gate.model } : {}),
				...(visualParse.gate?.threshold !== undefined ? { threshold: visualParse.gate.threshold } : {}),
				...(inheritedJev !== undefined ? { inheritedJev } : {}),
				agentDir,
				...(input.env !== undefined ? { env: input.env } : {}),
			});

	let extractor: VisualPageExtractor;
	if (isVisualPageExtractor(visualParse.extractor)) {
		extractor = visualParse.extractor;
	} else if (visualParse.extractor === undefined || visualParse.extractor.kind === "ocr") {
		extractor = createOcrVisualPageExtractor({
			languages: visualParse.extractor?.languages ?? input.languages,
			timeoutMs: visualParse.extractor?.timeoutMs,
			cachePath: visualParse.extractor?.cachePath ?? tesseractCachePathFor(input.workspaceRoot),
		});
	} else {
		if (visualParse.extractor.model === undefined) {
			throw new Error("visualParse.extractor.model is required when visualParse.extractor.kind is 'vlm'.");
		}
		const resolved = isResolvedModel(visualParse.extractor.model)
			? visualParse.extractor.model
			: await resolveVlmModelRef(visualParse.extractor.model, agentDir, input.env);
		extractor = createVlmVisualPageExtractor({
			model: resolved.model,
			...(resolved.apiKey !== undefined ? { apiKey: resolved.apiKey } : {}),
			...(visualParse.extractor.timeoutMs !== undefined ? { timeoutMs: visualParse.extractor.timeoutMs } : {}),
		});
	}

	const renderer =
		visualParse.renderer ??
		createDefaultPageRenderer({
			maxPagesPerFile: visualParse.maxPagesPerFile,
			maxWidthPx: visualParse.maxWidthPx,
		});

	return {
		gate,
		extractor,
		renderer,
		...(visualParse.maxPagesPerFile !== undefined ? { maxPagesPerFile: visualParse.maxPagesPerFile } : {}),
		...(visualParse.maxWidthPx !== undefined ? { maxWidthPx: visualParse.maxWidthPx } : {}),
	};
}

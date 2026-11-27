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
	createRemoteApiOcrExtractor,
	findOfficialRemoteOcrModel,
	isOfficialRemoteOcrProvider,
	type RemoteOcrProviderId,
} from "../parser/remote-ocr.ts";
import {
	createOcrVisualPageExtractor,
	createVlmVisualPageExtractor,
	type LocalOcrEngine,
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
	readonly engine?: LocalOcrEngine;
	readonly languages?: readonly LanguageTag[];
	readonly timeoutMs?: number;
	readonly cachePath?: string;
}

export interface VisualParseOcrAgentConfig {
	/**
	 * `"auto"` (PP-OCRv5 for ko/en else Tesseract), `"kordoc"` (PP-OCRv5), `"tesseract"`,
	 * `"api"` (official provider OCR/document-parsing API: Mistral OCR, NVIDIA Nemotron Parse, Upstage Document Parse, Z.AI GLM-OCR),
	 * or `"vlm"` (generic remote vision-model OCR).
	 */
	readonly engine?: LocalOcrEngine | "api" | "vlm";
	/** Official provider OCR API provider id when `engine === "api"` (e.g. `"mistral"`, `"nvidia"`, `"upstage"`, `"zai"`). */
	readonly provider?: string;
	/** Remote OCR model id (`engine === "api"`) or vision model reference (`engine === "vlm"`). */
	readonly model?: string | VisualParseExtractorModelRef | VisualParseResolvedModel;
	/** Optional base URL override for official remote OCR APIs. */
	readonly baseUrl?: string;
	/** Optional API key env var override for official remote OCR APIs. */
	readonly apiKeyEnv?: string;
	readonly languages?: readonly LanguageTag[];
	readonly timeoutMs?: number;
	readonly cachePath?: string;
}

export interface VisualParseAgentOptions {
	/** Opt-in switch. Absent means enabled when a visualParse block is supplied; `false` disables it. */
	readonly enabled?: boolean;
	/** Calibrated probability cutoff for the unified decision model (`jev`). Default 0.6. */
	readonly threshold?: number;
	readonly gate?: VisualParseGateAgentConfig | VisualPageGate;
	readonly extractor?: VisualParseExtractorAgentConfig | VisualPageExtractor;
	/** Independent OCR configuration for `ocr`-routed pages (local PP-OCRv5/Tesseract, remote OCR model, or `false` to disable OCR). */
	readonly ocr?: VisualParseOcrAgentConfig | false;
	/** Optional dedicated OCR extractor seam when `extractor` is VLM (defaults to local auto PP-OCRv5/Tesseract OCR). */
	readonly ocrExtractor?: VisualPageExtractor;
	readonly renderer?: PageRenderer;
	readonly maxPagesPerFile?: number;
	readonly maxWidthPx?: number;
	readonly concurrency?: number;
	readonly gateConcurrency?: number;
	readonly ocrConcurrency?: number;
	readonly fileConcurrency?: number;
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

async function resolveProviderApiKey(input: {
	readonly provider: string;
	readonly apiKeyEnv?: string;
	readonly defaultApiKeyEnv?: string;
	readonly agentDir: string;
	readonly env?: NodeJS.ProcessEnv;
}): Promise<string | undefined> {
	const env = input.env ?? process.env;
	if (input.apiKeyEnv !== undefined && env[input.apiKeyEnv]) return env[input.apiKeyEnv];
	if (input.defaultApiKeyEnv !== undefined && env[input.defaultApiKeyEnv]) return env[input.defaultApiKeyEnv];
	const envMap = Object.fromEntries(
		Object.entries(env).filter((entry): entry is [string, string] => entry[1] !== undefined),
	);
	const fromEnv = getEnvApiKey(input.provider, envMap) ?? env[envKeyNameForProvider(input.provider)];
	if (fromEnv) return fromEnv;
	if (env.AUTORAG_AUTH_MODE === "environment") return undefined;
	try {
		const runtime = await ModelRuntime.create({
			authPath: join(input.agentDir, "auth.json"),
			modelsPath: join(input.agentDir, "models.json"),
			allowModelNetwork: false,
		});
		const auth = await runtime.getAuth(input.provider);
		if (auth !== undefined && "key" in auth && typeof auth.key === "string") return auth.key;
		if (auth !== undefined && "access" in auth && typeof auth.access === "string") return auth.access;
		if (auth !== undefined && "auth" in auth) {
			const nested = (auth as { auth?: { apiKey?: unknown; access?: unknown } }).auth;
			if (typeof nested?.apiKey === "string") return nested.apiKey;
			if (typeof nested?.access === "string") return nested.access;
		}
	} catch {
		// Fall through; caller reports missing API key clearly.
	}
	return undefined;
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

	const threshold =
		visualParse.threshold ?? (!isVisualPageGate(visualParse.gate) ? visualParse.gate?.threshold : undefined);
	const gate: VisualPageGate = isVisualPageGate(visualParse.gate)
		? visualParse.gate
		: createVisualPageGate({
				...(visualParse.gate?.backend !== undefined ? { backend: visualParse.gate.backend } : {}),
				...(visualParse.gate?.model !== undefined ? { model: visualParse.gate.model } : {}),
				...(threshold !== undefined ? { threshold } : {}),
				...(inheritedJev !== undefined ? { inheritedJev } : {}),
				agentDir,
				...(input.env !== undefined ? { env: input.env } : {}),
			});

	const resolveOcrConfig = async (
		ocrConfig: VisualParseOcrAgentConfig | false | undefined,
		fallbackExtractorConfig: VisualParseExtractorAgentConfig | undefined,
	): Promise<VisualPageExtractor | undefined> => {
		if (visualParse.ocrExtractor !== undefined) return visualParse.ocrExtractor;
		if (ocrConfig === false) return undefined;
		const inferredProvider =
			ocrConfig?.provider ??
			(typeof ocrConfig?.model === "object" && ocrConfig.model !== null && !isResolvedModel(ocrConfig.model)
				? ocrConfig.model.provider
				: undefined);
		const inferredModelId =
			typeof ocrConfig?.model === "string"
				? ocrConfig.model
				: typeof ocrConfig?.model === "object" && ocrConfig.model !== null && !isResolvedModel(ocrConfig.model)
					? ocrConfig.model.id
					: undefined;
		const matchesOfficialApi =
			inferredProvider !== undefined &&
			inferredModelId !== undefined &&
			findOfficialRemoteOcrModel(inferredProvider, inferredModelId) !== undefined;
		const engine =
			ocrConfig?.engine ??
			(ocrConfig?.provider !== undefined || matchesOfficialApi
				? "api"
				: ocrConfig?.model !== undefined
					? "vlm"
					: (fallbackExtractorConfig?.engine ?? "auto"));
		if (engine === "api") {
			const providerId = inferredProvider;
			const modelId = inferredModelId;
			if (providerId === undefined || !isOfficialRemoteOcrProvider(providerId)) {
				throw new Error(
					`Unsupported visualParse.ocr.provider "${providerId ?? ""}". Supported official OCR API providers: mistral, nvidia, upstage, zai.`,
				);
			}
			if (modelId === undefined || modelId.trim().length === 0) {
				throw new Error(`visualParse.ocr.model is required when visualParse.ocr.engine is "api".`);
			}
			const spec = findOfficialRemoteOcrModel(providerId, modelId);
			const modelRef =
				typeof ocrConfig?.model === "object" && ocrConfig.model !== null && !isResolvedModel(ocrConfig.model)
					? ocrConfig.model
					: undefined;
			const apiKey = await resolveProviderApiKey({
				provider: providerId,
				apiKeyEnv: ocrConfig?.apiKeyEnv ?? modelRef?.apiKeyEnv,
				defaultApiKeyEnv: spec?.defaultApiKeyEnv,
				agentDir,
				env: input.env,
			});
			if (apiKey === undefined || apiKey.trim().length === 0) {
				throw new Error(
					`Missing API key for remote OCR provider "${providerId}" (expected ${ocrConfig?.apiKeyEnv ?? spec?.defaultApiKeyEnv ?? envKeyNameForProvider(providerId)}).`,
				);
			}
			return createRemoteApiOcrExtractor({
				provider: providerId as RemoteOcrProviderId,
				model: modelId,
				apiKey,
				baseUrl: ocrConfig?.baseUrl ?? modelRef?.baseUrl ?? spec?.defaultBaseUrl,
				...(ocrConfig?.timeoutMs !== undefined ? { timeoutMs: ocrConfig.timeoutMs } : {}),
			});
		}
		if (engine === "vlm") {
			const modelSpec = ocrConfig?.model ?? fallbackExtractorConfig?.model;
			if (modelSpec === undefined || typeof modelSpec === "string") {
				throw new Error("visualParse.ocr.model is required when visualParse.ocr.engine is 'vlm'.");
			}
			const resolved = isResolvedModel(modelSpec)
				? modelSpec
				: await resolveVlmModelRef(modelSpec, agentDir, input.env);
			return createVlmVisualPageExtractor({
				model: resolved.model,
				mode: "ocr",
				...(resolved.apiKey !== undefined ? { apiKey: resolved.apiKey } : {}),
				...(ocrConfig?.timeoutMs !== undefined ? { timeoutMs: ocrConfig.timeoutMs } : {}),
			});
		}
		return createOcrVisualPageExtractor({
			engine,
			languages: ocrConfig?.languages ?? fallbackExtractorConfig?.languages ?? input.languages,
			timeoutMs: ocrConfig?.timeoutMs ?? fallbackExtractorConfig?.timeoutMs,
			cachePath:
				ocrConfig?.cachePath ?? fallbackExtractorConfig?.cachePath ?? tesseractCachePathFor(input.workspaceRoot),
		});
	};

	let vlmExtractor: VisualPageExtractor | undefined;
	let ocrExtractor: VisualPageExtractor | undefined;
	if (isVisualPageExtractor(visualParse.extractor)) {
		if (visualParse.extractor.kind === "vlm") {
			vlmExtractor = visualParse.extractor;
			ocrExtractor =
				visualParse.ocr !== undefined
					? await resolveOcrConfig(visualParse.ocr, undefined)
					: visualParse.ocrExtractor;
		} else {
			ocrExtractor = visualParse.extractor;
		}
	} else if (visualParse.extractor === undefined) {
		ocrExtractor = await resolveOcrConfig(visualParse.ocr, undefined);
	} else if (visualParse.extractor.kind === "ocr") {
		ocrExtractor = await resolveOcrConfig(visualParse.ocr, visualParse.extractor);
	} else {
		if (visualParse.extractor.model === undefined) {
			throw new Error("visualParse.extractor.model is required when visualParse.extractor.kind is 'vlm'.");
		}
		const resolved = isResolvedModel(visualParse.extractor.model)
			? visualParse.extractor.model
			: await resolveVlmModelRef(visualParse.extractor.model, agentDir, input.env);
		vlmExtractor = createVlmVisualPageExtractor({
			model: resolved.model,
			...(resolved.apiKey !== undefined ? { apiKey: resolved.apiKey } : {}),
			...(visualParse.extractor.timeoutMs !== undefined ? { timeoutMs: visualParse.extractor.timeoutMs } : {}),
		});
		// When VLM mode is configured, `ocr`-routed pages use the configured OCR engine/model (or local auto OCR unless `ocr: false`).
		ocrExtractor = await resolveOcrConfig(visualParse.ocr, undefined);
	}
	if (vlmExtractor === undefined && ocrExtractor === undefined) {
		return undefined;
	}

	const renderer =
		visualParse.renderer ??
		createDefaultPageRenderer({
			maxPagesPerFile: visualParse.maxPagesPerFile,
			maxWidthPx: visualParse.maxWidthPx,
		});

	return {
		gate,
		...(vlmExtractor !== undefined ? { vlmExtractor } : {}),
		...(ocrExtractor !== undefined ? { ocrExtractor } : {}),
		renderer,
		...(visualParse.maxPagesPerFile !== undefined ? { maxPagesPerFile: visualParse.maxPagesPerFile } : {}),
		...(visualParse.maxWidthPx !== undefined ? { maxWidthPx: visualParse.maxWidthPx } : {}),
		...(visualParse.concurrency !== undefined ? { concurrency: visualParse.concurrency } : {}),
		...(visualParse.gateConcurrency !== undefined ? { gateConcurrency: visualParse.gateConcurrency } : {}),
		...(visualParse.ocrConcurrency !== undefined ? { ocrConcurrency: visualParse.ocrConcurrency } : {}),
		...(visualParse.fileConcurrency !== undefined ? { fileConcurrency: visualParse.fileConcurrency } : {}),
	};
}

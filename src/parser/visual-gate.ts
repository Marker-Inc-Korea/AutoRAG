import { basename } from "node:path";
import type { ClassifierApi, ClassifierModel, JsonObject } from "@earendil-works/pi-ai";
import { getEnvApiKey } from "@earendil-works/pi-ai/compat";
import { getAgentDir } from "@earendil-works/pi-coding-agent";
import {
	BIVIA_DEFAULT_MODEL,
	BIVIA_PROVIDER_ID,
	type DecisionModelRuntime,
	decisionRuntimeFor,
	getBiviaEnvApiKey,
} from "../agent/decision-backends.ts";
import { isDazziSignedIn } from "../cloud/credential.ts";
import { DazziJevBackend, type DazziJevBackendOptions } from "../cloud/jev-backend.ts";
import type { VisualImageMimeType } from "./page-render.ts";

/** Default calibrated P(visually_rich) cutoff for escalating a page to VLM/OCR extraction. */
export const DEFAULT_VISUAL_GATE_THRESHOLD = 0.6;

/** Prompt revision baked into the gate fingerprint so prompt updates invalidate stale decisions. */
const VISUAL_GATE_PROMPT_VERSION = "v1";

/** Maximum characters of base-extracted page text sent alongside the page image to keep state compact. */
const MAX_STATE_TEXT_CHARS = 4_000;

/** One attempt timeout for a visual decision call. */
const VISUAL_GATE_TIMEOUT_MS = 20_000;
const VISUAL_GATE_MAX_RETRIES = 2;

/**
 * Vision-capable default decision model per provider. Every entry here accepts image input
 * (`input: ["text", "image"]`) and answers typed yes/no questions over a page image + text state.
 */
export const VISION_DECISION_PROVIDER_DEFAULT_MODELS: Readonly<Record<string, string>> = {
	[BIVIA_PROVIDER_ID]: BIVIA_DEFAULT_MODEL,
	openrouter: "typesafe/jev-1.14",
	openai: "gpt-6-luna",
	typesafe: "jev-latest",
	"cloudflare-workers-ai": "@cf/cloudflare/clef-omni",
	dazzi: "typesafe/jev-1.14",
};

/** Provider resolution order when no explicit visual gate backend is named. */
export const VISION_DECISION_PROVIDER_ORDER: readonly string[] = [
	BIVIA_PROVIDER_ID,
	"openrouter",
	"openai",
	"typesafe",
	"cloudflare-workers-ai",
];

/** True when a pi-ai classifier model declares image input support. */
export function isVisionDecisionModel(model: Pick<ClassifierModel<ClassifierApi>, "input">): boolean {
	return model.input.includes("image");
}

export interface VisualGateInput {
	readonly virtualPath: string;
	readonly page: number;
	readonly image: { readonly data: Uint8Array; readonly mimeType: VisualImageMimeType };
	/** Base parser's extracted text for this page, if available. */
	readonly extractedText?: string;
	readonly signal?: AbortSignal;
}

export interface VisualGateVerdict {
	readonly visuallyRich: boolean;
	/** Calibrated P(visually_rich) in [0, 1]. */
	readonly probability: number;
	readonly model: string;
	readonly backend: string;
}

/** Gate deciding whether a rendered document page needs VLM/OCR visual enrichment. */
export interface VisualPageGate {
	/** Stable identity string for incremental page caching. */
	readonly fingerprint: string;
	judge(input: VisualGateInput): Promise<VisualGateVerdict>;
}

export interface VisualGateOptions {
	/** Decision backend id: `bivia`, `openrouter`, `openai`, `typesafe`, `cloudflare-workers-ai`, or `dazzi`. */
	readonly backend?: string;
	/** Decision model id within `backend`. Defaults to {@link VISION_DECISION_PROVIDER_DEFAULT_MODELS}. */
	readonly model?: string;
	/** Escalate pages whose P(visually_rich) >= threshold. Default {@link DEFAULT_VISUAL_GATE_THRESHOLD}. */
	readonly threshold?: number;
	/** Optional inherited `jev` config used when `backend` is omitted and the inherited backend supports vision. */
	readonly inheritedJev?: { readonly backend?: string; readonly model?: string };
	/** Programmatic runtime seam (tests / shared runtime). */
	readonly runtime?: DecisionModelRuntime;
	/** Dazzi backend seam (tests / custom baseUrl). */
	readonly dazzi?: DazziJevBackendOptions;
	/** Pi agent home (`auth.json`, `models.json`). */
	readonly agentDir?: string;
	/** Environment variables checked ahead of `process.env`. */
	readonly env?: Record<string, string | undefined>;
}

const VISUAL_RICH_QUESTION_ID = "visually_rich";
const VISUAL_RICH_INSTRUCTIONS =
	"Does this document page contain visually rich information — such as a chart, graph, plot, diagram, architecture figure, flowchart, map, photo or screenshot with data, complex multi-header/merged table, visual color highlight or red/marked annotation, or scanned/image-only text — whose important facts or structure are missing or distorted in the extracted text?";
const VISUAL_RICH_CRITERIA = {
	true: "The page has a chart, diagram, visual figure, complex layout table, meaningful colored highlight/annotation, or unreadable/scanned image text that needs VLM or OCR extraction to capture accurately.",
	false: "The page is standard prose, headings, lists, or simple text/tables already accurately captured in the extracted text (or purely decorative logos, headers, or footers).",
} as const;

function hasProviderKey(provider: string, env: Record<string, string>): boolean {
	if (provider === BIVIA_PROVIDER_ID) return getBiviaEnvApiKey(env) !== undefined;
	return getEnvApiKey(provider, env) !== undefined;
}

/**
 * Resolve which vision-capable decision backend and model will gate pages.
 * Prefers explicit `backend`/`model`, then inherits `jev` if its backend supports vision,
 * then picks signed-in Dazzi or the first vision decision provider with credentials.
 */
export function resolveVisualGateBackend(options: {
	readonly backend?: string;
	readonly model?: string;
	readonly inheritedJev?: { readonly backend?: string; readonly model?: string };
	readonly agentDir?: string;
	readonly env?: Record<string, string | undefined>;
}): { readonly backend: string; readonly model: string } {
	const env = options.env ?? process.env;
	const providerEnv = Object.fromEntries(
		Object.entries(env).filter((entry): entry is [string, string] => entry[1] !== undefined),
	);

	if (options.backend !== undefined) {
		const model = options.model ?? VISION_DECISION_PROVIDER_DEFAULT_MODELS[options.backend];
		if (model === undefined) {
			throw new Error(
				`Visual gate backend "${options.backend}" has no default vision decision model; specify visualParse.gate.model explicitly.`,
			);
		}
		return { backend: options.backend, model };
	}

	const inheritedBackend = options.inheritedJev?.backend;
	if (inheritedBackend !== undefined && Object.hasOwn(VISION_DECISION_PROVIDER_DEFAULT_MODELS, inheritedBackend)) {
		return {
			backend: inheritedBackend,
			model:
				options.model ??
				options.inheritedJev?.model ??
				(VISION_DECISION_PROVIDER_DEFAULT_MODELS[inheritedBackend] as string),
		};
	}

	if (isDazziSignedIn(options.agentDir)) {
		return {
			backend: "dazzi",
			model: options.model ?? (VISION_DECISION_PROVIDER_DEFAULT_MODELS.dazzi as string),
		};
	}

	const detected = VISION_DECISION_PROVIDER_ORDER.find((provider) => hasProviderKey(provider, providerEnv));
	if (detected !== undefined) {
		return {
			backend: detected,
			model: options.model ?? (VISION_DECISION_PROVIDER_DEFAULT_MODELS[detected] as string),
		};
	}

	throw new Error(
		"No vision decision model credentials found for visualParse.gate: set BIVIA_API_KEY, OPENROUTER_API_KEY, OPENAI_API_KEY, TYPESAFE_API_KEY, or CLOUDFLARE_API_KEY, or sign in to Dazzi Models.",
	);
}

function buildStateText(input: VisualGateInput): string {
	const text = input.extractedText?.trim() ?? "";
	const snippet = text.length > MAX_STATE_TEXT_CHARS ? `${text.slice(0, MAX_STATE_TEXT_CHARS)}\n…(truncated)` : text;
	return [
		`Document: ${basename(input.virtualPath)}`,
		`Page: ${input.page}`,
		"Base parser extracted text on this page:",
		snippet.length > 0 ? snippet : "(no text extracted)",
	].join("\n");
}

function toBase64(bytes: Uint8Array): string {
	return Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString("base64");
}

/**
 * Create a visual decision gate backed by a vision-capable decision model (Bivia, Jev, GPT-6 Luna Decisions,
 * Clef Omni, or hosted Dazzi `/v1/jev`).
 */
export function createVisualPageGate(options: VisualGateOptions = {}): VisualPageGate {
	const threshold = options.threshold ?? DEFAULT_VISUAL_GATE_THRESHOLD;
	if (!Number.isFinite(threshold) || threshold <= 0 || threshold > 1) {
		throw new RangeError(`visualParse.gate.threshold must be in (0, 1], received ${String(threshold)}`);
	}
	const resolved = resolveVisualGateBackend({
		backend: options.backend,
		model: options.model,
		inheritedJev: options.inheritedJev,
		agentDir: options.agentDir,
		env: options.env,
	});
	const fingerprint = `visual-gate:${VISUAL_GATE_PROMPT_VERSION}:${resolved.backend}:${resolved.model}:t${threshold}`;
	const cleanEnv =
		options.env === undefined
			? undefined
			: Object.fromEntries(
					Object.entries(options.env).filter((entry): entry is [string, string] => entry[1] !== undefined),
				);

	let dazziBackend: DazziJevBackend | undefined;

	return {
		fingerprint,
		async judge(input: VisualGateInput): Promise<VisualGateVerdict> {
			const stateText = buildStateText(input);
			const base64 = toBase64(input.image.data);

			if (resolved.backend === "dazzi") {
				dazziBackend ??= new DazziJevBackend({
					...(options.agentDir !== undefined ? { agentDir: options.agentDir } : {}),
					...(options.env !== undefined ? { env: options.env } : {}),
					...options.dazzi,
					defaultModel: resolved.model,
				});
				const dataUrl = `data:${input.image.mimeType};base64,${base64}`;
				const multimodalState = [{ type: "image_url", image_url: { url: dataUrl } }, stateText];
				const response = await dazziBackend.judge({
					model: resolved.model,
					state: multimodalState as unknown as string,
					questions: [
						{
							id: VISUAL_RICH_QUESTION_ID,
							type: "noul",
							question: VISUAL_RICH_INSTRUCTIONS,
							criteria: VISUAL_RICH_CRITERIA,
						},
					],
				});
				const answer = response.answers[0];
				const probability = typeof answer?.answer === "number" ? answer.answer : undefined;
				if (probability === undefined || !Number.isFinite(probability)) {
					throw new Error(`Dazzi visual gate returned no numeric probability for ${resolved.model}`);
				}
				return {
					visuallyRich: probability >= threshold,
					probability,
					model: response.model ?? resolved.model,
					backend: resolved.backend,
				};
			}

			const runtime = options.runtime ?? (await decisionRuntimeFor(options.agentDir ?? getAgentDir()));
			const catalog = runtime.getModelsOfType("classifier", resolved.backend);
			const template = catalog[0];
			if (template === undefined) {
				throw new Error(`Decision backend "${resolved.backend}" has no classifier models registered`);
			}
			const model = catalog.find((candidate) => candidate.id === resolved.model) ?? {
				...template,
				id: resolved.model,
				name: resolved.model,
				input: ["text", "image"] as Array<"text" | "image">,
			};
			if (!isVisionDecisionModel(model)) {
				throw new Error(
					`Decision model "${resolved.backend}/${model.id}" does not support image input; choose a vision-capable decision model (e.g. bivia/dec-1, openrouter/typesafe/jev-1.14, openai/gpt-6-luna).`,
				);
			}

			const result = await runtime.classify(
				model,
				{
					state: stateText as unknown as JsonObject,
					images: [{ type: "image", data: base64, mimeType: input.image.mimeType }],
					questions: {
						[VISUAL_RICH_QUESTION_ID]: {
							type: "bool",
							instructions: VISUAL_RICH_INSTRUCTIONS,
							criteria: VISUAL_RICH_CRITERIA,
						},
					},
				},
				{
					timeoutMs: VISUAL_GATE_TIMEOUT_MS,
					maxRetries: VISUAL_GATE_MAX_RETRIES,
					...(input.signal !== undefined ? { signal: input.signal } : {}),
					...(cleanEnv !== undefined ? { env: cleanEnv } : {}),
				},
			);
			if (result.stopReason !== "stop") {
				throw new Error(
					`Visual gate ${resolved.backend}/${model.id} failed: ${result.errorMessage ?? result.stopReason}`,
				);
			}
			const answer = result.answers[VISUAL_RICH_QUESTION_ID];
			if (answer === undefined || answer.type !== "bool" || !Number.isFinite(answer.probability)) {
				throw new Error(`Visual gate ${resolved.backend}/${model.id} returned no usable boolean verdict`);
			}
			return {
				visuallyRich: answer.probability >= threshold,
				probability: answer.probability,
				model: result.model,
				backend: resolved.backend,
			};
		},
	};
}

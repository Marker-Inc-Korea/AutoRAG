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

/** Default calibrated probability cutoff for escalating a page to `vlm` or `ocr` extraction. */
export const DEFAULT_VISUAL_GATE_THRESHOLD = 0.6;

/** Prompt revision baked into the gate fingerprint so prompt updates invalidate stale decisions. */
const VISUAL_GATE_PROMPT_VERSION = "v2-3way";

/** Route chosen by the vision decision gate for one page. */
export type VisualPageRoute = "vlm" | "ocr" | "plain";

/** Maximum characters of base-extracted page text sent alongside the page image to keep state compact. */
const MAX_STATE_TEXT_CHARS = 4_000;

/** One attempt timeout for a visual decision call. */
const VISUAL_GATE_TIMEOUT_MS = 20_000;
const VISUAL_GATE_MAX_RETRIES = 2;

/**
 * Vision-capable default decision model per provider. Every entry here accepts image input
 * (`input: ["text", "image"]`) and answers typed yes/no questions over a page image + text state.
 * Defaults match AutoRAG's unified decision model (`jev`), keeping Jev (`jev-1.13` / `typesafe/jev-1.13`)
 * as the single default across both search routing and visual-rich page gating.
 */
export const VISION_DECISION_PROVIDER_DEFAULT_MODELS: Readonly<Record<string, string>> = {
	openrouter: "typesafe/jev-1.13",
	openai: "gpt-6-luna",
	typesafe: "jev-latest",
	"cloudflare-workers-ai": "@cf/cloudflare/clef-omni",
	[BIVIA_PROVIDER_ID]: BIVIA_DEFAULT_MODEL,
	dazzi: "jev-1.13",
};

/** Provider resolution order when no explicit visual gate backend is named (matches DECISION_PROVIDER_ORDER, Jev first). */
export const VISION_DECISION_PROVIDER_ORDER: readonly string[] = [
	"openrouter",
	"openai",
	"typesafe",
	"cloudflare-workers-ai",
	BIVIA_PROVIDER_ID,
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
	/** 3-way route: `vlm` (genuinely visually rich), `ocr` (scanned/handwritten/missing text), or `plain` (base parser is enough). */
	readonly route: VisualPageRoute;
	/** True when `route` is `vlm` or `ocr` (i.e. this page needs a visual/OCR enrichment entry). */
	readonly visuallyRich: boolean;
	/** Probability of the chosen route (or P(escalate) when routed to `vlm`/`ocr`), in [0, 1]. */
	readonly probability: number;
	/** Per-route probability distribution when returned by the decision model. */
	readonly probabilities?: Readonly<Record<VisualPageRoute, number>>;
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

const VISUAL_ROUTE_QUESTION_ID = "parse_route";
const VISUAL_ROUTE_INSTRUCTIONS =
	"Choose the single best parsing route for this document page given both the page image and the base parser's extracted text.";
const VISUAL_ROUTE_OPTIONS: Readonly<Record<VisualPageRoute, string>> = {
	vlm: "Page contains genuinely visually rich data that OCR or plain text cannot accurately capture: charts, graphs, plots, data diagrams, flowcharts, org charts, infographics, maps, complex multi-header/merged tables or calendar grids, or meaningful color-coded/red-marked annotations. Do NOT choose vlm for decorative stock photos or illustrations without data.",
	ocr: "Page contains primarily text (scanned document, fax, handwriting, screenshot/image of text, or form) where readable text is visibly present on the page image but is missing, empty, or severely garbled in the base parser's extracted text.",
	plain: "Base parser's extracted text already captures the page's substantive content accurately (ordinary prose, headings, simple lists/tables, cover pages, or pages with only decorative photos/logos/headers).",
};

/** Resolve a 3-way choice answer into a calibrated {@link VisualGateVerdict}. */
function verdictFromChoice(
	choice: string,
	distribution: Readonly<Record<string, number>> | undefined,
	threshold: number,
	model: string,
	backend: string,
): VisualGateVerdict {
	const pVlm =
		typeof distribution?.vlm === "number" && Number.isFinite(distribution.vlm)
			? distribution.vlm
			: choice === "vlm"
				? 1
				: 0;
	const pOcr =
		typeof distribution?.ocr === "number" && Number.isFinite(distribution.ocr)
			? distribution.ocr
			: choice === "ocr"
				? 1
				: 0;
	const pPlain =
		typeof distribution?.plain === "number" && Number.isFinite(distribution.plain)
			? distribution.plain
			: Math.max(0, 1 - pVlm - pOcr);

	let route: VisualPageRoute = "plain";
	let probability = pPlain;
	if (choice === "vlm" && pVlm >= threshold) {
		route = "vlm";
		probability = pVlm;
	} else if (choice === "ocr" && pOcr >= threshold) {
		route = "ocr";
		probability = pOcr;
	} else if ((choice === "vlm" || choice === "ocr") && pVlm + pOcr >= threshold) {
		// Combined visual/OCR signal exceeds threshold even if probability splits between vlm and ocr.
		route = pVlm >= pOcr ? "vlm" : "ocr";
		probability = route === "vlm" ? pVlm : pOcr;
	}

	return {
		route,
		visuallyRich: route !== "plain",
		probability,
		probabilities: { vlm: pVlm, ocr: pOcr, plain: pPlain },
		model,
		backend,
	};
}

function hasProviderKey(provider: string, env: Record<string, string>): boolean {
	if (provider === BIVIA_PROVIDER_ID) return getBiviaEnvApiKey(env) !== undefined;
	const resolved = getEnvApiKey(provider, env);
	return typeof resolved === "string" && resolved.length > 0;
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
							id: VISUAL_ROUTE_QUESTION_ID,
							type: "choice",
							question: VISUAL_ROUTE_INSTRUCTIONS,
							options: { ...VISUAL_ROUTE_OPTIONS },
						},
					],
				});
				const answer = response.answers[0];
				const choice = typeof answer?.answer === "string" ? answer.answer : undefined;
				if (choice === undefined) {
					throw new Error(`Dazzi visual gate returned no route choice for ${resolved.model}`);
				}
				return verdictFromChoice(
					choice,
					answer?.distribution,
					threshold,
					response.model ?? resolved.model,
					resolved.backend,
				);
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
					`Decision model "${resolved.backend}/${model.id}" does not support image input; choose a vision-capable decision model (e.g. openrouter/typesafe/jev-1.13, bivia/dec-1, openai/gpt-6-luna).`,
				);
			}

			const result = await runtime.classify(
				model,
				{
					state: stateText as unknown as JsonObject,
					images: [{ type: "image", data: base64, mimeType: input.image.mimeType }],
					questions: {
						[VISUAL_ROUTE_QUESTION_ID]: {
							type: "choice",
							instructions: VISUAL_ROUTE_INSTRUCTIONS,
							criteria: { ...VISUAL_ROUTE_OPTIONS },
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
			const answer = result.answers[VISUAL_ROUTE_QUESTION_ID];
			if (answer === undefined || answer.type !== "choice") {
				throw new Error(`Visual gate ${resolved.backend}/${model.id} returned no usable route choice`);
			}
			return verdictFromChoice(answer.choice, answer.probabilities, threshold, result.model, resolved.backend);
		},
	};
}

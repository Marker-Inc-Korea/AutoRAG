import { basename } from "node:path";
import type { Api, AssistantMessage, Context, Model } from "@earendil-works/pi-ai";
import { completeSimple } from "@earendil-works/pi-ai/compat";
import { DEFAULT_LANGUAGES, type LanguageTag } from "../language.ts";
import { createTesseractOcrProvider, type OcrProvider } from "./ocr-engines.ts";
import type { VisualImageMimeType } from "./page-render.ts";
import { normalizeMarkdown } from "./text.ts";

export type VisualExtractorKind = "vlm" | "ocr";

export interface VisualExtractInput {
	readonly virtualPath: string;
	readonly page: number;
	readonly image: { readonly data: Uint8Array; readonly mimeType: VisualImageMimeType };
	/** Base parser's extracted text for this page, as context so identifiers and units stay faithful. */
	readonly extractedText?: string;
	readonly signal?: AbortSignal;
}

export interface VisualExtractResult {
	readonly markdown: string;
	readonly extractorKind: VisualExtractorKind;
	readonly extractorModel: string;
}

/** Pluggable extractor turning a visually-rich document page image into searchable Markdown. */
export interface VisualPageExtractor {
	readonly kind: VisualExtractorKind;
	readonly modelId: string;
	/** Stable identity for incremental caching; changes trigger re-extraction of visually rich pages. */
	readonly fingerprint: string;
	extract(input: VisualExtractInput): Promise<VisualExtractResult>;
}

const VLM_PROMPT_VERSION = "v1";
const OCR_PROMPT_VERSION = "v1";
const DEFAULT_VLM_TIMEOUT_MS = 60_000;
const DEFAULT_VLM_MAX_TOKENS = 4_096;
const MAX_CONTEXT_TEXT_CHARS = 3_000;

const VLM_SYSTEM_PROMPT = [
	"You are a precision document visual parser building a searchable Markdown mirror for a retrieval system.",
	"Given one rendered document page image, transcribe and structure ALL visual information and its surrounding headings/context so every chart, table, and diagram is self-contained and searchable:",
	"- Charts, plots, and graphs: title, axis labels, units, legend, every visible series and numeric value (as a Markdown table when tabular), and key annotations.",
	"- Diagrams, flowcharts, org charts, infographics, and architecture figures: every component/node label, hierarchy, connection, and directional relationship.",
	"- Complex tables or calendars (multi-level headers, merged cells, column-aligned grids): accurate Markdown tables preserving column alignment, headers, units, totals, and footnotes.",
	"- Visual annotations, colored highlights, underlines, or red-marked sections: explicitly note what text or region is highlighted/marked and quote the marked text verbatim.",
	"- Embedded photos, screenshots, callouts, formulas, or scanned text: verbatim visible text and concrete factual content.",
	"Rules:",
	"- Preserve the document's original language (e.g., Korean, English, Japanese) and exact numbers, dates, currencies, percentages, and identifiers.",
	"- Never invent numbers, labels, or facts not visible in the page image.",
	"- Output clean Markdown only, without meta-commentary or markdown code fences wrapping the whole response.",
].join("\n");

export type VlmCompleteFn = (
	model: Model<Api>,
	context: Context,
	options?: { readonly apiKey?: string; readonly maxTokens?: number; readonly signal?: AbortSignal },
) => Promise<AssistantMessage>;

export interface VlmVisualPageExtractorOptions {
	readonly model: Model<Api>;
	readonly apiKey?: string;
	readonly timeoutMs?: number;
	readonly maxTokens?: number;
	/** Programmatic seam for tests or custom transports. */
	readonly complete?: VlmCompleteFn;
}

function extractAssistantText(message: AssistantMessage): string {
	if (message.stopReason === "error") {
		throw new Error(message.errorMessage ?? "VLM visual extraction failed");
	}
	return message.content
		.filter((block): block is { type: "text"; text: string } => block.type === "text")
		.map((block) => block.text)
		.join("\n")
		.trim();
}

function buildVlmUserText(input: VisualExtractInput): string {
	const baseText = input.extractedText?.trim() ?? "";
	const snippet =
		baseText.length > MAX_CONTEXT_TEXT_CHARS
			? `${baseText.slice(0, MAX_CONTEXT_TEXT_CHARS)}\n…(truncated)`
			: baseText;
	const parts = [
		`Document: ${basename(input.virtualPath)} (Page ${input.page})`,
		"Extract all visually rich content (charts, diagrams, figures, complex tables, and embedded/scanned text) from this page image into accurate Markdown.",
	];
	if (snippet.length > 0) {
		parts.push(
			"Reference text already extracted by the base parser (use it to preserve exact spellings/terms, and focus on capturing visual details, data values, layout tables, and image text accurately):",
			snippet,
		);
	}
	return parts.join("\n\n");
}

function cleanExtractedMarkdown(raw: string): string {
	return normalizeMarkdown(
		raw
			.replace(/\r\n?/gu, "\n")
			.split("\n")
			.map((line) => line.trimEnd())
			.join("\n")
			.trim(),
	);
}

/**
 * Create a VLM-backed visual page extractor over any vision-capable pi-ai chat model
 * (OpenRouter, OpenAI, Anthropic, Google, Dazzi Models, or local OpenAI-compatible VLM servers).
 */
export function createVlmVisualPageExtractor(options: VlmVisualPageExtractorOptions): VisualPageExtractor {
	const { model } = options;
	if (!model.input.includes("image")) {
		throw new Error(
			`VLM visual extractor requires a vision-capable model (input must include "image"), received "${model.provider}/${model.id}".`,
		);
	}
	const modelId = `${model.provider}/${model.id}`;
	const timeoutMs = options.timeoutMs ?? DEFAULT_VLM_TIMEOUT_MS;
	const maxTokens = options.maxTokens ?? DEFAULT_VLM_MAX_TOKENS;
	const complete: VlmCompleteFn =
		options.complete ??
		((targetModel, context, callOptions) =>
			completeSimple(targetModel, context, {
				...(callOptions?.apiKey !== undefined ? { apiKey: callOptions.apiKey } : {}),
				...(callOptions?.maxTokens !== undefined ? { maxTokens: callOptions.maxTokens } : {}),
				...(callOptions?.signal !== undefined ? { signal: callOptions.signal } : {}),
			}));

	return {
		kind: "vlm",
		modelId,
		fingerprint: `vlm:${VLM_PROMPT_VERSION}:${modelId}`,
		async extract(input: VisualExtractInput): Promise<VisualExtractResult> {
			const base64 = Buffer.from(
				input.image.data.buffer,
				input.image.data.byteOffset,
				input.image.data.byteLength,
			).toString("base64");
			const controller = new AbortController();
			const onAbort = () => controller.abort(input.signal?.reason);
			input.signal?.addEventListener("abort", onAbort, { once: true });
			const timer = setTimeout(
				() => controller.abort(new Error(`VLM visual extraction timed out after ${timeoutMs}ms`)),
				timeoutMs,
			);
			try {
				const response = await complete(
					model,
					{
						systemPrompt: VLM_SYSTEM_PROMPT,
						messages: [
							{
								role: "user",
								timestamp: Date.now(),
								content: [
									{ type: "text", text: buildVlmUserText(input) },
									{ type: "image", data: base64, mimeType: input.image.mimeType },
								],
							},
						],
					},
					{
						...(options.apiKey !== undefined ? { apiKey: options.apiKey } : {}),
						maxTokens,
						signal: controller.signal,
					},
				);
				return {
					markdown: cleanExtractedMarkdown(extractAssistantText(response)),
					extractorKind: "vlm",
					extractorModel: modelId,
				};
			} finally {
				clearTimeout(timer);
				input.signal?.removeEventListener("abort", onAbort);
			}
		},
	};
}

export interface OcrVisualPageExtractorOptions {
	readonly languages?: readonly LanguageTag[];
	readonly timeoutMs?: number;
	readonly cachePath?: string;
	/** Programmatic seam for tests or custom OCR engines. */
	readonly ocrProvider?: OcrProvider;
}

/** Create a local OCR-backed visual page extractor (Tesseract by default, or an injected OCR engine). */
export function createOcrVisualPageExtractor(options: OcrVisualPageExtractorOptions = {}): VisualPageExtractor {
	const languages = options.languages && options.languages.length > 0 ? options.languages : DEFAULT_LANGUAGES;
	const langKey = [...new Set(languages)].sort().join("+");
	const modelId = `tesseract:${langKey}`;
	const provider =
		options.ocrProvider ??
		createTesseractOcrProvider({
			languages,
			timeoutMs: options.timeoutMs,
			cachePath: options.cachePath,
		});

	return {
		kind: "ocr",
		modelId,
		fingerprint: `ocr:${OCR_PROMPT_VERSION}:${modelId}`,
		async extract(input: VisualExtractInput): Promise<VisualExtractResult> {
			const raw = await provider(input.image.data, input.page, input.image.mimeType);
			return {
				markdown: cleanExtractedMarkdown(raw),
				extractorKind: "ocr",
				extractorModel: modelId,
			};
		},
	};
}

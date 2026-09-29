import type { LanguageTag } from "../language.ts";
import { createTesseractOcrOperation, withTimeout } from "./ocr.ts";

const TESSERACT_LANGUAGES: Record<LanguageTag, string> = {
	ko: "kor",
	en: "eng",
	ja: "jpn",
	"zh-hans": "chi_sim",
	"zh-hant": "chi_tra",
	fr: "fra",
	de: "deu",
	es: "spa",
	ru: "rus",
	it: "ita",
	pt: "por",
	vi: "vie",
	th: "tha",
	ar: "ara",
	hi: "hin",
};

export type OcrProvider = (
	pageImage: Uint8Array,
	pageNumber: number,
	mimeType: "image/png" | "image/jpeg" | "image/webp",
) => Promise<string>;

export type TesseractOcrEngine = (
	language: string,
	pageImage: Uint8Array,
	pageNumber: number,
	mimeType: "image/png" | "image/jpeg" | "image/webp",
) => Promise<string>;

export interface TesseractOcrProviderOptions {
	readonly languages: readonly LanguageTag[];
	readonly timeoutMs?: number;
	readonly engine?: TesseractOcrEngine;
}

const DEFAULT_TESSERACT_TIMEOUT_MS = 30_000;

/** Traineddata codes for the configured languages, order preserved and deduplicated. */
export function tesseractLanguageListFor(tags: readonly LanguageTag[]): string[] {
	const languages = [...new Set(tags)].map((tag) => TESSERACT_LANGUAGES[tag]);
	if (languages.length === 0) {
		throw new Error("OCR provider requires at least one language.");
	}
	return languages;
}

export function tesseractLanguageFor(tags: readonly LanguageTag[]): string {
	return tesseractLanguageListFor(tags).join("+");
}

export function createTesseractOcrProvider(options: TesseractOcrProviderOptions): OcrProvider {
	const language = tesseractLanguageFor(options.languages);
	const timeoutMs = options.timeoutMs ?? DEFAULT_TESSERACT_TIMEOUT_MS;
	const engine =
		options.engine ??
		((resolvedLanguage, pageImage, pageNumber, mimeType) =>
			defaultTesseractEngine(resolvedLanguage, pageImage, pageNumber, mimeType, timeoutMs));
	return (pageImage, pageNumber, mimeType) => engine(language, pageImage, pageNumber, mimeType);
}

async function defaultTesseractEngine(
	language: string,
	pageImage: Uint8Array,
	_pageNumber: number,
	_mimeType: "image/png" | "image/jpeg" | "image/webp",
	timeoutMs: number,
): Promise<string> {
	const controller = new AbortController();
	const operation = createTesseractOcrOperation({
		bytes: pageImage,
		languages: [language],
		timeoutMs,
		signal: controller.signal,
	});
	return withTimeout(operation, timeoutMs, () => controller.abort());
}

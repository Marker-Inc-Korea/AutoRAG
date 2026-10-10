import { mkdir } from "node:fs/promises";
import { createWorker, type Worker } from "tesseract.js";
import type { LanguageTag } from "../language.ts";

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

/**
 * Languages kordoc's built-in PP-OCRv5 model reads reliably. Its recognizer
 * dictionary is Korean plus basic Latin: it has no kana, han, Cyrillic, Thai,
 * Arabic or Devanagari, and on rendered text it drops diacritics (accented
 * character recall fr 0.64, de 0.82, es 0.45 versus Tesseract 1.00, 0.91, 1.00).
 * Any configured language outside this set switches OCR to Tesseract.
 */
export const BUILTIN_OCR_LANGUAGES: ReadonlySet<LanguageTag> = new Set<LanguageTag>(["ko", "en"]);

/** OCR switch and Tesseract settings. `enabled` is the single opt-in for image files and scanned pages. */
export interface OcrOptions {
	readonly enabled: boolean;
	readonly timeoutMs?: number;
	/**
	 * Directory where Tesseract stores downloaded traineddata. tesseract.js
	 * falls back to the process working directory when unset.
	 */
	readonly cachePath?: string;
}

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
	readonly cachePath?: string;
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
		((resolvedLanguage, pageImage) =>
			recognizeWithTesseract({
				language: resolvedLanguage,
				bytes: pageImage,
				timeoutMs,
				cachePath: options.cachePath,
			}));
	return (pageImage, pageNumber, mimeType) => engine(language, pageImage, pageNumber, mimeType);
}

interface TesseractRecognition {
	readonly language: string;
	readonly bytes: Uint8Array;
	readonly timeoutMs: number;
	readonly cachePath: string | undefined;
}

interface OcrOperation {
	readonly result: Promise<string>;
	/** Settles once the worker is terminated, so a timeout never leaks a live worker. */
	readonly cleanup: Promise<void>;
}

function recognizeWithTesseract(recognition: TesseractRecognition): Promise<string> {
	const controller = new AbortController();
	const operation = startTesseract(recognition, controller.signal);
	return withTimeout(operation, recognition.timeoutMs, () => controller.abort());
}

function startTesseract(recognition: TesseractRecognition, signal: AbortSignal): OcrOperation {
	let cleanupResolve: () => void = () => undefined;
	let cleanupReject: (reason: unknown) => void = () => undefined;
	// tsconfig lib is ES2022, which predates Promise.withResolvers.
	const cleanup = new Promise<void>((resolve, reject) => {
		cleanupResolve = resolve;
		cleanupReject = reject;
	});
	const result = runTesseract(recognition, signal, cleanupResolve, cleanupReject);
	return { result, cleanup };
}

async function runTesseract(
	recognition: TesseractRecognition,
	signal: AbortSignal,
	cleanupResolve: () => void,
	cleanupReject: (reason: unknown) => void,
): Promise<string> {
	let worker: Worker | undefined;
	let termination: Promise<void> | undefined;
	const terminate = async (): Promise<void> => {
		if (worker === undefined) return;
		termination ??= worker.terminate().then(() => undefined);
		await termination;
	};
	const abort = () => {
		if (worker !== undefined) {
			void terminate().then(cleanupResolve, cleanupReject);
		}
	};
	signal.addEventListener("abort", abort, { once: true });
	try {
		const { cachePath } = recognition;
		if (cachePath !== undefined) await mkdir(cachePath, { recursive: true });
		worker = await createWorker(recognition.language, undefined, cachePath === undefined ? {} : { cachePath });
		if (signal.aborted) throw new Error("OCR aborted before worker was ready");
		const result = await worker.recognize(Buffer.from(recognition.bytes));
		return result.data.text;
	} finally {
		signal.removeEventListener("abort", abort);
		await terminate().then(cleanupResolve, cleanupReject);
	}
}

function withTimeout(operation: OcrOperation, timeoutMs: number, onTimeout: () => void): Promise<string> {
	// tsconfig lib is ES2022, which predates Promise.withResolvers.
	return new Promise((resolve, reject) => {
		const timeout = setTimeout(() => {
			onTimeout();
			operation.cleanup.finally(() => reject(new Error(`OCR timed out after ${timeoutMs}ms`)));
		}, timeoutMs);
		operation.result.then(
			(value) => {
				clearTimeout(timeout);
				resolve(value);
			},
			(error: unknown) => {
				clearTimeout(timeout);
				operation.cleanup.then(
					() => reject(error),
					(cleanupError: unknown) => reject(cleanupError),
				);
			},
		);
	});
}

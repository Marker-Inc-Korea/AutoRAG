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

/** Leptonica will not scale an image whose width or height is below this. */
const MIN_LEPTONICA_SIDE_PIXELS = 3;

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
		// Header-only. A truncated or unrecognized image still goes to Tesseract; errorHandler contains that failure.
		const tooSmall = ocrImageTooSmallMessage(recognition.bytes);
		if (tooSmall !== undefined) throw new Error(tooSmall);
		const { cachePath } = recognition;
		if (cachePath !== undefined) await mkdir(cachePath, { recursive: true });
		worker = await createWorker(recognition.language, undefined, {
			...(cachePath === undefined ? {} : { cachePath }),
			// createWorker.js rejects the job with this payload, then throws from the worker message
			// handler when no errorHandler is set. That throw is outside recognize()'s promise.
			errorHandler: containTesseractWorkerError,
		});
		if (signal.aborted) throw new Error("OCR aborted before worker was ready");
		const result = await worker.recognize(Buffer.from(recognition.bytes));
		return result.data.text;
	} finally {
		signal.removeEventListener("abort", abort);
		await terminate().then(cleanupResolve, cleanupReject);
	}
}

/**
 * tesseract.js already rejected the in-flight job with the worker payload.
 * Rethrowing here runs outside that promise and kills the process.
 */
function containTesseractWorkerError(_payload: unknown): void {}

/**
 * PNG, JPEG, or WebP whose header decodes both sides and either side is below
 * leptonica's minimum. Undefined when the header is truncated, unrecognized,
 * or both sides are at least 3 pixels.
 */
export function ocrImageTooSmallMessage(bytes: Uint8Array): string | undefined {
	const size = readImagePixelSize(bytes);
	if (size === undefined || (size.width >= MIN_LEPTONICA_SIDE_PIXELS && size.height >= MIN_LEPTONICA_SIDE_PIXELS)) {
		return undefined;
	}
	return `Image is too small to scale (${size.width}x${size.height} vs minimum of ${MIN_LEPTONICA_SIDE_PIXELS} pixels).`;
}

interface ImagePixelSize {
	readonly width: number;
	readonly height: number;
}

function readImagePixelSize(bytes: Uint8Array): ImagePixelSize | undefined {
	if (bytes.length >= 8 && bytes[0] === 0x89 && bytes[1] === 0x50) return readPngSize(bytes);
	if (bytes.length >= 2 && bytes[0] === 0xff && bytes[1] === 0xd8) return readJpegSize(bytes);
	if (bytes.length >= 12 && bytes[0] === 0x52 && bytes[1] === 0x49) return readWebpSize(bytes);
	return undefined;
}

function readPngSize(bytes: Uint8Array): ImagePixelSize | undefined {
	if (bytes.length < 24) return undefined;
	if (
		bytes[0] !== 0x89 ||
		bytes[1] !== 0x50 ||
		bytes[2] !== 0x4e ||
		bytes[3] !== 0x47 ||
		bytes[4] !== 0x0d ||
		bytes[5] !== 0x0a ||
		bytes[6] !== 0x1a ||
		bytes[7] !== 0x0a
	) {
		return undefined;
	}
	// IHDR is the first chunk. Width and height are big-endian at the start of its data.
	if (bytes[12] !== 0x49 || bytes[13] !== 0x48 || bytes[14] !== 0x44 || bytes[15] !== 0x52) return undefined;
	const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	return { width: view.getUint32(16), height: view.getUint32(20) };
}

function readJpegSize(bytes: Uint8Array): ImagePixelSize | undefined {
	if (bytes.length < 4 || bytes[0] !== 0xff || bytes[1] !== 0xd8) return undefined;
	const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	let offset = 2;
	while (offset + 1 < bytes.length) {
		if (bytes[offset] !== 0xff) return undefined;
		offset += 1;
		while (offset < bytes.length && bytes[offset] === 0xff) offset += 1;
		if (offset >= bytes.length) return undefined;
		const marker = bytes[offset] ?? 0;
		offset += 1;
		// Stuffed 0xFF00, or a marker that carries no length (TEM, RST, SOI, EOI).
		if (marker === 0x00) return undefined;
		if (marker === 0x01 || marker === 0xd9 || (marker >= 0xd0 && marker <= 0xd8)) continue;
		if (offset + 2 > bytes.length) return undefined;
		const segmentLength = view.getUint16(offset);
		if (segmentLength < 2) return undefined;
		if (isJpegStartOfFrame(marker)) {
			if (segmentLength < 7 || offset + 7 > bytes.length) return undefined;
			return { width: view.getUint16(offset + 5), height: view.getUint16(offset + 3) };
		}
		// Entropy-coded scan. No start-of-frame in the header we can trust.
		if (marker === 0xda) return undefined;
		const next = offset + segmentLength;
		if (next <= offset || next > bytes.length) return undefined;
		offset = next;
	}
	return undefined;
}

/** SOF markers carry width and height. DHT (C4), JPG (C8), and DAC (CC) do not. */
function isJpegStartOfFrame(marker: number): boolean {
	return marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc;
}

function readWebpSize(bytes: Uint8Array): ImagePixelSize | undefined {
	if (bytes.length < 20 || !asciiAt(bytes, 0, "RIFF") || !asciiAt(bytes, 8, "WEBP")) return undefined;
	const chunkSize = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(16, true);
	if (asciiAt(bytes, 12, "VP8X")) return readVp8xSize(bytes, chunkSize);
	if (asciiAt(bytes, 12, "VP8L")) return readVp8lSize(bytes, chunkSize);
	if (asciiAt(bytes, 12, "VP8 ")) return readVp8Size(bytes, chunkSize);
	return undefined;
}

function readVp8xSize(bytes: Uint8Array, chunkSize: number): ImagePixelSize | undefined {
	// Canvas width and height are 24-bit little-endian, stored minus one, after flags and reserved bytes.
	if (!webpPayloadFits(bytes, chunkSize, 10)) return undefined;
	const payload = 20;
	const width = 1 + readU24LE(bytes, payload + 4);
	const height = 1 + readU24LE(bytes, payload + 7);
	return { width, height };
}

function readVp8lSize(bytes: Uint8Array, chunkSize: number): ImagePixelSize | undefined {
	if (!webpPayloadFits(bytes, chunkSize, 5) || bytes[20] !== 0x2f) return undefined;
	const b1 = bytes[21] ?? 0;
	const b2 = bytes[22] ?? 0;
	const b3 = bytes[23] ?? 0;
	const b4 = bytes[24] ?? 0;
	const width = 1 + (b1 | ((b2 & 0x3f) << 8));
	const height = 1 + ((b2 >> 6) | (b3 << 2) | ((b4 & 0x0f) << 10));
	return { width, height };
}

function readVp8Size(bytes: Uint8Array, chunkSize: number): ImagePixelSize | undefined {
	// Key frame only: 3-byte frame tag, 0x9d012a start code, then 14-bit width and height.
	if (!webpPayloadFits(bytes, chunkSize, 10) || (bytes[20] & 0x01) !== 0) return undefined;
	if (bytes[23] !== 0x9d || bytes[24] !== 0x01 || bytes[25] !== 0x2a) return undefined;
	const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	return { width: view.getUint16(26, true) & 0x3fff, height: view.getUint16(28, true) & 0x3fff };
}

function webpPayloadFits(bytes: Uint8Array, chunkSize: number, needed: number): boolean {
	return chunkSize >= needed && bytes.length >= 20 + needed;
}

function readU24LE(bytes: Uint8Array, offset: number): number {
	return (bytes[offset] ?? 0) | ((bytes[offset + 1] ?? 0) << 8) | ((bytes[offset + 2] ?? 0) << 16);
}

function asciiAt(bytes: Uint8Array, offset: number, text: string): boolean {
	if (offset + text.length > bytes.length) return false;
	for (let index = 0; index < text.length; index += 1) {
		if (bytes[offset + index] !== text.charCodeAt(index)) return false;
	}
	return true;
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

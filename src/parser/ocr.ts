import { createWorker } from "tesseract.js";
import { ParseError } from "./errors.ts";
import { normalizeMarkdown } from "./text.ts";
import { type ParseInput, type ParseOutput, Parser } from "./types.ts";

export interface OcrEngineInput {
	readonly bytes: Uint8Array;
	readonly languages: readonly string[];
	readonly timeoutMs: number;
	readonly signal: AbortSignal;
}

export type OcrEngine = (input: OcrEngineInput) => Promise<string>;
type OcrEngineWithCleanup = (input: OcrEngineInput) => {
	readonly result: Promise<string>;
	readonly cleanup: Promise<void>;
};

export interface OcrParserOptions {
	readonly enabled: boolean;
	readonly languages?: readonly string[];
	readonly timeoutMs?: number;
	readonly maxBytes?: number;
	readonly engine?: OcrEngine;
}

const DEFAULT_OCR_TIMEOUT_MS = 30_000;
const DEFAULT_OCR_LANGUAGES = ["eng"] as const;

export class ImageOcrParser extends Parser {
	readonly name = "image-ocr";
	readonly extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"] as const;
	private readonly languages: readonly string[];
	private readonly timeoutMs: number;
	private readonly maxBytes: number | undefined;
	private readonly engine: OcrEngineWithCleanup;

	constructor(options: OcrParserOptions) {
		super();
		this.languages = options.languages ?? DEFAULT_OCR_LANGUAGES;
		this.timeoutMs = options.timeoutMs ?? DEFAULT_OCR_TIMEOUT_MS;
		this.maxBytes = options.maxBytes;
		this.engine = options.engine ? engineWithNoopCleanup(options.engine) : tesseractOcr;
	}

	async parse(input: ParseInput): Promise<ParseOutput> {
		const controller = new AbortController();
		try {
			if (this.maxBytes !== undefined && input.bytes.byteLength > this.maxBytes) {
				throw new Error(`OCR input exceeds maxBytes budget of ${this.maxBytes}`);
			}
			const operation = this.engine({
				bytes: input.bytes,
				languages: this.languages,
				timeoutMs: this.timeoutMs,
				signal: controller.signal,
			});
			const markdown = await withTimeout(operation, this.timeoutMs, () => controller.abort());
			return {
				markdown: normalizeMarkdown(markdown),
				metadata: { parser: this.name, languages: [...this.languages] },
			};
		} catch (cause) {
			throw new ParseError(this.name, input.virtualPath, cause);
		} finally {
			controller.abort();
		}
	}
}

function engineWithNoopCleanup(engine: OcrEngine): OcrEngineWithCleanup {
	return (input) => ({ result: engine(input), cleanup: Promise.resolve() });
}

function tesseractOcr(input: OcrEngineInput): ReturnType<OcrEngineWithCleanup> {
	let cleanupResolve: () => void = () => undefined;
	let cleanupReject: (reason: unknown) => void = () => undefined;
	const cleanup = new Promise<void>((resolve, reject) => {
		cleanupResolve = resolve;
		cleanupReject = reject;
	});
	const result = runTesseractOcr(input, cleanupResolve, cleanupReject);
	return { result, cleanup };
}

async function runTesseractOcr(
	input: OcrEngineInput,
	cleanupResolve: () => void,
	cleanupReject: (reason: unknown) => void,
): Promise<string> {
	let worker: Awaited<ReturnType<typeof createWorker>> | undefined;
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
	input.signal.addEventListener("abort", abort, { once: true });
	try {
		worker = await createWorker(input.languages.join("+"));
		if (input.signal.aborted) throw new Error("OCR aborted before worker was ready");
		const result = await worker.recognize(Buffer.from(input.bytes));
		return result.data.text;
	} finally {
		input.signal.removeEventListener("abort", abort);
		await terminate().then(cleanupResolve, cleanupReject);
	}
}

function withTimeout(
	operation: ReturnType<OcrEngineWithCleanup>,
	timeoutMs: number,
	onTimeout: () => void,
): Promise<string> {
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

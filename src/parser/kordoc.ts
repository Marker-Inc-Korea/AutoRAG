import { parse as kordocParse } from "kordoc";
import { ParseError } from "./errors.ts";
import { type ParseDiagnostic, type ParseInput, type ParseOutput, Parser } from "./types.ts";

/** Image mime types kordoc hands to an injected OCR provider. */
export type KordocOcrMimeType = "image/png" | "image/jpeg" | "image/webp";

/**
 * OCR seam kordoc calls per page that needs recognition. AutoRAG injects a
 * tesseract-backed provider so recognition follows the configured languages
 * instead of kordoc's Korean+Latin built-in model.
 */
export type KordocOcrProvider = (
	pageImage: Uint8Array,
	pageNumber: number,
	mimeType: KordocOcrMimeType,
) => Promise<string>;

export interface KordocWarning {
	readonly code?: string;
	readonly message: string;
}

export interface KordocParseSuccess {
	readonly success: true;
	readonly fileType?: string;
	readonly markdown: string;
	readonly blocks?: readonly unknown[];
	readonly warnings?: readonly KordocWarning[];
}

export interface KordocParseFailure {
	readonly success: false;
	readonly fileType?: string;
	readonly error: string;
	readonly code?: string;
}

export type KordocParseResult = KordocParseSuccess | KordocParseFailure;

export interface KordocParseCallOptions {
	readonly ocr?: boolean | "force" | KordocOcrProvider;
}

/** Injection seam: the real kordoc `parse`, or a stub in tests. */
export type KordocParseFn = (bytes: Uint8Array, options: KordocParseCallOptions) => Promise<KordocParseResult>;

export interface KordocParserOptions {
	/** OCR mode handed to kordoc: off, built-in engine, or an injected provider. */
	readonly ocr?: boolean | "force" | KordocOcrProvider;
	/** Also claim standalone image files; kordoc OCRs them whenever it parses them, so this follows the OCR switch. */
	readonly images?: boolean;
	readonly parse?: KordocParseFn;
}

/** Document extensions kordoc parses. `.pptx` and `.eml` stay on their own parsers. */
export const KORDOC_EXTENSIONS = [".hwp", ".hwpx", ".hml", ".hwpml", ".pdf", ".docx", ".xlsx", ".xls"] as const;

/**
 * Standalone image formats kordoc OCRs directly. `.bmp` and `.tiff` return
 * `UNSUPPORTED_FORMAT` from kordoc, so they are deliberately not claimed.
 */
export const KORDOC_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"] as const;

/** kordoc accepts Buffer/ArrayBuffer/string; wrap without copying when possible. */
function toKordocInput(bytes: Uint8Array): Buffer {
	return Buffer.isBuffer(bytes) ? bytes : Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

const defaultParse: KordocParseFn = async (bytes, options) => {
	const result = await kordocParse(toKordocInput(bytes), {
		// Plain markdown keeps heading/list/table structure while dropping image
		// placeholders and inline styling: kordoc's documented indexing/RAG profile.
		plain: true,
		...(options.ocr === undefined ? {} : { ocr: options.ocr }),
	});
	if (result.success) {
		return {
			success: true,
			fileType: result.fileType,
			markdown: result.markdown,
			warnings: result.warnings,
		};
	}
	return { success: false, fileType: result.fileType, error: result.error, code: result.code };
};

/**
 * Parses HWP/HWPX/HWPML, PDF, DOCX and XLSX/XLS through the kordoc library.
 *
 * kordoc reports failures as a result object; AutoRAG's contract is a thrown
 * `ParseError`, so the code and message are forwarded verbatim (AGENTS.md error
 * transparency) and warnings become `parser-warning` diagnostics.
 */
export class KordocParser extends Parser {
	readonly name = "kordoc";
	readonly extensions: readonly string[];

	readonly #parse: KordocParseFn;
	readonly #ocr: KordocParserOptions["ocr"];

	constructor(options: KordocParserOptions = {}) {
		super();
		this.#parse = options.parse ?? defaultParse;
		this.#ocr = options.ocr;
		this.extensions =
			options.images === true ? [...KORDOC_EXTENSIONS, ...KORDOC_IMAGE_EXTENSIONS] : KORDOC_EXTENSIONS;
	}

	async parse(input: ParseInput): Promise<ParseOutput> {
		let result: KordocParseResult;
		try {
			result = await this.#parse(input.bytes, { ocr: this.#ocr });
		} catch (cause) {
			throw new ParseError(this.name, input.virtualPath, cause);
		}

		if (!result.success) {
			throw new ParseError(
				this.name,
				input.virtualPath,
				new Error(`${result.code ?? "PARSE_ERROR"}: ${result.error}`),
			);
		}

		const diagnostics: ParseDiagnostic[] = (result.warnings ?? []).map((warning) => ({
			code: "parser-warning",
			severity: "warning",
			message: warning.code ? `${warning.code}: ${warning.message}` : warning.message,
		}));

		return {
			markdown: result.markdown,
			metadata: { parser: this.name, format: result.fileType ?? "unknown" },
			...(diagnostics.length > 0 ? { diagnostics } : {}),
		};
	}
}

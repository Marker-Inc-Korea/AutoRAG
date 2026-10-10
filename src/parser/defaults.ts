import { join } from "node:path";
import { DEFAULT_LANGUAGES, type LanguageTag } from "../language.ts";
import { EmlParser } from "./eml.ts";
import { KordocParser, type KordocParserOptions } from "./kordoc.ts";
import { BUILTIN_OCR_LANGUAGES, createTesseractOcrProvider, type OcrOptions } from "./ocr-engines.ts";
import { PptxParser } from "./office.ts";
import { PlainTextParser } from "./plain-text.ts";
import { ParserRegistry } from "./registry.ts";

export interface DefaultParserRegistryOptions {
	readonly ocr?: OcrOptions;
	readonly kordoc?: KordocParserOptions;
	/** Global document languages; selects the OCR engine when OCR is enabled. */
	readonly languages?: readonly LanguageTag[];
}

/** Workspace-relative directory for Tesseract traineddata, so models never land in the process working directory. */
export function tesseractCachePathFor(workspacePath: string): string {
	return join(workspacePath, ".autorag", "models", "tessdata");
}

/**
 * Merges the globally configured document languages into registry options.
 * An explicit `parserOptions.languages` wins over the global setting. When a
 * workspace is known, Tesseract's model cache is pinned under it unless the
 * caller set `ocr.cachePath`.
 */
export function resolveParserOptions(
	parserOptions: DefaultParserRegistryOptions | undefined,
	languages: readonly LanguageTag[],
	workspacePath?: string,
): DefaultParserRegistryOptions {
	const resolved = { ...parserOptions, languages: parserOptions?.languages ?? languages };
	if (workspacePath === undefined || parserOptions?.ocr === undefined || parserOptions.ocr.cachePath !== undefined) {
		return resolved;
	}
	return { ...resolved, ocr: { ...parserOptions.ocr, cachePath: tesseractCachePathFor(workspacePath) } };
}

/**
 * `ocr.enabled` is the single opt-in. Off: image files are not parsed and
 * scanned pages are left unrecognized. On: kordoc owns both, and the
 * configured languages pick the engine. kordoc's built-in PP-OCRv5 model is
 * used only when every language is one it reads reliably; otherwise Tesseract
 * is injected as kordoc's OCR provider. An explicit `kordoc.ocr` always wins.
 */
export function createDefaultParserRegistry(options: DefaultParserRegistryOptions = {}): ParserRegistry {
	const languages = options.languages ?? DEFAULT_LANGUAGES;
	const ocrEnabled = options.ocr?.enabled === true;
	const builtinCovers = languages.every((language) => BUILTIN_OCR_LANGUAGES.has(language));
	const kordocOptions: KordocParserOptions = {
		...options.kordoc,
		images: ocrEnabled,
		...(options.kordoc?.ocr === undefined && ocrEnabled
			? {
					ocr: builtinCovers
						? true
						: createTesseractOcrProvider({
								languages,
								timeoutMs: options.ocr?.timeoutMs,
								cachePath: options.ocr?.cachePath,
							}),
				}
			: {}),
	};

	return new ParserRegistry([
		new PlainTextParser(),
		new KordocParser(kordocOptions),
		new PptxParser(),
		new EmlParser(),
	]);
}

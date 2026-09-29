import { DEFAULT_LANGUAGES, type LanguageTag } from "../language.ts";
import { EmlParser } from "./eml.ts";
import { KordocParser, type KordocParserOptions } from "./kordoc.ts";
import { ImageOcrParser, type OcrParserOptions } from "./ocr.ts";
import { createTesseractOcrProvider, tesseractLanguageListFor } from "./ocr-engines.ts";
import { PptxParser } from "./office.ts";
import { PlainTextParser } from "./plain-text.ts";
import { ParserRegistry } from "./registry.ts";
import type { Parser } from "./types.ts";

export interface DefaultParserRegistryOptions {
	readonly ocr?: OcrParserOptions;
	readonly kordoc?: KordocParserOptions;
	/** Global document languages; drives OCR engine selection. */
	readonly languages?: readonly LanguageTag[];
}

/**
 * Merges the globally configured document languages into registry options.
 * An explicit `parserOptions.languages` wins over the global setting.
 */
export function resolveParserOptions(
	parserOptions: DefaultParserRegistryOptions | undefined,
	languages: readonly LanguageTag[],
): DefaultParserRegistryOptions {
	return { ...parserOptions, languages: parserOptions?.languages ?? languages };
}

export function createDefaultParserRegistry(options: DefaultParserRegistryOptions = {}): ParserRegistry {
	const languages = options.languages ?? DEFAULT_LANGUAGES;
	const ocrEnabled = options.ocr?.enabled === true;
	// OCR stays opt-in: without it, parsing never reaches a recognition engine
	// and never downloads a model.
	const kordocOptions: KordocParserOptions = {
		...options.kordoc,
		...(options.kordoc?.ocr === undefined && ocrEnabled
			? { ocr: createTesseractOcrProvider({ languages, timeoutMs: options.ocr?.timeoutMs }) }
			: {}),
	};

	const parsers: Parser[] = [
		new PlainTextParser(),
		new KordocParser(kordocOptions),
		new PptxParser(),
		new EmlParser(),
	];
	if (ocrEnabled) {
		parsers.push(
			new ImageOcrParser({
				...options.ocr,
				enabled: true,
				languages: options.ocr?.languages ?? tesseractLanguageListFor(languages),
			}),
		);
	}
	return new ParserRegistry(parsers);
}

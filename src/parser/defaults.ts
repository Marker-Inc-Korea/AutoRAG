import { EmlParser } from "./eml.ts";
import { HwpParser, type HwpParserOptions } from "./hwp.ts";
import { ImageOcrParser, type OcrParserOptions } from "./ocr.ts";
import { DocxParser, PptxParser, XlsxParser } from "./office.ts";
import { OpendataloaderPdfParser, type OpendataloaderPdfParserOptions } from "./opendataloader-pdf.ts";
import { PlainTextParser } from "./plain-text.ts";
import { ParserRegistry } from "./registry.ts";
import type { Parser } from "./types.ts";

export interface DefaultParserRegistryOptions {
	readonly ocr?: OcrParserOptions;
	readonly hwp?: HwpParserOptions;
	readonly thinExtract?: OpendataloaderPdfParserOptions["thinExtract"];
}

export function createDefaultParserRegistry(options: DefaultParserRegistryOptions = {}): ParserRegistry {
	const parsers: Parser[] = [
		new PlainTextParser(),
		new OpendataloaderPdfParser({ ocr: options.ocr, thinExtract: options.thinExtract }),
		new DocxParser(),
		new PptxParser(),
		new XlsxParser(),
		new HwpParser(options.hwp),
		new EmlParser(),
	];
	if (options.ocr?.enabled) parsers.push(new ImageOcrParser(options.ocr));
	return new ParserRegistry(parsers);
}

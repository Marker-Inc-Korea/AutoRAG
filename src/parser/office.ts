import { extname } from "node:path";
import { ParseError } from "./errors.ts";
import { normalizeMarkdown } from "./text.ts";
import { type ParseInput, type ParseOutput, Parser } from "./types.ts";
import { readZipXmlText } from "./xml-text.ts";

export class DocxParser extends Parser {
	readonly name = "docx";
	readonly extensions = [".docx"] as const;

	async parse(input: ParseInput): Promise<ParseOutput> {
		try {
			const chunks = await readZipXmlText(input.bytes, /^word\/(?:document|header\d+|footer\d+)\.xml$/);
			return formatMarkdown(this.name, chunks);
		} catch (cause) {
			throw new ParseError(this.name, input.virtualPath, cause);
		}
	}
}

export class PptxParser extends Parser {
	readonly name = "pptx";
	readonly extensions = [".pptx"] as const;

	async parse(input: ParseInput): Promise<ParseOutput> {
		try {
			const chunks = await readZipXmlText(
				input.bytes,
				/^ppt\/(?:slides\/slide\d+|notesSlides\/notesSlide\d+)\.xml$/,
			);
			return formatMarkdown(this.name, chunks);
		} catch (cause) {
			throw new ParseError(this.name, input.virtualPath, cause);
		}
	}
}

export class XlsxParser extends Parser {
	readonly name = "xlsx";
	readonly extensions = [".xlsx", ".xls"] as const;

	async parse(input: ParseInput): Promise<ParseOutput> {
		try {
			if (extname(input.virtualPath).toLowerCase() === ".xls") {
				throw new Error("legacy XLS binary parsing is not supported by the pure JavaScript parser");
			}
			const sharedStrings = await readZipXmlText(input.bytes, /^xl\/sharedStrings\.xml$/);
			const sheetText = await readZipXmlText(input.bytes, /^xl\/worksheets\/sheet\d+\.xml$/);
			return formatMarkdown(this.name, [...sharedStrings, ...sheetText]);
		} catch (cause) {
			throw new ParseError(this.name, input.virtualPath, cause);
		}
	}
}

function formatMarkdown(parserName: string, chunks: readonly string[]): ParseOutput {
	const markdown = normalizeMarkdown(chunks.filter((chunk) => chunk.trim().length > 0).join("\n\n"));
	return { markdown, metadata: { parser: parserName } };
}

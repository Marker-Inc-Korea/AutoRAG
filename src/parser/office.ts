import { extname } from "node:path";
import * as XLSX from "xlsx";
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
				return formatLegacyXls(input.bytes);
			}
			const sharedStrings = await readZipXmlText(input.bytes, /^xl\/sharedStrings\.xml$/);
			const sheetText = await readZipXmlText(input.bytes, /^xl\/worksheets\/sheet\d+\.xml$/);
			return formatMarkdown(this.name, [...sharedStrings, ...sheetText]);
		} catch (cause) {
			throw new ParseError(this.name, input.virtualPath, cause);
		}
	}
}

function formatLegacyXls(bytes: Uint8Array): ParseOutput {
	const oleMagic = [0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1];
	if (bytes.length < oleMagic.length || !oleMagic.every((byte, index) => bytes[index] === byte)) {
		throw new Error("legacy XLS input is not an OLE compound document");
	}
	const workbook = XLSX.read(Buffer.from(bytes), { type: "buffer", cellText: true, cellDates: true });
	const markdown = workbook.SheetNames.map((sheetName) => {
		const worksheet = workbook.Sheets[sheetName];
		if (!worksheet) return `## ${sheetName}`;
		const rows = XLSX.utils.sheet_to_json<readonly unknown[]>(worksheet, {
			header: 1,
			raw: false,
			blankrows: false,
			defval: "",
		});
		const lines = rows
			.map((row) =>
				row
					.map((cell) => String(cell ?? "").trim())
					.join(" | ")
					.replace(/\s+\|$/, ""),
			)
			.filter((row) => row.trim().length > 0);
		return [`## ${sheetName}`, ...lines].join("\n");
	}).join("\n\n");
	return { markdown: normalizeMarkdown(markdown), metadata: { parser: "xlsx", format: "xls" } };
}

function formatMarkdown(parserName: string, chunks: readonly string[]): ParseOutput {
	const markdown = normalizeMarkdown(chunks.filter((chunk) => chunk.trim().length > 0).join("\n\n"));
	return { markdown, metadata: { parser: parserName } };
}

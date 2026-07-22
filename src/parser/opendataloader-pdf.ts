import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join, parse } from "node:path";
import type { ConvertOptions } from "@opendataloader/pdf";
import { convert } from "@opendataloader/pdf";
import { ParseError } from "./errors.ts";
import { normalizeMarkdown } from "./text.ts";
import { type ParseInput, type ParseOutput, Parser } from "./types.ts";

export type PdfConverter = (inputPath: string, options: ConvertOptions) => Promise<string>;

export interface OpendataloaderPdfParserOptions {
	readonly converter?: PdfConverter;
	readonly ocr?: {
		readonly enabled: boolean;
		readonly timeoutMs?: number;
		readonly hybrid?: string;
		readonly hybridMode?: string;
		readonly maxBytes?: number;
	};
}

export class OpendataloaderPdfParser extends Parser {
	readonly name = "opendataloader-pdf";
	readonly extensions = [".pdf"] as const;

	private readonly options: OpendataloaderPdfParserOptions;
	private readonly converter: PdfConverter;

	constructor(options: OpendataloaderPdfParserOptions = {}) {
		super();
		this.options = options;
		this.converter = options.converter ?? convert;
	}

	async parse(input: ParseInput): Promise<ParseOutput> {
		const tempRoot = await mkdtemp(join(tmpdir(), "autorag-pdf-"));
		try {
			const inputPath = input.sourcePath ?? join(tempRoot, basename(input.virtualPath));
			if (!input.sourcePath) {
				await writeFile(inputPath, input.bytes);
			}
			if (
				this.options.ocr?.enabled &&
				this.options.ocr.maxBytes !== undefined &&
				input.bytes.byteLength > this.options.ocr.maxBytes
			) {
				throw new Error(`PDF OCR input exceeds maxBytes budget of ${this.options.ocr.maxBytes}`);
			}

			await this.converter(inputPath, {
				outputDir: tempRoot,
				format: "markdown",
				quiet: true,
				imageOutput: "off",
				...(this.options.ocr?.enabled
					? {
							hybrid: this.options.ocr.hybrid ?? "docling-fast",
							hybridMode: this.options.ocr.hybridMode ?? "auto",
							hybridTimeout: String(this.options.ocr.timeoutMs ?? 30_000),
						}
					: {}),
			});

			return {
				markdown: normalizeMarkdown(await readFile(markdownOutputPath(tempRoot, inputPath), "utf8")),
				metadata: { parser: this.name },
			};
		} catch (cause) {
			throw new ParseError(this.name, input.virtualPath, cause);
		} finally {
			await rm(tempRoot, { recursive: true, force: true });
		}
	}
}

function markdownOutputPath(outputDir: string, inputPath: string): string {
	return join(outputDir, `${parse(inputPath).name}.md`);
}

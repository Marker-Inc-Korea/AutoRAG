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
	/** Quality gate for multi-page PDFs whose local markdown yield is abnormally thin. */
	readonly thinExtract?: {
		readonly enabled?: boolean;
		readonly minPages?: number;
		readonly minChars?: number;
		readonly minCharsPerPage?: number;
		readonly timeoutMs?: number;
		readonly hybrid?: string;
		readonly hybridMode?: string;
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

			const localOptions: ConvertOptions = {
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
			};
			await this.converter(inputPath, localOptions);
			const localMarkdown = normalizeMarkdown(await readMarkdown(tempRoot, inputPath));
			const pages = countPdfPages(input.bytes);
			const thin = this.options.thinExtract;
			const shouldRetry =
				this.options.ocr?.enabled !== true &&
				thin?.enabled !== false &&
				pages >= (thin?.minPages ?? 3) &&
				(localMarkdown.length < (thin?.minChars ?? 800) ||
					localMarkdown.length / pages < (thin?.minCharsPerPage ?? 40));
			if (shouldRetry) {
				try {
					await withTimeout(
						this.converter(inputPath, {
							...localOptions,
							hybrid: thin?.hybrid ?? "docling-fast",
							hybridMode: thin?.hybridMode ?? "auto",
							hybridTimeout: String(thin?.timeoutMs ?? 30_000),
						}),
						thin?.timeoutMs ?? 30_000,
					);
					return {
						markdown: normalizeMarkdown(await readMarkdown(tempRoot, inputPath)),
						metadata: { parser: this.name, pages, pdfExtract: "hybrid-retry" },
					};
				} catch {
					return {
						markdown: localMarkdown,
						metadata: {
							parser: this.name,
							pages,
							pdfExtract: "thin-local-fallback",
						},
						diagnostics: [
							{
								code: "pdf-extract-thin",
								severity: "warning",
								message: `Local PDF extraction yielded only ${localMarkdown.length} characters across ${pages} pages.`,
							},
							{
								code: "pdf-hybrid-unavailable",
								severity: "warning",
								message: "Hybrid PDF extraction was unavailable; local markdown was retained.",
							},
						],
					};
				}
			}
			return {
				markdown: localMarkdown,
				metadata: { parser: this.name, pages },
			};
		} catch (cause) {
			throw new ParseError(this.name, input.virtualPath, cause);
		} finally {
			await rm(tempRoot, { recursive: true, force: true });
		}
	}
}

async function readMarkdown(outputDir: string, inputPath: string): Promise<string> {
	return readFile(markdownOutputPath(outputDir, inputPath), "utf8");
}

function countPdfPages(bytes: Uint8Array): number {
	const text = Buffer.from(bytes).toString("latin1");
	return Math.max(1, [...text.matchAll(/\/Type\s*\/Page\b/gu)].length);
}

async function withTimeout<T>(operation: Promise<T>, timeoutMs: number): Promise<T> {
	let timer: ReturnType<typeof setTimeout> | undefined;
	try {
		return await Promise.race([
			operation,
			new Promise<T>((_, reject) => {
				timer = setTimeout(
					() => reject(new Error(`PDF hybrid extraction timed out after ${timeoutMs}ms`)),
					timeoutMs,
				);
			}),
		]);
	} finally {
		if (timer !== undefined) clearTimeout(timer);
	}
}

function markdownOutputPath(outputDir: string, inputPath: string): string {
	return join(outputDir, `${parse(inputPath).name}.md`);
}

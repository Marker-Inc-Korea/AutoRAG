import { extname } from "node:path";
import { ParseError } from "./errors.ts";
import { renderHwpMarkdown } from "./hwp-markdown.ts";
import { createRhwpExtractor, type HwpExtractionLimits, type HwpExtractor } from "./rhwp-adapter.ts";
import { normalizeMarkdown } from "./text.ts";
import { type ParseInput, type ParseOutput, Parser } from "./types.ts";
import { readZipXmlText } from "./xml-text.ts";

export interface HwpParserOptions {
	readonly limits?: HwpExtractionLimits;
	readonly extractor?: HwpExtractor;
}

export class HwpParser extends Parser {
	readonly name = "hwp";
	readonly extensions = [".hwpx", ".hwp"] as const;
	private readonly extractor: HwpExtractor;

	constructor(options: HwpParserOptions = {}) {
		super();
		this.extractor = options.extractor ?? createRhwpExtractor(undefined, options.limits);
	}

	async parse(input: ParseInput): Promise<ParseOutput> {
		try {
			if (extname(input.virtualPath).toLowerCase() === ".hwp") {
				const document = await this.extractor(input.bytes);
				return {
					markdown: normalizeMarkdown(renderHwpMarkdown(document)),
					metadata: { parser: this.name, format: "hwp5" },
				};
			}
			const chunks = await readZipXmlText(input.bytes, /^(?:Contents|content)\/.*\.xml$/i);
			return {
				markdown: normalizeMarkdown(chunks.join("\n\n")),
				metadata: { parser: this.name, format: "hwpx" },
			};
		} catch (cause) {
			throw new ParseError(this.name, input.virtualPath, cause);
		}
	}
}

import { ParseError } from "./errors.ts";
import { normalizeMarkdown } from "./text.ts";
import { type ParseInput, type ParseOutput, Parser } from "./types.ts";
import { readZipXmlText } from "./xml-text.ts";

/** PPTX stays on AutoRAG's own reader: kordoc detects pptx but does not parse it. */
export class PptxParser extends Parser {
	readonly name = "pptx";
	readonly extensions = [".pptx"] as const;

	async parse(input: ParseInput): Promise<ParseOutput> {
		try {
			const chunks = await readZipXmlText(
				input.bytes,
				/^ppt\/(?:slides\/slide\d+|notesSlides\/notesSlide\d+)\.xml$/,
			);
			const markdown = normalizeMarkdown(chunks.filter((chunk) => chunk.trim().length > 0).join("\n\n"));
			return { markdown, metadata: { parser: this.name } };
		} catch (cause) {
			throw new ParseError(this.name, input.virtualPath, cause);
		}
	}
}

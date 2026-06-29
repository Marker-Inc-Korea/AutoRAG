import { extname } from "node:path";
import { ParseError } from "./errors.ts";
import { normalizeMarkdown } from "./text.ts";
import { type ParseInput, type ParseOutput, Parser } from "./types.ts";
import { readZipXmlText } from "./xml-text.ts";

export class HwpParser extends Parser {
	readonly name = "hwp";
	readonly extensions = [".hwpx", ".hwp"] as const;

	async parse(input: ParseInput): Promise<ParseOutput> {
		try {
			if (extname(input.virtualPath).toLowerCase() === ".hwp") {
				throw new Error("legacy HWP5 binary parsing is not supported by the pure JavaScript parser");
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

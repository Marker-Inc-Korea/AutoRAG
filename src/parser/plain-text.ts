import { type ParseInput, type ParseOutput, Parser } from "./types.ts";

export class PlainTextParser extends Parser {
	readonly name = "plain-text";
	readonly extensions = [".txt", ".text", ".md", ".markdown"] as const;

	async parse(input: ParseInput): Promise<ParseOutput> {
		return { markdown: Buffer.from(input.bytes).toString("utf8") };
	}
}

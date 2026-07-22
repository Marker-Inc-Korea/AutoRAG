import type { AddressObject } from "mailparser";
import { simpleParser } from "mailparser";
import { ParseError } from "./errors.ts";
import { normalizeMarkdown } from "./text.ts";
import { type ParseInput, type ParseOutput, Parser } from "./types.ts";

export class EmlParser extends Parser {
	readonly name = "eml";
	readonly extensions = [".eml"] as const;

	async parse(input: ParseInput): Promise<ParseOutput> {
		try {
			const parsed = await simpleParser(Buffer.from(input.bytes));
			const from = addressText(parsed.from);
			const to = addressText(parsed.to);
			const lines = [
				parsed.subject ? `# ${parsed.subject}` : undefined,
				from ? `From: ${from}` : undefined,
				to ? `To: ${to}` : undefined,
				parsed.date ? `Date: ${parsed.date.toISOString()}` : undefined,
				parsed.text?.trim(),
			].filter((line): line is string => line !== undefined && line.length > 0);
			return {
				markdown: normalizeMarkdown(lines.join("\n\n")),
				metadata: {
					parser: this.name,
					subject: parsed.subject,
					from,
					to,
					date: parsed.date?.toISOString(),
				},
			};
		} catch (cause) {
			throw new ParseError(this.name, input.virtualPath, cause);
		}
	}
}

function addressText(address: AddressObject | AddressObject[] | undefined): string | undefined {
	if (address === undefined) return undefined;
	if (Array.isArray(address)) {
		return address
			.map((item) => item.text)
			.filter((text) => text.length > 0)
			.join(", ");
	}
	return address.text;
}

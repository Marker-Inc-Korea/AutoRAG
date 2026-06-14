import { PlainTextParser } from "./plain-text.ts";
import { ParserRegistry } from "./registry.ts";

export function createDefaultParserRegistry(): ParserRegistry {
	return new ParserRegistry([new PlainTextParser()]);
}

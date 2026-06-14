import { describe, expect, it } from "vitest";
import { createDefaultParserRegistry, Parser, ParserRegistry, PlainTextParser } from "../../src/parser/index.ts";

class UppercaseParser extends Parser {
	readonly name = "uppercase";
	readonly extensions = [".up"];

	async parse(input: { readonly bytes: Uint8Array }): Promise<{ readonly markdown: string }> {
		return { markdown: Buffer.from(input.bytes).toString("utf8").toUpperCase() };
	}
}

describe("ParserRegistry", () => {
	it("routes by lowercased extension through Parser subclasses", async () => {
		const registry = new ParserRegistry([new UppercaseParser()]);
		const parser = registry.getForVirtualPath("/docs/NOTE.UP");

		expect(parser).toBeInstanceOf(UppercaseParser);
		await expect(parser?.parse({ virtualPath: "/docs/NOTE.UP", bytes: Buffer.from("alpha") })).resolves.toEqual({
			markdown: "ALPHA",
		});
	});

	it("rejects duplicate extension ownership", () => {
		const first = new PlainTextParser();
		const second = new PlainTextParser();

		expect(() => new ParserRegistry([first, second])).toThrow('Parser extension ".txt" is already registered');
	});

	it("default registry supports text and markdown but skips unsupported binary files", () => {
		const registry = createDefaultParserRegistry();

		expect(registry.getForVirtualPath("/docs/a.txt")).toBeInstanceOf(PlainTextParser);
		expect(registry.getForVirtualPath("/docs/a.md")).toBeInstanceOf(PlainTextParser);
		expect(registry.getForVirtualPath("/docs/a.pdf")).toBeUndefined();
	});
});

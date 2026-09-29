import { describe, expect, it } from "vitest";
import { createDefaultParserRegistry, resolveParserOptions } from "../../src/parser/defaults.ts";
import type { KordocParseCallOptions } from "../../src/parser/kordoc.ts";

describe("createDefaultParserRegistry language wiring", () => {
	it("maps configured languages onto the image OCR parser", async () => {
		const registry = createDefaultParserRegistry({
			languages: ["ja", "en"],
			ocr: { enabled: true, engine: async () => "recognized text" },
		});
		const parser = registry.getForVirtualPath("/docs/scan.png");

		expect(parser?.name).toBe("image-ocr");
		const parsed = await parser?.parse({ virtualPath: "/docs/scan.png", bytes: new Uint8Array([1, 2, 3]) });

		expect(parsed?.metadata).toMatchObject({ parser: "image-ocr", languages: ["jpn", "eng"] });
	});

	it("routes .webp images to the OCR parser", () => {
		const registry = createDefaultParserRegistry({ ocr: { enabled: true } });
		expect(registry.getForVirtualPath("/docs/shot.webp")?.name).toBe("image-ocr");
	});

	it("hands kordoc a language-aware OCR provider only when OCR is enabled", async () => {
		const captured: Array<KordocParseCallOptions["ocr"]> = [];
		const stub = async (_bytes: Uint8Array, options: KordocParseCallOptions) => {
			captured.push(options.ocr);
			return { success: true as const, fileType: "pdf", markdown: "body" };
		};

		const withOcr = createDefaultParserRegistry({
			languages: ["ko"],
			ocr: { enabled: true },
			kordoc: { parse: stub },
		});
		await withOcr.getForVirtualPath("/docs/a.pdf")?.parse({ virtualPath: "/docs/a.pdf", bytes: new Uint8Array([1]) });

		const withoutOcr = createDefaultParserRegistry({ languages: ["ko"], kordoc: { parse: stub } });
		await withoutOcr
			.getForVirtualPath("/docs/b.pdf")
			?.parse({ virtualPath: "/docs/b.pdf", bytes: new Uint8Array([1]) });

		expect(typeof captured[0]).toBe("function");
		expect(captured[1]).toBeUndefined();
	});

	it("keeps OCR off by default so parsing never downloads a model unasked", async () => {
		const captured: Array<KordocParseCallOptions["ocr"]> = [];
		const registry = createDefaultParserRegistry({
			kordoc: {
				parse: async (_bytes, options) => {
					captured.push(options.ocr);
					return { success: true as const, fileType: "hwp", markdown: "body" };
				},
			},
		});

		await registry
			.getForVirtualPath("/docs/a.hwp")
			?.parse({ virtualPath: "/docs/a.hwp", bytes: new Uint8Array([1]) });

		expect(captured).toEqual([undefined]);
		expect(registry.getForVirtualPath("/docs/scan.png")).toBeUndefined();
	});
});

describe("resolveParserOptions", () => {
	it("fills the registry languages from the configured global languages", () => {
		expect(resolveParserOptions(undefined, ["ja", "en"])).toEqual({ languages: ["ja", "en"] });
	});

	it("keeps caller-provided parser options and adds languages", () => {
		const resolved = resolveParserOptions({ ocr: { enabled: true } }, ["ko"]);
		expect(resolved).toEqual({ ocr: { enabled: true }, languages: ["ko"] });
	});

	it("lets an explicit parser-level language list win", () => {
		const resolved = resolveParserOptions({ languages: ["th"] }, ["ko", "en"]);
		expect(resolved.languages).toEqual(["th"]);
	});
});

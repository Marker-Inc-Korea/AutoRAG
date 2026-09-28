import { describe, expect, it, vi } from "vitest";
import { AUTORAG_LANGUAGE_TAGS, DEFAULT_LANGUAGES, type LanguageTag } from "../../src/language.ts";
import { createTesseractOcrProvider, tesseractLanguageFor } from "../../src/parser/ocr-engines.ts";

const expectedLanguages: Record<LanguageTag, string> = {
	ko: "kor",
	en: "eng",
	ja: "jpn",
	"zh-hans": "chi_sim",
	"zh-hant": "chi_tra",
	fr: "fra",
	de: "deu",
	es: "spa",
	ru: "rus",
	it: "ita",
	pt: "por",
	vi: "vie",
	th: "tha",
	ar: "ara",
	hi: "hin",
};

describe("OCR engines", () => {
	it("maps every configured language tag to traineddata", () => {
		for (const tag of AUTORAG_LANGUAGE_TAGS) {
			expect(tesseractLanguageFor([tag])).toBe(expectedLanguages[tag]);
		}
	});

	it("joins languages in order and collapses duplicates", () => {
		expect(tesseractLanguageFor(["ko", "en"])).toBe("kor+eng");
		expect(tesseractLanguageFor(["ko", "ja"])).toBe("kor+jpn");
		expect(tesseractLanguageFor(["ko", "en", "ko", "en"])).toBe("kor+eng");
	});

	it("creates a provider that forwards the resolved language and page metadata", async () => {
		const engine = vi.fn(async (language: string, pageImage: Uint8Array, pageNumber: number, mimeType: string) => {
			expect(language).toBe("kor+eng");
			expect(pageImage).toEqual(new Uint8Array([1, 2, 3]));
			expect(pageNumber).toBe(4);
			expect(mimeType).toBe("image/png");
			return "recognized text";
		});
		const provider = createTesseractOcrProvider({ languages: DEFAULT_LANGUAGES, engine });

		await expect(provider(new Uint8Array([1, 2, 3]), 4, "image/png")).resolves.toBe("recognized text");
		expect(engine).toHaveBeenCalledOnce();
	});

	it("preserves injected engine errors", async () => {
		const engine = vi.fn(async () => {
			throw new Error("traineddata exploded at /tmp/tessdata");
		});
		const provider = createTesseractOcrProvider({ languages: ["en"], engine });

		await expect(provider(new Uint8Array([1]), 1, "image/jpeg")).rejects.toThrow(
			"traineddata exploded at /tmp/tessdata",
		);
	});

	it("rejects an empty language list", () => {
		expect(() => createTesseractOcrProvider({ languages: [] })).toThrow(/at least one language/i);
		expect(() => tesseractLanguageFor([])).toThrow(/at least one language/i);
	});
});

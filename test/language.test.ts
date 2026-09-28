import { describe, expect, it } from "vitest";
import {
	AUTORAG_LANGUAGE_TAGS,
	DEFAULT_LANGUAGES,
	LanguageError,
	normalizeLanguages,
	parseLanguageList,
} from "../src/language.ts";

describe("normalizeLanguages", () => {
	it("defaults to Korean + English when nothing is configured", () => {
		expect(DEFAULT_LANGUAGES).toEqual(["ko", "en"]);
		expect(normalizeLanguages(undefined)).toEqual(["ko", "en"]);
	});

	it("accepts a single tag and keeps order", () => {
		expect(normalizeLanguages(["ja"])).toEqual(["ja"]);
		expect(normalizeLanguages(["en", "ko"])).toEqual(["en", "ko"]);
	});

	it("lowercases, trims, and removes duplicates while preserving first position", () => {
		expect(normalizeLanguages([" KO ", "en", "ko", "EN"])).toEqual(["ko", "en"]);
	});

	it("accepts script-qualified Chinese tags", () => {
		expect(normalizeLanguages(["zh-Hans"])).toEqual(["zh-hans"]);
		expect(normalizeLanguages(["ZH-HANT", "ko"])).toEqual(["zh-hant", "ko"]);
	});

	it("rejects an unsupported tag with the supported set in the message", () => {
		let caught: unknown;
		try {
			normalizeLanguages(["kr"]);
		} catch (error) {
			caught = error;
		}
		expect(caught).toBeInstanceOf(LanguageError);
		expect((caught as LanguageError).message).toContain('Unsupported language "kr"');
		expect((caught as LanguageError).message).toContain("ko");
		expect((caught as LanguageError).message).toContain("zh-hans");
	});

	it("rejects a non-array, non-string value", () => {
		expect(() => normalizeLanguages(42 as unknown)).toThrow(LanguageError);
		expect(() => normalizeLanguages({} as unknown)).toThrow(LanguageError);
	});

	it("rejects an empty list instead of silently defaulting", () => {
		expect(() => normalizeLanguages([])).toThrow(LanguageError);
		expect(() => normalizeLanguages(["", "  "])).toThrow(LanguageError);
	});

	it("exposes the curated tag set", () => {
		expect(AUTORAG_LANGUAGE_TAGS).toContain("ko");
		expect(AUTORAG_LANGUAGE_TAGS).toContain("ja");
		expect(AUTORAG_LANGUAGE_TAGS).toContain("zh-hant");
		expect(AUTORAG_LANGUAGE_TAGS).not.toContain("kr");
	});
});

describe("parseLanguageList", () => {
	it("splits a comma separated CLI/env value", () => {
		expect(parseLanguageList("ko,en")).toEqual(["ko", "en"]);
		expect(parseLanguageList(" ja , zh-Hans ")).toEqual(["ja", "zh-hans"]);
	});

	it("rejects a blank value so a typo never silently falls back to the default", () => {
		expect(() => parseLanguageList("")).toThrow(LanguageError);
		expect(() => parseLanguageList(" , ")).toThrow(LanguageError);
	});
});

import { describe, expect, it } from "vitest";
import type { LanguageTag } from "../../src/language.ts";
import { type MinSyncLexicalLanguage, minSyncLexicalLanguage } from "../../src/minsync/lexical-language.ts";

describe("minSyncLexicalLanguage", () => {
	const cases: ReadonlyArray<readonly [readonly LanguageTag[], MinSyncLexicalLanguage]> = [
		// Korean is a primary corpus language: it selects the morphological analyzer even next to English.
		[["ko"], "ko"],
		[["ko", "en"], "ko"],
		[["en", "ko"], "ko"],
		[["ja"], "ja"],
		[["zh-hans"], "zh"],
		[["zh-hant", "en"], "zh"],
		[["ar"], "ar"],
		// Both Chinese tags share one tokenizer, so they are still a single preset.
		[["zh-hans", "zh-hant"], "zh"],
		// Languages MinSync has no analyzer for fall back to whitespace tokenization.
		[["en"], "simple"],
		[["en", "fr", "de"], "simple"],
		[["ru", "hi", "th", "vi"], "simple"],
		// Several analyzed scripts need the combined preset.
		[["ko", "ja"], "multilingual"],
		[["ko", "zh-hans", "en"], "multilingual"],
		[["ja", "ar"], "multilingual"],
	];

	it.each(cases)("maps %j to %s", (languages, expected) => {
		expect(minSyncLexicalLanguage(languages)).toBe(expected);
	});
});

import type { LanguageTag } from "../language.ts";

/** BM25 tokenizer presets MinSync accepts for `[lexical].language` (`minsync init --language`). */
export type MinSyncLexicalLanguage = "simple" | "ko" | "ja" | "zh" | "ar" | "multilingual";

/**
 * The MinSync analyzer for each AutoRAG language tag that has one. Every other tag (en, fr, de, ...) is split on
 * whitespace by the `simple` preset, which is also all MinSync offers for them.
 */
const ANALYZER_BY_LANGUAGE: Readonly<
	Partial<Record<LanguageTag, Exclude<MinSyncLexicalLanguage, "simple" | "multilingual">>>
> = {
	ko: "ko",
	ja: "ja",
	"zh-hans": "zh",
	"zh-hant": "zh",
	ar: "ar",
};

/**
 * Chooses MinSync's BM25 tokenizer from the global `languages` setting. Without an analyzer, Korean text is split on
 * whitespace only, so a particle-suffixed word (인증서를) never matches its stem (인증서).
 *
 * One analyzed script selects that analyzer: each preset also indexes Latin words, so `["ko", "en"]` is `ko`. Several
 * analyzed scripts select `multilingual`, which runs every analyzer over every document and so costs index size and
 * adds some cross-script noise; it is used only when the configured languages actually need more than one.
 */
export function minSyncLexicalLanguage(languages: readonly LanguageTag[]): MinSyncLexicalLanguage {
	const analyzers = new Set<MinSyncLexicalLanguage>();
	for (const language of languages) {
		const analyzer = ANALYZER_BY_LANGUAGE[language];
		if (analyzer !== undefined) analyzers.add(analyzer);
	}
	if (analyzers.size === 0) return "simple";
	if (analyzers.size === 1) return [...analyzers][0] as MinSyncLexicalLanguage;
	return "multilingual";
}

/** Whether MinSync's tokenizer for `language` loads the Kiwi native library (`ko`, and `multilingual` which includes it). */
export function requiresKiwi(language: MinSyncLexicalLanguage): boolean {
	return language === "ko" || language === "multilingual";
}

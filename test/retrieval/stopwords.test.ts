import { describe, expect, it } from "vitest";
import { removeQueryStopwords, STOPWORD_LANGUAGES } from "../../src/retrieval/stopwords.ts";

describe("removeQueryStopwords: Latin-script languages are detected from the query itself", () => {
	it("drops English question words, auxiliaries, pronouns and prepositions, keeping order and casing", () => {
		expect(removeQueryStopwords(["When", "did", "I", "read", "about", "the", "Inverted", "index"])).toEqual([
			"read",
			"Inverted",
			"index",
		]);
		expect(removeQueryStopwords(["Where", "DID", "I", "read", "Altavista"])).toEqual(["read", "Altavista"]);
	});

	it("handles German, French, Spanish, Italian and Portuguese", () => {
		expect(removeQueryStopwords(["wo", "habe", "ich", "über", "datenbanken", "gelesen"])).toEqual([
			"datenbanken",
			"gelesen",
		]);
		expect(
			removeQueryStopwords(["où", "ai", "je", "lu", "des", "articles", "sur", "les", "bases", "de", "données"]),
		).toEqual(["lu", "articles", "bases", "données"]);
		expect(removeQueryStopwords(["donde", "leí", "sobre", "el", "índice"])).toEqual(["leí", "índice"]);
		expect(removeQueryStopwords(["dove", "ho", "letto", "dell", "indice", "del", "database"])).toEqual([
			"letto",
			"indice",
			"database",
		]);
		expect(removeQueryStopwords(["onde", "li", "sobre", "o", "índice", "do", "banco"])).toEqual([
			"li",
			"índice",
			"banco",
		]);
	});

	it("matches a stopword typed without its accent", () => {
		expect(removeQueryStopwords(["donde", "lei", "sobre", "las", "bases", "de", "datos"])).toEqual([
			"lei",
			"bases",
			"datos",
		]);
		expect(removeQueryStopwords(["où", "est", "la", "recette"])).toEqual(["recette"]);
		expect(removeQueryStopwords(["ou", "est", "la", "recette"])).toEqual(["recette"]);
	});

	it("needs two stopwords of one language, so a word that is a stopword elsewhere survives", () => {
		// `die` is German "the"; `war` is German "was"; `son` is French "his"; `come` is Italian "how".
		expect(removeQueryStopwords(["die", "hard"])).toEqual(["die", "hard"]);
		expect(removeQueryStopwords(["war", "in", "ukraine"])).toEqual(["war", "in", "ukraine"]);
		expect(removeQueryStopwords(["son", "of", "anarchy"])).toEqual(["son", "of", "anarchy"]);
		expect(removeQueryStopwords(["come", "together", "lyrics"])).toEqual(["come", "together", "lyrics"]);
	});

	it("never applies another language's list to a query that is clearly in one language", () => {
		// German `die`/`war` are English content words once the query is English.
		expect(removeQueryStopwords(["when", "did", "the", "war", "end", "and", "who", "did", "die"])).toEqual([
			"war",
			"end",
			"die",
		]);
		// English `was` / `will` / `do` are German words, but this query is German.
		expect(removeQueryStopwords(["was", "ist", "der", "invertierte", "index"])).toEqual(["invertierte", "index"]);
	});
});

describe("removeQueryStopwords: scripts that identify their language need no detection", () => {
	it("Korean question and request forms", () => {
		expect(removeQueryStopwords(["어디서", "읽었던", "인증서를", "찾아줘"])).toEqual(["읽었던", "인증서를"]);
		expect(removeQueryStopwords(["역색인", "에", "대해", "알려줘"])).toEqual(["역색인"]);
	});

	it("Japanese and Chinese", () => {
		expect(removeQueryStopwords(["転置インデックス", "について", "教えて"])).toEqual(["転置インデックス"]);
		expect(removeQueryStopwords(["什么", "是", "倒排索引"])).toEqual(["倒排索引"]);
		expect(removeQueryStopwords(["什麼", "是", "倒排索引"])).toEqual(["倒排索引"]);
	});

	it("Russian, Arabic, Hindi, Thai and Vietnamese", () => {
		expect(removeQueryStopwords(["где", "я", "читал", "про", "индекс"])).toEqual(["читал", "индекс"]);
		expect(removeQueryStopwords(["أين", "قرأت", "عن", "الفهرس"])).toEqual(["قرأت", "الفهرس"]);
		expect(removeQueryStopwords(["क्या", "है", "इंडेक्स"])).toEqual(["इंडेक्स"]);
		expect(removeQueryStopwords(["อะไร", "คือ", "ดัชนี"])).toContain("ดัชนี");
		expect(removeQueryStopwords(["tôi", "đã", "đọc", "về", "chỉ", "mục", "ở", "đâu"])).toEqual(["đọc", "chỉ", "mục"]);
	});

	it("mixes scripts: each token is judged by its own script", () => {
		expect(removeQueryStopwords(["nogon", "검색", "어떻게", "하는지"])).toEqual(["nogon", "하는지"]);
		expect(removeQueryStopwords(["where", "did", "I", "read", "인증서", "에", "대해"])).toEqual(["read", "인증서"]);
	});
});

describe("removeQueryStopwords: it never empties or distorts a query", () => {
	it("keeps the original terms when every one is a stopword", () => {
		expect(removeQueryStopwords(["the", "of", "and"])).toEqual(["the", "of", "and"]);
		expect(removeQueryStopwords(["어디", "어떻게"])).toEqual(["어디", "어떻게"]);
		expect(removeQueryStopwords(["was", "ist", "das"])).toEqual(["was", "ist", "das"]);
	});

	it("leaves one-word and empty queries alone", () => {
		expect(removeQueryStopwords(["the"])).toEqual(["the"]);
		expect(removeQueryStopwords(["where"])).toEqual(["where"]);
		expect(removeQueryStopwords([])).toEqual([]);
	});

	it("leaves wildcard terms and terms with digits or underscores alone", () => {
		expect(removeQueryStopwords(["wh*", "the", "is", "foo_bar", "2024"])).toEqual(["wh*", "foo_bar", "2024"]);
	});

	it("exports the languages it covers, which are the languages AutoRAG supports", () => {
		expect([...STOPWORD_LANGUAGES].sort()).toEqual(
			["ar", "de", "en", "es", "fr", "hi", "it", "ja", "ko", "pt", "ru", "th", "vi", "zh"].sort(),
		);
	});
});

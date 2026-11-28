/**
 * Query stopwords for keyword (lexical) retrieval, in the languages AutoRAG supports.
 *
 * A keyword engine ranks by term weight, and a natural-language question ("where did I read about the inverted
 * index") puts function words next to the one word that matters. When the question is sent as alternatives, the
 * function words match nearly every page, so they swamp the rare term: on a 1,600-page Nogon index the page that
 * contained the rare word ranked 24th. Dropping them before the query is built removes that noise.
 *
 * The lists are curated function words (articles, pronouns, prepositions, auxiliaries, question words, and in Korean
 * and Japanese the request forms "찾아줘" / "教えて"), not a full linguistics stopword set. They are not a dependency:
 * a new runtime dependency needs its own review.
 *
 * Two rules keep this safe:
 * - A script that identifies its language (Hangul, kana, Han, Cyrillic, Arabic, Devanagari, Thai) is filtered token
 *   by token against that language's list.
 * - Latin script does not: German `die` / `war` and French `son` are ordinary English words. So one Latin language is
 *   chosen from the query as a whole, only when at least two of its unambiguous stopwords appear, and only that
 *   language's list is applied. A short or mixed query that proves nothing is left alone.
 *
 * A query is never emptied: if every term is a stopword, the original terms are returned.
 */

export type StopwordLanguage =
	| "ar"
	| "de"
	| "en"
	| "es"
	| "fr"
	| "hi"
	| "it"
	| "ja"
	| "ko"
	| "pt"
	| "ru"
	| "th"
	| "vi"
	| "zh";

const words = (list: string): string[] => list.split(/\s+/u).filter((word) => word.length > 0);

const EN = words(
	"a an the and or but if of at by for with about against between into through during before after above below to from up down in out on off over under " +
		"again further then once here there when where why how all any both each few more most other some such no nor not only own same so than too very " +
		"can will just should now i me my myself we our ours you your yours he him his she her hers it its they them their theirs what which who whom " +
		"this that these those am is are was were be been being have has had having do does did doing would could shall might must may also please via " +
		"s t d ll re ve m don doesn didn isn wasn aren weren won wouldn couldn shouldn haven hasn hadn",
);

const FR = words(
	"le la les l un une des du de d et ou où mais si que qu qui quoi dont quand comment pourquoi quel quelle quels quelles ce cet cette ces ça cela " +
		"mon ma mes ton ta tes son sa ses notre nos votre vos leur leurs je j tu il elle on nous vous ils elles me m te t se s lui y en à au aux avec sans " +
		"sur sous dans par pour vers chez entre est suis es sommes êtes sont été être avoir ai as a avons avez ont avait avais avez n ne pas plus très " +
		"aussi alors donc car ni c",
);

const DE = words(
	"der die das den dem des ein eine einen einem einer eines und oder aber wenn dass daß weil als wie wo wann warum was wer wen wem welche welcher " +
		"welches ich du er sie es wir ihr mein dein sein unser euer mich dich sich mir dir ihm ihn uns euch ist sind war waren bin bist hat haben habe " +
		"hatte hast wird werden wurde kann können muss müssen soll nicht kein keine auch noch nur sehr schon zu zum zur von vom mit nach bei aus auf an " +
		"in im um für über unter vor durch gegen ohne bis",
);

const ES = words(
	"el la los las un una unos unas de del al y o u pero si que quién quien qué cuál cuándo dónde donde cómo como por porque para con sin sobre " +
		"entre desde hasta en a mi tu su mis tus sus nuestro este esta estos estas ese esa eso esto aquel yo tú él ella nosotros ellos ellas me te se lo " +
		"le les nos es son era eran fue ser estar está están he ha han hay había muy también más menos ya no",
);

const IT = words(
	"il lo la i gli le un uno una di del dello della dei degli delle dell all alla alle ai agli al a da dal dallo dalla dalle in nel nello nella nei " +
		"nelle con su sul sulla sulle per tra fra e ed o ma se che chi cosa cui dove quando come perché perche quale quali questo questa questi queste " +
		"quello quella io tu lui lei noi voi loro mi ti ci vi si è sono era erano essere ha hanno ho hai abbiamo non più molto anche l d un",
);

const PT = words(
	"o a os as um uma uns umas de do da dos das em no na nos nas por pelo pela pelos pelas para com sem sobre entre até e ou mas se que quem qual " +
		"quais onde quando como porque eu tu ele ela nós vós eles elas me te lhe meu minha meus minhas seu sua seus suas este esta esse essa isso isto " +
		"aquele é são era foi ser estar está estão há tem têm muito também mais menos já não",
);

const VI = words(
	"và của là có không được cho một những các trong để với này đó khi đã sẽ cũng như tôi bạn anh chị em họ gì nào đâu sao thế ai bao giờ ở tại " +
		"từ đến về rất hay hoặc nhưng nếu thì mà bị",
);

const RU = words(
	"и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее её мне было вот от меня еще ещё нет о из ему теперь " +
		"когда даже ну ли если уже или ни быть был него до вас там потом себя ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без " +
		"тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее неё сейчас были куда зачем " +
		"всех при два об другой после над больше тот через эти нас про всего них какая много разве три эту моя этой перед том такой им более всегда " +
		"конечно всю между",
);

const AR = words(
	"في من على إلى الى عن مع هذا هذه ذلك تلك التي الذي الذين هو هي هم هن أنا انا نحن أنت انت ما ماذا أين اين متى كيف لماذا هل كان كانت يكون لا لم لن " +
		"قد أو او و ثم لكن إذا اذا إن ان أن كل بعد قبل بين حتى منذ",
);

const HI = words(
	"का की के में से को पर और या है हैं था थी थे यह वह ये वे मैं तुम हम आप क्या कहाँ कहां कब कैसे क्यों कौन कि जो भी नहीं एक साथ लिए तो ही इस उस",
);

const TH = words(
	"ที่ และ ของ ใน เป็น มี ไม่ ได้ ให้ จะ ว่า กับ แต่ ก็ ไป มา นี้ นั้น อะไร ที่ไหน เมื่อไร เมื่อไหร่ อย่างไร ทำไม ใคร คือ หรือ ซึ่ง โดย",
);

const KO = words(
	"어디 어디서 어디에 어디에서 무엇 뭐 뭘 누가 누구 언제 왜 어떻게 어떤 어느 얼마나 찾아 찾아줘 찾아줘요 찾아주세요 알려 알려줘 알려줘요 알려주세요 " +
		"보여줘 보여줘요 보여주세요 검색 검색해줘 검색해줘요 검색해주세요 관련 대한 대해 대해서 관한 관해 있는 있는지 있어 있나 있나요 했던 했는지 " +
		"이거 그거 저거 이것 그것 저것 이런 그런 저런 그리고 그런데 하지만 그래서 또는 및 등 좀 것 수 더 잘 매우 너무 정말 제발 나 내 저 제 우리 " +
		"은 는 이 가 을 를 의 에 에서 로 으로 와 과 도 만 부터 까지 하고 이랑",
);

const JA = words(
	"の は が を に で と も や か へ から まで より です ます でした ました する した して いる ある これ それ あれ この その あの ここ そこ どこ なに " +
		"何 いつ どう どの どれ なぜ だれ 誰 ください 下さい 教えて 教えてください 探して 見せて 検索 について に関する に関して とは って という",
);

const ZH = words(
	"的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己 这 那 什么 怎么 哪里 哪 谁 为什么 吗 呢 吧 请 帮 查找 搜索 关于 " +
		"說 會 著 沒有 這 什麼 怎麼 哪裡 誰 為什麼 嗎 請 幫 搜尋 關於 我們 你們 他們 它",
);

/**
 * Words that are ordinary English content words. A non-English list that holds one (German `die`, `war`; French
 * `son`; Italian `come`) must not count it as evidence that the query is in that language.
 */
const ENGLISH_CONTENT_WORDS = new Set(
	words("die war son come once bin hat kind gift art pain sale rat fin rest bad".replace(/\s+/gu, " ")),
);

const LATIN_ORDER = ["en", "fr", "de", "es", "it", "pt", "vi"] as const;
type LatinLanguage = (typeof LATIN_ORDER)[number];

/**
 * Lowercased, NFKC; Latin letters also lose their accents so `donde` matches `dónde`. Not used for Vietnamese, where
 * the tone marks tell words apart (`chỉ` "only" and `chị` "older sister" would collapse into one).
 */
function fold(token: string): string {
	return token.normalize("NFKC").toLowerCase().normalize("NFD").replace(/\p{M}/gu, "").normalize("NFC");
}

function lower(token: string): string {
	return token.normalize("NFKC").toLowerCase();
}

const foldedSet = (list: readonly string[]): ReadonlySet<string> => new Set(list.map(fold));
const plainSet = (list: readonly string[]): ReadonlySet<string> => new Set(list.map(lower));

const LATIN_LISTS: Readonly<Record<LatinLanguage, ReadonlySet<string>>> = {
	en: foldedSet(EN),
	fr: foldedSet(FR),
	de: foldedSet(DE),
	es: foldedSet(ES),
	it: foldedSet(IT),
	pt: foldedSet(PT),
	vi: plainSet(VI),
};

/** How a term is looked up in a language's list: accent-folded, except Vietnamese, which is matched exactly. */
const keyFor = (language: LatinLanguage, term: string): string => (language === "vi" ? lower(term) : fold(term));

/**
 * What counts as proof of a language: its list minus words that are also English stopwords or common English words.
 * English itself counts its whole list.
 */
const LATIN_EVIDENCE: Readonly<Record<LatinLanguage, ReadonlySet<string>>> = Object.fromEntries(
	LATIN_ORDER.map((language): [LatinLanguage, ReadonlySet<string>] => [
		language,
		language === "en"
			? LATIN_LISTS.en
			: new Set(
					[...LATIN_LISTS[language]].filter(
						(word) => !LATIN_LISTS.en.has(word) && !ENGLISH_CONTENT_WORDS.has(word),
					),
				),
	]),
) as Record<LatinLanguage, ReadonlySet<string>>;

const KO_SET = plainSet(KO);
const JA_SET = plainSet(JA);
const HAN_SET = plainSet([...ZH, ...JA.filter((word) => !/[\p{Script=Hiragana}\p{Script=Katakana}]/u.test(word))]);
const RU_SET = plainSet(RU);
const AR_SET = plainSet(AR);
const HI_SET = plainSet(HI);
const TH_SET = plainSet(TH);

export const STOPWORD_LANGUAGES: readonly StopwordLanguage[] = [
	"ar",
	"de",
	"en",
	"es",
	"fr",
	"hi",
	"it",
	"ja",
	"ko",
	"pt",
	"ru",
	"th",
	"vi",
	"zh",
];

/** A language is chosen for Latin text only with at least this many of its unambiguous stopwords in the query. */
const MIN_LATIN_EVIDENCE = 2;

const HANGUL = /\p{Script=Hangul}/u;
const KANA = /[\p{Script=Hiragana}\p{Script=Katakana}]/u;
const HAN = /\p{Script=Han}/u;
const CYRILLIC = /\p{Script=Cyrillic}/u;
const ARABIC = /\p{Script=Arabic}/u;
const DEVANAGARI = /\p{Script=Devanagari}/u;
const THAI = /\p{Script=Thai}/u;
const LATIN_WORD = /^\p{Script=Latin}+$/u;

function isScriptStopword(token: string): boolean {
	const key = lower(token);
	if (HANGUL.test(token)) return KO_SET.has(key);
	if (KANA.test(token)) return JA_SET.has(key);
	if (HAN.test(token)) return HAN_SET.has(key);
	if (CYRILLIC.test(token)) return RU_SET.has(key);
	if (ARABIC.test(token)) return AR_SET.has(key);
	if (DEVANAGARI.test(token)) return HI_SET.has(key);
	if (THAI.test(token)) return TH_SET.has(key);
	return false;
}

/** The one Latin language this query is written in, or undefined when it does not show enough to say. */
function detectLatinLanguage(terms: readonly string[]): LatinLanguage | undefined {
	const latin = terms.filter((term) => LATIN_WORD.test(term));
	if (latin.length < MIN_LATIN_EVIDENCE) return undefined;
	let best: LatinLanguage | undefined;
	let bestCount = MIN_LATIN_EVIDENCE - 1;
	for (const language of LATIN_ORDER) {
		const evidence = LATIN_EVIDENCE[language];
		const count = latin.filter((term) => evidence.has(keyFor(language, term))).length;
		if (count > bestCount) {
			best = language;
			bestCount = count;
		}
	}
	return best;
}

/**
 * `terms` without their stopwords, in the original order and casing. Returns `terms` unchanged when that would leave
 * nothing, so a question made only of function words still searches for something.
 */
export function removeQueryStopwords(terms: readonly string[]): string[] {
	const latin = detectLatinLanguage(terms);
	const kept = terms.filter((term) => {
		if (LATIN_WORD.test(term)) return latin === undefined || !LATIN_LISTS[latin].has(keyFor(latin, term));
		return !isScriptStopword(term);
	});
	return kept.length === 0 ? [...terms] : kept;
}

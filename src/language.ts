/**
 * Global language configuration for AutoRAG.
 *
 * The tag set is curated rather than free-form BCP-47: every accepted tag must
 * map to a Tesseract traineddata code (see src/parser/ocr-engines.ts), which
 * is the OCR engine for every language kordoc's built-in model cannot read.
 * A tag we cannot map would be a promise the parser stack cannot keep.
 */

export const AUTORAG_LANGUAGE_TAGS = [
	"ko",
	"en",
	"ja",
	"zh-hans",
	"zh-hant",
	"fr",
	"de",
	"es",
	"ru",
	"it",
	"pt",
	"vi",
	"th",
	"ar",
	"hi",
] as const;

export type LanguageTag = (typeof AUTORAG_LANGUAGE_TAGS)[number];

/** Korean + English: AutoRAG's primary corpus languages. */
export const DEFAULT_LANGUAGES: readonly LanguageTag[] = ["ko", "en"];

export class LanguageError extends Error {
	readonly name = "LanguageError";
}

function supportedList(): string {
	return AUTORAG_LANGUAGE_TAGS.join(", ");
}

function toTag(raw: string): LanguageTag {
	const tag = raw.trim().toLowerCase();
	if (!(AUTORAG_LANGUAGE_TAGS as readonly string[]).includes(tag)) {
		throw new LanguageError(`Unsupported language "${raw.trim()}". Supported: ${supportedList()}.`);
	}
	return tag as LanguageTag;
}

function dedupe(tags: readonly LanguageTag[]): LanguageTag[] {
	return [...new Set(tags)];
}

/**
 * Normalizes a configured language value into a non-empty, deduplicated,
 * lowercase tag list. The first entry is the primary language.
 */
export function normalizeLanguages(raw: unknown): LanguageTag[] {
	if (raw === undefined || raw === null) return [...DEFAULT_LANGUAGES];
	if (typeof raw === "string") return parseLanguageList(raw);
	if (!Array.isArray(raw)) {
		throw new LanguageError(`languages must be a string or an array of language tags; received ${typeof raw}.`);
	}
	const entries = raw.map((entry, index) => {
		if (typeof entry !== "string") {
			throw new LanguageError(`languages[${index}] must be a string language tag; received ${typeof entry}.`);
		}
		return entry;
	});
	const tags = entries.filter((entry) => entry.trim() !== "").map(toTag);
	if (tags.length === 0) {
		throw new LanguageError(`languages must list at least one language. Supported: ${supportedList()}.`);
	}
	return dedupe(tags);
}

/** Parses a comma-separated CLI flag or environment variable value. */
export function parseLanguageList(value: string): LanguageTag[] {
	const tags = value
		.split(",")
		.map((entry) => entry.trim())
		.filter((entry) => entry !== "")
		.map(toTag);
	if (tags.length === 0) {
		throw new LanguageError(`languages must list at least one language. Supported: ${supportedList()}.`);
	}
	return dedupe(tags);
}

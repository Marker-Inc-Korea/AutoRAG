/**
 * HTML entity decoding shared by the web search providers and the fetch
 * pipeline.
 *
 * Single-pass by construction: one alternation regex decodes each entity
 * exactly once, so an entity produced by decoding (`&amp;lt;` → `&lt;`) is
 * never re-decoded into markup (CodeQL js/double-escaping). Unknown entities
 * and bare ampersands pass through untouched.
 */

const ENTITY_PATTERN = /&(#x?[0-9a-f]+|[a-z][a-z0-9]+);/gi;

export function decodeHtmlEntities(value: string): string {
	return value.replace(ENTITY_PATTERN, (match, body: string) => {
		const lower = body.toLowerCase();
		if (lower.startsWith("#x")) {
			const codePoint = Number.parseInt(lower.slice(2), 16);
			return Number.isFinite(codePoint) ? String.fromCodePoint(codePoint) : match;
		}
		if (lower.startsWith("#")) {
			const codePoint = Number.parseInt(lower.slice(1), 10);
			return Number.isFinite(codePoint) ? String.fromCodePoint(codePoint) : match;
		}
		switch (lower) {
			case "amp":
				return "&";
			case "lt":
				return "<";
			case "gt":
				return ">";
			case "quot":
				return '"';
			case "apos":
				return "'";
			case "nbsp":
				return " ";
			default:
				return match;
		}
	});
}

const MAX_QUERY_LENGTH = 4096;

/**
 * The deterministic signatures shared by inbound L0 and the later outbound
 * scan. These expressions intentionally have no global/sticky flag, so callers
 * can safely reuse them without resetting RegExp.lastIndex.
 */
export const INJECTION_SIGNATURES: readonly RegExp[] = Object.freeze([
	/ignore (all |any |the )?(previous|prior|above) (instructions|prompts)/i,
	/\b(system|assistant|human)\s*:/i,
	/\[INST\]|<<SYS>>|<\|im_start\|>|<\|im_end\|>|<start_of_turn>|<end_of_turn>|<\/?s>/i,
	/\b(disregard|override|forget)\b.{0,40}\b(instructions?|rules?|prompt)/i,
	/\byou are (now )?(a|an|the)\b/i,
	/\bnew instructions?\b/i,
	/\b(jailbreak|DAN)\b/i,
]);

/** Alias for callers that refer to these as instruction signatures. */
export const INSTRUCTION_SIGNATURES = INJECTION_SIGNATURES;

const STRIPPED_FORMAT_CHARACTERS = /[\u061C\u180E\u200B-\u200F\u202A-\u202E\u2060-\u206F\uFEFF]/g;
const DISALLOWED_CONTROL_CHARACTERS = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]/;
const UNPAIRED_SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/;
const ENCODED_TOKEN = /[A-Za-z0-9+/=]{120,}/g;
const BASE64_TOKEN = /^[A-Za-z0-9+/]+={0,2}$/;
const HEX_TOKEN = /^[0-9a-f]+$/i;

const CONFUSABLES: Readonly<Record<string, string>> = {
	// Fullwidth ASCII is not changed by NFC. Fold it for detection only; the
	// canonical query returned to the caller remains NFC-normalized text.
	"\u3000": " ",
	// Common Cyrillic/Greek lookalikes used in Latin instruction words.
	а: "a",
	е: "e",
	і: "i",
	о: "o",
	р: "p",
	с: "c",
	у: "y",
	х: "x",
	ѕ: "s",
	ο: "o",
	ι: "i",
	ρ: "p",
	ν: "v",
	χ: "x",
	А: "a",
	Е: "e",
	І: "i",
	О: "o",
	Р: "p",
	С: "c",
	У: "y",
	Х: "x",
	Ѕ: "s",
	Ο: "o",
	Ι: "i",
	Ρ: "p",
	Ν: "v",
	Χ: "x",
};

export type InjectionGateResult =
	| {
			readonly ok: true;
			readonly canonicalQuery: string;
	  }
	| {
			readonly ok: false;
			readonly code: "injection-detected";
			/** An internal category or signature; never the complete query. */
			readonly matched: string;
	  };

export interface InjectionGateOptions {
	/**
	 * Optional deterministic detector seam for callers/tests. A truthy return
	 * rejects the query. Any thrown error fails closed.
	 */
	readonly detector?: (canonicalQuery: string) => unknown;
}

/** NFC-normalize input and remove invisible format/control-direction markers. */
export function canonicalizeQuery(query: string): string {
	if (typeof query !== "string") throw new TypeError("Query must be a string.");
	return query.normalize("NFC").replace(STRIPPED_FORMAT_CHARACTERS, "");
}

function foldConfusables(value: string): string {
	let folded = "";
	for (const character of value.normalize("NFKC")) {
		const codePoint = character.codePointAt(0);
		if (codePoint !== undefined && codePoint >= 0xff01 && codePoint <= 0xff5e) {
			folded += String.fromCodePoint(codePoint - 0xfee0);
			continue;
		}
		folded += CONFUSABLES[character] ?? character;
	}
	return folded;
}

function signatureIndex(value: string): number | undefined {
	for (let index = 0; index < INJECTION_SIGNATURES.length; index += 1) {
		if (INJECTION_SIGNATURES[index]?.test(value)) return index;
	}
	return undefined;
}

function detectionForm(value: string): string {
	return foldConfusables(value).replace(/\s+/g, " ");
}

function decodedText(bytes: Uint8Array): string | undefined {
	try {
		return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
	} catch {
		return undefined;
	}
}

function encodedSignatureIndex(token: string): number | undefined {
	if (BASE64_TOKEN.test(token) && token.length % 4 !== 1) {
		const unpaddedToken = token.replace(/=+$/, "");
		const paddedToken = unpaddedToken.padEnd(Math.ceil(unpaddedToken.length / 4) * 4, "=");
		const bytes = Buffer.from(paddedToken, "base64");
		if (bytes.length > 0 && bytes.toString("base64").replace(/=+$/, "") === unpaddedToken) {
			const decoded = decodedText(bytes);
			if (decoded !== undefined) {
				const canonical = canonicalizeQuery(decoded);
				const index = signatureIndex(detectionForm(canonical));
				if (index !== undefined) return index;
			}
		}
	}

	if (HEX_TOKEN.test(token) && token.length % 2 === 0) {
		const bytes = Buffer.from(token, "hex");
		const decoded = decodedText(bytes);
		if (decoded !== undefined) {
			const canonical = canonicalizeQuery(decoded);
			const index = signatureIndex(detectionForm(canonical));
			if (index !== undefined) return index;
		}
	}

	return undefined;
}

/** Detect a signature or encoded signature without applying inbound size rules. */
export function detectInjectionSignature(value: string): string | undefined {
	const detectionText = detectionForm(canonicalizeQuery(value));
	const directSignature = signatureIndex(detectionText);
	if (directSignature !== undefined) return `signature-${directSignature}`;

	ENCODED_TOKEN.lastIndex = 0;
	for (const match of detectionText.matchAll(ENCODED_TOKEN)) {
		if (encodedSignatureIndex(match[0]) !== undefined) return "encoded-signature";
	}
	return undefined;
}

export const findInjectionSignature = detectInjectionSignature;

function reject(matched: string): InjectionGateResult {
	return { ok: false, code: "injection-detected", matched };
}

/**
 * Screen an untrusted peer query before it reaches the agent loop.
 *
 * Every validation/detector exception is converted to a generic rejection.
 * The result contains only an internal category, never the offending query.
 */
export function screenInboundQuery(query: unknown, options: InjectionGateOptions = {}): InjectionGateResult {
	try {
		if (typeof query !== "string") return reject("invalid-query");
		if (query.length > MAX_QUERY_LENGTH) return reject("query-too-long");
		if (DISALLOWED_CONTROL_CHARACTERS.test(query)) return reject("control-character");
		if (UNPAIRED_SURROGATE.test(query)) return reject("malformed-unicode");

		const canonicalQuery = canonicalizeQuery(query);
		if (canonicalQuery.length > MAX_QUERY_LENGTH) return reject("query-too-long");

		const customDetection = options.detector?.(canonicalQuery);
		if (customDetection) return reject("custom-detector");

		const matched = detectInjectionSignature(canonicalQuery);
		if (matched !== undefined) return reject(matched);

		return { ok: true, canonicalQuery };
	} catch {
		return reject("detector-error");
	}
}

export const screenQuery = screenInboundQuery;

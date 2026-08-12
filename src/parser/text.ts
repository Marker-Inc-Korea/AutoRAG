import { detect } from "chardet";
import iconv from "iconv-lite";

// Text decoding and normalization applied to mirror content before BM25 chunking: the stored
// artifact contains exactly the text produced here, so a change invalidates every stored index.
const WINDOWS_949_LABELS = new Set(["EUC-KR", "ISO-2022-KR", "windows-949", "CP949"]);

export function decodeText(bytes: Uint8Array): string {
	const buffer = Buffer.from(bytes);
	if (isPureAscii(bytes)) {
		// Bytes 0x01-0x7F decode to the same code points under utf8, cp949, and every single-byte
		// encoding chardet can name, and NFC normalization is the identity on them, so the result
		// is fixed without consulting chardet. ESC (0x1B) is excluded: chardet can claim
		// ISO-2022-JP/CN for buffers containing its escape sequences, which iconv-lite cannot
		// decode, so those bytes must keep the legacy path. Buffers shorter than one 4-byte unit
		// are excluded too: chardet's UTF-32LE heuristic can claim UTF-32LE for control-character
		// and DEL bytes, and iconv-lite then drops the incomplete trailing unit, making the legacy
		// output "". Pure printable ASCII decodes normally ("A"/"AB"/"ABC"), so excluding every
		// sub-4-byte buffer is just the conservative choice and must keep the legacy path.
		return iconv.decode(buffer, "ascii");
	}
	const utf8 = iconv.decode(buffer, "utf8");
	if (replacementCount(utf8) > 0) {
		const cp949 = iconv.decode(buffer, "cp949");
		if (replacementCount(cp949) < replacementCount(utf8)) return cp949.normalize("NFC");
	}
	const detected = detect(buffer);
	const encoding = detected && WINDOWS_949_LABELS.has(detected) ? "cp949" : (detected ?? "utf8");
	return iconv.decode(buffer, encoding).normalize("NFC");
}

export function normalizeMarkdown(markdown: string): string {
	return markdown.normalize("NFC");
}

function replacementCount(value: string): number {
	return [...value].filter((character) => character === "\uFFFD").length;
}

function isPureAscii(bytes: Uint8Array): boolean {
	if (bytes.length === 0) return true;
	if (bytes.length < 4) return false;
	for (let i = 0; i < bytes.length; i += 1) {
		const byte = bytes[i] as number;
		if (byte < 0x01 || byte === 0x1b || byte > 0x7f) return false;
	}
	return true;
}

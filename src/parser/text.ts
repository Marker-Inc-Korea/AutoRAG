import { detect } from "chardet";
import iconv from "iconv-lite";

const WINDOWS_949_LABELS = new Set(["EUC-KR", "ISO-2022-KR", "windows-949", "CP949"]);

export function decodeText(bytes: Uint8Array): string {
	const buffer = Buffer.from(bytes);
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

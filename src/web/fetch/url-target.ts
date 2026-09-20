/**
 * URL normalization, MIME classification, and binary detection — ported from
 * oh-my-pi's `packages/coding-agent/src/tools/fetch.ts` (MIT licensed).
 */

import { extname } from "node:path";

/**
 * MIME types that AutoRAG's parser registry can convert (PDF, DOCX, PPTX,
 * XLSX). EPUB has no registered parser and falls through to the binary-notice
 * fallback.
 */
const CONVERTIBLE_MIMES = new Set([
	"application/pdf",
	"application/vnd.openxmlformats-officedocument.wordprocessingml.document",
	"application/vnd.openxmlformats-officedocument.presentationml.presentation",
	"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
]);

const CONVERTIBLE_EXTENSIONS = new Set([".pdf", ".docx", ".pptx", ".xlsx"]);

/**
 * Repair a URL whose scheme `//` collapsed to a single `/`.
 */
function repairCollapsedScheme(value: string): string {
	const m = value.match(/^(https?):\/(?!\/)/i);
	return m ? `${m[1]}://${value.slice(m[0].length)}` : value;
}

/**
 * Normalize a URL: repair a collapsed scheme, then add `https://` if missing.
 */
export function normalizeUrl(url: string): string {
	url = repairCollapsedScheme(url);
	if (!url.match(/^https?:\/\//i)) {
		return `https://${url}`;
	}
	return url;
}

/**
 * Normalize a MIME type (lowercase, strip charset/params).
 */
export function normalizeMime(contentType: string): string {
	return contentType.split(";")[0].trim().toLowerCase();
}

export function getFilenameExtensionHint(filename: string): string {
	const lower = filename.toLowerCase();
	if (lower.endsWith(".tar.gz")) return ".tar.gz";
	return extname(filename).toLowerCase();
}

/**
 * Get extension from URL or Content-Disposition.
 */
export function getExtensionHint(url: string, contentDisposition?: string): string {
	if (contentDisposition) {
		const match = contentDisposition.match(/filename[*]?=["']?([^"';\n]+)/i);
		if (match) {
			const ext = getFilenameExtensionHint(match[1]);
			if (ext) return ext;
		}
	}
	try {
		const pathname = new URL(url).pathname;
		const ext = getFilenameExtensionHint(pathname);
		if (ext) return ext;
	} catch {}
	return "";
}

/**
 * Check if content type is convertible via AutoRAG's parser registry.
 */
export function isConvertible(mime: string, extensionHint: string): boolean {
	if (CONVERTIBLE_MIMES.has(mime)) return true;
	if (mime === "application/octet-stream" && CONVERTIBLE_EXTENSIONS.has(extensionHint)) return true;
	if (CONVERTIBLE_EXTENSIONS.has(extensionHint)) return true;
	return false;
}

const BINARY_SAMPLE_CHARS = 4096;

/**
 * Heuristic: does the text sample look like binary (NUL bytes or high U+FFFD density)?
 */
export function sampleLooksBinary(text: string): boolean {
	const limit = Math.min(text.length, BINARY_SAMPLE_CHARS);
	if (limit === 0) return false;

	let replacementCount = 0;
	for (let index = 0; index < limit; index++) {
		const code = text.charCodeAt(index);
		if (code === 0) return true;
		if (code === 0xfffd) replacementCount++;
	}

	return replacementCount >= 3 && replacementCount / limit > 0.01;
}

function formatBytes(bytes: number): string {
	if (bytes < 1024) return `${bytes} B`;
	if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
	return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

function binaryContentType(mime: string): string {
	return mime || "application/octet-stream";
}

/**
 * Build a textual binary notice for unsupported binary payloads.
 */
export function buildBinaryNotice(finalUrl: string, mime: string, byteLength?: number): string {
	const size = byteLength === undefined ? "unknown size" : formatBytes(byteLength);
	return `[Binary content: ${binaryContentType(mime)}, ${size}] ${finalUrl}`;
}

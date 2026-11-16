import { createHash } from "node:crypto";
import { readFileSync, rmSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import type { PageSpan, ParsedPages } from "../parser/pages.ts";
import { writeFileAtomic } from "./atomic-write.ts";

const SIDECAR_VERSION = 1;

interface PagesSidecar {
	readonly version: typeof SIDECAR_VERSION;
	/** SHA-256 of the markdown the spans were measured on; a sidecar for any other text is ignored. */
	readonly markdownSha256: string;
	readonly pages: ParsedPages;
}

/**
 * The page index of a parsed mirror entry: `parsed/files/<hash>.md` → `parsed/pages/<hash>.json`. It sits
 * beside `files/`, not in it: that directory is MinSync's stage output root and holds only parsed markdown.
 */
export function pagesSidecarPath(mirrorPath: string): string {
	return join(dirname(dirname(mirrorPath)), "pages", `${basename(mirrorPath).replace(/\.md$/u, "")}.json`);
}

function digest(markdown: string): string {
	return createHash("sha256").update(markdown).digest("hex");
}

/**
 * Persist the page index measured on `markdown`, the exact text written to `mirrorPath`; no pages removes
 * a previous sidecar so a re-parse that lost its pages never leaves the old ones behind.
 */
export function writePagesSidecar(mirrorPath: string, markdown: string, pages: ParsedPages | undefined): void {
	const path = pagesSidecarPath(mirrorPath);
	if (pages === undefined) {
		rmSync(path, { force: true });
		return;
	}
	const sidecar: PagesSidecar = { version: SIDECAR_VERSION, markdownSha256: digest(markdown), pages };
	writeFileAtomic(path, JSON.stringify(sidecar));
}

function isPositiveInteger(value: unknown): value is number {
	return typeof value === "number" && Number.isInteger(value) && value > 0;
}

function isSpan(value: unknown): value is PageSpan {
	if (typeof value !== "object" || value === null) return false;
	const span = value as Record<string, unknown>;
	return (
		isPositiveInteger(span.page) &&
		isPositiveInteger(span.lineStart) &&
		isPositiveInteger(span.lineEnd) &&
		span.lineEnd >= span.lineStart
	);
}

function parsePages(value: unknown): ParsedPages | undefined {
	if (typeof value !== "object" || value === null) return undefined;
	const pages = value as Record<string, unknown>;
	if (pages.mode !== "layout" && pages.mode !== "section") return undefined;
	if (!Array.isArray(pages.spans) || pages.spans.length === 0 || !pages.spans.every(isSpan)) return undefined;
	const spans = pages.spans as PageSpan[];
	for (let index = 1; index < spans.length; index++) {
		const previous = spans[index - 1] as PageSpan;
		const current = spans[index] as PageSpan;
		if (current.page <= previous.page || current.lineStart < previous.lineEnd) return undefined;
	}
	const pageCount = isPositiveInteger(pages.pageCount) ? pages.pageCount : undefined;
	return { mode: pages.mode, ...(pageCount !== undefined ? { pageCount } : {}), spans };
}

/**
 * The page index for the mirror entry whose text is `markdown`. Undefined when there is none, it is
 * unreadable or malformed, or it was measured on different text: a page is never reported from a guess.
 */
export function readPagesSidecar(mirrorPath: string, markdown: string): ParsedPages | undefined {
	let raw: unknown;
	try {
		raw = JSON.parse(readFileSync(pagesSidecarPath(mirrorPath), "utf8"));
	} catch {
		return undefined;
	}
	if (typeof raw !== "object" || raw === null) return undefined;
	const sidecar = raw as Record<string, unknown>;
	if (sidecar.version !== SIDECAR_VERSION || sidecar.markdownSha256 !== digest(markdown)) return undefined;
	return parsePages(sidecar.pages);
}

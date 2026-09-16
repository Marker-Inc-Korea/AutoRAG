/**
 * RSS/Atom feed parsing and JSON formatting — ported from oh-my-pi's
 * `packages/coding-agent/src/tools/fetch.ts` (MIT licensed).
 *
 * Feed parsing uses `fast-xml-parser` (already an AutoRAG dependency) instead
 * of oh-my-pi's `parseHTML`/DOM utils, which are not available here.
 */

import { extname } from "node:path";
import { XMLParser } from "fast-xml-parser";
import { decodeHtmlEntities } from "../entities.ts";

/**
 * Read a single HTML attribute from a tag string.
 */
export function getHtmlAttribute(tag: string, attribute: string): string | null {
	const pattern = new RegExp(`\\b${attribute}\\s*=\\s*(?:"([^"]*)"|'([^']*)'|([^\\s"'=<>]+))`, "i");
	const match = tag.match(pattern);
	if (!match) return null;
	return (match[1] ?? match[2] ?? match[3] ?? "").trim();
}

/**
 * Extract bounded `<head>` markup to avoid expensive whole-page parsing.
 */
export function extractHeadHtml(html: string): string {
	const SCAN_LIMIT = 256 * 1024;
	const window = html.length > SCAN_LIMIT ? html.slice(0, SCAN_LIMIT) : html;
	const headStart = window.search(/<head[\s>]/i);
	if (headStart === -1) {
		return html.slice(0, 32 * 1024);
	}

	const headTagEnd = html.indexOf(">", headStart);
	if (headTagEnd === -1 || headTagEnd - headStart > 4096) {
		return html.slice(headStart, headStart + 32 * 1024);
	}

	const tail = html.slice(headTagEnd + 1, headTagEnd + 1 + 128 * 1024);
	const relativeEnd = tail.search(/<\/head\s*>/i);
	if (relativeEnd === -1) {
		return html.slice(headStart, headTagEnd + 1 + tail.length);
	}
	return html.slice(headStart, headTagEnd + 1 + relativeEnd + 7);
}

/**
 * Parse alternate links from HTML head.
 */
export function parseAlternateLinks(html: string, pageUrl: string): string[] {
	const links: string[] = [];

	try {
		const pagePath = new URL(pageUrl).pathname;
		const headHtml = extractHeadHtml(html);
		const linkTags = headHtml.match(/<link\b[^>]*>/gi) ?? [];

		for (const tag of linkTags) {
			const rel = getHtmlAttribute(tag, "rel")?.toLowerCase() ?? "";
			const relTokens = rel.split(/\s+/).filter(Boolean);
			if (!relTokens.includes("alternate")) continue;

			const href = getHtmlAttribute(tag, "href");
			const type = getHtmlAttribute(tag, "type")?.toLowerCase() ?? "";
			if (!href) continue;

			if (
				href.includes("RecentChanges") ||
				href.includes("Special:") ||
				href.includes("/feed/") ||
				href.includes("action=feed")
			) {
				continue;
			}

			if (type.includes("markdown")) {
				links.push(href);
			} else if (
				(type.includes("rss") || type.includes("atom") || type.includes("feed")) &&
				(href.includes(pagePath) || href.includes("comments"))
			) {
				links.push(href);
			}
		}
	} catch {}

	return links;
}

const CONVERTIBLE_EXTENSIONS = new Set([".pdf", ".docx", ".pptx", ".xlsx"]);

/**
 * Extract document links from HTML (for PDF/DOCX wrapper pages).
 */
export function extractDocumentLinks(html: string, baseUrl: string): string[] {
	const links: string[] = [];
	const seen = new Set<string>();

	try {
		const anchorTags = html.slice(0, 512 * 1024).match(/<a\b[^>]*>/gi) ?? [];
		for (const tag of anchorTags) {
			const href = getHtmlAttribute(tag, "href");
			if (!href) continue;

			const ext = extname(href).toLowerCase();
			if (!CONVERTIBLE_EXTENSIONS.has(ext)) continue;

			const resolved = href.startsWith("http") ? href : new URL(href, baseUrl).href;
			if (seen.has(resolved)) continue;
			seen.add(resolved);
			links.push(resolved);
			if (links.length >= 20) break;
		}
	} catch {}

	return links;
}

/**
 * Strip a CDATA wrapper and clean text. Only a full-string wrapper is
 * unwrapped (anchored): a stray `]]>` inside content is data, not markup.
 * Entity decoding is single-pass (see `../entities.ts`).
 */
export function cleanFeedText(text: string): string {
	const trimmed = text.trim();
	const cdata = /^<!\[CDATA\[([\s\S]*?)\]\]>$/.exec(trimmed);
	const inner = cdata ? (cdata[1] ?? "") : trimmed;
	// Tag stripping runs to a fixed point: removing one tag may reveal another
	// (`<scr<script>ipt>`), so a single global pass is insufficient (CodeQL
	// js/incomplete-multi-character-sanitization). Each pass strictly shrinks
	// the string, so the loop terminates; the bound is defense in depth.
	let stripped = decodeHtmlEntities(inner);
	for (let pass = 0; pass < 100 && /<[^>]+>/.test(stripped); pass++) {
		stripped = stripped.replace(/<[^>]+>/g, "");
	}
	return stripped.trim();
}

interface FeedNode {
	title?: string;
	link?: string;
	href?: string;
	pubDate?: string;
	updated?: string;
	description?: string;
	summary?: string;
	content?: string;
}

function textOf(value: unknown): string {
	if (typeof value === "string") return value;
	if (value && typeof value === "object" && "#text" in value) {
		return String((value as Record<string, unknown>)["#text"] ?? "");
	}
	return "";
}

/**
 * Parse RSS/Atom feed XML to markdown using fast-xml-parser.
 */
export function parseFeedToMarkdown(content: string, maxItems = 10): string {
	try {
		const parser = new XMLParser({ ignoreAttributes: true, trimValues: true });
		const doc = parser.parse(content) as Record<string, unknown>;

		const root = doc.rss ?? doc.feed ?? doc["rdf:RDF"];
		if (!root || typeof root !== "object") return content;

		// RSS
		const rssRoot = root as Record<string, unknown>;
		const channel = rssRoot.channel;
		if (channel && typeof channel === "object") {
			const ch = channel as Record<string, unknown>;
			const title = cleanFeedText(textOf(ch.title) || "RSS Feed");
			const items = (ch.item ?? []) as FeedNode | FeedNode[];
			const itemList = Array.isArray(items) ? items : [items];

			let md = `# ${title}\n\n`;
			for (const item of itemList.slice(0, maxItems)) {
				const itemTitle = cleanFeedText(textOf(item.title) || "Untitled");
				const link = cleanFeedText(textOf(item.link) || "");
				const pubDate = cleanFeedText(textOf(item.pubDate) || "");
				const desc = cleanFeedText(textOf(item.description) || "");

				md += `## ${itemTitle}\n`;
				if (pubDate) md += `*${pubDate}*\n\n`;
				if (desc) md += `${desc.slice(0, 500)}${desc.length > 500 ? "..." : ""}\n\n`;
				if (link) md += `[Read more](${link})\n\n`;
				md += "---\n\n";
			}
			return md;
		}

		// Atom
		const feed = root as Record<string, unknown>;
		const title = cleanFeedText(textOf(feed.title) || "Atom Feed");
		const entries = (feed.entry ?? []) as FeedNode | FeedNode[];
		const entryList = Array.isArray(entries) ? entries : [entries];

		let md = `# ${title}\n\n`;
		for (const entry of entryList.slice(0, maxItems)) {
			const entryTitle = cleanFeedText(textOf(entry.title) || "Untitled");
			const link = cleanFeedText(entry.href || textOf(entry.link) || "");
			const updated = cleanFeedText(textOf(entry.updated) || "");
			const summary = cleanFeedText(textOf(entry.summary) || textOf(entry.content) || "");

			md += `## ${entryTitle}\n`;
			if (updated) md += `*${updated}*\n\n`;
			if (summary) md += `${summary.slice(0, 500)}${summary.length > 500 ? "..." : ""}\n\n`;
			if (link) md += `[Read more](${link})\n\n`;
			md += "---\n\n";
		}
		return md;
	} catch {}

	return content;
}

/**
 * Pretty-print JSON content.
 */
export function formatJson(content: string): string {
	try {
		return JSON.stringify(JSON.parse(content), null, 2);
	} catch {
		return content;
	}
}

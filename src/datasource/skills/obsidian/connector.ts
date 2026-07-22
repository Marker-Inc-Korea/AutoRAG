/**
 * Obsidian vault connector (issue #1314).
 *
 * Walks a trusted, server-configured vault directory for markdown notes,
 * extracting frontmatter/inline tags, wiki links, and embeds. No network
 * access. Per-file read failures aggregate into a count-only warning;
 * absolute paths never appear in warnings (vault-relative segments are the
 * vault's own logical hierarchy and are allowed in doc ids).
 */

import type { Dirent } from "node:fs";
import { readdir, readFile, stat } from "node:fs/promises";
import { basename, extname, join, relative, sep } from "node:path";
import type { ConnectorDocument, ConnectorFetchResult, DatasourceConnector } from "../../connector.ts";

export interface ObsidianConnectorOptions {
	/** Absolute path to the vault root. Required trusted configuration. */
	readonly vaultPath?: string;
	readonly maxDocuments?: number;
	/** Files larger than this are skipped. Default 2 MiB. */
	readonly maxBytesPerFile?: number;
}

const DEFAULT_MAX_DOCUMENTS = 1000;
const DEFAULT_MAX_BYTES_PER_FILE = 2 * 1024 * 1024;
const MAX_CONTENT_CHARS = 100_000;
const SKIPPED_DIRS = new Set([".obsidian", ".trash", ".git", "node_modules"]);
const INLINE_TAG_PATTERN = /(?:^|\s)#([A-Za-z0-9_/-]+)/gu;
const WIKI_LINK_PATTERN = /(!?)\[\[([^\]|]+)(?:\|[^\]]*)?\]\]/gu;

export class ObsidianConnector implements DatasourceConnector {
	private readonly options: ObsidianConnectorOptions;

	constructor(options: ObsidianConnectorOptions = {}) {
		this.options = options;
	}

	async fetch(): Promise<ConnectorFetchResult> {
		const vaultPath = this.options.vaultPath;
		if (vaultPath === undefined || vaultPath.length === 0) {
			return { ok: false, reason: "not-configured", message: "vault path not configured" };
		}
		try {
			const info = await stat(vaultPath);
			if (!info.isDirectory()) return { ok: false, reason: "unavailable", message: "vault path is not a directory" };
		} catch {
			return { ok: false, reason: "unavailable", message: "vault path is not readable" };
		}
		const maxDocuments = this.options.maxDocuments ?? DEFAULT_MAX_DOCUMENTS;
		const maxBytes = this.options.maxBytesPerFile ?? DEFAULT_MAX_BYTES_PER_FILE;

		// 1. Walk the vault for markdown files, skipping vault-internal dirs.
		const files: string[] = [];
		const walk = async (dir: string): Promise<void> => {
			let entries: Dirent[];
			try {
				entries = await readdir(dir, { withFileTypes: true });
			} catch {
				return;
			}
			for (const entry of entries) {
				if (entry.isDirectory()) {
					if (!SKIPPED_DIRS.has(entry.name)) await walk(join(dir, entry.name));
				} else if (entry.isFile() && extname(entry.name).toLowerCase() === ".md") {
					files.push(join(dir, entry.name));
				}
			}
		};
		await walk(vaultPath);

		// 2. Parse notes into documents.
		const documents: ConnectorDocument[] = [];
		let readFailures = 0;
		let skippedLarge = 0;
		for (const file of files.sort()) {
			if (documents.length >= maxDocuments) break;
			let text: string;
			try {
				const info = await stat(file);
				if (info.size > maxBytes) {
					skippedLarge += 1;
					continue;
				}
				text = await readFile(file, "utf8");
				const relativePath = relative(vaultPath, file);
				const segments = relativePath.split(sep);
				const directorySegments = segments.slice(0, -1);
				const stem = basename(file, ".md");
				const { body, frontmatterTags } = stripFrontmatter(text);
				const inlineTags = matchAll(body, INLINE_TAG_PATTERN, 1);
				const links: string[] = [];
				const embeds: string[] = [];
				for (const match of body.matchAll(WIKI_LINK_PATTERN)) {
					const target = match[2]?.trim();
					if (target === undefined || target.length === 0) continue;
					(match[1] === "!" ? embeds : links).push(target);
				}
				const heading = /^#\s+(.+)$/mu.exec(body)?.[1]?.trim();
				const content = body.trim().slice(0, MAX_CONTENT_CHARS);
				if (content.length === 0) continue;
				documents.push({
					docId: relativePath.split(sep).join("__").replace(/\.md$/u, ""),
					hierarchy: ["folders", ...directorySegments],
					title: heading ?? stem,
					content,
					publishedAt: info.mtimeMs,
					metadata: {
						tags: [...new Set([...frontmatterTags, ...inlineTags])],
						links,
						embeds,
						folder: directorySegments.join("/"),
					},
				});
			} catch {
				readFailures += 1;
			}
		}

		const warnings: string[] = [];
		if (skippedLarge > 0) warnings.push(`${skippedLarge} note(s) exceeded the size limit and were skipped`);
		if (readFailures > 0) warnings.push(`${readFailures} note(s) failed to read`);
		return { ok: true, documents, ...(warnings.length > 0 ? { warnings } : {}) };
	}
}

function stripFrontmatter(text: string): { body: string; frontmatterTags: readonly string[] } {
	if (!text.startsWith("---")) return { body: text, frontmatterTags: [] };
	const end = text.indexOf("\n---", 3);
	if (end === -1) return { body: text, frontmatterTags: [] };
	const frontmatter = text.slice(3, end);
	const body = text.slice(text.indexOf("\n", end + 1) + 1);
	return { body, frontmatterTags: parseFrontmatterTags(frontmatter) };
}

/** Line-based `tags:` extraction — inline lists, comma/space strings, and dash lists. */
function parseFrontmatterTags(frontmatter: string): readonly string[] {
	const lines = frontmatter.split("\n");
	const tags: string[] = [];
	for (const [index, line] of lines.entries()) {
		const match = /^tags:\s*(.*)$/u.exec(line.trim());
		if (match === null) continue;
		const inline = match[1]?.trim() ?? "";
		if (inline.length > 0) {
			const cleaned = inline.replace(/^\[/u, "").replace(/\]$/u, "");
			for (const tag of cleaned.split(/[\s,]+/u)) {
				const trimmed = tag.trim().replace(/^["'#]+|["']+$/gu, "");
				if (trimmed.length > 0) tags.push(trimmed);
			}
			break;
		}
		// Dash-list form on following lines.
		for (let cursor = index + 1; cursor < lines.length; cursor += 1) {
			const item = /^\s*-\s*(.+)$/u.exec(lines[cursor] ?? "");
			if (item === null) break;
			const trimmed = item[1]?.trim().replace(/^["'#]+|["']+$/gu, "") ?? "";
			if (trimmed.length > 0) tags.push(trimmed);
		}
		break;
	}
	return tags;
}

function matchAll(text: string, pattern: RegExp, group: number): readonly string[] {
	const out: string[] = [];
	for (const match of text.matchAll(pattern)) {
		const value = match[group]?.trim();
		if (value !== undefined && value.length > 0) out.push(value);
	}
	return out;
}

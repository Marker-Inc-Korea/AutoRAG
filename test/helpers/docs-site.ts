import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { allPages, SECTIONS, sourceFileFor } from "../../docs/.vitepress/sidebar.mts";

export const REPO_ROOT = fileURLToPath(new URL("../..", import.meta.url));
export const DOCS_ROOT = join(REPO_ROOT, "docs");

/** Top-level docs entries that are tooling, not pages. */
const NON_PAGE_ENTRIES = new Set([".vitepress", "node_modules", "public", "package.json", "bun.lock"]);

export interface DocPage {
	/** Path relative to `docs/`, forward slashes. */
	readonly file: string;
	readonly absolute: string;
	readonly text: string;
}

function walk(dir: string, out: string[]): void {
	for (const entry of readdirSync(dir, { withFileTypes: true })) {
		if (dir === DOCS_ROOT && NON_PAGE_ENTRIES.has(entry.name)) continue;
		const full = join(dir, entry.name);
		if (entry.isDirectory()) walk(full, out);
		else if (entry.name.endsWith(".md")) out.push(full);
	}
}

/** Every Markdown file under `docs/` (the home page and README included), LF-normalised. */
export function readDocPages(): DocPage[] {
	const files: string[] = [];
	walk(DOCS_ROOT, files);
	return files.sort().map((absolute) => ({
		file: relative(DOCS_ROOT, absolute).split("\\").join("/"),
		absolute,
		text: readFileSync(absolute, "utf8").replace(/\r\n?/g, "\n"),
	}));
}

/** Pages the sidebar publishes, as `docs/`-relative source files. */
export function sidebarFiles(): string[] {
	return allPages().map((item) => sourceFileFor(item.link));
}

export { allPages, SECTIONS };

/** Markdown without fenced code blocks, so links and headings inside examples are not parsed as prose. */
export function stripFences(markdown: string): string {
	return markdown.replace(/^(```|~~~)[^\n]*\n[\s\S]*?^\1[^\n]*$/gm, "");
}

/** Fenced code bodies only. */
export function fencedBlocks(markdown: string): string[] {
	return [...markdown.matchAll(/^(```|~~~)[^\n]*\n([\s\S]*?)^\1[^\n]*$/gm)].map((m) => m[2]);
}

/** Inline code spans outside fences. */
export function inlineCode(markdown: string): string[] {
	return [...stripFences(markdown).matchAll(/`([^`\n]+)`/g)].map((m) => m[1]);
}

const SPECIAL = /[\s~`!@#$%^&*()\-_+=[\]{}|\\;:"'“”‘’<>,.?/]+/g;

/** The slug VitePress (`@mdit-vue/shared`) gives a heading, so anchors can be checked without building the site. */
export function slugify(heading: string): string {
	return heading
		.normalize("NFKD")
		.replace(/[\u0000-\u001f]/g, "")
		.replace(SPECIAL, "-")
		.replace(/-{2,}/g, "-")
		.replace(/^-+|-+$/g, "")
		.replace(/^(\d)/, "_$1")
		.toLowerCase();
}

/** Anchor ids a page exposes: headings (with `{#custom}` ids honoured). */
export function anchorsOf(markdown: string): Set<string> {
	const anchors = new Set<string>();
	for (const match of stripFences(markdown).matchAll(/^#{1,6}\s+(.+?)\s*$/gm)) {
		let text = match[1];
		const custom = /\{#([^}\s]+)\}\s*$/.exec(text);
		if (custom) {
			anchors.add(custom[1]);
			continue;
		}
		text = text.replace(/\[([^\]]*)\]\([^)]*\)/g, "$1").replace(/`([^`]*)`/g, "$1");
		// Strip tags until the text stops changing, so nested fragments like "<<b>b>" cannot survive one pass.
		for (let previous = ""; previous !== text; ) {
			previous = text;
			text = text.replace(/<[^<>]+>/g, "");
		}
		text = text.replace(/[*_]{1,3}([^*_]+)[*_]{1,3}/g, "$1");
		anchors.add(slugify(text));
	}
	return anchors;
}

export interface DocLink {
	readonly raw: string;
	readonly target: string;
	readonly hash: string;
}

/** Markdown links `[text](target)` outside code, split into path and `#hash`. Images are included. */
export function linksOf(markdown: string): DocLink[] {
	const links: DocLink[] = [];
	for (const match of stripFences(markdown)
		.replace(/`[^`\n]*`/g, "")
		.matchAll(/!?\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g)) {
		const raw = match[1];
		const hashAt = raw.indexOf("#");
		links.push({
			raw,
			target: hashAt < 0 ? raw : raw.slice(0, hashAt),
			hash: hashAt < 0 ? "" : raw.slice(hashAt + 1),
		});
	}
	return links;
}

export function isExternal(target: string): boolean {
	return /^[a-z][a-z0-9+.-]*:/i.test(target) || target.startsWith("//");
}

/** Resolve a relative page link from `fromFile` (docs-relative) to an absolute path. */
export function resolveDocLink(fromFile: string, target: string): string {
	return resolve(dirname(join(DOCS_ROOT, fromFile)), target);
}

export function exists(path: string): boolean {
	return existsSync(path);
}

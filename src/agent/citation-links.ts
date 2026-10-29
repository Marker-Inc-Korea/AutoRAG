import { basename, isAbsolute } from "node:path";
import { pathToFileURL } from "node:url";
import type { ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { answerMarkers } from "./citations.ts";
import { type EvidenceLedger, isRegularFile } from "./evidence-ledger.ts";

/** Markdown punctuation that would change the meaning of a link label. */
const MARKDOWN_SPECIALS = /[\\`*_{}[\]<>()#+\-.!|~]/gu;

/** `file://` URL of an absolute local path, an http(s) URL as is, otherwise undefined (a datasource id, a message id). */
export function openableTarget(source: string): string | undefined {
	if (/^https?:\/\//iu.test(source)) {
		try {
			return new URL(source).href;
		} catch {
			return undefined;
		}
	}
	return isAbsolute(source) ? pathToFileURL(source).href : undefined;
}

/**
 * Rewrite the citation markers of an assistant answer into Markdown links, so
 * pi renders them as terminal hyperlinks that open the cited file or page.
 *
 * `[e3]` and `[e3, e7]` keep their brackets and ids; each id that resolves to
 * an openable source becomes a link. `[file:/abs/path]` becomes `[<file name>]`
 * linked to that file. Markers that resolve to nothing openable stay as the
 * model wrote them.
 *
 * `sourceOf` maps an evidence id to its recorded source; `isFile` says whether
 * a `[file:...]` path is a real local file.
 */
export function linkCitations(
	markdown: string,
	sourceOf: (id: string) => string | undefined,
	isFile: (path: string) => boolean,
): string {
	let out = "";
	let cursor = 0;
	for (const marker of answerMarkers(markdown, isFile)) {
		if (marker.kind === "number") continue;
		const labels =
			marker.kind === "file"
				? [{ label: basename(marker.path), source: marker.path }]
				: marker.ids.map((id) => ({ label: id, source: sourceOf(id) }));
		let linked = false;
		const parts = labels.map(({ label, source }) => {
			const target = source === undefined ? undefined : openableTarget(source);
			if (target === undefined) return label;
			linked = true;
			// The angle form keeps `(`, `)` and spaces in a target from ending the link early.
			return `[${label.replace(MARKDOWN_SPECIALS, "\\$&")}](<${target}>)`;
		});
		if (!linked) continue;
		const raw = markdown.slice(marker.start, marker.end);
		const leading = raw.slice(0, raw.length - raw.trimStart().length);
		out += `${markdown.slice(cursor, marker.start)}${leading}[${parts.join(", ")}]`;
		cursor = marker.end;
	}
	return out + markdown.slice(cursor);
}

/**
 * pi extension that makes the citations in the interactive transcript
 * clickable. The model cites by evidence id; the ledger recorded the source
 * each id names, so the link target is never model-written text.
 */
export function createCitationLinks(ledger: EvidenceLedger): ExtensionFactory {
	return (pi) => {
		const fileCache = new Map<string, boolean>();
		pi.registerMarkdownTransformer((markdown, context) =>
			context.messageType !== "assistant"
				? markdown
				: linkCitations(
						markdown,
						(id) => ledger.lookup(id, { allowLocalFiles: false })[0]?.source,
						(path) => {
							if (!isAbsolute(path)) return false;
							let known = fileCache.get(path);
							if (known === undefined) {
								known = isRegularFile(path);
								fileCache.set(path, known);
							}
							return known;
						},
					),
		);
	};
}

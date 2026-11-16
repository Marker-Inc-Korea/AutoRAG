import { XMLParser } from "fast-xml-parser";
import { ParseError } from "./errors.ts";
import { joinPages, type PageMarkdownInput } from "./pages.ts";
import { normalizeMarkdown } from "./text.ts";
import { type ParseInput, type ParseOutput, Parser } from "./types.ts";
import { readZipXmlFiles, readZipXmlText, readZipXmlTextParts } from "./xml-text.ts";

const xml = new XMLParser({ ignoreAttributes: false, parseTagValue: false, trimValues: true });

const PRESENTATION_PART =
	/^ppt\/(?:presentation\.xml|_rels\/presentation\.xml\.rels|slides\/_rels\/slide\d+\.xml\.rels)$/;
const SLIDE_PART = /^ppt\/slides\/slide\d+\.xml$/;
const SLIDE_AND_NOTES = /^ppt\/(?:slides\/slide\d+|notesSlides\/notesSlide\d+)\.xml$/;
/** Structure parts are small and one per slide; they do not count against the text-part limit. */
const MAX_STRUCTURE_PARTS = 2_200;

interface Relationship {
	readonly type: string;
	readonly target: string;
}

function asArray<T>(value: T | readonly T[] | undefined): readonly T[] {
	return value === undefined ? [] : Array.isArray(value) ? value : [value as T];
}

function attribute(node: unknown, name: string): string | undefined {
	if (typeof node !== "object" || node === null) return undefined;
	const value = (node as Record<string, unknown>)[`@_${name}`];
	return typeof value === "string" ? value : undefined;
}

/** `dir` joined with a relationship target, with `.`/`..` resolved against the archive root. */
function resolvePart(dir: string, target: string): string {
	const parts = target.startsWith("/") ? [] : dir.split("/").filter((part) => part.length > 0);
	for (const segment of target.split("/")) {
		if (segment === "" || segment === ".") continue;
		if (segment === "..") parts.pop();
		else parts.push(segment);
	}
	return parts.join("/");
}

function relationships(source: string | undefined, dir: string): ReadonlyMap<string, Relationship> {
	const map = new Map<string, Relationship>();
	if (source === undefined) return map;
	const root = xml.parse(source) as { Relationships?: { Relationship?: unknown } };
	for (const rel of asArray(root.Relationships?.Relationship)) {
		const id = attribute(rel, "Id");
		const target = attribute(rel, "Target");
		if (id === undefined || target === undefined || attribute(rel, "TargetMode") === "External") continue;
		map.set(id, { type: attribute(rel, "Type") ?? "", target: resolvePart(dir, target) });
	}
	return map;
}

/**
 * Slide parts in presentation order and the notes part each carries, from `ppt/presentation.xml`. Undefined
 * when the deck names no slide order: the slide file number is not the slide number once a deck is reordered,
 * so a page is never derived from it.
 */
async function slideOrder(
	bytes: Uint8Array,
): Promise<readonly { readonly slide: string; readonly notes?: string }[] | undefined> {
	const parts = new Map(
		(await readZipXmlFiles(bytes, PRESENTATION_PART, { maxFiles: MAX_STRUCTURE_PARTS })).map((part) => [
			part.name,
			part.xml,
		]),
	);
	const presentation = parts.get("ppt/presentation.xml");
	if (presentation === undefined) return undefined;
	const rels = relationships(parts.get("ppt/_rels/presentation.xml.rels"), "ppt");
	const root = xml.parse(presentation) as { "p:presentation"?: Record<string, unknown> } & Record<string, unknown>;
	const body = (root["p:presentation"] ?? Object.values(root).find((value) => typeof value === "object")) as
		| Record<string, unknown>
		| undefined;
	const list = (body?.["p:sldIdLst"] ?? body?.sldIdLst) as Record<string, unknown> | undefined;
	const ordered: { slide: string; notes?: string }[] = [];
	for (const entry of asArray(list?.["p:sldId"] ?? list?.sldId)) {
		const id = attribute(entry, "r:id");
		const slide = id === undefined ? undefined : rels.get(id)?.target;
		if (slide === undefined || !SLIDE_PART.test(slide)) continue;
		const slideRels = relationships(
			parts.get(`ppt/slides/_rels/${slide.slice(slide.lastIndexOf("/") + 1)}.rels`),
			"ppt/slides",
		);
		const notes = [...slideRels.values()].find((rel) => rel.type.endsWith("/notesSlide"))?.target;
		ordered.push({ slide, ...(notes === undefined ? {} : { notes }) });
	}
	return ordered.length === 0 ? undefined : ordered;
}

/**
 * PPTX stays on AutoRAG's own reader. kordoc parses a deck too, but this reader keeps every `a:t` text run
 * and its slide boundaries come straight from the presentation's slide list, so a slide number is the
 * slide's position in the deck, never a file name.
 */
export class PptxParser extends Parser {
	readonly name = "pptx";
	readonly extensions = [".pptx"] as const;
	/** 2: slides are read in presentation order and the output carries the slide index. */
	override readonly version = "2";

	async parse(input: ParseInput): Promise<ParseOutput> {
		try {
			const order = await slideOrder(input.bytes);
			if (order === undefined) {
				const chunks = await readZipXmlText(input.bytes, SLIDE_AND_NOTES);
				const markdown = normalizeMarkdown(chunks.filter((chunk) => chunk.trim().length > 0).join("\n\n"));
				return { markdown, metadata: { parser: this.name } };
			}
			const texts = new Map(
				(await readZipXmlTextParts(input.bytes, SLIDE_AND_NOTES)).map((part) => [part.name, part.chunks]),
			);
			const slides: PageMarkdownInput[] = [];
			for (const { slide, notes } of order) {
				const chunks = texts.get(slide);
				if (chunks === undefined) continue;
				const own = [...chunks, ...(notes === undefined ? [] : (texts.get(notes) ?? []))];
				slides.push({
					page: slides.length + 1,
					markdown: normalizeMarkdown(
						own
							.map((chunk) => chunk.trim())
							.filter((chunk) => chunk.length > 0)
							.join("\n\n"),
					),
				});
			}
			const joined = joinPages(slides, "layout", slides.length);
			return {
				markdown: joined.markdown,
				metadata: { parser: this.name },
				...(joined.pages === undefined ? {} : { pages: joined.pages }),
			};
		} catch (cause) {
			throw new ParseError(this.name, input.virtualPath, cause);
		}
	}
}

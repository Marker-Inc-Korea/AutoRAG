/**
 * File-kind map — the reference's `KD` table (DESIGN.md §2.9).
 *
 * `tileToken` / `foregroundToken` are custom-property names from tokens.css;
 * components resolve them with var(), never with a literal color.
 */

export type FileKind =
	| "folder"
	| "xlsx"
	| "csv"
	| "pdf"
	| "docx"
	| "pptx"
	| "md"
	| "png"
	| "zip"
	| "slack"
	| "mail"
	| "notion"
	| "discord"
	| "telegram"
	| "contact";

export interface KindMeta {
	/** Tile letter; empty for folders, which render the folder glyph. */
	readonly letter: string;
	readonly tileToken: string;
	readonly foregroundToken: string;
	readonly label: string;
}

export const KIND_ORDER: readonly FileKind[] = [
	"folder",
	"xlsx",
	"csv",
	"pdf",
	"docx",
	"pptx",
	"md",
	"png",
	"zip",
	"slack",
	"mail",
	"notion",
	"discord",
	"telegram",
	"contact",
];

const ON_TILE = "--text-on-tile";

export const KIND_META: Readonly<Record<FileKind, KindMeta>> = {
	folder: { letter: "", tileToken: "--folder", foregroundToken: ON_TILE, label: "Folder" },
	xlsx: { letter: "X", tileToken: "--tile-xlsx", foregroundToken: ON_TILE, label: "Excel Spreadsheet" },
	csv: { letter: "C", tileToken: "--tile-csv", foregroundToken: ON_TILE, label: "CSV Document" },
	pdf: { letter: "P", tileToken: "--tile-pdf", foregroundToken: ON_TILE, label: "PDF Document" },
	docx: { letter: "W", tileToken: "--tile-docx", foregroundToken: ON_TILE, label: "Word Document" },
	pptx: { letter: "P", tileToken: "--tile-pptx", foregroundToken: ON_TILE, label: "PowerPoint" },
	md: { letter: "M", tileToken: "--tile-md", foregroundToken: ON_TILE, label: "Markdown" },
	png: { letter: "I", tileToken: "--tile-png", foregroundToken: ON_TILE, label: "PNG Image" },
	zip: { letter: "Z", tileToken: "--tile-zip", foregroundToken: ON_TILE, label: "ZIP Archive" },
	slack: { letter: "S", tileToken: "--tile-slack", foregroundToken: ON_TILE, label: "Slack Thread" },
	mail: { letter: "M", tileToken: "--tile-mail", foregroundToken: ON_TILE, label: "Email" },
	notion: { letter: "N", tileToken: "--tile-notion", foregroundToken: "--tile-notion-fg", label: "Notion Page" },
	discord: { letter: "D", tileToken: "--tile-discord", foregroundToken: ON_TILE, label: "Discord Thread" },
	telegram: { letter: "T", tileToken: "--tile-telegram", foregroundToken: ON_TILE, label: "Telegram Message" },
	contact: { letter: "@", tileToken: "--tile-contact", foregroundToken: ON_TILE, label: "Contact" },
};

/** Unknown kinds fall back to the md entry (DESIGN.md §2.9). */
export function kindMeta(kind: string): KindMeta {
	return KIND_META[kind as FileKind] ?? KIND_META.md;
}

const EXTENSION_KINDS: Readonly<Record<string, FileKind>> = {
	xlsx: "xlsx",
	xlsm: "xlsx",
	xls: "xlsx",
	numbers: "xlsx",
	csv: "csv",
	tsv: "csv",
	pdf: "pdf",
	doc: "docx",
	docx: "docx",
	rtf: "docx",
	pages: "docx",
	hwp: "docx",
	hwpx: "docx",
	ppt: "pptx",
	pptx: "pptx",
	key: "pptx",
	md: "md",
	markdown: "md",
	txt: "md",
	png: "png",
	jpg: "png",
	jpeg: "png",
	gif: "png",
	webp: "png",
	heic: "png",
	svg: "png",
	tiff: "png",
	zip: "zip",
	gz: "zip",
	tgz: "zip",
	tar: "zip",
	rar: "zip",
	"7z": "zip",
	eml: "mail",
};

export function kindFromExtension(extension: string): FileKind {
	return EXTENSION_KINDS[extension.toLowerCase()] ?? "md";
}

import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { extname, join } from "node:path";
import { promisify } from "node:util";
import { renderDocument as kordocRenderDocument } from "kordoc";
import { abortReason } from "../process/abort-reason.ts";

const execFileAsync = promisify(execFile);

export type VisualImageMimeType = "image/png" | "image/jpeg" | "image/webp";

/** One rendered page image ready for a visual decision gate or VLM/OCR extractor. */
export interface RenderedPage {
	/** 1-based page number. */
	readonly page: number;
	readonly data: Uint8Array;
	readonly mimeType: VisualImageMimeType;
	/** SHA-256 of `data`, used for page-level incremental caching. */
	readonly sha256: string;
	readonly width?: number;
	readonly height?: number;
}

export interface PageRenderInput {
	readonly virtualPath: string;
	readonly sourcePath?: string;
	readonly bytes: Uint8Array;
	/** Optional 1-based page subset to render; omitted renders up to `maxPagesPerFile`. */
	readonly pageNumbers?: readonly number[];
	readonly maxPagesPerFile?: number;
	readonly maxWidthPx?: number;
	readonly signal?: AbortSignal;
}

export interface PageRenderer {
	/** Stable identity for incremental cache validation. */
	readonly fingerprint: string;
	supports(virtualPath: string): boolean;
	renderPages(input: PageRenderInput): Promise<readonly RenderedPage[]>;
}

/** Default cap on pages rendered per document during visual gating. */
export const DEFAULT_VISUAL_MAX_PAGES_PER_FILE = 50;
/** Default max rendered image width in pixels; sharp enough for charts and tables while keeping token cost low. */
export const DEFAULT_VISUAL_MAX_WIDTH_PX = 1024;
/** Maximum pixel area per rendered page to keep memory bounded on poster-sized PDFs. */
const MAX_RENDER_PIXELS = 4_000_000;

const IMAGE_MIME_BY_EXT: Readonly<Record<string, VisualImageMimeType>> = {
	".png": "image/png",
	".jpg": "image/jpeg",
	".jpeg": "image/jpeg",
	".webp": "image/webp",
};

const KORDOC_RENDER_EXTENSIONS = new Set([".hwp", ".hwpx"]);

/** Formats the built-in visual page renderer can rasterize into per-page images. */
export function supportsVisualPageRendering(virtualPath: string): boolean {
	const ext = extname(virtualPath).toLowerCase();
	return ext === ".pdf" || KORDOC_RENDER_EXTENSIONS.has(ext) || Object.hasOwn(IMAGE_MIME_BY_EXT, ext);
}

function sha256Of(data: Uint8Array): string {
	return createHash("sha256").update(data).digest("hex");
}

function normalizePageSelection(
	totalPages: number,
	requested: readonly number[] | undefined,
	maxPages: number,
): number[] {
	if (requested !== undefined) {
		const unique = [...new Set(requested.filter((p) => Number.isInteger(p) && p >= 1 && p <= totalPages))].sort(
			(a, b) => a - b,
		);
		return unique.slice(0, maxPages);
	}
	const count = Math.min(totalPages, maxPages);
	return Array.from({ length: count }, (_item, index) => index + 1);
}

interface PdfiumPageLike {
	getOriginalSize(): { originalWidth: number; originalHeight: number };
	render(options: {
		scale: number;
		render: (args: { data: Uint8Array; width: number; height: number }) => Promise<Uint8Array>;
	}): Promise<{ data: Uint8Array; width: number; height: number }>;
}

interface PdfiumDocLike {
	getPageCount(): number;
	getPage(index: number): PdfiumPageLike;
	destroy(): void;
}

interface PdfiumLibraryLike {
	loadDocument(bytes: Uint8Array): Promise<PdfiumDocLike>;
	destroy(): void;
}

interface PdfiumModuleLike {
	PDFiumLibrary: { init(): Promise<PdfiumLibraryLike> };
}

type SharpFn = (
	input: Uint8Array,
	options: { raw: { width: number; height: number; channels: 4 } },
) => { png(): { toBuffer(): Promise<Buffer> } };

async function tryDynamicImport<T>(specifier: string): Promise<T | undefined> {
	try {
		return (await import(specifier)) as T;
	} catch {
		return undefined;
	}
}

/** PDFium outputs BGRA pixels; swap R and B channels in-place into RGBA for sharp. */
function bgraToRgba(bgra: Uint8Array): Uint8Array {
	const rgba = new Uint8Array(bgra.length);
	for (let i = 0; i + 3 < bgra.length; i += 4) {
		rgba[i] = bgra[i + 2] ?? 0;
		rgba[i + 1] = bgra[i + 1] ?? 0;
		rgba[i + 2] = bgra[i] ?? 0;
		rgba[i + 3] = bgra[i + 3] ?? 255;
	}
	return rgba;
}

async function renderPdfWithPdfium(
	bytes: Uint8Array,
	pageNumbers: readonly number[] | undefined,
	maxPages: number,
	maxWidthPx: number,
	signal: AbortSignal | undefined,
): Promise<RenderedPage[] | undefined> {
	const pdfiumMod = await tryDynamicImport<PdfiumModuleLike>("@hyzyla/pdfium");
	const sharpMod = await tryDynamicImport<{ default?: SharpFn } | SharpFn>("sharp");
	const sharpFn: SharpFn | undefined = typeof sharpMod === "function" ? sharpMod : sharpMod?.default;
	if (pdfiumMod === undefined || sharpFn === undefined) return undefined;

	const library = await pdfiumMod.PDFiumLibrary.init();
	const doc = await library.loadDocument(bytes);
	try {
		const targets = normalizePageSelection(doc.getPageCount(), pageNumbers, maxPages);
		const pages: RenderedPage[] = [];
		for (const pageNumber of targets) {
			if (signal?.aborted) throw abortReason(signal);
			const page = doc.getPage(pageNumber - 1);
			const { originalWidth, originalHeight } = page.getOriginalSize();
			const widthScale = originalWidth > 0 ? maxWidthPx / originalWidth : 1.5;
			const areaScale = Math.sqrt(MAX_RENDER_PIXELS / Math.max(1, originalWidth * originalHeight));
			const scale = Math.max(0.5, Math.min(2, widthScale, areaScale));
			const rendered = await page.render({
				scale,
				render: async ({ data }) => data,
			});
			const rgba = bgraToRgba(rendered.data);
			const png = await sharpFn(rgba, {
				raw: { width: rendered.width, height: rendered.height, channels: 4 },
			})
				.png()
				.toBuffer();
			const pngBytes = new Uint8Array(png.buffer, png.byteOffset, png.byteLength);
			pages.push({
				page: pageNumber,
				data: pngBytes,
				mimeType: "image/png",
				sha256: sha256Of(pngBytes),
				width: rendered.width,
				height: rendered.height,
			});
		}
		return pages;
	} finally {
		doc.destroy();
		library.destroy();
	}
}

async function renderPdfWithPdftoppm(
	bytes: Uint8Array,
	pageNumbers: readonly number[] | undefined,
	maxPages: number,
	maxWidthPx: number,
	signal: AbortSignal | undefined,
): Promise<RenderedPage[]> {
	const workDir = await mkdtemp(join(tmpdir(), "autorag-visual-pdf-"));
	try {
		const inputPdf = join(workDir, "input.pdf");
		await writeFile(inputPdf, bytes);
		const prefix = join(workDir, "page");
		const args = ["-png", "-scale-to-x", String(maxWidthPx), "-scale-to-y", "-1"];
		if (pageNumbers !== undefined && pageNumbers.length > 0) {
			const sorted = [...new Set(pageNumbers.filter((p) => Number.isInteger(p) && p >= 1))].sort((a, b) => a - b);
			const first = sorted[0];
			const last = sorted[sorted.length - 1];
			if (first !== undefined && last !== undefined) {
				args.push("-f", String(first), "-l", String(last));
			}
		} else {
			args.push("-f", "1", "-l", String(maxPages));
		}
		args.push(inputPdf, prefix);
		await execFileAsync("pdftoppm", args, { signal, maxBuffer: 8 * 1024 * 1024 });
		const entries = (await readdir(workDir))
			.filter((name) => name.startsWith("page-") && name.endsWith(".png"))
			.map((name) => {
				const match = /^page-(\d+)\.png$/u.exec(name);
				return match === null ? undefined : { name, page: Number(match[1]) };
			})
			.filter((item): item is { name: string; page: number } => item !== undefined)
			.sort((a, b) => a.page - b.page);

		const wantedSet = pageNumbers === undefined ? undefined : new Set(pageNumbers);
		const pages: RenderedPage[] = [];
		for (const entry of entries) {
			if (wantedSet !== undefined && !wantedSet.has(entry.page)) continue;
			if (pages.length >= maxPages) break;
			if (signal?.aborted) throw abortReason(signal);
			const buffer = await readFile(join(workDir, entry.name));
			const data = new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
			pages.push({
				page: entry.page,
				data,
				mimeType: "image/png",
				sha256: sha256Of(data),
			});
		}
		return pages;
	} finally {
		await rm(workDir, { recursive: true, force: true });
	}
}

async function renderHwpPages(
	bytes: Uint8Array,
	pageNumbers: readonly number[] | undefined,
	maxPages: number,
	maxWidthPx: number,
	signal: AbortSignal | undefined,
): Promise<RenderedPage[]> {
	if (signal?.aborted) throw abortReason(signal);
	const selectedPages =
		pageNumbers !== undefined
			? [...new Set(pageNumbers.filter((p) => Number.isInteger(p) && p >= 1))]
					.sort((a, b) => a - b)
					.slice(0, maxPages)
			: `1-${maxPages}`;
	const rendered = await kordocRenderDocument(Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength), {
		format: "png",
		pages: selectedPages,
		maxWidthPx,
	});
	const pages: RenderedPage[] = [];
	for (const asset of rendered.assets) {
		if (signal?.aborted) throw abortReason(signal);
		if (asset.page === undefined || typeof asset.data === "string") continue;
		const data = new Uint8Array(asset.data.buffer, asset.data.byteOffset, asset.data.byteLength);
		pages.push({
			page: asset.page,
			data,
			mimeType: "image/png",
			sha256: sha256Of(data),
			...(asset.width !== undefined ? { width: asset.width } : {}),
			...(asset.height !== undefined ? { height: asset.height } : {}),
		});
	}
	return pages.sort((a, b) => a.page - b.page);
}

/** Built-in document page renderer covering PDF (PDFium/sharp with pdftoppm fallback), HWP/HWPX (kordoc), and standalone images. */
export function createDefaultPageRenderer(
	defaults: { readonly maxPagesPerFile?: number; readonly maxWidthPx?: number } = {},
): PageRenderer {
	const defaultMaxPages = defaults.maxPagesPerFile ?? DEFAULT_VISUAL_MAX_PAGES_PER_FILE;
	const defaultMaxWidth = defaults.maxWidthPx ?? DEFAULT_VISUAL_MAX_WIDTH_PX;
	return {
		fingerprint: `default-page-renderer:v1:w${defaultMaxWidth}:p${defaultMaxPages}`,
		supports: supportsVisualPageRendering,
		async renderPages(input: PageRenderInput): Promise<readonly RenderedPage[]> {
			if (input.signal?.aborted) throw abortReason(input.signal);
			const ext = extname(input.virtualPath).toLowerCase();
			const maxPages = input.maxPagesPerFile ?? defaultMaxPages;
			const maxWidthPx = input.maxWidthPx ?? defaultMaxWidth;

			const imageMime = IMAGE_MIME_BY_EXT[ext];
			if (imageMime !== undefined) {
				if (input.pageNumbers !== undefined && !input.pageNumbers.includes(1)) return [];
				return [
					{
						page: 1,
						data: input.bytes,
						mimeType: imageMime,
						sha256: sha256Of(input.bytes),
					},
				];
			}
			if (ext === ".pdf") {
				const fromPdfium = await renderPdfWithPdfium(
					input.bytes,
					input.pageNumbers,
					maxPages,
					maxWidthPx,
					input.signal,
				);
				if (fromPdfium !== undefined) return fromPdfium;
				return renderPdfWithPdftoppm(input.bytes, input.pageNumbers, maxPages, maxWidthPx, input.signal);
			}
			if (KORDOC_RENDER_EXTENSIONS.has(ext)) {
				return renderHwpPages(input.bytes, input.pageNumbers, maxPages, maxWidthPx, input.signal);
			}
			return [];
		},
	};
}

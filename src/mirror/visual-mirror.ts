import { createHash } from "node:crypto";
import { existsSync, readFileSync, rmSync } from "node:fs";
import {
	createDefaultPageRenderer,
	DEFAULT_VISUAL_MAX_PAGES_PER_FILE,
	DEFAULT_VISUAL_MAX_WIDTH_PX,
	type PageRenderer,
} from "../parser/page-render.ts";
import { joinPages, type PageMarkdownInput, type ParsedPages } from "../parser/pages.ts";
import { normalizeMarkdown } from "../parser/text.ts";
import type { VisualExtractorKind, VisualPageExtractor } from "../parser/visual-extractor.ts";
import type { VisualPageGate, VisualPageRoute } from "../parser/visual-gate.ts";
import { abortReason } from "../process/abort-reason.ts";
import { writeFileAtomic } from "./atomic-write.ts";
import { readPagesSidecar, writePagesSidecar } from "./pages-sidecar.ts";
import { parsedVisualManifestPath, parsedVisualOutputPath } from "./paths.ts";

const VISUAL_MANIFEST_VERSION = 1;

/**
 * Default parallel API workers for per-page visual gating and VLM extraction within one file.
 * Decoupled from local OCR concurrency so remote HTTP calls (decision gate + VLM) fly in parallel
 * while CPU-heavy local Tesseract OCR stays bounded.
 */
export const DEFAULT_VISUAL_PAGE_CONCURRENCY = 8;
/** Default parallel workers for the lightweight vision decision gate. */
export const DEFAULT_VISUAL_GATE_CONCURRENCY = 12;
/** Default parallel workers for local CPU-bound OCR extraction (Tesseract). */
export const DEFAULT_VISUAL_OCR_CONCURRENCY = 2;
/** Default parallel file workers when enriching multiple documents in one Stage 2 visual pass. */
export const DEFAULT_VISUAL_FILE_CONCURRENCY = 3;

export class AsyncSemaphore {
	private active = 0;
	private readonly queue: Array<() => void> = [];

	constructor(private readonly max: number) {}

	async run<T>(fn: () => Promise<T>): Promise<T> {
		if (this.active >= this.max) {
			await new Promise<void>((resolve) => {
				this.queue.push(resolve);
			});
		}
		this.active += 1;
		try {
			return await fn();
		} finally {
			this.active -= 1;
			this.queue.shift()?.();
		}
	}
}

/** Shared cross-file concurrency pools so parallel files still obey global gate/VLM/OCR API limits. */
export interface VisualMirrorSharedLimiters {
	readonly gate: AsyncSemaphore;
	readonly vlm: AsyncSemaphore;
	readonly ocr: AsyncSemaphore;
}

export function createVisualMirrorSharedLimiters(options: VisualMirrorSyncOptions): VisualMirrorSharedLimiters {
	const vlmConcurrency = Math.max(1, Math.floor(options.concurrency ?? DEFAULT_VISUAL_PAGE_CONCURRENCY));
	const gateConcurrency = Math.max(
		1,
		Math.floor(
			options.gateConcurrency ??
				(options.concurrency !== undefined
					? Math.max(vlmConcurrency, DEFAULT_VISUAL_GATE_CONCURRENCY)
					: DEFAULT_VISUAL_GATE_CONCURRENCY),
		),
	);
	const ocrConcurrency = Math.max(1, Math.floor(options.ocrConcurrency ?? DEFAULT_VISUAL_OCR_CONCURRENCY));
	return {
		gate: new AsyncSemaphore(gateConcurrency),
		vlm: new AsyncSemaphore(vlmConcurrency),
		ocr: new AsyncSemaphore(ocrConcurrency),
	};
}

/** Per-page cached visual-gating and VLM/OCR extraction record. */
export interface VisualPageRecord {
	readonly page: number;
	readonly imageSha256: string;
	readonly baseTextSha256: string;
	readonly gateFingerprint: string;
	readonly gateProbability: number;
	/** 3-way route (`vlm`, `ocr`, or `plain`); older manifests without `route` infer from `visuallyRich`. */
	readonly route?: VisualPageRoute;
	readonly visuallyRich: boolean;
	readonly extractorFingerprint?: string;
	readonly extractorKind?: VisualExtractorKind;
	readonly extractorModel?: string;
	readonly markdown?: string;
	readonly updatedAt: string;
}

/** Durable per-file manifest accumulating visual gate decisions and VLM/OCR extractions by page. */
export interface VisualMirrorManifest {
	readonly version: typeof VISUAL_MANIFEST_VERSION;
	readonly virtualPath: string;
	readonly sourcePath: string;
	readonly sourceVersion: string;
	readonly pageCount: number;
	readonly rendererFingerprint: string;
	readonly gateFingerprint: string;
	/** Combined fingerprint of active VLM and OCR extractors. */
	readonly extractorFingerprint: string;
	readonly maxPagesPerFile: number;
	readonly maxWidthPx: number;
	/** Keyed by 1-based page number string (`"1"`, `"2"`, …). */
	readonly pages: Readonly<Record<string, VisualPageRecord>>;
	readonly updatedAt: string;
}

export interface VisualMirrorSyncOptions {
	readonly gate: VisualPageGate;
	/** Primary extractor (legacy single-extractor seam, or VLM extractor when `vlmExtractor` is omitted). */
	readonly extractor?: VisualPageExtractor;
	/** Dedicated VLM extractor for pages routed to `vlm` (charts, diagrams, complex tables, visual annotations). */
	readonly vlmExtractor?: VisualPageExtractor;
	/** Dedicated OCR extractor for pages routed to `ocr` (scans, handwriting, text missed by base parser). */
	readonly ocrExtractor?: VisualPageExtractor;
	readonly renderer?: PageRenderer;
	readonly maxPagesPerFile?: number;
	readonly maxWidthPx?: number;
	/** Max concurrent VLM/page extraction API calls per file (default {@link DEFAULT_VISUAL_PAGE_CONCURRENCY}). */
	readonly concurrency?: number;
	/** Max concurrent vision decision-gate API calls (default max(concurrency, {@link DEFAULT_VISUAL_GATE_CONCURRENCY})). */
	readonly gateConcurrency?: number;
	/** Max concurrent local OCR workers (default {@link DEFAULT_VISUAL_OCR_CONCURRENCY}). */
	readonly ocrConcurrency?: number;
	/** Max files visually enriched concurrently across a batch (default {@link DEFAULT_VISUAL_FILE_CONCURRENCY}). */
	readonly fileConcurrency?: number;
}

function resolveExtractors(options: VisualMirrorSyncOptions): {
	readonly vlm: VisualPageExtractor | undefined;
	readonly ocr: VisualPageExtractor | undefined;
	readonly combinedFingerprint: string;
} {
	const vlm = options.vlmExtractor ?? (options.extractor?.kind === "vlm" ? options.extractor : undefined);
	const ocr = options.ocrExtractor ?? (options.extractor?.kind === "ocr" ? options.extractor : undefined);
	if (vlm === undefined && ocr === undefined) {
		throw new Error(
			"VisualMirrorSyncOptions requires at least one extractor (vlmExtractor, ocrExtractor, or extractor).",
		);
	}
	const combinedFingerprint = [
		vlm !== undefined ? `vlm=${vlm.fingerprint}` : "vlm=none",
		ocr !== undefined ? `ocr=${ocr.fingerprint}` : "ocr=none",
	].join("|");
	return { vlm, ocr, combinedFingerprint };
}

function pickExtractorForRoute(
	route: VisualPageRoute,
	extractors: { readonly vlm: VisualPageExtractor | undefined; readonly ocr: VisualPageExtractor | undefined },
): VisualPageExtractor | undefined {
	if (route === "plain") return undefined;
	if (route === "vlm") return extractors.vlm ?? extractors.ocr;
	return extractors.ocr ?? extractors.vlm;
}

export interface VisualMirrorDiagnostic {
	readonly code: "visual-render-failed" | "visual-gate-failed" | "visual-extract-failed";
	readonly page?: number;
	readonly message: string;
}

export interface VisualMirrorFileOutcome {
	/** True when the visual mirror file was written, updated, or removed on disk. */
	readonly changed: boolean;
	readonly pagesRendered: number;
	readonly pagesGated: number;
	readonly pagesReused: number;
	readonly pagesExtracted: number;
	readonly visualMirrorExists: boolean;
	readonly diagnostics: readonly VisualMirrorDiagnostic[];
}

function sha256Text(text: string): string {
	return createHash("sha256").update(text).digest("hex");
}

/** Read a persisted visual manifest for `virtualPath`, or undefined when absent/corrupt. */
export function loadVisualManifest(root: string, virtualPath: string): VisualMirrorManifest | undefined {
	const path = parsedVisualManifestPath(root, virtualPath);
	if (!existsSync(path)) return undefined;
	try {
		const raw: unknown = JSON.parse(readFileSync(path, "utf8"));
		if (typeof raw !== "object" || raw === null) return undefined;
		const record = raw as Record<string, unknown>;
		if (record.version !== VISUAL_MANIFEST_VERSION || record.virtualPath !== virtualPath) return undefined;
		if (typeof record.pages !== "object" || record.pages === null || Array.isArray(record.pages)) return undefined;
		return record as unknown as VisualMirrorManifest;
	} catch {
		return undefined;
	}
}

export function saveVisualManifest(root: string, virtualPath: string, manifest: VisualMirrorManifest): void {
	const sortedPages = Object.fromEntries(Object.entries(manifest.pages).sort(([a], [b]) => Number(a) - Number(b)));
	writeFileAtomic(
		parsedVisualManifestPath(root, virtualPath),
		`${JSON.stringify({ ...manifest, pages: sortedPages }, null, 2)}\n`,
	);
}

/** Remove the visual mirror markdown, its layout pages sidecar, and its incremental manifest. */
export function removeVisualMirror(root: string, virtualPath: string): boolean {
	const visualPath = parsedVisualOutputPath(root, virtualPath);
	const manifestPath = parsedVisualManifestPath(root, virtualPath);
	const existed = existsSync(visualPath) || existsSync(manifestPath);
	rmSync(visualPath, { force: true });
	writePagesSidecar(visualPath, "", undefined);
	rmSync(manifestPath, { force: true });
	return existed;
}

/**
 * Slice the base mirror's markdown into per-page snippets using its verified layout pages sidecar.
 * Pages without a layout span return empty string; the gate still sees the full page image.
 */
export function extractBasePageTexts(baseMirrorPath: string): ReadonlyMap<number, string> {
	const byPage = new Map<number, string>();
	if (!existsSync(baseMirrorPath)) return byPage;
	let markdown: string;
	try {
		markdown = readFileSync(baseMirrorPath, "utf8");
	} catch {
		return byPage;
	}
	const pages = readPagesSidecar(baseMirrorPath, markdown);
	if (pages === undefined || pages.spans.length === 0) return byPage;
	const lines = markdown.split("\n");
	for (const span of pages.spans) {
		const slice = lines
			.slice(span.lineStart - 1, span.lineEnd)
			.join("\n")
			.trim();
		byPage.set(span.page, slice);
	}
	return byPage;
}

/** Render all visually-rich pages accumulated in `manifest` into one deterministic Markdown mirror + layout page spans. */
export function renderVisualMirrorMarkdown(manifest: VisualMirrorManifest): {
	readonly markdown: string;
	readonly pages: ParsedPages | undefined;
} {
	const richPages: PageMarkdownInput[] = Object.values(manifest.pages)
		.filter((record): record is VisualPageRecord & { readonly markdown: string } => {
			return record.visuallyRich && typeof record.markdown === "string" && record.markdown.trim().length > 0;
		})
		.sort((a, b) => a.page - b.page)
		.map((record) => {
			const extractorTag =
				record.extractorKind !== undefined && record.extractorModel !== undefined
					? `${record.extractorKind}:${record.extractorModel}`
					: "visual";
			const body = normalizeMarkdown(record.markdown.trim());
			return {
				page: record.page,
				markdown: `## Page ${record.page} (Visual Extraction — ${extractorTag})\n\n${body}`,
			};
		});

	if (richPages.length === 0) {
		return { markdown: "", pages: undefined };
	}
	const pageCount = Math.max(manifest.pageCount, richPages[richPages.length - 1]?.page ?? 0);
	const joined = joinPages(richPages, "layout", pageCount);
	return { markdown: normalizeMarkdown(joined.markdown), pages: joined.pages };
}

/** True when a source file's visual manifest already covers this exact source version and gate/extractor settings. */
export function isVisualMirrorUpToDate(
	root: string,
	virtualPath: string,
	sourceVersion: string,
	options: VisualMirrorSyncOptions,
): boolean {
	const renderer =
		options.renderer ??
		createDefaultPageRenderer({
			maxPagesPerFile: options.maxPagesPerFile,
			maxWidthPx: options.maxWidthPx,
		});
	if (!renderer.supports(virtualPath)) return true;
	const manifest = loadVisualManifest(root, virtualPath);
	if (manifest === undefined) return false;
	const maxPages = options.maxPagesPerFile ?? DEFAULT_VISUAL_MAX_PAGES_PER_FILE;
	const maxWidth = options.maxWidthPx ?? DEFAULT_VISUAL_MAX_WIDTH_PX;
	const extractors = resolveExtractors(options);
	if (
		manifest.sourceVersion !== sourceVersion ||
		manifest.rendererFingerprint !== renderer.fingerprint ||
		manifest.gateFingerprint !== options.gate.fingerprint ||
		manifest.extractorFingerprint !== extractors.combinedFingerprint ||
		manifest.maxPagesPerFile !== maxPages ||
		manifest.maxWidthPx !== maxWidth
	) {
		return false;
	}
	const hasRich = Object.values(manifest.pages).some(
		(record) => record.visuallyRich && typeof record.markdown === "string" && record.markdown.trim().length > 0,
	);
	const visualPath = parsedVisualOutputPath(root, virtualPath);
	return hasRich ? existsSync(visualPath) : !existsSync(visualPath);
}

/** Write (or remove) the visual mirror markdown + pages sidecar from `manifest`; returns whether on-disk content changed. */
function materializeVisualMirror(
	root: string,
	virtualPath: string,
	manifest: VisualMirrorManifest,
): {
	readonly changed: boolean;
	readonly exists: boolean;
} {
	const visualPath = parsedVisualOutputPath(root, virtualPath);
	const { markdown, pages } = renderVisualMirrorMarkdown(manifest);
	if (markdown.length === 0) {
		const existed = existsSync(visualPath);
		rmSync(visualPath, { force: true });
		writePagesSidecar(visualPath, "", undefined);
		return { changed: existed, exists: false };
	}
	const previousText = existsSync(visualPath) ? readFileSync(visualPath, "utf8") : undefined;
	const unchanged = previousText === markdown && readPagesSidecar(visualPath, markdown) !== undefined;
	if (!unchanged) {
		writeFileAtomic(visualPath, markdown);
		writePagesSidecar(visualPath, markdown, pages);
	}
	return { changed: !unchanged, exists: true };
}

/**
 * Incrementally gate and visually extract pages for one source file into its dedicated visual mirror.
 * Unchanged pages (identical rendered `imageSha256`, base text hash, and gate/extractor fingerprints)
 * are reused from the manifest with zero model calls.
 */
export async function syncVisualMirrorForFile(input: {
	readonly root: string;
	readonly virtualPath: string;
	readonly sourcePath: string;
	readonly sourceVersion: string;
	readonly baseMirrorPath: string;
	readonly readBytes: () => Promise<Uint8Array>;
	readonly options: VisualMirrorSyncOptions;
	readonly sharedLimiters?: VisualMirrorSharedLimiters;
	readonly force?: boolean;
	readonly signal?: AbortSignal;
}): Promise<VisualMirrorFileOutcome> {
	const maxPagesPerFile = input.options.maxPagesPerFile ?? DEFAULT_VISUAL_MAX_PAGES_PER_FILE;
	const maxWidthPx = input.options.maxWidthPx ?? DEFAULT_VISUAL_MAX_WIDTH_PX;
	const renderer = input.options.renderer ?? createDefaultPageRenderer({ maxPagesPerFile, maxWidthPx });

	if (!renderer.supports(input.virtualPath)) {
		return {
			changed: false,
			pagesRendered: 0,
			pagesGated: 0,
			pagesReused: 0,
			pagesExtracted: 0,
			visualMirrorExists: false,
			diagnostics: [],
		};
	}

	if (
		input.force !== true &&
		isVisualMirrorUpToDate(input.root, input.virtualPath, input.sourceVersion, input.options)
	) {
		return {
			changed: false,
			pagesRendered: 0,
			pagesGated: 0,
			pagesReused: 0,
			pagesExtracted: 0,
			visualMirrorExists: existsSync(parsedVisualOutputPath(input.root, input.virtualPath)),
			diagnostics: [],
		};
	}

	const diagnostics: VisualMirrorDiagnostic[] = [];
	const bytes = await input.readBytes();
	if (input.signal?.aborted) throw abortReason(input.signal);

	let renderedPages: readonly Awaited<ReturnType<PageRenderer["renderPages"]>>[number][];
	try {
		renderedPages = await renderer.renderPages({
			virtualPath: input.virtualPath,
			sourcePath: input.sourcePath,
			bytes,
			maxPagesPerFile,
			maxWidthPx,
			signal: input.signal,
		});
	} catch (error) {
		if (input.signal?.aborted) throw abortReason(input.signal);
		diagnostics.push({
			code: "visual-render-failed",
			message: error instanceof Error ? error.message : String(error),
		});
		return {
			changed: false,
			pagesRendered: 0,
			pagesGated: 0,
			pagesReused: 0,
			pagesExtracted: 0,
			visualMirrorExists: existsSync(parsedVisualOutputPath(input.root, input.virtualPath)),
			diagnostics,
		};
	}

	const previousManifest = input.force === true ? undefined : loadVisualManifest(input.root, input.virtualPath);
	const extractors = resolveExtractors(input.options);
	const baseTexts = extractBasePageTexts(input.baseMirrorPath);
	const nextPages: Record<string, VisualPageRecord> = {};
	let pagesGated = 0;
	let pagesReused = 0;
	let pagesExtracted = 0;

	const persistPartial = (completeSourceVersion: string): VisualMirrorManifest => {
		const manifest: VisualMirrorManifest = {
			version: VISUAL_MANIFEST_VERSION,
			virtualPath: input.virtualPath,
			sourcePath: input.sourcePath,
			sourceVersion: completeSourceVersion,
			pageCount: renderedPages.reduce((max, page) => Math.max(max, page.page), 0),
			rendererFingerprint: renderer.fingerprint,
			gateFingerprint: input.options.gate.fingerprint,
			extractorFingerprint: extractors.combinedFingerprint,
			maxPagesPerFile,
			maxWidthPx,
			pages: nextPages,
			updatedAt: new Date().toISOString(),
		};
		saveVisualManifest(input.root, input.virtualPath, manifest);
		return manifest;
	};

	let hadPageFailure = false;
	const limiters = input.sharedLimiters ?? createVisualMirrorSharedLimiters(input.options);
	const gateLimiter = limiters.gate;
	const vlmLimiter = limiters.vlm;
	const ocrLimiter = limiters.ocr;
	const extractPromises: Promise<void>[] = [];

	// Two-lane pipeline:
	// 1) Gate lane runs fast vision-classifier calls at `gateConcurrency` (default 12), and as soon as a page is
	//    classified as `vlm` or `ocr`, immediately enqueues its extraction into the VLM (default 8) or OCR (default 2)
	//    lane without blocking gate workers from screening the remaining pages.
	// 2) Both lanes overlap completely, so slow VLM/OCR pages never stall gate screening for the rest of the document.
	const scheduleExtraction = (args: {
		readonly rendered: (typeof renderedPages)[number];
		readonly pageKey: string;
		readonly extractedText: string;
		readonly baseTextSha256: string;
		readonly cached: VisualPageRecord | undefined;
		readonly route: Exclude<VisualPageRoute, "plain">;
		readonly probability: number;
		readonly targetExtractor: VisualPageExtractor;
	}): void => {
		const limiter = args.targetExtractor.kind === "ocr" ? ocrLimiter : vlmLimiter;
		const job = limiter.run(async () => {
			if (input.signal?.aborted) throw abortReason(input.signal);
			try {
				const extracted = await args.targetExtractor.extract({
					virtualPath: input.virtualPath,
					page: args.rendered.page,
					image: { data: args.rendered.data, mimeType: args.rendered.mimeType },
					extractedText: args.extractedText,
					signal: input.signal,
				});
				pagesExtracted += 1;
				nextPages[args.pageKey] = {
					page: args.rendered.page,
					imageSha256: args.rendered.sha256,
					baseTextSha256: args.baseTextSha256,
					gateFingerprint: input.options.gate.fingerprint,
					gateProbability: args.probability,
					route: args.route,
					visuallyRich: true,
					extractorFingerprint: args.targetExtractor.fingerprint,
					extractorKind: extracted.extractorKind,
					extractorModel: extracted.extractorModel,
					markdown: extracted.markdown,
					updatedAt: new Date().toISOString(),
				};
				// Persist incrementally as visually-rich/OCR pages finish so an abort never wastes completed calls.
				persistPartial(`${input.sourceVersion}:in-progress`);
			} catch (error) {
				if (input.signal?.aborted) throw abortReason(input.signal);
				hadPageFailure = true;
				diagnostics.push({
					code: "visual-extract-failed",
					page: args.rendered.page,
					message: `Page ${args.rendered.page} visual extraction (${args.route}) failed: ${error instanceof Error ? error.message : String(error)}`,
				});
				if (args.cached !== undefined) nextPages[args.pageKey] = args.cached;
			}
		});
		extractPromises.push(job);
	};

	const screenPage = async (rendered: (typeof renderedPages)[number]): Promise<void> => {
		if (input.signal?.aborted) throw abortReason(input.signal);
		const pageKey = String(rendered.page);
		const extractedText = baseTexts.get(rendered.page) ?? "";
		const baseTextSha256 = sha256Text(extractedText);
		const cached = previousManifest?.pages[pageKey];

		const gateCacheHit =
			cached !== undefined &&
			cached.imageSha256 === rendered.sha256 &&
			cached.baseTextSha256 === baseTextSha256 &&
			cached.gateFingerprint === input.options.gate.fingerprint;

		const cachedRoute: VisualPageRoute =
			cached?.route ?? (cached?.visuallyRich ? (cached.extractorKind ?? "vlm") : "plain");
		if (gateCacheHit && cachedRoute === "plain") {
			nextPages[pageKey] = cached;
			pagesReused += 1;
			return;
		}

		const cachedExpectedExtractor = gateCacheHit ? pickExtractorForRoute(cachedRoute, extractors) : undefined;
		if (
			gateCacheHit &&
			cachedRoute !== "plain" &&
			cachedExpectedExtractor !== undefined &&
			cached.extractorFingerprint === cachedExpectedExtractor.fingerprint &&
			typeof cached.markdown === "string"
		) {
			nextPages[pageKey] = cached;
			pagesReused += 1;
			return;
		}

		let route: VisualPageRoute;
		let probability: number;
		if (gateCacheHit) {
			route = cachedRoute;
			probability = cached.gateProbability;
			pagesReused += 1;
		} else {
			try {
				const verdict = await gateLimiter.run(async () => {
					if (input.signal?.aborted) throw abortReason(input.signal);
					return input.options.gate.judge({
						virtualPath: input.virtualPath,
						page: rendered.page,
						image: { data: rendered.data, mimeType: rendered.mimeType },
						extractedText,
						signal: input.signal,
					});
				});
				route = verdict.route ?? (verdict.visuallyRich ? "vlm" : "plain");
				probability = verdict.probability;
				pagesGated += 1;
			} catch (error) {
				if (input.signal?.aborted) throw abortReason(input.signal);
				hadPageFailure = true;
				diagnostics.push({
					code: "visual-gate-failed",
					page: rendered.page,
					message: `Page ${rendered.page} visual gate failed: ${error instanceof Error ? error.message : String(error)}`,
				});
				if (cached !== undefined) nextPages[pageKey] = cached;
				return;
			}
		}

		const targetExtractor = pickExtractorForRoute(route, extractors);
		if (route === "plain" || targetExtractor === undefined) {
			nextPages[pageKey] = {
				page: rendered.page,
				imageSha256: rendered.sha256,
				baseTextSha256,
				gateFingerprint: input.options.gate.fingerprint,
				gateProbability: probability,
				route: "plain",
				visuallyRich: false,
				updatedAt: new Date().toISOString(),
			};
			return;
		}

		scheduleExtraction({
			rendered,
			pageKey,
			extractedText,
			baseTextSha256,
			cached,
			route,
			probability,
			targetExtractor,
		});
	};

	await Promise.all(renderedPages.map((rendered) => screenPage(rendered)));
	await Promise.all(extractPromises);

	// Keep the final sourceVersion only when every rendered page completed, so a transient page error retries next run
	// while still reusing every already-succeeded page from the manifest.
	const finalManifest = persistPartial(hadPageFailure ? `${input.sourceVersion}:partial` : input.sourceVersion);
	const materialized = materializeVisualMirror(input.root, input.virtualPath, finalManifest);

	return {
		changed: materialized.changed,
		pagesRendered: renderedPages.length,
		pagesGated,
		pagesReused,
		pagesExtracted,
		visualMirrorExists: materialized.exists,
		diagnostics,
	};
}

/**
 * Targeted incremental accumulation: extract specific pages (e.g. `[2, 4]`) via VLM/OCR and merge them into
 * the document's existing visual mirror without discarding previously accumulated pages.
 */
export async function accumulateVisualPages(input: {
	readonly root: string;
	readonly virtualPath: string;
	readonly sourcePath: string;
	readonly sourceVersion: string;
	readonly bytes: Uint8Array;
	readonly pageNumbers: readonly number[];
	readonly pageCount?: number;
	readonly baseMirrorPath?: string;
	readonly renderer?: PageRenderer;
	readonly extractor: VisualPageExtractor;
	readonly maxWidthPx?: number;
	readonly concurrency?: number;
	readonly signal?: AbortSignal;
}): Promise<{ readonly extractedPages: readonly number[]; readonly visualMirrorPath: string }> {
	const maxWidthPx = input.maxWidthPx ?? DEFAULT_VISUAL_MAX_WIDTH_PX;
	const renderer = input.renderer ?? createDefaultPageRenderer({ maxWidthPx });
	const rendered = await renderer.renderPages({
		virtualPath: input.virtualPath,
		sourcePath: input.sourcePath,
		bytes: input.bytes,
		pageNumbers: input.pageNumbers,
		maxPagesPerFile: Math.max(input.pageNumbers.length, DEFAULT_VISUAL_MAX_PAGES_PER_FILE),
		maxWidthPx,
		signal: input.signal,
	});
	const existing = loadVisualManifest(input.root, input.virtualPath);
	const baseTexts =
		input.baseMirrorPath !== undefined ? extractBasePageTexts(input.baseMirrorPath) : new Map<number, string>();
	const mergedPages: Record<string, VisualPageRecord> = { ...(existing?.pages ?? {}) };
	const extractedPages: number[] = [];
	const defaultConcurrency =
		input.extractor.kind === "ocr" ? DEFAULT_VISUAL_OCR_CONCURRENCY : DEFAULT_VISUAL_PAGE_CONCURRENCY;
	const limiter = new AsyncSemaphore(Math.max(1, Math.floor(input.concurrency ?? defaultConcurrency)));

	await Promise.all(
		rendered.map((page) =>
			limiter.run(async () => {
				if (input.signal?.aborted) throw abortReason(input.signal);
				const extractedText = baseTexts.get(page.page) ?? "";
				const result = await input.extractor.extract({
					virtualPath: input.virtualPath,
					page: page.page,
					image: { data: page.data, mimeType: page.mimeType },
					extractedText,
					signal: input.signal,
				});
				mergedPages[String(page.page)] = {
					page: page.page,
					imageSha256: page.sha256,
					baseTextSha256: sha256Text(extractedText),
					gateFingerprint: existing?.gateFingerprint ?? "targeted",
					gateProbability: 1,
					route: result.extractorKind,
					visuallyRich: true,
					extractorFingerprint: input.extractor.fingerprint,
					extractorKind: result.extractorKind,
					extractorModel: result.extractorModel,
					markdown: result.markdown,
					updatedAt: new Date().toISOString(),
				};
				extractedPages.push(page.page);
			}),
		),
	);
	extractedPages.sort((a, b) => a - b);

	const maxSeenPage = Object.values(mergedPages).reduce((max, record) => Math.max(max, record.page), 0);
	const manifest: VisualMirrorManifest = {
		version: VISUAL_MANIFEST_VERSION,
		virtualPath: input.virtualPath,
		sourcePath: input.sourcePath,
		sourceVersion: input.sourceVersion,
		pageCount: Math.max(input.pageCount ?? 0, existing?.pageCount ?? 0, maxSeenPage),
		rendererFingerprint: existing?.rendererFingerprint ?? renderer.fingerprint,
		gateFingerprint: existing?.gateFingerprint ?? "targeted",
		extractorFingerprint: input.extractor.fingerprint,
		maxPagesPerFile: existing?.maxPagesPerFile ?? DEFAULT_VISUAL_MAX_PAGES_PER_FILE,
		maxWidthPx: existing?.maxWidthPx ?? maxWidthPx,
		pages: mergedPages,
		updatedAt: new Date().toISOString(),
	};
	saveVisualManifest(input.root, input.virtualPath, manifest);
	materializeVisualMirror(input.root, input.virtualPath, manifest);
	return {
		extractedPages,
		visualMirrorPath: parsedVisualOutputPath(input.root, input.virtualPath),
	};
}

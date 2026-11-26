import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createParsedMirrorMapper } from "../../src/agent/parsed-mirror-tools.ts";
import { minSyncDocumentPath, minSyncVisualDocumentPath, minSyncWorkspaceRoot } from "../../src/minsync/paths.ts";
import { buildMinSyncPathMap, syncMinSyncWorkspace } from "../../src/minsync/workspace.ts";
import {
	loadMirrorIndex,
	parsedVisualManifestPath,
	parsedVisualOutputPath,
	readPagesSidecar,
	syncParsedMirrors,
} from "../../src/mirror/index.ts";
import {
	accumulateVisualPages,
	loadVisualManifest,
	type VisualMirrorSyncOptions,
} from "../../src/mirror/visual-mirror.ts";
import type { PageRenderer, RenderedPage } from "../../src/parser/page-render.ts";
import { joinPages } from "../../src/parser/pages.ts";
import { ParserRegistry } from "../../src/parser/registry.ts";
import { type ParseInput, type ParseOutput, Parser } from "../../src/parser/types.ts";
import type { VisualPageExtractor } from "../../src/parser/visual-extractor.ts";
import type { VisualPageGate } from "../../src/parser/visual-gate.ts";

class StubPagedPdfParser extends Parser {
	readonly name = "stub-pdf";
	readonly extensions = [".pdf"] as const;

	async parse(input: ParseInput): Promise<ParseOutput> {
		const raw = Buffer.from(input.bytes).toString("utf8");
		const pages = raw.split("\f").map((text, idx) => ({ page: idx + 1, markdown: text.trim() }));
		const joined = joinPages(pages, "layout", pages.length);
		return {
			markdown: joined.markdown,
			...(joined.pages ? { pages: joined.pages } : {}),
		};
	}
}

function fakeRenderedPage(page: number, marker: string): RenderedPage {
	const data = new Uint8Array(Buffer.from(`PNG:${page}:${marker}`));
	return {
		page,
		data,
		mimeType: "image/png",
		sha256: `sha:${page}:${marker}`,
	};
}

let root: string;
let docs: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-visual-mirror-test-"));
	docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

describe("visual-rich page gating and multi-mirror incremental sync", () => {
	it("gates per page, creates a second visual mirror for visually rich pages only, and reuses cache incrementally", async () => {
		const pdfPath = join(docs, "quarterly.pdf");
		// Keep a sibling file so deleting quarterly.pdf does not trigger empty-root preservation.
		writeFileSync(join(docs, "readme.txt"), "Quarterly workspace notes");
		// 3 pages separated by form-feed (\f): page 1 text, page 2 chart, page 3 text
		writeFileSync(pdfPath, "Page 1 executive summary\fPage 2 chart caption only\fPage 3 appendix");

		let pageMarkers: Record<number, string> = { 1: "text-v1", 2: "chart-v1", 3: "text-v1" };
		const renderer: PageRenderer = {
			fingerprint: "stub-renderer:v1",
			supports: (virtualPath) => virtualPath.endsWith(".pdf"),
			renderPages: vi.fn(async ({ pageNumbers }) => {
				const wanted = pageNumbers ?? [1, 2, 3];
				return wanted.map((p: number) => fakeRenderedPage(p, pageMarkers[p] ?? "blank"));
			}),
		};

		const gateJudge = vi.fn(async ({ image }) => {
			const tag = Buffer.from(image.data).toString("utf8");
			const rich = tag.includes("chart");
			return {
				visuallyRich: rich,
				probability: rich ? 0.96 : 0.04,
				model: "dec-1",
				backend: "bivia",
			};
		});
		const gate: VisualPageGate = { fingerprint: "bivia:dec-1:0.6:v1", judge: gateJudge };

		const extractFn = vi.fn(async ({ page, image }) => {
			const tag = Buffer.from(image.data).toString("utf8");
			return {
				markdown: `Extracted chart values on page ${page} (${tag}): Revenue = 48.2M`,
				extractorKind: "vlm" as const,
				extractorModel: "openrouter/google/gemini-3.5-flash-lite",
			};
		});
		const extractor: VisualPageExtractor = {
			kind: "vlm",
			modelId: "openrouter/google/gemini-3.5-flash-lite",
			fingerprint: "vlm:openrouter/google/gemini-3.5-flash-lite:v1",
			extract: extractFn,
		};

		const registry = new ParserRegistry([new StubPagedPdfParser()]);
		const visualParse: VisualMirrorSyncOptions = { renderer, gate, extractor };

		// 1. First sync: gates all 3 pages, extracts ONLY page 2
		const first = await syncParsedMirrors({ root, searchPaths: [docs], registry, visualParse });
		expect(first.written).toBe(1);
		expect(gateJudge).toHaveBeenCalledTimes(3);
		expect(extractFn).toHaveBeenCalledTimes(1);
		expect(extractFn).toHaveBeenCalledWith(expect.objectContaining({ page: 2 }));

		const virtualPath = "/docs/quarterly.pdf";
		const index = loadMirrorIndex(root);
		const baseMirrorPath = index.entries[virtualPath]!.outputPath;
		const visualMirrorPath = parsedVisualOutputPath(root, virtualPath);
		expect(existsSync(baseMirrorPath)).toBe(true);
		expect(existsSync(visualMirrorPath)).toBe(true);

		const visualMarkdown = readFileSync(visualMirrorPath, "utf8");
		expect(visualMarkdown).toContain("Revenue = 48.2M");
		expect(visualMarkdown).not.toContain("Page 1 executive summary");

		// Visual mirror has its own layout pages sidecar pointing to exact page 2
		const visualPages = readPagesSidecar(visualMirrorPath, visualMarkdown);
		expect(visualPages?.mode).toBe("layout");
		expect(visualPages?.spans).toHaveLength(1);
		expect(visualPages?.spans[0]?.page).toBe(2);

		// 2. Second sync with unchanged file: 0 gate calls, 0 extract calls!
		gateJudge.mockClear();
		extractFn.mockClear();
		const second = await syncParsedMirrors({ root, searchPaths: [docs], registry, visualParse });
		expect(second.written).toBe(0);
		expect(gateJudge).not.toHaveBeenCalled();
		expect(extractFn).not.toHaveBeenCalled();

		// 3. Edit source so ONLY page 3 becomes a chart (pages 1 and 2 images stay identical!)
		await new Promise((r) => setTimeout(r, 20));
		writeFileSync(pdfPath, "Page 1 executive summary\fPage 2 chart caption only\fPage 3 new diagram");
		pageMarkers = { 1: "text-v1", 2: "chart-v1", 3: "chart-diagram-v2" };
		gateJudge.mockClear();
		extractFn.mockClear();

		const third = await syncParsedMirrors({ root, searchPaths: [docs], registry, visualParse });
		expect(third.written).toBe(1);
		// Only page 3 changed imageSha256/baseTextSha256 -> pages 1 and 2 reused from manifest!
		expect(gateJudge).toHaveBeenCalledTimes(1);
		expect(gateJudge).toHaveBeenCalledWith(expect.objectContaining({ page: 3 }));
		expect(extractFn).toHaveBeenCalledTimes(1);
		expect(extractFn).toHaveBeenCalledWith(expect.objectContaining({ page: 3 }));

		const updatedVisualMarkdown = readFileSync(visualMirrorPath, "utf8");
		expect(updatedVisualMarkdown).toContain("Extracted chart values on page 2");
		expect(updatedVisualMarkdown).toContain("Extracted chart values on page 3");
		const updatedVisualPages = readPagesSidecar(visualMirrorPath, updatedVisualMarkdown);
		expect(updatedVisualPages?.spans.map((s) => s.page)).toEqual([2, 3]);

		// 4. MinSync workspace stages BOTH mirrors for the single source file and maps both back to original sourcePath
		const ws = minSyncWorkspaceRoot(root);
		const staged = syncMinSyncWorkspace(root, { workspacePath: ws });
		const stagedText = minSyncDocumentPath(ws, virtualPath);
		const stagedVisual = minSyncVisualDocumentPath(ws, virtualPath);
		expect(existsSync(stagedText)).toBe(true);
		expect(existsSync(stagedVisual)).toBe(true);
		expect(staged.entries.some((e) => e.minSyncPath === stagedText)).toBe(true);
		expect(staged.entries.some((e) => e.minSyncPath === stagedVisual)).toBe(true);

		const pathMap = buildMinSyncPathMap(root, ws);
		const textMapped = pathMap.get(stagedText);
		const visualMapped = pathMap.get(stagedVisual);
		expect(textMapped?.virtualPath).toBe(virtualPath);
		expect(textMapped?.mirrorKind).toBe("text");
		expect(visualMapped?.virtualPath).toBe(virtualPath);
		expect(visualMapped?.sourcePath).toBe(pdfPath);
		expect(visualMapped?.mirrorKind).toBe("visual");
		expect(visualMapped?.parsedOutputPath).toBe(visualMirrorPath);

		// 5. ParsedMirrorMapper resolves staged visual mirror back to original source and locates its page
		const mapper = createParsedMirrorMapper({ workspaceRoot: root, searchPaths: [docs] });
		const resolved = mapper.toSourcePath(stagedVisual);
		expect(resolved?.virtualPath).toBe(virtualPath);
		expect(resolved?.pagesMirrorPath).toBe(visualMirrorPath);
		const lineOfPage3 = updatedVisualMarkdown.split("\n").findIndex((l) => l.includes("page 3")) + 1;
		const located = mapper.locateLine(stagedVisual, lineOfPage3);
		expect(located?.pageStart).toBe(3);
		expect(located?.pageEnd).toBe(3);

		// 6. Delete source file -> both base mirror, visual mirror, visual manifest, and staged files are cleaned up
		unlinkSync(pdfPath);
		await syncParsedMirrors({ root, searchPaths: [docs], registry, visualParse });
		expect(existsSync(visualMirrorPath)).toBe(false);
		expect(existsSync(parsedVisualManifestPath(root, virtualPath))).toBe(false);
		syncMinSyncWorkspace(root, { workspacePath: ws });
		expect(existsSync(stagedText)).toBe(false);
		expect(existsSync(stagedVisual)).toBe(false);
	});

	it("enriches already-indexed files when visualParse is enabled later without requiring force", async () => {
		const pdfPath = join(docs, "existing.pdf");
		writeFileSync(pdfPath, "Cover page\fFinancial chart page");
		const registry = new ParserRegistry([new StubPagedPdfParser()]);

		// Initial sync WITHOUT visualParse
		await syncParsedMirrors({ root, searchPaths: [docs], registry });
		const virtualPath = "/docs/existing.pdf";
		expect(existsSync(parsedVisualOutputPath(root, virtualPath))).toBe(false);

		// Later sync WITH visualParse enabled (force = false)
		const onBatch = vi.fn(async () => undefined);
		const renderer: PageRenderer = {
			fingerprint: "stub-renderer:v1",
			supports: (vp) => vp.endsWith(".pdf"),
			renderPages: async () => [fakeRenderedPage(1, "cover"), fakeRenderedPage(2, "chart")],
		};
		const gate: VisualPageGate = {
			fingerprint: "bivia:dec-1:0.6:v1",
			judge: async ({ page }) => ({
				visuallyRich: page === 2,
				probability: page === 2 ? 0.92 : 0.05,
				model: "dec-1",
				backend: "bivia",
			}),
		};
		const extractor: VisualPageExtractor = {
			kind: "ocr",
			modelId: "tesseract:en",
			fingerprint: "ocr:tesseract:en:v1",
			extract: async ({ page }) => ({
				markdown: `OCR chart page ${page}: CAGR 19.4%`,
				extractorKind: "ocr",
				extractorModel: "tesseract:en",
			}),
		};

		const second = await syncParsedMirrors({
			root,
			searchPaths: [docs],
			registry,
			visualParse: { renderer, gate, extractor },
			onBatch,
		});
		expect(second.written).toBe(1);
		expect(onBatch).toHaveBeenCalled();
		const visualPath = parsedVisualOutputPath(root, virtualPath);
		expect(existsSync(visualPath)).toBe(true);
		expect(readFileSync(visualPath, "utf8")).toContain("CAGR 19.4%");
		expect(loadVisualManifest(root, virtualPath)?.pages["2"]?.visuallyRich).toBe(true);
	});

	it("supports targeted incremental accumulation of specific pages onto an existing visual mirror", async () => {
		const pdfPath = join(docs, "manual.pdf");
		writeFileSync(pdfPath, "P1\fP2\fP3\fP4");
		const virtualPath = "/docs/manual.pdf";
		const renderer: PageRenderer = {
			fingerprint: "stub-renderer:v1",
			supports: () => true,
			renderPages: async ({ pageNumbers }) =>
				(pageNumbers ?? [1, 2, 3, 4]).map((p) => fakeRenderedPage(p, `img-${p}`)),
		};
		const extractor: VisualPageExtractor = {
			kind: "vlm",
			modelId: "openrouter/google/gemini-3.5-flash-lite",
			fingerprint: "vlm:gemini:v1",
			extract: async ({ page }) => ({
				markdown: `Visual details for page ${page}`,
				extractorKind: "vlm",
				extractorModel: "openrouter/google/gemini-3.5-flash-lite",
			}),
		};

		await accumulateVisualPages({
			root,
			virtualPath,
			sourcePath: pdfPath,
			sourceVersion: "100:1",
			bytes: readFileSync(pdfPath),
			pageNumbers: [2],
			pageCount: 4,
			renderer,
			extractor,
		});
		expect(readFileSync(parsedVisualOutputPath(root, virtualPath), "utf8")).toContain("Visual details for page 2");

		// Accumulate page 4 onto the SAME visual mirror — page 2 must be preserved!
		await accumulateVisualPages({
			root,
			virtualPath,
			sourcePath: pdfPath,
			sourceVersion: "100:1",
			bytes: readFileSync(pdfPath),
			pageNumbers: [4],
			pageCount: 4,
			renderer,
			extractor,
		});
		const combined = readFileSync(parsedVisualOutputPath(root, virtualPath), "utf8");
		expect(combined).toContain("Visual details for page 2");
		expect(combined).toContain("Visual details for page 4");
		const pages = readPagesSidecar(parsedVisualOutputPath(root, virtualPath), combined);
		expect(pages?.spans.map((s) => s.page)).toEqual([2, 4]);
	});
});

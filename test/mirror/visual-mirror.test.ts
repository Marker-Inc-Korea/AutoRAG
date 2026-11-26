import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createParsedMirrorMapper, runMirrorGrep } from "../../src/agent/parsed-mirror-tools.ts";
import { coalesceCrossMirrorPageHits } from "../../src/minsync/method.ts";
import { minSyncDocumentPath, minSyncVisualDocumentPath, minSyncWorkspaceRoot } from "../../src/minsync/paths.ts";
import { buildMinSyncPathMap, syncMinSyncWorkspace } from "../../src/minsync/workspace.ts";
import {
	detectMirrorStaleness,
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
import { ParseError } from "../../src/parser/errors.ts";
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

	it("handles simultaneous multi-file add, modify, delete, visual-removal, parse-failure, staleness, and single-file grep across both mirrors", async () => {
		class FailingOnCorruptParser extends StubPagedPdfParser {
			override async parse(input: ParseInput): Promise<ParseOutput> {
				if (Buffer.from(input.bytes).toString("utf8").includes("CORRUPT")) {
					throw new ParseError(this.name, input.virtualPath, new Error("corrupt pdf"));
				}
				return super.parse(input);
			}
		}

		// Seed 6 files: a(visual), b(text), c(visual->will delete), d(text->will delete), e(visual->will become text), f(visual->will corrupt)
		writeFileSync(join(docs, "a.pdf"), "A intro\fA CHART_TOKEN");
		writeFileSync(join(docs, "b.pdf"), "B plain page 1\fB plain page 2");
		writeFileSync(join(docs, "c.pdf"), "C CHART_OLD");
		writeFileSync(join(docs, "d.pdf"), "D plain");
		writeFileSync(join(docs, "e.pdf"), "E CHART_TEMP");
		writeFileSync(join(docs, "f.pdf"), "F CHART_VALID_INITIAL");

		const renderer: PageRenderer = {
			fingerprint: "churn-renderer:v1",
			supports: (vp) => vp.endsWith(".pdf"),
			renderPages: async ({ bytes, pageNumbers }) => {
				const rawPages = Buffer.from(bytes).toString("utf8").split("\f");
				const wanted = pageNumbers ?? rawPages.map((_, idx) => idx + 1);
				return wanted.map((p) => fakeRenderedPage(p, rawPages[p - 1] ?? ""));
			},
		};
		const gateJudge = vi.fn(async ({ image }: { readonly image: { readonly data: Uint8Array } }) => {
			const text = Buffer.from(image.data).toString("utf8");
			const rich = text.includes("CHART_");
			return {
				visuallyRich: rich,
				probability: rich ? 0.95 : 0.05,
				model: "typesafe/jev-1.13",
				backend: "openrouter",
			};
		});
		const extractFn = vi.fn(
			async ({ page, image }: { readonly page: number; readonly image: { readonly data: Uint8Array } }) => {
				const text = Buffer.from(image.data).toString("utf8");
				return {
					markdown: `Visual extraction p.${page}: ${text.replace(/^PNG:\d+:/u, "")}`,
					extractorKind: "vlm" as const,
					extractorModel: "openrouter/google/gemini-3.5-flash-lite",
				};
			},
		);

		const registry = new ParserRegistry([new FailingOnCorruptParser()]);
		const visualParse: VisualMirrorSyncOptions = {
			renderer,
			gate: { fingerprint: "jev-1.13:v1", judge: gateJudge },
			extractor: { kind: "vlm", modelId: "gemini", fingerprint: "gemini:v1", extract: extractFn },
			concurrency: 3,
		};
		const ws = minSyncWorkspaceRoot(root);

		// 1. Initial sync WITHOUT visualParse, then check staleness WITH visualParse -> must report visual-parse-pending!
		await syncParsedMirrors({ root, searchPaths: [docs], registry, batchSize: 2 });
		const staleBefore = await detectMirrorStaleness({ root, searchPaths: [docs], registry, visualParse });
		expect(staleBefore.length).toBe(6);
		expect(staleBefore.every((d) => d.reason === "visual-parse-pending")).toBe(true);

		// 2. Sync WITH visualParse and per-batch MinSync staging (batchSize = 2)
		let batchRuns = 0;
		await syncParsedMirrors({
			root,
			searchPaths: [docs],
			registry,
			visualParse,
			batchSize: 2,
			onBatch: async () => {
				batchRuns += 1;
				syncMinSyncWorkspace(root, { workspacePath: ws });
				return undefined;
			},
		});
		expect(batchRuns).toBeGreaterThanOrEqual(3);
		expect(await detectMirrorStaleness({ root, searchPaths: [docs], registry, visualParse })).toEqual([]);
		expect(existsSync(minSyncVisualDocumentPath(ws, "/docs/a.pdf"))).toBe(true);
		expect(existsSync(minSyncVisualDocumentPath(ws, "/docs/b.pdf"))).toBe(false);
		expect(existsSync(minSyncVisualDocumentPath(ws, "/docs/c.pdf"))).toBe(true);
		expect(existsSync(minSyncVisualDocumentPath(ws, "/docs/e.pdf"))).toBe(true);
		expect(existsSync(minSyncVisualDocumentPath(ws, "/docs/f.pdf"))).toBe(true);

		// Verify single-file grep AND glob-filtered grep find hits inside BOTH base and visual mirrors!
		const mapper = createParsedMirrorMapper({ workspaceRoot: root, searchPaths: [docs] });
		const fileTarget = mapper.toMirrorTarget(join(docs, "a.pdf"));
		expect(fileTarget?.kind).toBe("file");
		const singleFileGrep = await runMirrorGrep({
			mapper,
			target: fileTarget!,
			pattern: "Visual extraction p\\.2",
		});
		expect(singleFileGrep.matchCount).toBe(1);
		expect(singleFileGrep.output).toContain("p.2");

		const globGrep = await runMirrorGrep({
			mapper,
			target: mapper.toMirrorTarget(docs)!,
			pattern: "CHART_TOKEN",
			glob: "*.pdf",
		});
		// Matches BOTH base mirror and visual mirror of a.pdf
		expect(globGrep.matchCount).toBe(2);

		// 3. Simultaneous churn: add g(visual) & h(text), delete c & d, edit a(add p.3 visual keeping p.1/p.2 cached),
		// edit e(no longer visual), corrupt f(parse failure)
		await new Promise((r) => setTimeout(r, 25));
		unlinkSync(join(docs, "c.pdf"));
		unlinkSync(join(docs, "d.pdf"));
		writeFileSync(join(docs, "a.pdf"), "A intro\fA CHART_TOKEN\fA CHART_NEW_P3");
		writeFileSync(join(docs, "e.pdf"), "E plain text now");
		writeFileSync(join(docs, "f.pdf"), "CORRUPT");
		writeFileSync(join(docs, "g.pdf"), "G cover\fG CHART_ADDED");
		writeFileSync(join(docs, "h.pdf"), "H plain added");

		gateJudge.mockClear();
		extractFn.mockClear();
		await syncParsedMirrors({
			root,
			searchPaths: [docs],
			registry,
			visualParse,
			batchSize: 2,
			onBatch: async () => {
				syncMinSyncWorkspace(root, { workspacePath: ws });
				return undefined;
			},
		});

		// Deleted files c & d: both base and visual mirrors + manifests + staged files are gone
		expect(existsSync(minSyncDocumentPath(ws, "/docs/c.pdf"))).toBe(false);
		expect(existsSync(minSyncVisualDocumentPath(ws, "/docs/c.pdf"))).toBe(false);
		expect(existsSync(parsedVisualManifestPath(root, "/docs/c.pdf"))).toBe(false);
		expect(existsSync(minSyncDocumentPath(ws, "/docs/d.pdf"))).toBe(false);

		// Corrupted file f: stale visual mirror and manifest were removed by skipFile
		expect(existsSync(parsedVisualOutputPath(root, "/docs/f.pdf"))).toBe(false);
		expect(existsSync(parsedVisualManifestPath(root, "/docs/f.pdf"))).toBe(false);
		expect(existsSync(minSyncVisualDocumentPath(ws, "/docs/f.pdf"))).toBe(false);

		// File e became plain text: base mirror remains, visual mirror cleanly removed
		expect(existsSync(minSyncDocumentPath(ws, "/docs/e.pdf"))).toBe(true);
		expect(existsSync(minSyncVisualDocumentPath(ws, "/docs/e.pdf"))).toBe(false);

		// File a: pages 1 & 2 reused from cache; page 3 accumulated!
		const aVisual = readFileSync(minSyncVisualDocumentPath(ws, "/docs/a.pdf"), "utf8");
		expect(aVisual).toContain("CHART_TOKEN");
		expect(aVisual).toContain("CHART_NEW_P3");
		const aPages = readPagesSidecar(parsedVisualOutputPath(root, "/docs/a.pdf"), aVisual);
		expect(aPages?.spans.map((s) => s.page)).toEqual([2, 3]);

		// Added files g & h: g has BOTH mirrors, h has ONLY base mirror
		expect(existsSync(minSyncDocumentPath(ws, "/docs/g.pdf"))).toBe(true);
		expect(existsSync(minSyncVisualDocumentPath(ws, "/docs/g.pdf"))).toBe(true);
		expect(existsSync(minSyncDocumentPath(ws, "/docs/h.pdf"))).toBe(true);
		expect(existsSync(minSyncVisualDocumentPath(ws, "/docs/h.pdf"))).toBe(false);

		// Unchanged file b was NOT re-gated during churn!
		const gatedImages = gateJudge.mock.calls.map(([arg]) => Buffer.from(arg.image.data).toString("utf8"));
		expect(gatedImages.some((s) => s.includes("B plain"))).toBe(false);
		// Pages 1 & 2 of a.pdf were NOT re-gated!
		expect(gatedImages.some((s) => s.includes("A intro"))).toBe(false);
		expect(gatedImages.some((s) => s.includes("CHART_TOKEN"))).toBe(false);
		// Only genuinely new/changed pages were gated: a p.3, e p.1, g p.1-2, h p.1 = 5 calls total!
		expect(gateJudge).toHaveBeenCalledTimes(5);
	});

	it("coalesces same-page text + visual hits into one slot without crowding out other pages", () => {
		const coalesced = coalesceCrossMirrorPageHits([
			{
				id: "1",
				source: "/docs/report.pdf",
				content: "Page 7 prose text",
				score: 0.95,
				metadata: { pageStart: 7, pageEnd: 7 },
			},
			{
				id: "2",
				source: "/docs/report.pdf",
				content: "Page 7 bar chart: Q4 = 48.2M",
				score: 0.9,
				metadata: { pageStart: 7, pageEnd: 7, mirrorKind: "visual" },
			},
			{
				id: "3",
				source: "/docs/report.pdf",
				content: "Page 7 chart footnote: USD millions",
				score: 0.85,
				metadata: { pageStart: 7, pageEnd: 7, mirrorKind: "visual" },
			},
			{
				id: "4",
				source: "/docs/report.pdf",
				content: "Page 12 conclusion",
				score: 0.7,
				metadata: { pageStart: 12, pageEnd: 12 },
			},
		]);
		expect(coalesced).toHaveLength(2);
		expect(coalesced[0]?.content).toContain("Page 7 prose text");
		expect(coalesced[0]?.content).toContain("Page 7 bar chart: Q4 = 48.2M");
		expect(coalesced[0]?.content).toContain("Page 7 chart footnote");
		expect(coalesced[0]?.metadata?.hasVisualComplement).toBe(true);
		expect(coalesced[1]?.metadata?.pageStart).toBe(12);
	});
});

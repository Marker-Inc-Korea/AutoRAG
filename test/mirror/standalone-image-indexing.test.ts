import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
	EMBEDDINGGEMMA2_PASSAGE_PREFIX,
	EMBEDDINGGEMMA2_QUERY_PREFIX,
	resolveEmbeddingPrefixes,
} from "../../src/embedding/prefixes.ts";
import {
	defaultNativeEmbedderDimension,
	isMultimodalEmbedder,
	isNativeEmbedderId,
	MULTIMODAL_EMBEDDINGGEMMA2_DIMENSION,
	MULTIMODAL_EMBEDDINGGEMMA2_ID,
} from "../../src/minsync/embedder-config.ts";
import { coalesceCrossMirrorPageHits } from "../../src/minsync/method.ts";
import { buildMinSyncPathMap, syncMinSyncWorkspace } from "../../src/minsync/workspace.ts";
import { parsedVisualOutputPath } from "../../src/mirror/paths.ts";
import { syncParsedMirrors } from "../../src/mirror/sync.ts";
import { createDefaultParserRegistry } from "../../src/parser/defaults.ts";
import { ImageMetadataParser, isStandaloneImagePath, readImageDimensions } from "../../src/parser/image-metadata.ts";
import type { VisualPageExtractor } from "../../src/parser/visual-extractor.ts";
import type { VisualPageGate } from "../../src/parser/visual-gate.ts";
import type { RetrievalResult } from "../../src/retrieval/types.ts";

/** Minimal valid 16x12 PNG header + IEND chunk. */
function makePngBytes(width = 16, height = 12): Buffer {
	const buf = Buffer.alloc(33);
	buf.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a], 0);
	buf.writeUInt32BE(13, 8);
	buf.write("IHDR", 12, "ascii");
	buf.writeUInt32BE(width, 16);
	buf.writeUInt32BE(height, 20);
	buf[24] = 8;
	buf[25] = 2;
	return buf;
}

const tempDirs: string[] = [];
function makeTempDir(prefix: string): string {
	const dir = mkdtempSync(join(tmpdir(), prefix));
	tempDirs.push(dir);
	return dir;
}

afterEach(() => {
	while (tempDirs.length > 0) {
		rmSync(tempDirs.pop()!, { recursive: true, force: true });
	}
});

describe("standalone image indexing & multimodal embedder support", () => {
	it("ImageMetadataParser extracts format, dimensions, and byte size without OCR", async () => {
		const parser = new ImageMetadataParser();
		const bytes = makePngBytes(320, 240);
		expect(readImageDimensions(bytes)).toEqual({ width: 320, height: 240 });
		expect(isStandaloneImagePath("/docs/photo.PNG")).toBe(true);
		expect(isStandaloneImagePath("/docs/report.pdf")).toBe(false);

		const out = await parser.parse({
			virtualPath: "/docs/architecture.png",
			sourcePath: "/tmp/architecture.png",
			bytes,
		});
		expect(out.markdown).toContain("# architecture.png");
		expect(out.markdown).toContain("Format: PNG");
		expect(out.markdown).toContain("Dimensions: 320x240");
		expect(out.metadata?.parser).toBe("image-metadata");
	});

	it("createDefaultParserRegistry registers ImageMetadataParser only when images:true and ocr is off", () => {
		const defaultReg = createDefaultParserRegistry();
		expect(defaultReg.getForVirtualPath("/docs/cat.png")).toBeUndefined();

		const imgReg = createDefaultParserRegistry({ images: true });
		expect(imgReg.getForVirtualPath("/docs/cat.png")?.name).toBe("image-metadata");
		expect(imgReg.getForVirtualPath("/docs/dog.jpg")?.name).toBe("image-metadata");
		expect(imgReg.getForVirtualPath("/docs/chart.webp")?.name).toBe("image-metadata");

		const ocrReg = createDefaultParserRegistry({ images: true, ocr: { enabled: true } });
		expect(ocrReg.getForVirtualPath("/docs/cat.png")?.name).toBe("kordoc");
	});

	it("recognizes native-multimodal:google/embeddinggemma-2 dimension and retrieval prefixes", () => {
		expect(isNativeEmbedderId(MULTIMODAL_EMBEDDINGGEMMA2_ID)).toBe(true);
		expect(isMultimodalEmbedder({ id: MULTIMODAL_EMBEDDINGGEMMA2_ID })).toBe(true);
		expect(isMultimodalEmbedder({ id: "native:Qwen/Qwen3-Embedding-0.6B" })).toBe(false);
		expect(defaultNativeEmbedderDimension(MULTIMODAL_EMBEDDINGGEMMA2_ID)).toBe(MULTIMODAL_EMBEDDINGGEMMA2_DIMENSION);
		expect(resolveEmbeddingPrefixes({ id: MULTIMODAL_EMBEDDINGGEMMA2_ID })).toEqual({
			query: EMBEDDINGGEMMA2_QUERY_PREFIX,
			passage: EMBEDDINGGEMMA2_PASSAGE_PREFIX,
		});
	});

	it("routes standalone images through 3-way visual gate (screenshot->OCR, rich photo->VLM) and stages raw images for multimodal MinSync", async () => {
		const root = makeTempDir("autorag-img-idx-");
		const docsDir = join(root, "docs");
		mkdirSync(docsDir, { recursive: true });
		writeFileSync(join(docsDir, "screenshot.png"), makePngBytes(800, 600));
		writeFileSync(
			join(docsDir, "mountain-sunset.jpg"),
			Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46]),
		);
		writeFileSync(join(docsDir, "notes.txt"), "meeting notes about product launch");

		const gateCalls: string[] = [];
		const vlmCalls: string[] = [];
		const ocrCalls: string[] = [];

		const gate: VisualPageGate = {
			fingerprint: "test-gate:v1",
			async judge(input) {
				gateCalls.push(input.virtualPath);
				if (input.virtualPath.endsWith("screenshot.png")) {
					return {
						route: "ocr",
						visuallyRich: true,
						probability: 0.95,
						model: "test-jev",
						backend: "test",
					};
				}
				return {
					route: "vlm",
					visuallyRich: true,
					probability: 0.92,
					model: "test-jev",
					backend: "test",
				};
			},
		};
		const vlmExtractor: VisualPageExtractor = {
			kind: "vlm",
			modelId: "test/vlm",
			remote: true,
			fingerprint: "vlm:test",
			async extract(input) {
				vlmCalls.push(input.virtualPath);
				return {
					markdown: "Snow-capped mountain peak glowing orange at alpine sunset over a calm lake.",
					extractorKind: "vlm",
					extractorModel: "test/vlm",
				};
			},
		};
		const ocrExtractor: VisualPageExtractor = {
			kind: "ocr",
			modelId: "test/ocr",
			remote: false,
			fingerprint: "ocr:test",
			async extract(input) {
				ocrCalls.push(input.virtualPath);
				return {
					markdown: "Invoice #INV-2026-9912\nTotal Due: $4,250.00",
					extractorKind: "ocr",
					extractorModel: "test/ocr",
				};
			},
		};

		// Stage 2 visual parse automatically includes standalone images even with ocr.enabled off.
		const syncRes = await syncParsedMirrors({
			root,
			searchPaths: [docsDir],
			visualParse: { gate, vlmExtractor, ocrExtractor },
		});

		expect(syncRes.scanned).toBe(3);
		expect(syncRes.written).toBe(3);
		expect(gateCalls.sort()).toEqual(["/docs/mountain-sunset.jpg", "/docs/screenshot.png"]);
		expect(ocrCalls).toEqual(["/docs/screenshot.png"]);
		expect(vlmCalls).toEqual(["/docs/mountain-sunset.jpg"]);

		const screenshotVisual = readFileSync(parsedVisualOutputPath(root, "/docs/screenshot.png"), "utf8");
		expect(screenshotVisual).toContain("INV-2026-9912");
		const mountainVisual = readFileSync(parsedVisualOutputPath(root, "/docs/mountain-sunset.jpg"), "utf8");
		expect(mountainVisual).toContain("Snow-capped mountain peak");

		// Stage MinSync workspace with multimodal raw image staging enabled:
		// boilerplate image-metadata `.md` stubs are omitted so only raw pixels + visual OCR/VLM mirrors are indexed.
		const staged = syncMinSyncWorkspace(root, { stageRawImages: true });
		const stagedKinds = staged.entries.map((e) => `${e.virtualPath}:${e.mirrorKind}`).sort();
		expect(stagedKinds).toEqual([
			"/docs/mountain-sunset.jpg:image",
			"/docs/mountain-sunset.jpg:visual",
			"/docs/notes.txt:text",
			"/docs/screenshot.png:image",
			"/docs/screenshot.png:visual",
		]);
		for (const entry of staged.entries) {
			expect(existsSync(entry.minSyncPath)).toBe(true);
		}

		// Reverse path map resolves image, visual, and text hits back to original sourcePath.
		const pathMap = buildMinSyncPathMap(root, staged.workspacePath);
		expect(pathMap.get("files/docs/mountain-sunset.jpg")?.sourcePath).toBe(join(docsDir, "mountain-sunset.jpg"));
		expect(pathMap.get("files/docs/mountain-sunset.jpg")?.mirrorKind).toBe("image");
		expect(pathMap.get("files/docs/mountain-sunset.jpg.autorag-visual.md")?.mirrorKind).toBe("visual");

		// Incremental re-run with no changes reuses gate & extractor cache (zero extra calls).
		await syncParsedMirrors({
			root,
			searchPaths: [docsDir],
			visualParse: { gate, vlmExtractor, ocrExtractor },
		});
		expect(gateCalls.length).toBe(2);
		expect(vlmCalls.length).toBe(1);
		expect(ocrCalls.length).toBe(1);
	});

	it("coalesces standalone image direct-embedding hit with VLM/OCR visual mirror hit into one slot", () => {
		const imageHit: RetrievalResult = {
			id: "minsync:/docs/sunset.jpg:img1",
			content: "Image file: sunset.jpg (direct multimodal visual embedding match)",
			source: "/tmp/docs/sunset.jpg",
			score: 0.91,
			metadata: { mirrorKind: "image", virtualPath: "/docs/sunset.jpg" },
		};
		const visualHit: RetrievalResult = {
			id: "minsync:/docs/sunset.jpg:vis1",
			content: "Golden hour sunset behind Mount Fuji.",
			source: "/tmp/docs/sunset.jpg",
			score: 0.85,
			metadata: { mirrorKind: "visual", virtualPath: "/docs/sunset.jpg", pageStart: 1, pageEnd: 1 },
		};
		const textMetaHit: RetrievalResult = {
			id: "minsync:/docs/sunset.jpg:txt1",
			content: "# sunset.jpg\n\nImage file (Format: JPEG, Size: 1024 bytes).",
			source: "/tmp/docs/sunset.jpg",
			score: 0.55,
			metadata: { virtualPath: "/docs/sunset.jpg" },
		};
		const merged = coalesceCrossMirrorPageHits([imageHit, visualHit, textMetaHit]);
		expect(merged).toHaveLength(1);
		expect(merged[0]?.score).toBe(0.91);
		expect(merged[0]?.content).toContain("Golden hour sunset behind Mount Fuji");
		expect(merged[0]?.metadata?.hasVisualComplement).toBe(true);

		// When lightweight metadata mirror outranks raw image hit in RRF, upgrading to mirrorKind:"image" keeps direct visual protection.
		const mergedMetaFirst = coalesceCrossMirrorPageHits([textMetaHit, imageHit]);
		expect(mergedMetaFirst).toHaveLength(1);
		expect(mergedMetaFirst[0]?.metadata?.mirrorKind).toBe("image");
		expect(mergedMetaFirst[0]?.content).toContain("direct multimodal visual embedding match");
	});
});

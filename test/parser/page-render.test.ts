import { chmodSync, mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { MINSYNC_MIN_VERSION } from "../../src/minsync/version.ts";
import { createDefaultPageRenderer, supportsVisualPageRendering } from "../../src/parser/page-render.ts";
import { createMultiPagePdfBuffer } from "../fixtures/minimal-pdf.ts";
import { fakeMinSyncStageSimulatorPath } from "../helpers/fake-minsync.ts";

const PNG_MAGIC = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
const tempDirs: string[] = [];

function installSearchableFakeMinSync(dir: string): string {
	mkdirSync(dir, { recursive: true });
	const bin = join(dir, "fake-minsync.mjs");
	writeFileSync(
		bin,
		`#!/usr/bin/env node
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, relative } from "node:path";
const require = createRequire(import.meta.url);
const args = process.argv.slice(2);
const config = join(process.cwd(), ".minsync", "config.toml");
const cursor = join(process.cwd(), ".minsync", "cursor.json");

function walk(root, out = []) {
  if (!existsSync(root)) return out;
  for (const entry of readdirSync(root, { withFileTypes: true })) {
    const full = join(root, entry.name);
    if (entry.isDirectory()) walk(full, out);
    else if (entry.isFile()) out.push(full);
  }
  return out;
}

if (args[0] === "--version") {
  console.log("minsync ${MINSYNC_MIN_VERSION}");
} else if (args[0] === "stage") {
  const eq = args.find((a) => a.startsWith("--state-dir="));
  const idx = args.indexOf("--state-dir");
  const stateDir = eq ? eq.slice("--state-dir=".length) : idx >= 0 ? args[idx + 1] : args[1];
  require(${JSON.stringify(fakeMinSyncStageSimulatorPath)}).runStage({ stateDir });
} else if (args[0] === "init") {
  mkdirSync(dirname(config), { recursive: true });
  writeFileSync(config, "[embedder]\\nid = \\"none\\"\\n");
  console.log(JSON.stringify({ initialized: true }));
} else if (args[0] === "check") {
  console.log(JSON.stringify({ vectorstore_ok: true, embedder_ok: true }));
} else if (args[0] === "sync") {
  const files = walk(join(process.cwd(), "files"));
  mkdirSync(dirname(cursor), { recursive: true });
  writeFileSync(cursor, JSON.stringify({ ready: true, files: files.length }));
  console.log(JSON.stringify({ files_added: files.length, files_modified: 0, files_deleted: 0, chunks_added: files.length, chunks_deleted: 0 }));
} else if (args[0] === "query") {
  const q = (args[args.length - 1] ?? "").toLowerCase();
  const terms = q.split(/\\s+/).filter(Boolean);
  const filesRoot = join(process.cwd(), "files");
  const hits = [];
  for (const file of walk(filesRoot)) {
    const text = readFileSync(file, "utf8");
    const lower = text.toLowerCase();
    const score = terms.reduce((acc, term) => acc + (lower.includes(term) ? 1 : 0), 0);
    if (score > 0) {
      const rel = "files/" + relative(filesRoot, file).split("\\\\").join("/");
      hits.push({
        doc_id: createHash("sha256").update(rel).digest("hex"),
        path: rel,
        heading_path: rel,
        text: rel + "\\n" + text,
        score,
      });
    }
  }
  hits.sort((a, b) => b.score - a.score);
  console.log(JSON.stringify({ results: hits }));
} else {
  console.log(JSON.stringify({ ok: true }));
}
`,
	);
	chmodSync(bin, 0o755);
	return bin;
}

afterEach(() => {
	while (tempDirs.length > 0) {
		const dir = tempDirs.pop();
		if (dir !== undefined) rmSync(dir, { recursive: true, force: true });
	}
});

describe("default visual page renderer + end-to-end multi-mirror retrieval", () => {
	it("renders real multi-page PDF pages into PNG images with deterministic hashes", async () => {
		expect(supportsVisualPageRendering("/docs/a.pdf")).toBe(true);
		expect(supportsVisualPageRendering("/docs/a.hwpx")).toBe(true);
		expect(supportsVisualPageRendering("/docs/a.png")).toBe(true);
		expect(supportsVisualPageRendering("/docs/a.txt")).toBe(false);

		const pdf = createMultiPagePdfBuffer([["Page one summary"], ["Page two bar chart"], ["Page three appendix"]]);
		const renderer = createDefaultPageRenderer({ maxWidthPx: 512 });
		const pages = await renderer.renderPages({
			virtualPath: "/docs/deck.pdf",
			bytes: new Uint8Array(pdf),
		});

		expect(pages).toHaveLength(3);
		for (const [index, page] of pages.entries()) {
			expect(page.page).toBe(index + 1);
			expect(page.mimeType).toBe("image/png");
			expect(Array.from(page.data.slice(0, 8))).toEqual(PNG_MAGIC);
			expect(page.sha256).toMatch(/^[0-9a-f]{64}$/u);
		}
		// Distinct page content renders to distinct image hashes
		expect(pages[0]?.sha256).not.toBe(pages[1]?.sha256);

		// Subset rendering and deterministic hash reuse
		const subset = await renderer.renderPages({
			virtualPath: "/docs/deck.pdf",
			bytes: new Uint8Array(pdf),
			pageNumbers: [2],
		});
		expect(subset).toHaveLength(1);
		expect(subset[0]?.page).toBe(2);
		expect(subset[0]?.sha256).toBe(pages[1]?.sha256);
	});

	it("retrieves hits from BOTH base text mirror and visual supplement mirror via AutoRAGAgent", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-visual-e2e-"));
		tempDirs.push(root);
		const docs = join(root, "docs");
		mkdirSync(docs, { recursive: true });
		const pdfPath = join(docs, "metrics.pdf");
		writeFileSync(
			pdfPath,
			createMultiPagePdfBuffer([["Annual Operating Overview for Alpha Corp"], ["Figure 1 Quarterly Bar Chart"]]),
		);

		const fakeMinSync = installSearchableFakeMinSync(join(root, "fake-bin"));
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: { binaryPath: fakeMinSync, autoInstall: false, embedder: { id: "none" } },
			jikji: false,
			everything: false,
			fsearch: false,
			dupey: false,
			filee: false,
			visualParse: {
				enabled: true,
				gate: {
					fingerprint: "stub-gate:v2",
					judge: async ({ page }) => ({
						route: page === 2 ? "vlm" : "plain",
						visuallyRich: page === 2,
						probability: page === 2 ? 0.97 : 0.95,
						model: "typesafe/jev-1.13",
						backend: "openrouter",
					}),
				},
				extractor: {
					kind: "vlm",
					modelId: "stub/vlm",
					fingerprint: "stub-vlm:v1",
					extract: async ({ page }) => ({
						markdown: `Chart Data Page ${page}: QuantumWidget Q4 ARR reached 94.7M USD.`,
						extractorKind: "vlm",
						extractorModel: "stub/vlm",
					}),
				},
			},
		});

		// Stage 1 (default/onboarding refresh): runs ONLY fast base parsing — visual mirror must NOT exist yet!
		await agent.refresh(false, { methods: ["parsed", "minsync"] });
		const engine = agent.getRetrievalEngine();
		const beforeVisual = await engine.retrieve("QuantumWidget 94.7M", { topK: 10 });
		expect(beforeVisual.results.some((hit) => hit.content.includes("QuantumWidget"))).toBe(false);

		// Stage 2 (opt-in advanced visual parsing): layers visual/OCR mirror on top of Stage 1 mirrors!
		await agent.refreshVisual();

		const visualHits = await engine.retrieve("QuantumWidget 94.7M", { topK: 10 });
		const visualHit = visualHits.results.find((hit) => hit.content.includes("QuantumWidget"));
		expect(visualHit).toBeDefined();
		expect(visualHit?.source).toBe(realpathSync(pdfPath));
		expect(visualHit?.metadata?.mirrorKind).toBe("visual");
		expect(visualHit?.metadata?.pageStart).toBe(2);
		expect(visualHit?.metadata?.pageEnd).toBe(2);

		const textHits = await engine.retrieve("Alpha Corp", { topK: 10 });
		const textHit = textHits.results.find((hit) => hit.content.includes("Alpha Corp"));
		expect(textHit).toBeDefined();
		expect(textHit?.source).toBe(realpathSync(pdfPath));
		expect(textHit?.metadata?.mirrorKind).toBeUndefined();
		expect(textHit?.metadata?.pageStart).toBe(1);
	});
});

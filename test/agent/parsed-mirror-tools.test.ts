import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, realpathSync, rmSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import type { ExtensionAPI, ToolDefinition } from "@earendil-works/pi-coding-agent";
import { MockBackend } from "jev-use";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { deriveResultsFromAnswer } from "../../src/agent/answer-citations.ts";
import { CitationNumbers } from "../../src/agent/citation-numbers.ts";
import { EvidenceLedger } from "../../src/agent/evidence-ledger.ts";
import {
	createParsedMirrorMapper,
	createParsedMirrorToolsExtension,
	rewriteMirrorGlob,
	runMirrorGrep,
	shouldPreferMirrorForRead,
} from "../../src/agent/parsed-mirror-tools.ts";
import {
	DECOMPOSE_QUESTION_ID,
	FOLLOW_UP_QUESTION_ID,
	QUERY_ROUTE_QUESTION_ID,
} from "../../src/agent/query-routing.ts";
import { buildSystemPrompt } from "../../src/agent/system-prompt.ts";
import { planSourceRoots, sourceIdentifier } from "../../src/filesystem/source-paths.ts";
import { minSyncDocumentPath, minSyncWorkspaceRoot } from "../../src/minsync/paths.ts";
import { writePagesSidecar } from "../../src/mirror/pages-sidecar.ts";
import { parsedOutputPath } from "../../src/mirror/paths.ts";
import { citationsRelevant } from "../helpers/citations-relevant.ts";

let root: string;
let docs: string;
let notes: string;
let workspace: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-mirror-map-"));
	docs = join(root, "docs");
	notes = join(root, "notes");
	workspace = join(root, "workspace");
	mkdirSync(join(docs, "contracts"), { recursive: true });
	mkdirSync(notes, { recursive: true });
	mkdirSync(workspace, { recursive: true });
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function stageMirror(
	searchPaths: readonly string[],
	sourceFile: string,
	markdown: string,
	pages?: {
		readonly mode: "layout" | "section";
		readonly pageCount?: number;
		readonly spans: { page: number; lineStart: number; lineEnd: number }[];
	},
): { virtualPath: string; mirrorPath: string; pagesMirrorPath: string } {
	const roots = planSourceRoots(searchPaths);
	const owner = roots.find((r) => resolve(sourceFile).startsWith(resolve(r.rootPath)));
	if (!owner) throw new Error(`No source root owns ${sourceFile}`);
	const virtualPath = sourceIdentifier(owner, sourceFile);
	const mirrorPath = minSyncDocumentPath(minSyncWorkspaceRoot(workspace), virtualPath);
	mkdirSync(join(mirrorPath, ".."), { recursive: true });
	writeFileSync(mirrorPath, markdown, "utf8");
	const pagesMirrorPath = parsedOutputPath(workspace, virtualPath);
	if (pages !== undefined) {
		mkdirSync(join(pagesMirrorPath, ".."), { recursive: true });
		writePagesSidecar(pagesMirrorPath, markdown, pages);
	}
	return { virtualPath, mirrorPath, pagesMirrorPath };
}

function registerExtensionTools(
	factory: ReturnType<typeof createParsedMirrorToolsExtension>,
): Map<string, ToolDefinition> {
	const registered = new Map<string, ToolDefinition>();
	const piStub = {
		registerTool: (def: ToolDefinition) => {
			registered.set(def.name, def);
		},
	} as unknown as ExtensionAPI;
	factory(piStub);
	return registered;
}

describe("ParsedMirrorMapper", () => {
	it("maps a source file inside searchPaths to its deterministic MinSync mirror file and back", () => {
		const pdf = join(docs, "contracts", "alpha.pdf");
		writeFileSync(pdf, "%PDF-1.7 binary stub");
		const { mirrorPath } = stageMirror([docs, notes], pdf, "# Alpha Contract\nPenalty code: ERR-9910\n");

		const mapper = createParsedMirrorMapper({ workspaceRoot: workspace, searchPaths: [docs, notes] });
		const resolved = mapper.toMirrorFile(pdf);
		expect(resolved).toBeDefined();
		expect(resolved?.mirrorPath).toBe(mirrorPath);
		expect(resolved?.sourcePath).toBe(resolve(pdf));
		expect(resolved?.virtualPath).toBe("/docs/contracts/alpha.pdf");

		const reverse = mapper.toSourcePath(mirrorPath);
		expect(reverse?.sourcePath).toBe(resolve(pdf));
		expect(reverse?.virtualPath).toBe("/docs/contracts/alpha.pdf");
	});

	it("maps a directory inside searchPaths to its corresponding mirror directory", () => {
		const pdf = join(docs, "contracts", "beta.pdf");
		writeFileSync(pdf, "%PDF-1.7");
		stageMirror([docs, notes], pdf, "Beta body\n");

		const mapper = createParsedMirrorMapper({ workspaceRoot: workspace, searchPaths: [docs, notes] });
		const dir = mapper.toMirrorTarget(join(docs, "contracts"));
		expect(dir).toEqual({
			kind: "directory",
			mirrorPath: join(minSyncWorkspaceRoot(workspace), "files", "docs", "contracts"),
			sourceRootPath: resolve(join(docs, "contracts")),
		});
	});

	it("maps the workspace root or omitted path to the whole MinSync files root when multiple roots exist", () => {
		const pdf = join(docs, "a.pdf");
		const hwp = join(notes, "b.hwp");
		writeFileSync(pdf, "%PDF");
		writeFileSync(hwp, "HWP");
		stageMirror([docs, notes], pdf, "alpha\n");
		stageMirror([docs, notes], hwp, "beta\n");

		const mapper = createParsedMirrorMapper({ workspaceRoot: workspace, searchPaths: [docs, notes] });
		const all = mapper.toMirrorTarget(undefined);
		expect(all?.kind).toBe("corpus");
		expect(all?.mirrorPath).toBe(join(minSyncWorkspaceRoot(workspace), "files"));
	});

	it("returns undefined for paths outside searchPaths", () => {
		const outside = join(root, "other", "secret.txt");
		mkdirSync(join(root, "other"), { recursive: true });
		writeFileSync(outside, "hello");
		const pdf = join(docs, "a.pdf");
		writeFileSync(pdf, "%PDF");
		stageMirror([docs], pdf, "alpha\n");

		const mapper = createParsedMirrorMapper({ workspaceRoot: workspace, searchPaths: [docs] });
		expect(mapper.toMirrorTarget(outside)).toBeUndefined();
		expect(mapper.toMirrorTarget(join(root, "other"))).toBeUndefined();
	});

	it("handles duplicate basename search roots (/docs and /docs-2) without collision", () => {
		const docsA = join(root, "a", "docs");
		const docsB = join(root, "b", "docs");
		mkdirSync(docsA, { recursive: true });
		mkdirSync(docsB, { recursive: true });
		const fileA = join(docsA, "same.pdf");
		const fileB = join(docsB, "same.pdf");
		writeFileSync(fileA, "%PDF-A");
		writeFileSync(fileB, "%PDF-B");
		const stagedA = stageMirror([docsA, docsB], fileA, "from A\n");
		const stagedB = stageMirror([docsA, docsB], fileB, "from B\n");

		const mapper = createParsedMirrorMapper({ workspaceRoot: workspace, searchPaths: [docsA, docsB] });
		expect(mapper.toMirrorFile(fileA)?.mirrorPath).toBe(stagedA.mirrorPath);
		expect(mapper.toMirrorFile(fileB)?.mirrorPath).toBe(stagedB.mirrorPath);
		expect(mapper.toSourcePath(stagedA.mirrorPath)?.sourcePath).toBe(resolve(fileA));
		expect(mapper.toSourcePath(stagedB.mirrorPath)?.sourcePath).toBe(resolve(fileB));
	});

	it("reads layout pages sidecar for a mirrored document and resolves line-to-page and section", () => {
		const pdf = join(docs, "policy.pdf");
		writeFileSync(pdf, "%PDF");
		const markdown = "# Section One\nIntro line\n\n## Clause 4\nExact code: POL-404\nTail line\n";
		const { mirrorPath } = stageMirror([docs], pdf, markdown, {
			mode: "layout",
			pageCount: 2,
			spans: [
				{ page: 1, lineStart: 1, lineEnd: 3 },
				{ page: 2, lineStart: 4, lineEnd: 6 },
			],
		});

		const mapper = createParsedMirrorMapper({ workspaceRoot: workspace, searchPaths: [docs] });
		const loc = mapper.locateLine(mirrorPath, 5);
		expect(loc).toEqual({
			pageStart: 2,
			pageEnd: 2,
			section: "Clause 4",
		});
	});

	it("does not report page numbers for section-mode sidecars", () => {
		const hwp = join(docs, "draft.hwp");
		writeFileSync(hwp, "HWP");
		const markdown = "# Heading\nBody line\n";
		const { mirrorPath } = stageMirror([docs], hwp, markdown, {
			mode: "section",
			pageCount: 1,
			spans: [{ page: 1, lineStart: 1, lineEnd: 2 }],
		});

		const mapper = createParsedMirrorMapper({ workspaceRoot: workspace, searchPaths: [docs] });
		const loc = mapper.locateLine(mirrorPath, 2);
		expect(loc?.pageStart).toBeUndefined();
		expect(loc?.section).toBe("Heading");
	});

	it("detects stale mirror when original source was modified after the mirror file", () => {
		const pdf = join(docs, "updated.pdf");
		writeFileSync(pdf, "%PDF-v1");
		const { mirrorPath } = stageMirror([docs], pdf, "old parsed content\n");
		const past = new Date(Date.now() - 60_000);
		utimesSync(mirrorPath, past, past);
		writeFileSync(pdf, "%PDF-v2");

		const mapper = createParsedMirrorMapper({ workspaceRoot: workspace, searchPaths: [docs] });
		const resolved = mapper.toMirrorFile(pdf);
		expect(resolved?.stale).toBe(true);
	});
});

describe("rewriteMirrorGlob", () => {
	it("rewrites source file extension globs so they match .md mirror filenames", () => {
		expect(rewriteMirrorGlob("*.pdf")).toBe("*.pdf.md");
		expect(rewriteMirrorGlob("contracts/**/*.hwp")).toBe("contracts/**/*.hwp.md");
		expect(rewriteMirrorGlob("*.{pdf,docx,hwpx}")).toBe("*.{pdf,docx,hwpx}.md");
		expect(rewriteMirrorGlob("!*.xlsx")).toBe("!*.xlsx.md");
		expect(rewriteMirrorGlob("./reports/*.pdf")).toBe("reports/*.pdf.md");
	});

	it("leaves already-.md-suffixed or trailing-wildcard globs matching properly", () => {
		expect(rewriteMirrorGlob("*")).toBe("*");
		expect(rewriteMirrorGlob("contracts/**")).toBe("contracts/**");
		expect(rewriteMirrorGlob("report*")).toBe("report*");
		expect(rewriteMirrorGlob("*.md")).toBe("*.md.md");
	});
});

describe("shouldPreferMirrorForRead", () => {
	it("prefers mirror for parsed document formats (pdf, hwp, hwpx, docx, xlsx, pptx, eml)", () => {
		for (const ext of [".pdf", ".hwp", ".hwpx", ".hml", ".docx", ".xlsx", ".xls", ".ods", ".pptx", ".eml"]) {
			expect(shouldPreferMirrorForRead(`/docs/sample${ext}`)).toBe(true);
		}
	});

	it("does not prefer mirror for plain text, markdown, code, or image files", () => {
		for (const ext of [".txt", ".md", ".ts", ".json", ".yaml", ".csv", ".png", ".jpg", ".webp"]) {
			expect(shouldPreferMirrorForRead(`/docs/sample${ext}`)).toBe(false);
		}
	});
});

describe("runMirrorGrep & parsed-mirror grep/read extension", () => {
	it("finds exact strings inside binary PDF/HWP mirrors, returns original source paths + page/section + [eN] ids", async () => {
		const pdf = join(docs, "contracts", "vendor-2026.pdf");
		const hwp = join(notes, "memo.hwp");
		// Original files are binary gibberish — raw grep would find nothing in them.
		writeFileSync(pdf, Buffer.from([0x25, 0x50, 0x44, 0x46, 0x00, 0xff, 0xfe, 0x01]));
		writeFileSync(hwp, Buffer.from([0xd0, 0xcf, 0x11, 0xe0, 0x00, 0x00]));
		stageMirror(
			[docs, notes],
			pdf,
			"# Master Agreement\nIntro\n\n## Payment Terms\nInvoice code: INV-88421-KR\nNet 30 days\n",
			{
				mode: "layout",
				pageCount: 2,
				spans: [
					{ page: 1, lineStart: 1, lineEnd: 3 },
					{ page: 2, lineStart: 4, lineEnd: 6 },
				],
			},
		);
		stageMirror([docs, notes], hwp, "# 회의록\n결제 코드 INV-88421-KR 확인 완료\n");

		const ledger = new EvidenceLedger();
		const tools = registerExtensionTools(
			createParsedMirrorToolsExtension({
				workspaceRoot: workspace,
				searchPaths: [docs, notes],
				ledger,
			}),
		);
		const grepTool = tools.get("grep");
		expect(grepTool).toBeDefined();

		const res = await grepTool!.execute("call-1", { pattern: "INV-88421-KR", context: 1 }, undefined, undefined, {
			cwd: workspace,
		} as never);
		const text = res.content.find((c) => c.type === "text")?.text ?? "";
		expect(text).toContain(resolve(pdf));
		expect(text).toContain(resolve(hwp));
		expect(text).toContain("page 2");
		expect(text).toContain('section "Payment Terms"');
		expect(text).toContain("[e1]");
		expect(text).toContain("[e2]");
		expect(text).not.toContain(".autorag/minsync/files");

		// Evidence ids registered by mirror grep resolve cleanly during answer citation derivation.
		const derived = deriveResultsFromAnswer(
			"Invoice code is INV-88421-KR [e1] and confirmed in notes [e2].",
			ledger,
			{
				allowLocalFiles: true,
				numbers: new CitationNumbers(),
			},
		);
		expect(derived.diagnostics).toEqual([]);
		expect(derived.details.mapping.map((m) => m.source).sort()).toEqual([resolve(pdf), resolve(hwp)].sort());
	});

	it("filters by original extension glob (*.pdf) on parsed mirrors", async () => {
		const pdf = join(docs, "contracts", "one.pdf");
		const hwp = join(docs, "contracts", "two.hwp");
		writeFileSync(pdf, Buffer.from([0x25, 0x50, 0x44, 0x46, 0x00]));
		writeFileSync(hwp, Buffer.from([0xd0, 0xcf, 0x00]));
		stageMirror([docs], pdf, "Token: SHARED-777\n");
		stageMirror([docs], hwp, "Token: SHARED-777\n");

		const tools = registerExtensionTools(
			createParsedMirrorToolsExtension({
				workspaceRoot: workspace,
				searchPaths: [docs],
			}),
		);
		const res = await tools
			.get("grep")!
			.execute("call-glob", { pattern: "SHARED-777", path: docs, glob: "*.pdf" }, undefined, undefined, {
				cwd: workspace,
			} as never);
		const text = res.content.find((c) => c.type === "text")?.text ?? "";
		expect(text).toContain(resolve(pdf));
		expect(text).not.toContain(resolve(hwp));
	});

	it("flags matched sources changed since the last refresh in directory-wide searches", async () => {
		const pdf = join(docs, "contracts", "changed.pdf");
		const fresh = join(docs, "contracts", "fresh.pdf");
		writeFileSync(pdf, Buffer.from([0x25, 0x50, 0x44, 0x46, 0x00]));
		writeFileSync(fresh, Buffer.from([0x25, 0x50, 0x44, 0x46, 0x01]));
		const { mirrorPath } = stageMirror([docs], pdf, "Token: STALE-CHECK-1\n");
		stageMirror([docs], fresh, "Token: STALE-CHECK-1\n");
		const past = new Date(Date.now() - 60_000);
		utimesSync(mirrorPath, past, past);
		writeFileSync(pdf, Buffer.from([0x25, 0x50, 0x44, 0x46, 0x02]));

		const mapper = createParsedMirrorMapper({ workspaceRoot: workspace, searchPaths: [docs] });
		const res = await runMirrorGrep({ mapper, target: mapper.toMirrorTarget(docs)!, pattern: "STALE-CHECK-1" });
		expect(res.output).toContain("changed since the last refresh");
		expect(res.output).toContain(resolve(pdf));
		// Only the changed file is named in the notice; the fresh one is not.
		const notice = res.output.slice(res.output.indexOf("changed since the last refresh"));
		expect(notice).not.toContain(resolve(fresh));
	});

	it("works via the pure-JS fallback when rg is unavailable", async () => {
		const pdf = join(docs, "contracts", "fallback.pdf");
		writeFileSync(pdf, Buffer.from([0x25, 0x50, 0x44, 0x46, 0x00]));
		stageMirror([docs], pdf, "Line 1\nNeedle: FALLBACK-42\nLine 3\n");

		const mapper = createParsedMirrorMapper({ workspaceRoot: workspace, searchPaths: [docs] });
		const target = mapper.toMirrorTarget(docs)!;
		const res = await runMirrorGrep({
			mapper,
			target,
			pattern: "FALLBACK-42",
			context: 1,
			rgPath: "/nonexistent/rg-binary",
		});
		expect(res.matchCount).toBe(1);
		expect(res.matchedSources).toEqual([resolve(pdf)]);
		expect(res.output).toContain("L2: Needle: FALLBACK-42");
		expect(res.output).toContain("L1- Line 1");
	});

	it("falls back to raw filesystem grep for files outside searchPaths and for newer plain-text files", async () => {
		const outside = join(root, "outside");
		mkdirSync(outside, { recursive: true });
		const outsideFile = join(outside, "code.ts");
		writeFileSync(outsideFile, "export const secret = 'OUTSIDE-123';\n");

		const txt = join(docs, "live.txt");
		writeFileSync(txt, "old text\n");
		const { mirrorPath } = stageMirror([docs], txt, "old text\n");
		const past = new Date(Date.now() - 60_000);
		utimesSync(mirrorPath, past, past);
		writeFileSync(txt, "brand new text LIVE-999\n");

		const tools = registerExtensionTools(
			createParsedMirrorToolsExtension({
				workspaceRoot: workspace,
				searchPaths: [docs],
			}),
		);
		const grepTool = tools.get("grep")!;

		const outRes = await grepTool.execute("c1", { pattern: "OUTSIDE-123", path: outside }, undefined, undefined, {
			cwd: workspace,
		} as never);
		expect(outRes.content[0]?.type === "text" ? outRes.content[0].text : "").toContain("OUTSIDE-123");

		const freshRes = await grepTool.execute("c2", { pattern: "LIVE-999", path: txt }, undefined, undefined, {
			cwd: workspace,
		} as never);
		expect(freshRes.content[0]?.type === "text" ? freshRes.content[0].text : "").toContain("LIVE-999");
	});

	it("read transparently serves parsed markdown mirror with offset/limit, page info, and [eN] id for PDFs, and raw file for plain text", async () => {
		const pdf = join(docs, "contracts", "manual.pdf");
		writeFileSync(pdf, Buffer.from([0x25, 0x50, 0x44, 0x46, 0x00, 0xff]));
		stageMirror(
			[docs],
			pdf,
			["# Manual", "Line two", "## Installation", "Run autorag refresh", "Verify output"].join("\n"),
			{
				mode: "layout",
				pageCount: 2,
				spans: [
					{ page: 1, lineStart: 1, lineEnd: 2 },
					{ page: 2, lineStart: 3, lineEnd: 5 },
				],
			},
		);
		const txt = join(docs, "plain.txt");
		writeFileSync(txt, "raw plain content\n");
		stageMirror([docs], txt, "stale mirror content\n");

		const ledger = new EvidenceLedger();
		const tools = registerExtensionTools(
			createParsedMirrorToolsExtension({
				workspaceRoot: workspace,
				searchPaths: [docs],
				ledger,
			}),
		);
		const readTool = tools.get("read")!;

		const pdfRes = await readTool.execute("r1", { path: pdf, offset: 3, limit: 2 }, undefined, undefined, {
			cwd: workspace,
		} as never);
		const pdfText = pdfRes.content.find((c) => c.type === "text")?.text ?? "";
		expect(pdfText).toContain("[e1 | Parsed mirror of ");
		expect(pdfText).toContain("page 2");
		expect(pdfText).toContain('section "Installation"');
		expect(pdfText).toContain("## Installation");
		expect(pdfText).toContain("Run autorag refresh");
		expect(pdfText).not.toContain("Line two");

		// Plain text reads directly from disk so edits are immediately visible.
		const txtRes = await readTool.execute("r2", { path: txt }, undefined, undefined, {
			cwd: workspace,
		} as never);
		const txtText = txtRes.content.find((c) => c.type === "text")?.text ?? "";
		expect(txtText).toContain("raw plain content");
		expect(txtText).not.toContain("stale mirror content");
	});

	it("read serves parsed mirror even for plain text when the file lives under a cloud root (avoids placeholder download)", async () => {
		const cloudDir = join(root, "cloud-drive");
		mkdirSync(cloudDir, { recursive: true });
		const cloudTxt = join(cloudDir, "notes.txt");
		writeFileSync(cloudTxt, "DO NOT HYDRATE ORIGINAL\n");
		stageMirror([cloudDir], cloudTxt, "MIRRORED CLOUD TEXT\n");

		const tools = registerExtensionTools(
			createParsedMirrorToolsExtension({
				workspaceRoot: workspace,
				searchPaths: [cloudDir],
				cloudRoots: async () => [{ path: cloudDir, kind: "file-provider", provider: "icloud" }],
			}),
		);
		const res = await tools.get("read")!.execute("r-cloud", { path: cloudTxt }, undefined, undefined, {
			cwd: workspace,
		} as never);
		const text = res.content.find((c) => c.type === "text")?.text ?? "";
		expect(text).toContain("MIRRORED CLOUD TEXT");
		expect(text).not.toContain("DO NOT HYDRATE ORIGINAL");
	});

	it("integrates into AutoRAGAgent session: model calls grep -> read on a PDF and cites [eN] cleanly", async () => {
		const pdf = join(docs, "contracts", "sla.pdf");
		writeFileSync(pdf, Buffer.from([0x25, 0x50, 0x44, 0x46, 0x00, 0x01, 0x02]));
		stageMirror([docs], pdf, "# SLA\n\n## Penalty\nDowntime penalty code is SLA-PEN-9001 (15% credit).\n", {
			mode: "layout",
			pageCount: 1,
			spans: [{ page: 1, lineStart: 1, lineEnd: 4 }],
		});

		const reg = registerFauxProvider({ api: `faux-mirror-grep-${randomUUID()}`, models: [{ id: "mirror-model" }] });
		reg.setResponses([
			fauxAssistantMessage([fauxToolCall("grep", { i: "Search SLA code", pattern: "SLA-PEN-9001", path: docs })], {
				stopReason: "toolUse",
			}),
			fauxAssistantMessage([fauxToolCall("read", { i: "Read SLA clause", path: pdf, offset: 3, limit: 2 })], {
				stopReason: "toolUse",
			}),
			fauxAssistantMessage("The downtime penalty code is SLA-PEN-9001 with a 15% credit [e1]."),
		]);

		const jev = citationsRelevant(
			new MockBackend({
				[QUERY_ROUTE_QUESTION_ID]: {
					answer: "extra",
					distribution: { local: 0.1, organize: 0.1, extra: 0.8 },
					confidence: 0.8,
				},
				[DECOMPOSE_QUESTION_ID]: { answer: 0.1 },
				[FOLLOW_UP_QUESTION_ID]: { answer: 0.1 },
			}),
		);

		const agent = new AutoRAGAgent({
			model: reg.getModel(),
			apiKey: "test-key",
			searchPaths: [docs],
			workspacePath: workspace,
			memoryPath: join(root, "memory.json"),
			minSync: { autoInstall: false },
			jikji: false,
			jev: { backend: jev },
		});

		const observedTools: string[] = [];
		agent.subscribe((event) => {
			if (event.type === "tool_execution_end") observedTools.push(event.toolName);
		});

		try {
			const response = await agent.searchDocuments("What is SLA-PEN-9001?");
			expect(response.answer).toContain("SLA-PEN-9001");
			expect(response.results.length).toBe(1);
			expect(response.results[0]?.source).toBe(realpathSync(pdf));
			expect(observedTools).toEqual(["grep", "read"]);
		} finally {
			agent.stopAutoRefresh();
			reg.unregister();
		}
	});
});

describe("buildSystemPrompt parsed-mirror guidance", () => {
	it("instructs the agent to actively use parsed-mirror grep and read instead of bash recursive grep", () => {
		const prompt = buildSystemPrompt({
			toolNames: ["read", "bash", "grep", "search_all_documents", "fsearch_search"],
		});
		expect(prompt).toContain("Parsed-mirror `grep` and `read`");
		expect(prompt).toContain("PDF, HWP/HWPX, DOCX, XLSX, PPTX, EML");
		expect(prompt).toContain("offset=L#");
		expect(prompt).toContain("Never recursively run `bash` `find`/`grep`/`rg`");
	});
});

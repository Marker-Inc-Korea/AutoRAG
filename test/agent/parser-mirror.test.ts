import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { clearWorkspaceCache } from "../../src/agentdir/workspace.ts";
import { loadMirrorIndex } from "../../src/mirror/index.ts";
import { createMinimalPdfBuffer } from "../fixtures/minimal-pdf.ts";

let root: string;

const pdfMarker = "OpenDataLoader AutoRAG PDF marker refund policy alpha";

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-agent-mirror-"));
	clearWorkspaceCache();
});

afterEach(() => {
	clearWorkspaceCache();
	rmSync(root, { recursive: true, force: true });
});

describe("AutoRAGAgent parsed mirror integration", () => {
	it("refresh() syncs parsed markdown mirrors for supported files", async () => {
		const docs = join(root, "docs");
		mkdirSync(docs, { recursive: true });
		writeFileSync(join(docs, "note.txt"), "Mirror me\n");
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
		});

		await agent.refresh(true);
		const index = loadMirrorIndex(root);
		const outputPath = index.entries["/docs/note.txt"]?.outputPath;

		expect(outputPath).toBeDefined();
		if (!outputPath) throw new Error("expected parsed mirror output path");
		expect(existsSync(outputPath)).toBe(true);
		expect(readFileSync(outputPath, "utf8")).toBe("Mirror me\n");
	});

	it("refresh(true) syncs parsed markdown mirrors for PDF files without leaking source paths", async () => {
		// Given: a document source containing a minimal PDF with marker text.
		const docs = join(root, "docs");
		mkdirSync(docs, { recursive: true });
		writeFileSync(join(docs, "report.pdf"), createMinimalPdfBuffer(pdfMarker));
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
		});

		// When: the agent refreshes with hash verification enabled.
		await agent.refresh(true);
		const index = loadMirrorIndex(root);
		const outputPath = index.entries["/docs/report.pdf"]?.outputPath;

		// Then: the parsed mirror contains PDF text and persisted mirror artifacts stay source-path opaque.
		expect(outputPath).toBeDefined();
		if (!outputPath) throw new Error("expected parsed mirror output path for PDF");
		const parsedMarkdown = readFileSync(outputPath, "utf8");

		expect(parsedMarkdown).toContain(pdfMarker);
		expect(parsedMarkdown).not.toContain(docs);
	});

	it("refresh(true) keeps adjacent mirrors when a PDF cannot be parsed", async () => {
		// Given: a valid text file next to a malformed PDF.
		const docs = join(root, "docs");
		mkdirSync(docs, { recursive: true });
		writeFileSync(join(docs, "note.txt"), "Plain text survives bad PDF\n");
		writeFileSync(join(docs, "broken.pdf"), Buffer.from("not a pdf"));
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
		});

		// When: the collection refresh reaches the malformed PDF.
		await expect(agent.refresh(true)).resolves.toBeDefined();
		const index = loadMirrorIndex(root);
		const textOutput = index.entries["/docs/note.txt"]?.outputPath;

		// Then: the malformed PDF is skipped and the adjacent text mirror remains searchable.
		expect(index.entries["/docs/broken.pdf"]).toBeUndefined();
		expect(textOutput).toBeDefined();
		if (!textOutput) throw new Error("expected parsed mirror output path for adjacent text");
		expect(readFileSync(textOutput, "utf8")).toBe("Plain text survives bad PDF\n");
	});
});

/**
 * Live manual QA for parsed-mirror `grep` and `read` (Option A).
 *
 * Uses a real mixed-format corpus (PDF, DOCX, XLSX, PPTX, TXT), real `kordoc`/parser
 * refresh, real `minsync` staging/indexing, real `rg`, and the real `AutoRAGAgent`
 * session with the parsed-mirror extension wired in.
 *
 * Isolation: every path (`AUTORAG_HOME`, `AUTORAG_CONFIG`, workspace, corpus) lives
 * under a fresh temp directory, and the real `~/.autorag/config.json` hash is checked
 * before and after.
 *
 * Run: bun scripts/manual-qa/run-qa-parsed-mirror-grep-live.ts
 */

import { createHash, randomUUID } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { MockBackend } from "jev-use";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import {
	DECOMPOSE_QUESTION_ID,
	FOLLOW_UP_QUESTION_ID,
	QUERY_ROUTE_QUESTION_ID,
} from "../../src/agent/query-routing.ts";
import { minSyncWorkspaceRoot } from "../../src/minsync/paths.ts";
import { createDocxFixture, createPptxFixture, createXlsxFixture } from "../../test/fixtures/document-formats.ts";
import { createMultiPagePdfBuffer } from "../../test/fixtures/minimal-pdf.ts";
import { citationsRelevant } from "../../test/helpers/citations-relevant.ts";

function shaOfIfExists(path: string): string {
	if (!existsSync(path)) return "absent";
	return createHash("sha1").update(readFileSync(path)).digest("hex");
}

const realConfigPath = join(homedir(), ".autorag", "config.json");
const realConfigBefore = shaOfIfExists(realConfigPath);

const tmpRoot = realpathSync(mkdtempSync(join(tmpdir(), "autorag-mirror-grep-qa-")));
const docs = join(tmpRoot, "docs");
const contracts = join(docs, "contracts");
const workspace = join(tmpRoot, "workspace");
const autoragHome = join(tmpRoot, ".autorag-home");
process.env.AUTORAG_HOME = autoragHome;
process.env.AUTORAG_CONFIG = join(autoragHome, "config.json");
mkdirSync(contracts, { recursive: true });
mkdirSync(workspace, { recursive: true });
mkdirSync(autoragHome, { recursive: true });

let failures = 0;
function check(name: string, pass: boolean, detail?: string): void {
	if (!pass) failures += 1;
	console.log(`${pass ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
}

try {
	const pdfPath = join(contracts, "refund-policy.pdf");
	const docxPath = join(contracts, "vendor-sla.docx");
	const xlsxPath = join(docs, "budget-2026.xlsx");
	const pptxPath = join(docs, "kickoff.pptx");
	const txtPath = join(docs, "notes.txt");

	writeFileSync(
		pdfPath,
		createMultiPagePdfBuffer([
			["Refund Policy Overview", "Applies to all regional offices."],
			["Exception Code REF-7721-DIRECTOR", "Refund exceptions require director approval before payout."],
		]),
	);
	writeFileSync(docxPath, await createDocxFixture("Vendor SLA agreement. Escalation ticket code: SLA-4409-OPS."));
	writeFileSync(xlsxPath, await createXlsxFixture("Q4 capex reserve code: CAPEX-9021-KR"));
	writeFileSync(pptxPath, await createPptxFixture("Launch milestone code: MILESTONE-3302"));
	writeFileSync(txtPath, "Plain text note referencing REF-7721-DIRECTOR.\n", "utf8");

	const reg = registerFauxProvider({ api: `faux-qa-mirror-${randomUUID()}`, models: [{ id: "qa-model" }] });
	reg.setResponses([
		fauxAssistantMessage(
			[fauxToolCall("grep", { i: "Search exact code REF-7721-DIRECTOR", pattern: "REF-7721-DIRECTOR", path: docs, context: 1 })],
			{ stopReason: "toolUse" },
		),
		fauxAssistantMessage(
			[fauxToolCall("grep", { i: "Search DOCX SLA code", pattern: "SLA-4409-OPS", glob: "*.docx" })],
			{ stopReason: "toolUse" },
		),
		fauxAssistantMessage(
			[fauxToolCall("grep", { i: "Search XLSX capex code", pattern: "CAPEX-9021-KR", glob: "*.xlsx" })],
			{ stopReason: "toolUse" },
		),
		fauxAssistantMessage(
			[fauxToolCall("read", { i: "Read PDF exception clause", path: pdfPath, offset: 3, limit: 5 })],
			{ stopReason: "toolUse" },
		),
		fauxAssistantMessage(
			"Refund exceptions (REF-7721-DIRECTOR) require director approval on page 2 [e1]; SLA code is SLA-4409-OPS [e3] and capex reserve is CAPEX-9021-KR [e4].",
		),
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
		apiKey: "qa-key",
		searchPaths: [docs],
		workspacePath: workspace,
		memoryPath: join(autoragHome, "memory.json"),
		minSync: { autoInstall: false, embedder: { id: "none" } },
		jikji: false,
		everything: false,
		fsearch: false,
		jev: { backend: jev },
	});

	const refreshResult = await agent.refresh(true, { methods: ["parsed", "minsync"] });
	check(
		"real parsed + minsync refresh succeeded",
		refreshResult.minsync?.ok === true,
		`written=${refreshResult.written}`,
	);
	check(
		"minsync cursor exists in isolated workspace",
		existsSync(join(minSyncWorkspaceRoot(workspace), ".minsync", "cursor.json")),
	);

	const toolOutputs: { name: string; text: string }[] = [];
	agent.subscribe((event) => {
		if (event.type === "tool_execution_end") {
			const text = event.result.content
				.filter((c: { type: string }) => c.type === "text")
				.map((c: { text?: string }) => c.text ?? "")
				.join("\n");
			toolOutputs.push({ name: event.toolName, text });
		}
	});

	const response = await agent.searchDocuments(
		"Verify REF-7721-DIRECTOR, SLA-4409-OPS, and CAPEX-9021-KR across our documents.",
	);

	const pdfGrepOut = toolOutputs[0]?.text ?? "";
	const docxGrepOut = toolOutputs[1]?.text ?? "";
	const xlsxGrepOut = toolOutputs[2]?.text ?? "";
	const pdfReadOut = toolOutputs[3]?.text ?? "";

	check(
		"grep matched PDF binary via mirror with original path + layout page location",
		pdfGrepOut.includes(pdfPath) && (pdfGrepOut.includes("page 2") || pdfGrepOut.includes("pages 1-2")),
	);
	check("grep never leaked internal .autorag/minsync/files path", !pdfGrepOut.includes(".autorag/minsync/files"));
	check("grep --glob '*.docx' matched DOCX mirror", docxGrepOut.includes(docxPath) && docxGrepOut.includes("SLA-4409-OPS"));
	check("grep --glob '*.xlsx' matched XLSX mirror", xlsxGrepOut.includes(xlsxPath) && xlsxGrepOut.includes("CAPEX-9021-KR"));
	check("read on PDF served parsed markdown mirror with [eN] id", pdfReadOut.includes("Parsed mirror of") && pdfReadOut.includes("director approval"));
	check("final answer resolved all 3 cited sources cleanly", response.results.length === 3, `results=${response.results.length}`);
	check(
		"all cited sources exist on disk as OS-absolute paths",
		response.results.every((r) => typeof r.source === "string" && existsSync(r.source)),
	);

	reg.unregister();
} finally {
	rmSync(tmpRoot, { recursive: true, force: true });
}

const realConfigAfter = shaOfIfExists(realConfigPath);
check("real ~/.autorag/config.json untouched", realConfigBefore === realConfigAfter, `${realConfigBefore} === ${realConfigAfter}`);

if (failures > 0) {
	console.error(`\n${failures} check(s) failed.`);
	process.exit(1);
}
console.log("\nAll parsed-mirror live QA checks passed.");

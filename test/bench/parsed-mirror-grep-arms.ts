/**
 * Benchmark comparing baseline (raw grep + MinSync SPLADE/hybrid retrieval) against
 * parsed-mirror `grep` & `read` on a realistic mixed-format corpus (PDF, DOCX, XLSX, PPTX, TXT)
 * with exact identifiers, contract codes, and error numbers across nested folders.
 *
 * Isolation: uses a fresh temp root and points AUTORAG_HOME / AUTORAG_CONFIG inside it;
 * never touches the maintainer's ~/.autorag.
 *
 * Run: bun test/bench/parsed-mirror-grep-arms.ts [--docs N]
 */

import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { performance } from "node:perf_hooks";
import { parseArgs } from "node:util";
import { createGrepToolDefinition, createReadToolDefinition } from "@earendil-works/pi-coding-agent";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { EvidenceLedger } from "../../src/agent/evidence-ledger.ts";
import { createParsedMirrorMapper, runMirrorGrep } from "../../src/agent/parsed-mirror-tools.ts";
import { createDocxFixture, createPptxFixture, createXlsxFixture } from "../fixtures/document-formats.ts";
import { createMultiPagePdfBuffer } from "../fixtures/minimal-pdf.ts";

const { values } = parseArgs({
	options: {
		docs: { type: "string", default: "120" },
		queries: { type: "string", default: "24" },
	},
});

const totalDocs = Math.max(20, Number(values.docs));
const totalQueries = Math.max(8, Number(values.queries));

const tmpRoot = realpathSync(mkdtempSync(join(tmpdir(), "autorag-mirror-bench-")));
const corpusDir = join(tmpRoot, "docs");
const workspaceDir = join(tmpRoot, "workspace");
const autoragHome = join(tmpRoot, ".autorag-home");
process.env.AUTORAG_HOME = autoragHome;
process.env.AUTORAG_CONFIG = join(autoragHome, "config.json");
mkdirSync(corpusDir, { recursive: true });
mkdirSync(workspaceDir, { recursive: true });
mkdirSync(autoragHome, { recursive: true });

interface NeedleSpec {
	readonly code: string;
	readonly format: "pdf" | "docx" | "xlsx" | "pptx" | "txt";
	readonly filePath: string;
	readonly folderPath: string;
	readonly expectedPage?: number;
}

function percentile(sorted: readonly number[], p: number): number {
	if (sorted.length === 0) return 0;
	const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
	return sorted[idx] ?? 0;
}

function mean(valuesList: readonly number[]): number {
	if (valuesList.length === 0) return 0;
	return valuesList.reduce((acc, v) => acc + v, 0) / valuesList.length;
}

try {
	const folders = ["finance/2025", "finance/2026", "legal/contracts", "ops/incidents", "hr/policies", "eng/specs"];
	for (const folder of folders) mkdirSync(join(corpusDir, folder), { recursive: true });

	const needles: NeedleSpec[] = [];
	const formats: readonly NeedleSpec["format"][] = ["pdf", "docx", "xlsx", "pptx", "txt"];

	for (let i = 0; i < totalDocs; i += 1) {
		const folder = folders[i % folders.length] as string;
		const folderPath = join(corpusDir, folder);
		const fmt = formats[i % formats.length] as NeedleSpec["format"];
		const code = `ERR-${String(1000 + i).padStart(4, "0")}-${["KR", "US", "JP", "EU"][i % 4]}`;
		const sharedBoilerplate =
			"Quarterly governance review, compliance audit trail, payment SLA, vendor escalation procedure, budget variance note.";
		const filePath = join(folderPath, `record-${String(i).padStart(4, "0")}.${fmt}`);

		if (fmt === "pdf") {
			const buf = createMultiPagePdfBuffer([
				[`Title: Annual Governance Report ${i}`, "Governance summary line."],
				[`Clause 4.2 Reference Code ${code}`, `Director approval threshold is USD ${1000 + i * 25}`],
			]);
			writeFileSync(filePath, buf);
			needles.push({ code, format: fmt, filePath, folderPath, expectedPage: 2 });
		} else if (fmt === "docx") {
			const buf = await createDocxFixture(`Document ${i}. ${sharedBoilerplate} Reference code: ${code}.`);
			writeFileSync(filePath, buf);
			needles.push({ code, format: fmt, filePath, folderPath });
		} else if (fmt === "xlsx") {
			const buf = await createXlsxFixture(`${sharedBoilerplate} Code ${code}`);
			writeFileSync(filePath, buf);
			needles.push({ code, format: fmt, filePath, folderPath });
		} else if (fmt === "pptx") {
			const buf = await createPptxFixture(`Slide deck ${i}: ${sharedBoilerplate} Tracking ID ${code}`);
			writeFileSync(filePath, buf);
			needles.push({ code, format: fmt, filePath, folderPath });
		} else {
			writeFileSync(filePath, `# Record ${i}\n\n${sharedBoilerplate}\nExact identifier: ${code}\n`, "utf8");
			needles.push({ code, format: fmt, filePath, folderPath });
		}
	}

	const agent = new AutoRAGAgent({
		searchPaths: [corpusDir],
		workspacePath: workspaceDir,
		memoryPath: join(autoragHome, "memory.json"),
		minSync: { autoInstall: false, embedder: { id: "none" } },
		jikji: false,
		everything: false,
		fsearch: false,
	});

	const refreshStart = performance.now();
	const refreshResult = await agent.refresh(true, { methods: ["parsed", "minsync"] });
	const refreshMs = performance.now() - refreshStart;

	// Select a stratified query sample across all 5 formats.
	const sample: NeedleSpec[] = [];
	for (let i = 0; i < totalQueries && i < needles.length; i += 1) {
		sample.push(needles[i] as NeedleSpec);
	}

	const rawGrep = createGrepToolDefinition(workspaceDir);
	const rawRead = createReadToolDefinition(workspaceDir);
	const mapper = createParsedMirrorMapper({ workspaceRoot: workspaceDir, searchPaths: [corpusDir] });
	const ledger = new EvidenceLedger();

	// Arm 1: Raw pi grep on original files (the pre-change behavior)
	const rawGrepLatencies: number[] = [];
	let rawGrepHits = 0;
	let rawGrepBinaryHits = 0;
	let rawGrepBinaryTotal = 0;

	for (const q of sample) {
		if (q.format !== "txt") rawGrepBinaryTotal += 1;
		const t0 = performance.now();
		const res = await rawGrep.execute("raw", { pattern: q.code, path: corpusDir }, undefined, undefined, {
			cwd: workspaceDir,
		} as never);
		rawGrepLatencies.push(performance.now() - t0);
		const text = res.content.find((c) => c.type === "text")?.text ?? "";
		if (text.includes(q.code)) {
			rawGrepHits += 1;
			if (q.format !== "txt") rawGrepBinaryHits += 1;
		}
	}

	// Arm 2: MinSync SPLADE sparse retrieval (the existing indexed retrieval arm)
	const minSyncLatencies: number[] = [];
	let minSyncTop1Hits = 0;
	let minSyncTop10Hits = 0;
	for (const q of sample) {
		const t0 = performance.now();
		const res = await agent.retrieveWithDiagnostics(q.code, { topK: 10 });
		minSyncLatencies.push(performance.now() - t0);
		const target = realpathSync(q.filePath);
		if (res.results[0]?.source === target) minSyncTop1Hits += 1;
		if (res.results.some((r) => r.source === target)) minSyncTop10Hits += 1;
	}

	// Arm 3: Parsed-mirror grep (whole corpus & folder-scoped) + mirror read follow-up
	const mirrorGrepCorpusLatencies: number[] = [];
	const mirrorGrepFolderLatencies: number[] = [];
	let mirrorGrepHits = 0;
	let mirrorGrepBinaryHits = 0;
	let mirrorPageHits = 0;
	let mirrorPageTotal = 0;

	const corpusTarget = mapper.toMirrorTarget(corpusDir)!;
	for (const q of sample) {
		if (q.expectedPage !== undefined) mirrorPageTotal += 1;
		const t0 = performance.now();
		const res = await runMirrorGrep({
			mapper,
			target: corpusTarget,
			pattern: q.code,
			context: 1,
			ledger,
		});
		mirrorGrepCorpusLatencies.push(performance.now() - t0);
		const target = realpathSync(q.filePath);
		if (res.matchedSources.includes(target)) {
			mirrorGrepHits += 1;
			if (q.format !== "txt") mirrorGrepBinaryHits += 1;
			if (
				q.expectedPage !== undefined &&
				(res.output.includes(`page ${q.expectedPage}`) || res.output.includes(`pages 1-${q.expectedPage}`))
			) {
				mirrorPageHits += 1;
			}
		}

		const folderTarget = mapper.toMirrorTarget(q.folderPath)!;
		const t1 = performance.now();
		await runMirrorGrep({
			mapper,
			target: folderTarget,
			pattern: q.code,
			context: 1,
		});
		mirrorGrepFolderLatencies.push(performance.now() - t1);
	}

	// Arm 4: Read quality check on binary docs (raw read vs mirror read)
	const binarySample = sample.filter((q) => q.format !== "txt").slice(0, 10);
	let rawReadReadable = 0;
	let mirrorReadReadable = 0;
	const mirrorReadLatencies: number[] = [];
	for (const q of binarySample) {
		try {
			const rawRes = await rawRead.execute("r-raw", { path: q.filePath, limit: 40 }, undefined, undefined, {
				cwd: workspaceDir,
			} as never);
			const rawText = rawRes.content.find((c) => c.type === "text")?.text ?? "";
			if (rawText.includes(q.code) && !rawText.includes("\u0000") && !rawText.includes("PK\u0003\u0004")) {
				rawReadReadable += 1;
			}
		} catch {
			// binary read failure counts as unreadable
		}
		const mapping = mapper.toMirrorFile(q.filePath);
		if (mapping !== undefined) {
			const t0 = performance.now();
			const mirRes = await rawRead.execute("r-mir", { path: mapping.mirrorPath, limit: 40 }, undefined, undefined, {
				cwd: workspaceDir,
			} as never);
			mirrorReadLatencies.push(performance.now() - t0);
			const mirText = mirRes.content.find((c) => c.type === "text")?.text ?? "";
			if (mirText.includes(q.code)) mirrorReadReadable += 1;
		}
	}

	const sortNums = (arr: number[]): number[] => [...arr].sort((a, b) => a - b);
	const report = {
		corpus: {
			totalDocs,
			formats: ["pdf", "docx", "xlsx", "pptx", "txt"],
			written: refreshResult.written,
			skipped: refreshResult.skipped,
			refreshMs: Math.round(refreshMs),
		},
		sampleSize: sample.length,
		arms: {
			rawGrep: {
				recall: `${rawGrepHits}/${sample.length} (${((rawGrepHits / sample.length) * 100).toFixed(1)}%)`,
				binaryRecall: `${rawGrepBinaryHits}/${rawGrepBinaryTotal} (${((rawGrepBinaryHits / rawGrepBinaryTotal) * 100).toFixed(1)}%)`,
				meanMs: Number(mean(rawGrepLatencies).toFixed(2)),
				p50Ms: Number(percentile(sortNums(rawGrepLatencies), 50).toFixed(2)),
				p95Ms: Number(percentile(sortNums(rawGrepLatencies), 95).toFixed(2)),
			},
			minSyncSparse: {
				top1Recall: `${minSyncTop1Hits}/${sample.length} (${((minSyncTop1Hits / sample.length) * 100).toFixed(1)}%)`,
				top10Recall: `${minSyncTop10Hits}/${sample.length} (${((minSyncTop10Hits / sample.length) * 100).toFixed(1)}%)`,
				meanMs: Number(mean(minSyncLatencies).toFixed(2)),
				p50Ms: Number(percentile(sortNums(minSyncLatencies), 50).toFixed(2)),
				p95Ms: Number(percentile(sortNums(minSyncLatencies), 95).toFixed(2)),
			},
			parsedMirrorGrepWholeCorpus: {
				recall: `${mirrorGrepHits}/${sample.length} (${((mirrorGrepHits / sample.length) * 100).toFixed(1)}%)`,
				binaryRecall: `${mirrorGrepBinaryHits}/${rawGrepBinaryTotal} (${((mirrorGrepBinaryHits / rawGrepBinaryTotal) * 100).toFixed(1)}%)`,
				pdfPageVerified: `${mirrorPageHits}/${mirrorPageTotal}`,
				meanMs: Number(mean(mirrorGrepCorpusLatencies).toFixed(2)),
				p50Ms: Number(percentile(sortNums(mirrorGrepCorpusLatencies), 50).toFixed(2)),
				p95Ms: Number(percentile(sortNums(mirrorGrepCorpusLatencies), 95).toFixed(2)),
			},
			parsedMirrorGrepFolderScoped: {
				meanMs: Number(mean(mirrorGrepFolderLatencies).toFixed(2)),
				p50Ms: Number(percentile(sortNums(mirrorGrepFolderLatencies), 50).toFixed(2)),
				p95Ms: Number(percentile(sortNums(mirrorGrepFolderLatencies), 95).toFixed(2)),
			},
			binaryReadComparison: {
				sampleSize: binarySample.length,
				rawReadReadable: `${rawReadReadable}/${binarySample.length}`,
				mirrorReadReadable: `${mirrorReadReadable}/${binarySample.length}`,
				mirrorReadMeanMs: Number(mean(mirrorReadLatencies).toFixed(2)),
			},
		},
		speedup: {
			mirrorGrepVsMinSync: `${(mean(minSyncLatencies) / Math.max(0.01, mean(mirrorGrepCorpusLatencies))).toFixed(1)}x`,
			mirrorGrepFolderVsMinSync: `${(mean(minSyncLatencies) / Math.max(0.01, mean(mirrorGrepFolderLatencies))).toFixed(1)}x`,
		},
	};

	console.log(JSON.stringify(report, null, 2));
} finally {
	rmSync(tmpRoot, { recursive: true, force: true });
}

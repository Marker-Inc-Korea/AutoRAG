import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { detect } from "chardet";
import iconv from "iconv-lite";
import { syncParsedMirrors } from "../../src/mirror/sync.ts";
import { type ParseInput, type ParseOutput, Parser, ParserRegistry } from "../../src/parser/index.ts";
import { decodeText } from "../../src/parser/text.ts";

/**
 * Measures the decodeText fast path against the pre-change implementation.
 *
 * Table 1: per-document decode cost for the three document shapes that matter (pure ASCII,
 * Korean UTF-8, Korean CP949), before (legacy copy of git HEAD) and after (current source), plus
 * the two ingredients of the ASCII fast path in isolation (the purity scan and the ASCII iconv
 * decode). chardet's detect() was measured at ~670-710us on these shapes, which is what the fast
 * path must avoid.
 *
 * Table 2: full mirror refresh (syncParsedMirrors, serial schedule, force) at two corpus sizes for
 * ASCII and Korean corpora, legacy decode versus current decode. decodeText runs only in the
 * mirror parse stage, so the delta here is the whole-refresh delta; the BM25 stage is untouched by
 * this change.
 */

const WINDOWS_949_LABELS = new Set(["EUC-KR", "ISO-2022-KR", "windows-949", "CP949"]);

/** Copy of the pre-fast-path decodeText (git HEAD of src/parser/text.ts). */
function legacyDecodeText(bytes: Uint8Array): string {
	const buffer = Buffer.from(bytes);
	const utf8 = iconv.decode(buffer, "utf8");
	if (legacyReplacementCount(utf8) > 0) {
		const cp949 = iconv.decode(buffer, "cp949");
		if (legacyReplacementCount(cp949) < legacyReplacementCount(utf8)) return cp949.normalize("NFC");
	}
	const detected = detect(buffer);
	const encoding = detected && WINDOWS_949_LABELS.has(detected) ? "cp949" : (detected ?? "utf8");
	return iconv.decode(buffer, encoding).normalize("NFC");
}

function legacyReplacementCount(value: string): number {
	return [...value].filter((character) => character === "\uFFFD").length;
}

/** The fast-path purity scan in isolation (copy of the predicate in src/parser/text.ts). */
function isPureAsciiScan(bytes: Uint8Array): boolean {
	if (bytes.length === 0) return true;
	if (bytes.length < 4) return false;
	for (let i = 0; i < bytes.length; i += 1) {
		const byte = bytes[i] as number;
		if (byte < 0x01 || byte === 0x1b || byte > 0x7f) return false;
	}
	return true;
}

function median(times: readonly number[]): number {
	const sorted = [...times].sort((a, b) => a - b);
	return sorted[Math.floor(sorted.length / 2)] as number;
}

/** Median microseconds per call over `rounds` batches of `iterations` calls each. */
function measurePerCall(fn: () => string, rounds: number, iterations: number): number {
	const times: number[] = [];
	for (let round = 0; round < rounds; round += 1) {
		const started = performance.now();
		for (let i = 0; i < iterations; i += 1) fn();
		times.push(((performance.now() - started) * 1000) / iterations);
	}
	return Number(median(times).toFixed(1));
}

const asciiDoc = Buffer.from(
	"# Document 0\n\nRefund approval policy and escalation threshold.\nQuarterly chargeback report.\n".repeat(46),
);
const koreanText = [
	"# 문서 제목\n\n환불 정책과 승인 절차에 대한 안내입니다.\n",
	"매 분기마다 차지백 보고서를 작성하고, 승인 임계값을 검토합니다.\n",
	"고객 지원 팀은 접수된 문의를 우선순위에 따라 처리합니다.\n",
].join("\n");
const koreanUtf8Doc = Buffer.from(koreanText.repeat(22), "utf8");
const koreanCp949Doc = iconv.encode(koreanText.repeat(22), "cp949");

const docs = [
	{ label: "ASCII 4KB", bytes: asciiDoc },
	{ label: "Korean UTF-8 5.5KB", bytes: koreanUtf8Doc },
	{ label: "Korean CP949 5.5KB", bytes: koreanCp949Doc },
] as const;

console.log("| document type | bytes | legacy decode (us) | new decode (us) | speedup |");
console.log("|---|---|---|---|---|");
const decodeRows: Record<string, number | string>[] = [];
for (const doc of docs) {
	const legacy = measurePerCall(() => legacyDecodeText(doc.bytes), 5, 200);
	const current = measurePerCall(() => decodeText(doc.bytes), 5, 200);
	decodeRows.push({
		doc: doc.label,
		bytes: doc.bytes.length,
		legacyUs: legacy,
		newUs: current,
		speedup: Number((legacy / current).toFixed(1)),
	});
	console.log(`| ${doc.label} | ${doc.bytes.length} | ${legacy} | ${current} | ${(legacy / current).toFixed(2)}x |`);
}

// The two ingredients of the ASCII fast path, isolated, so the scan cost itself is on record.
const scanUs = measurePerCall(() => (isPureAsciiScan(asciiDoc) ? "x" : "y"), 5, 2000);
const asciiDecodeUs = measurePerCall(() => iconv.decode(asciiDoc, "ascii"), 5, 2000);
console.log(`\nASCII fast-path ingredients: purity scan ${scanUs}us, ascii iconv decode ${asciiDecodeUs}us (4KB doc)`);

interface Corpus {
	readonly root: string;
	readonly docs: string;
}

function buildCorpus(documentCount: number, body: Buffer): Corpus {
	const root = mkdtempSync(join(tmpdir(), "autorag-measure-decode-"));
	const docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	for (let i = 0; i < documentCount; i += 1) {
		writeFileSync(join(docs, `doc-${String(i).padStart(5, "0")}.md`), body);
	}
	return { root, docs };
}

class DecodeParser extends Parser {
	readonly name = "plain-text";
	readonly extensions = [".md"] as const;
	readonly decode: (bytes: Uint8Array) => string;

	constructor(decode: (bytes: Uint8Array) => string) {
		super();
		this.decode = decode;
	}

	async parse(input: ParseInput): Promise<ParseOutput> {
		return { markdown: this.decode(input.bytes) };
	}
}

async function medianRefreshMs(
	root: string,
	docs: string,
	decode: (bytes: Uint8Array) => string,
	samples: number,
): Promise<number> {
	const times: number[] = [];
	for (let i = 0; i < samples; i += 1) {
		const started = performance.now();
		await syncParsedMirrors({
			root,
			searchPaths: [docs],
			registry: new ParserRegistry([new DecodeParser(decode)]),
			force: true,
		});
		times.push(performance.now() - started);
	}
	return Number(median(times).toFixed(0));
}

console.log("\n| corpus | legacy refresh (ms) | new refresh (ms) | delta |");
console.log("|---|---|---|---|");
const refreshRows: Record<string, number | string>[] = [];
for (const [label, body] of [
	["ASCII 4KB x100", asciiDoc],
	["ASCII 4KB x400", asciiDoc],
	["Korean UTF-8 5.5KB x100", koreanUtf8Doc],
	["Korean UTF-8 5.5KB x400", koreanUtf8Doc],
] as const) {
	const count = label.includes("x400") ? 400 : 100;
	const sized = buildCorpus(count, body);
	try {
		const legacy = await medianRefreshMs(sized.root, sized.docs, legacyDecodeText, 3);
		const current = await medianRefreshMs(sized.root, sized.docs, decodeText, 3);
		const delta = Number((((current - legacy) / legacy) * 100).toFixed(1));
		refreshRows.push({ corpus: label, legacyMs: legacy, newMs: current, deltaPct: delta });
		console.log(`| ${label} | ${legacy} | ${current} | ${delta >= 0 ? "+" : ""}${delta}% |`);
	} finally {
		rmSync(sized.root, { recursive: true, force: true });
	}
}

console.log(`\nJSON: ${JSON.stringify({ decodeRows, scanUs, asciiDecodeUs, refreshRows })}`);

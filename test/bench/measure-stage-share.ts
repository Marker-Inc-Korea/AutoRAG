import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { syncParsedMirrors } from "../../src/mirror/sync.ts";
import { BM25Method } from "../../src/retrieval/methods/bm25.ts";

/**
 * Splits an unchanged refresh into its two local stages so the end-to-end effect of the BM25
 * fingerprint skip can be stated honestly.
 *
 * The mirror stage is untouched by this change; only the BM25 stage gets the skip. Reporting the
 * BM25 speedup alone would overstate what a user experiences, so this measures both stages and
 * derives the combined figure.
 */

function buildCorpus(documentCount: number): { root: string; searchPaths: string[] } {
	const root = mkdtempSync(join(tmpdir(), "autorag-stage-"));
	const docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	for (let i = 0; i < documentCount; i += 1) {
		const body = Array.from(
			{ length: 40 },
			(_, p) => `Paragraph ${p} of document ${i}. Refund approval policy and chargeback evidence retention.`,
		).join("\n\n");
		writeFileSync(join(docs, `doc-${String(i).padStart(5, "0")}.md`), `# Document ${i}\n\n${body}\n`);
	}
	return { root, searchPaths: [docs] };
}

async function median(samples: number, fn: () => Promise<void>): Promise<number> {
	const times: number[] = [];
	for (let i = 0; i < samples; i += 1) {
		const started = performance.now();
		await fn();
		times.push(performance.now() - started);
	}
	times.sort((a, b) => a - b);
	return times[Math.floor(times.length / 2)] as number;
}

const rows: Record<string, number | string>[] = [];

for (const size of [200, 800, 2000]) {
	const { root, searchPaths } = buildCorpus(size);
	try {
		await syncParsedMirrors({ root, searchPaths });
		const method = new BM25Method({ root, forceEngine: "typescript-fallback" });
		await method.sync();

		const mirrorMs = await median(5, async () => {
			await syncParsedMirrors({ root, searchPaths });
		});
		const bm25BeforeMs = await median(5, async () => {
			await method.sync({ force: true });
		});
		const bm25AfterMs = await median(5, async () => {
			await method.sync();
		});

		const totalBefore = mirrorMs + bm25BeforeMs;
		const totalAfter = mirrorMs + bm25AfterMs;
		rows.push({
			documents: size,
			mirrorMs: Number(mirrorMs.toFixed(1)),
			bm25BeforeMs: Number(bm25BeforeMs.toFixed(1)),
			bm25AfterMs: Number(bm25AfterMs.toFixed(2)),
			totalBeforeMs: Number(totalBefore.toFixed(1)),
			totalAfterMs: Number(totalAfter.toFixed(1)),
			bm25ShareBefore: `${((bm25BeforeMs / totalBefore) * 100).toFixed(0)}%`,
			endToEndSpeedup: `${(totalBefore / totalAfter).toFixed(2)}x`,
		});
	} finally {
		rmSync(root, { recursive: true, force: true });
	}
}

console.log(
	"\n| docs | mirror (ms) | BM25 before | BM25 after | total before | total after | BM25 share | end-to-end |",
);
console.log("|---|---|---|---|---|---|---|---|");
for (const r of rows) {
	console.log(
		`| ${r.documents} | ${r.mirrorMs} | ${r.bm25BeforeMs} | ${r.bm25AfterMs} | ${r.totalBeforeMs} | ${r.totalAfterMs} | ${r.bm25ShareBefore} | ${r.endToEndSpeedup} |`,
	);
}
console.log(`\nJSON: ${JSON.stringify(rows)}`);

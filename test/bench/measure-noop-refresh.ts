import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { syncParsedMirrors } from "../../src/mirror/sync.ts";
import { BM25Method } from "../../src/retrieval/methods/bm25.ts";

/**
 * Measures the cost of an unchanged ("noop") BM25 sync across corpus sizes, with the fingerprint
 * skip enabled and disabled, so the two curves can be compared directly.
 *
 * `force: true` reproduces the pre-change behaviour exactly: it takes the same code path the old
 * unconditional `sync()` took (loadChunks over every mirror, then a full artifact rewrite).
 */

type Row = {
	readonly documents: number;
	readonly chunks: number;
	readonly beforeMs: number;
	readonly afterMs: number;
	readonly speedup: number;
};

function buildCorpus(documentCount: number): { root: string; searchPaths: string[] } {
	const root = mkdtempSync(join(tmpdir(), "autorag-bench-"));
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

async function measure(documentCount: number, samples: number): Promise<Row> {
	const { root, searchPaths } = buildCorpus(documentCount);
	try {
		await syncParsedMirrors({ root, searchPaths });
		const method = new BM25Method({ root, forceEngine: "typescript-fallback" });
		const built = await method.sync();

		// force:true = the old unconditional path. force:false = the new fingerprint skip.
		const beforeMs = await median(samples, async () => {
			await method.sync({ force: true });
		});
		const afterMs = await median(samples, async () => {
			await method.sync();
		});

		return {
			documents: documentCount,
			chunks: built.indexedChunks,
			beforeMs,
			afterMs,
			speedup: beforeMs / afterMs,
		};
	} finally {
		rmSync(root, { recursive: true, force: true });
	}
}

const sizes = [50, 200, 800, 2000];
const rows: Row[] = [];
for (const size of sizes) {
	rows.push(await measure(size, 5));
}

console.log("\n| documents | chunks | before (noop, ms) | after (noop, ms) | speedup |");
console.log("|---|---|---|---|---|");
for (const row of rows) {
	console.log(
		`| ${row.documents} | ${row.chunks} | ${row.beforeMs.toFixed(1)} | ${row.afterMs.toFixed(2)} | ${row.speedup.toFixed(0)}x |`,
	);
}
console.log(`\nJSON: ${JSON.stringify(rows)}`);

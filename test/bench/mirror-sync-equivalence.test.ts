import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, relative } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
	loadMirrorIndex,
	type ParsedMirrorDiagnostic,
	type ParsedMirrorIndex,
	syncParsedMirrors,
} from "../../src/mirror/index.ts";
import { ParseError, type ParseInput, type ParseOutput, Parser, ParserRegistry } from "../../src/parser/index.ts";
import { decodeText } from "../../src/parser/text.ts";

/**
 * Equivalence safety net for `syncParsedMirrors()` before it is parallelized.
 *
 * The current loop in src/mirror/sync.ts parses one file at a time and writes each mirror
 * atomically. Parallelizing it (bounded concurrency over the sorted file list) must not change
 * any observable behavior, and the property that is easiest to break is ordering: diagnostics,
 * index insertion order, and the loop's failure isolation are all driven by the serial,
 * localeCompare-sorted walk. Every test below pins an observable contract that the parallel
 * implementation must keep, and the mutation checks at the end of the run (see test/bench/README.md)
 * prove the tests are not vacuous.
 *
 * Corpora are deliberately ASCII so that decode -> NFC normalize -> write is the identity: mirror
 * bytes then equal source bytes exactly, without re-implementing normalizeMarkdown in this file.
 * All workspaces live under the system temp directory and are removed after each test; nothing in
 * this file ever points `root` at the repository, so no `.autorag` can appear there.
 */

const roots: string[] = [];

function makeWorkspace(): { root: string; docs: string } {
	const root = mkdtempSync(join(tmpdir(), "autorag-equivalence-"));
	roots.push(root);
	const docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	return { root, docs };
}

afterEach(() => {
	for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

function writeCorpus(docs: string, files: Record<string, string>): void {
	for (const [name, content] of Object.entries(files)) {
		const path = join(docs, name);
		mkdirSync(dirname(path), { recursive: true });
		writeFileSync(path, content);
	}
}

/** Maps virtual path (/docs/<name>) to the exact source content the mirror must reproduce. */
function expectedVirtual(files: Record<string, string>): Record<string, string> {
	return Object.fromEntries(Object.entries(files).map(([name, content]) => [`/docs/${name}`, content]));
}

function requireValue<T>(value: T | undefined, label: string): T {
	if (value === undefined) throw new Error(`missing ${label}`);
	return value;
}

/** Code:severity:source per diagnostic. Array equality on these pins content AND order. */
function diagnosticTuples(diagnostics: readonly ParsedMirrorDiagnostic[]): string[] {
	return diagnostics.map((diagnostic) => `${diagnostic.code}:${diagnostic.severity}:${diagnostic.source}`);
}

function mirrorBytesByVirtualPath(index: ParsedMirrorIndex): Record<string, string> {
	const bytes: Record<string, string> = {};
	for (const entry of Object.values(index.entries)) {
		bytes[entry.virtualPath] = readFileSync(entry.outputPath, "utf8");
	}
	return bytes;
}

function sortedKeys(entries: Readonly<Record<string, unknown>>): string[] {
	const keys = Object.keys(entries);
	return [...keys].sort((a, b) => a.localeCompare(b));
}

/**
 * Controlled parser used across the suite. Fails with a typed `ParseError` for the configured
 * virtual paths and counts every parse call, which is how the no-reparse-on-resume contract is
 * observed.
 */
class ControlledParser extends Parser {
	readonly name = "controlled";
	readonly extensions = [".md", ".txt"] as const;
	parseCalls = 0;
	private readonly failPaths: ReadonlySet<string>;

	constructor(failPaths: ReadonlySet<string> = new Set()) {
		super();
		this.failPaths = failPaths;
	}

	async parse(input: ParseInput): Promise<ParseOutput> {
		this.parseCalls += 1;
		if (this.failPaths.has(input.virtualPath)) {
			throw new ParseError(this.name, input.virtualPath, new Error("deliberate failure for the safety net"));
		}
		return { markdown: decodeText(input.bytes) };
	}
}

/** Parser that blocks until the test releases it; used to observe mid-run checkpoint state. */
class BlockingParser extends Parser {
	readonly name = "blocking";
	readonly extensions = [".slow"] as const;
	parseCalls = 0;
	private readonly gate: Promise<void>;

	constructor(gate: Promise<void>) {
		super();
		this.gate = gate;
	}

	async parse(input: ParseInput): Promise<ParseOutput> {
		this.parseCalls += 1;
		await this.gate;
		return { markdown: decodeText(input.bytes) };
	}
}

function sleep(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitFor(condition: () => boolean, label: string, timeoutMs = 5_000): Promise<void> {
	const start = Date.now();
	while (!condition()) {
		if (Date.now() - start > timeoutMs) throw new Error(`timed out waiting for ${label}`);
		await sleep(5);
	}
}

/**
 * Project an index onto run-independent comparable fields.
 *
 * Excluded or normalized, with reasons:
 * - `sourcePath`, `outputPath`, `indexPath`: absolute paths that embed the per-run temp workspace
 *   root; they differ between two runs even for an identical corpus. Normalized to root-relative.
 * - `sourceMtimeNs`: wall-clock file metadata; the two runs write files at different instants, so
 *   the values are never equal. Only type/range-validated here.
 * - `updatedAt`: wall-clock ISO timestamp, likewise never equal across runs. Only format-validated.
 * Everything else (virtualPath, parserName, sourceSizeBytes, contentSha256) is compared as-is.
 */
function comparableEntries(root: string, index: ParsedMirrorIndex): unknown[] {
	return Object.values(index.entries)
		.map((entry) => {
			if (!Number.isFinite(entry.sourceMtimeNs)) {
				throw new Error(`entry ${entry.virtualPath} has a non-finite sourceMtimeNs`);
			}
			if (Number.isNaN(Date.parse(entry.updatedAt))) {
				throw new Error(`entry ${entry.virtualPath} has an invalid updatedAt`);
			}
			if (entry.contentSha256 === undefined) {
				throw new Error(`entry ${entry.virtualPath} has no contentSha256 after a forced sync`);
			}
			return {
				virtualPath: entry.virtualPath,
				parserName: entry.parserName,
				sourceSizeBytes: entry.sourceSizeBytes,
				contentSha256: entry.contentSha256,
				sourcePath: relative(root, entry.sourcePath),
				outputPath: relative(root, entry.outputPath),
			};
		})
		.sort((a, b) => a.virtualPath.localeCompare(b.virtualPath));
}

describe("syncParsedMirrors equivalence safety net", () => {
	it("(a) repeat runs are fully deterministic: counts, diagnostics in order, index fields, mirror bytes", async () => {
		const runOnce = async (): Promise<{
			root: string;
			result: Awaited<ReturnType<typeof syncParsedMirrors>>;
			index: ParsedMirrorIndex;
		}> => {
			const { root, docs } = makeWorkspace();
			writeCorpus(docs, {
				"a.md": "Alpha\n",
				"b-broken.md": "Beta broken\n",
				"c.md": "Gamma\n",
				"d-broken.md": "Delta broken\n",
				"e.md": "Epsilon\n",
				"f-big.md": "x".repeat(200),
			});
			const result = await syncParsedMirrors({
				root,
				searchPaths: [docs],
				// Failures and size skips are part of the corpus on purpose: an all-clean run would
				// leave the diagnostics order contract untested.
				registry: new ParserRegistry([new ControlledParser(new Set(["/docs/b-broken.md", "/docs/d-broken.md"]))]),
				maxSourceBytes: 40,
				force: true,
			});
			return { root, result, index: loadMirrorIndex(root) };
		};

		const first = await runOnce();
		const second = await runOnce();

		expect(second.result.scanned).toBe(first.result.scanned);
		expect(second.result.written).toBe(first.result.written);
		expect(second.result.deleted).toBe(first.result.deleted);
		expect(second.result.skipped).toBe(first.result.skipped);
		// Full-array equality: same length, same elements, same order.
		expect(second.result.diagnostics).toEqual(first.result.diagnostics);
		// Sanity: the corpus really produced the interesting shape (3 written, 2 failed, 1 oversized).
		expect(first.result).toMatchObject({ scanned: 6, written: 3, deleted: 0, skipped: 3 });
		expect(diagnosticTuples(first.result.diagnostics)).toEqual([
			"parser-failed:warning:/docs/b-broken.md",
			"parser-failed:warning:/docs/d-broken.md",
			"parser-skipped:warning:/docs/f-big.md",
		]);
		expect(comparableEntries(first.root, first.index)).toEqual(comparableEntries(second.root, second.index));
		// Index insertion order is the loop order, which is the localeCompare contract.
		expect(sortedKeys(first.index.entries)).toEqual(Object.keys(first.index.entries));
		expect(sortedKeys(second.index.entries)).toEqual(Object.keys(second.index.entries));
		expect(mirrorBytesByVirtualPath(first.index)).toEqual(mirrorBytesByVirtualPath(second.index));
	});

	describe("(b) corpus shapes", () => {
		it("(b1) fresh corpus: every supported file is written exactly once with exact bytes", async () => {
			const { root, docs } = makeWorkspace();
			const corpus = { "a.md": "Alpha\n", "b.md": "Beta\n", "c.md": "Gamma\n", "d.txt": "Delta\n" };
			writeCorpus(docs, corpus);

			const result = await syncParsedMirrors({
				root,
				searchPaths: [docs],
				registry: new ParserRegistry([new ControlledParser()]),
			});

			expect(result).toMatchObject({ scanned: 4, written: 4, deleted: 0, skipped: 0 });
			expect(result.diagnostics).toEqual([]);
			const index = loadMirrorIndex(root);
			const expected = expectedVirtual(corpus);
			expect(Object.keys(index.entries)).toHaveLength(4);
			for (const entry of Object.values(index.entries)) {
				// ASCII corpus: NFC normalization is the identity, so the mirror must equal the source.
				expect(readFileSync(entry.outputPath, "utf8")).toBe(expected[entry.virtualPath]);
				expect(entry.contentSha256).toBeDefined();
			}
		});

		it("(b2) partial change: only changed files are rewritten, unchanged mirrors keep their digest", async () => {
			const { root, docs } = makeWorkspace();
			const registry = new ParserRegistry([new ControlledParser()]);
			writeCorpus(docs, { "a.md": "Alpha\n", "b.md": "Beta\n", "c.md": "Gamma\n", "d.md": "Delta\n" });
			await syncParsedMirrors({ root, searchPaths: [docs], registry });
			const before = loadMirrorIndex(root);

			// Different content and different size, so the mtime+size staleness check fires.
			writeCorpus(docs, { "b.md": "Beta has grown much longer now\n", "d.md": "Delta changed\n" });
			const result = await syncParsedMirrors({ root, searchPaths: [docs], registry });

			expect(result).toMatchObject({ scanned: 4, written: 2, deleted: 0, skipped: 0 });
			expect(result.diagnostics).toEqual([]);
			const after = loadMirrorIndex(root);
			expect(after.entries["/docs/b.md"]?.contentSha256).not.toBe(before.entries["/docs/b.md"]?.contentSha256);
			expect(after.entries["/docs/d.md"]?.contentSha256).not.toBe(before.entries["/docs/d.md"]?.contentSha256);
			// Untouched files keep their digest: they were not reparsed or rewritten.
			expect(after.entries["/docs/a.md"]?.contentSha256).toBe(before.entries["/docs/a.md"]?.contentSha256);
			expect(after.entries["/docs/c.md"]?.contentSha256).toBe(before.entries["/docs/c.md"]?.contentSha256);
			expect(readFileSync(requireValue(after.entries["/docs/b.md"], "b.md").outputPath, "utf8")).toBe(
				"Beta has grown much longer now\n",
			);
			expect(readFileSync(requireValue(after.entries["/docs/d.md"], "d.md").outputPath, "utf8")).toBe(
				"Delta changed\n",
			);
		});

		it("(b3) deletion: a removed source drops its mirror file and emits a deleted-mirror diagnostic", async () => {
			const { root, docs } = makeWorkspace();
			const registry = new ParserRegistry([new ControlledParser()]);
			writeCorpus(docs, { "keep.txt": "Keep\n", "gone.md": "Gone\n", "stay.md": "Stay\n" });
			const first = await syncParsedMirrors({ root, searchPaths: [docs], registry });
			expect(first.written).toBe(3);
			const goneOutput = requireValue(
				loadMirrorIndex(root).entries["/docs/gone.md"],
				"gone.md output path",
			).outputPath;

			unlinkSync(join(docs, "gone.md"));
			const second = await syncParsedMirrors({ root, searchPaths: [docs], registry });

			expect(second).toMatchObject({ scanned: 2, written: 0, deleted: 1, skipped: 0 });
			expect(diagnosticTuples(second.diagnostics)).toEqual(["deleted-mirror:info:/docs/gone.md"]);
			const index = loadMirrorIndex(root);
			expect(index.entries["/docs/gone.md"]).toBeUndefined();
			expect(existsSync(goneOutput)).toBe(false);
		});

		it("(b4) parser failure: broken files are skipped with parser-failed diagnostics, every other mirror survives", async () => {
			const { root, docs } = makeWorkspace();
			const registry = new ParserRegistry([
				new ControlledParser(new Set(["/docs/broken-1.md", "/docs/broken-2.md"])),
			]);
			const corpus = {
				"broken-1.md": "Broken one\n",
				"broken-2.md": "Broken two\n",
				"good-a.md": "Good A\n",
				"good-b.md": "Good B\n",
				"good-c.md": "Good C\n",
			};
			writeCorpus(docs, corpus);

			const result = await syncParsedMirrors({ root, searchPaths: [docs], registry });

			expect(result).toMatchObject({ scanned: 5, written: 3, skipped: 2, deleted: 0 });
			// Full-array equality pins content AND order (virtualPath order, not file-system order).
			expect(diagnosticTuples(result.diagnostics)).toEqual([
				"parser-failed:warning:/docs/broken-1.md",
				"parser-failed:warning:/docs/broken-2.md",
			]);
			const index = loadMirrorIndex(root);
			expect(index.entries["/docs/broken-1.md"]).toBeUndefined();
			expect(index.entries["/docs/broken-2.md"]).toBeUndefined();
			const expected = expectedVirtual(corpus);
			for (const name of ["good-a.md", "good-b.md", "good-c.md"]) {
				const entry = requireValue(index.entries[`/docs/${name}`], name);
				expect(readFileSync(entry.outputPath, "utf8")).toBe(expected[`/docs/${name}`]);
			}
		});

		it("(b5) size cap: oversized sources are skipped with parser-skipped diagnostics, the rest are processed", async () => {
			const { root, docs } = makeWorkspace();
			const registry = new ParserRegistry([new ControlledParser()]);
			const corpus: Record<string, string> = {
				"big.md": "x".repeat(200),
				"small-a.md": "Small A\n",
				"small-b.md": "Small B\n",
			};
			writeCorpus(docs, corpus);

			const result = await syncParsedMirrors({ root, searchPaths: [docs], registry, maxSourceBytes: 40 });

			expect(result).toMatchObject({ scanned: 3, written: 2, skipped: 1, deleted: 0 });
			expect(diagnosticTuples(result.diagnostics)).toEqual(["parser-skipped:warning:/docs/big.md"]);
			const index = loadMirrorIndex(root);
			expect(index.entries["/docs/big.md"]).toBeUndefined();
			for (const name of ["small-a.md", "small-b.md"]) {
				const entry = requireValue(index.entries[`/docs/${name}`], name);
				expect(readFileSync(entry.outputPath, "utf8")).toBe(requireValue(corpus[name], name));
			}

			// Raising the cap later must recover the skipped file; the size gate must not poison it.
			const recovery = await syncParsedMirrors({ root, searchPaths: [docs], registry });
			expect(recovery).toMatchObject({ scanned: 3, written: 1, skipped: 0 });
			expect(loadMirrorIndex(root).entries["/docs/big.md"]).toBeDefined();
		});

		it("(b6) empty corpus: nothing scanned, nothing written, empty index", async () => {
			const { root, docs } = makeWorkspace();

			const result = await syncParsedMirrors({
				root,
				searchPaths: [docs],
				registry: new ParserRegistry([new ControlledParser()]),
			});

			expect(result).toMatchObject({ scanned: 0, written: 0, deleted: 0, skipped: 0 });
			expect(result.diagnostics).toEqual([]);
			expect(loadMirrorIndex(root).entries).toEqual({});
		});

		it("(b7) nested directories: subfolder documents are indexed under their virtual paths", async () => {
			const { root, docs } = makeWorkspace();
			const registry = new ParserRegistry([new ControlledParser()]);
			const corpus = { "root.md": "Root\n", "sub/a.md": "Sub A\n", "sub/deep/b.md": "Deep B\n" };
			writeCorpus(docs, corpus);

			const result = await syncParsedMirrors({ root, searchPaths: [docs], registry });

			expect(result).toMatchObject({ scanned: 3, written: 3, deleted: 0, skipped: 0 });
			const index = loadMirrorIndex(root);
			const keys = Object.keys(index.entries);
			expect(keys).toEqual(sortedKeys(index.entries));
			expect(new Set(keys)).toEqual(new Set(["/docs/root.md", "/docs/sub/a.md", "/docs/sub/deep/b.md"]));
			const expected = expectedVirtual(corpus);
			for (const entry of Object.values(index.entries)) {
				expect(readFileSync(entry.outputPath, "utf8")).toBe(expected[entry.virtualPath]);
			}
		});
	});

	it("(c) deterministic order: keys and diagnostics follow virtualPath localeCompare, not creation order", async () => {
		const { root, docs } = makeWorkspace();
		// Written deliberately in reverse-sorted order (z first, a last): a directory walk in
		// creation order would surface z first, and that must not leak into the index or diagnostics.
		const parser = new ControlledParser(new Set(["/docs/a-broken.md", "/docs/m-broken.md", "/docs/z-broken.md"]));
		const registry = new ParserRegistry([parser]);
		const corpus = {
			"z.md": "Z\n",
			"z-broken.md": "Z broken\n",
			"y.md": "Y\n",
			"y-broken.md": "Y broken\n",
			"m.md": "M\n",
			"m-broken.md": "M broken\n",
			"a.md": "A\n",
			"a-broken.md": "A broken\n",
		};
		writeCorpus(docs, corpus);

		const result = await syncParsedMirrors({ root, searchPaths: [docs], registry });
		const index = loadMirrorIndex(root);

		// The pin: production order equals what localeCompare produces, whatever the locale.
		expect(Object.keys(index.entries)).toEqual(sortedKeys(index.entries));
		const diagSources = result.diagnostics.map((diagnostic) => diagnostic.source);
		expect(diagSources).toEqual([...diagSources].sort((a, b) => a.localeCompare(b)));
		// The three failing sources differ in their leading letter, so this order is locale-stable.
		expect(diagSources).toEqual(["/docs/a-broken.md", "/docs/m-broken.md", "/docs/z-broken.md"]);
		expect(parser.parseCalls).toBe(8);
	});

	it("(d) checkpoint resume: mirrors written before a mid-run checkpoint are never reparsed afterwards", async () => {
		const { root, docs } = makeWorkspace();
		const fast = new ControlledParser();
		let releaseGate: () => void = () => {};
		const gate = new Promise<void>((resolve) => {
			releaseGate = resolve;
		});
		const blocking = new BlockingParser(gate);
		const registry = new ParserRegistry([fast, blocking]);
		// MIRROR_CHECKPOINT_EVERY in src/mirror/sync.ts is 25: with 31 files the checkpoint after
		// file 25 lands while file 31 (z-block.slow, last in virtualPath order) is still parsing,
		// so the checkpointed index can be observed on disk mid-run.
		const corpus: Record<string, string> = {};
		for (let i = 0; i < 30; i += 1) {
			corpus[`doc-${String(i).padStart(2, "0")}.md`] = `Doc ${i}\n`;
		}
		corpus["z-block.slow"] = "Blocked\n";
		writeCorpus(docs, corpus);

		const run = syncParsedMirrors({ root, searchPaths: [docs], registry });
		await waitFor(() => blocking.parseCalls === 1, "blocking parser to start");

		// The checkpoint is already on disk while the run is still in flight: a crash and restart
		// here would see exactly these 25 completed entries, each with its content digest.
		const checkpointed = loadMirrorIndex(root);
		const expectedCheckpointKeys = Array.from({ length: 25 }, (_, i) => `/docs/doc-${String(i).padStart(2, "0")}.md`);
		expect(Object.keys(checkpointed.entries)).toEqual(expectedCheckpointKeys);
		for (const entry of Object.values(checkpointed.entries)) {
			expect(entry.contentSha256).toBeDefined();
		}

		releaseGate();
		const result = await run;
		expect(result).toMatchObject({ scanned: 31, written: 31, deleted: 0, skipped: 0 });
		expect(fast.parseCalls + blocking.parseCalls).toBe(31);

		// The next sync is a no-op: every mirror already on disk is reused without a single parse.
		const again = await syncParsedMirrors({ root, searchPaths: [docs], registry });
		expect(again).toMatchObject({ scanned: 31, written: 0, deleted: 0, skipped: 0 });
		expect(fast.parseCalls + blocking.parseCalls).toBe(31);
	});

	describe("(e) invariants the parallel implementation must keep", () => {
		it("(e1) cross-contamination: each mirror file contains exactly its own source's parse output", async () => {
			const { root, docs } = makeWorkspace();
			// Every file has a unique body so any swap between mirrors is immediately visible.
			const corpus = {
				"alpha.md": "# Alpha\n\nalpha-only body one\n",
				"beta.md": "# Beta\n\nbeta-only body two\n",
				"gamma.txt": "gamma-only body three\n",
				"delta.md": "# Delta\n\ndelta-only body four\n",
				"epsilon.txt": "epsilon-only body five\n",
			};
			writeCorpus(docs, corpus);

			const result = await syncParsedMirrors({
				root,
				searchPaths: [docs],
				registry: new ParserRegistry([new ControlledParser()]),
			});

			expect(result.written).toBe(5);
			const expected = expectedVirtual(corpus);
			const index = loadMirrorIndex(root);
			expect(Object.keys(index.entries)).toHaveLength(5);
			for (const entry of Object.values(index.entries)) {
				expect(readFileSync(entry.outputPath, "utf8")).toBe(expected[entry.virtualPath]);
			}
		});

		it("(e2) contentSha256 is the sha256 of the exact bytes on disk", async () => {
			const { root, docs } = makeWorkspace();
			writeCorpus(docs, { "a.md": "Alpha\n", "b.txt": "Beta\n", "c.md": "Gamma\n" });

			await syncParsedMirrors({
				root,
				searchPaths: [docs],
				registry: new ParserRegistry([new ControlledParser()]),
			});

			const index = loadMirrorIndex(root);
			expect(Object.keys(index.entries)).toHaveLength(3);
			for (const entry of Object.values(index.entries)) {
				const digest = requireValue(entry.contentSha256, `${entry.virtualPath} contentSha256`);
				expect(createHash("sha256").update(readFileSync(entry.outputPath)).digest("hex")).toBe(digest);
			}
		});

		it("(e3) path-opaque diagnostics: source is a virtual path and never embeds the temp root", async () => {
			const { root, docs } = makeWorkspace();
			const registry = new ParserRegistry([new ControlledParser(new Set(["/docs/broken.md"]))]);
			writeCorpus(docs, { "broken.md": "B\n", "ok.md": "O\n", "too-big.md": "x".repeat(100) });

			const result = await syncParsedMirrors({ root, searchPaths: [docs], registry, maxSourceBytes: 20 });

			expect(result.diagnostics).toHaveLength(2);
			for (const diagnostic of result.diagnostics) {
				expect(diagnostic.source.startsWith("/")).toBe(true);
				expect(diagnostic.source).not.toContain(root);
			}
			expect(JSON.stringify(result.diagnostics)).not.toContain(root);
		});
	});
});

import { createHash } from "node:crypto";
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { loadMirrorIndex } from "../../src/mirror/index-store.ts";
import { syncParsedMirrors } from "../../src/mirror/sync.ts";
import { BM25Method, INDEX_SEMANTICS_VERSION } from "../../src/retrieval/methods/bm25.ts";
import type { RetrievalResult } from "../../src/retrieval/types.ts";

/**
 * Guard and proof harness for the BM25 index-semantics layers.
 *
 * The fingerprint skip trusts a stored artifact without opening it, so anything that changes what
 * an index *means* must invalidate the stored fingerprint or stale results get reused silently.
 * Two layers close the gap:
 * - Runtime: mirror entries (membership, content digest, parser name), MAX_CHUNK_CHARS, and
 *   INDEX_SEMANTICS_VERSION are hashed into the fingerprint, so tuning the chunking constant
 *   invalidates stored indexes automatically.
 * - CI: every INDEX_SEMANTIC_REGION block in the guarded sources is digested below in
 *   deterministic path-sorted order. The guarded sources are src/retrieval/methods/bm25.ts
 *   (fallback persistence schema, tantivy schema and document construction, fallback
 *   serialization/deserialization, chunk creation, chunk-id hashing, fingerprint material, and
 *   the sidecar trust path that decides whether a stored artifact is reused: fingerprint
 *   computation, equality, engine and artifact-size checks, and fingerprint recording) and
 *   src/parser/text.ts (decodeText and its helpers, which turn file bytes into the text stored in
 *   every chunk, and normalizeMarkdown, which shapes that text). Bundling
 *   bundle and force rebuilds when nothing changed; a CI-time digest of the real files has no such
 *   problem. The guard fails until the developer bumps INDEX_SEMANTICS_VERSION or consciously
 *   refreshes the recorded digest for a pure refactor.
 *
 * Query-time semantics (tokenize, bm25Score, BM25_K1/BM25_B) are deliberately excluded from both
 * layers: they run against the raw chunks already stored in the artifact, so a change applies to
 * existing artifacts immediately and never requires a rebuild. A test below fixes that contract
 * end to end by swapping the tokenizer regex in a copy of bm25.ts and proving that the artifact is
 * reused byte-for-byte while the query results change.
 */
/**
 * Tracks artifact commits performed through node:fs during a sync.
 *
 * `sync()` must commit the chosen artifact and record its fingerprint exactly once. The mock wraps
 * only writeFileSync/renameSync, delegating everything else to the real module, so every test in
 * this file runs against the real filesystem behavior. `renamedDestinations` records every rename
 * destination so tests can count artifact commits directly.
 */
const fsSpy = vi.hoisted(() => ({
	writeFileSync: undefined as
		| ((
				...args: Parameters<typeof import("node:fs").writeFileSync>
		  ) => ReturnType<typeof import("node:fs").writeFileSync>)
		| undefined,
	renameSync: undefined as
		| ((...args: Parameters<typeof import("node:fs").renameSync>) => ReturnType<typeof import("node:fs").renameSync>)
		| undefined,
	renamedDestinations: [] as string[],
}));

vi.mock("node:fs", async (importOriginal) => {
	const actual = await importOriginal<typeof import("node:fs")>();
	return {
		...actual,
		writeFileSync: (...args: Parameters<typeof actual.writeFileSync>) => {
			if (fsSpy.writeFileSync) return fsSpy.writeFileSync(...args);
			return actual.writeFileSync(...args);
		},
		renameSync: (...args: Parameters<typeof actual.renameSync>) => {
			fsSpy.renamedDestinations.push(String(args[1]));
			if (fsSpy.renameSync) return fsSpy.renameSync(...args);
			return actual.renameSync(...args);
		},
	};
});

const BM25_SOURCE_PATH = fileURLToPath(new URL("../../src/retrieval/methods/bm25.ts", import.meta.url));
const TEXT_SOURCE_PATH = fileURLToPath(new URL("../../src/parser/text.ts", import.meta.url));

/** Every file the guard digests; `semanticDigest` sorts by path, so the order is deterministic. */
const GUARDED_SOURCE_PATHS = [BM25_SOURCE_PATH, TEXT_SOURCE_PATH] as const;

const REGION_START_MARKER = "// INDEX_SEMANTIC_REGION_START";
const REGION_END_MARKER = "// INDEX_SEMANTIC_REGION_END";

/**
 * Artifact-producing declarations that must live inside a marked region, keyed by the file that
 * owns them. Each entry is a declaration-only string so call sites elsewhere in the file cannot
 * satisfy the check.
 */
const ARTIFACT_SYMBOLS: Readonly<Record<string, readonly string[]>> = {
	[BM25_SOURCE_PATH]: [
		"type IndexedChunk = {",
		"type FallbackIndex = {",
		"type IndexFingerprint = {",
		"private async writeTantivyIndex(",
		"private writeFallbackIndex(",
		"private readFallbackIndex(",
		"private computeFingerprint(",
		"private reuseExistingIndex(",
		"private readFingerprint(",
		"private writeFingerprint(",
		"private fallbackArtifactBytes(",
		"function loadChunks(",
		"function chunkMarkdown(",
		"function hash(",
	],
	[TEXT_SOURCE_PATH]: [
		"const WINDOWS_949_LABELS =",
		"function decodeText(",
		"function normalizeMarkdown(",
		"function replacementCount(",
		"function isPureAscii(",
	],
};

/** Query-time declarations that must stay OUTSIDE the marked regions. */
const QUERY_TIME_SYMBOLS = ["function tokenize(", "function bm25Score("] as const;

/**
 * Version and digest recorded when this test last passed.
 *
 * The digest covers every marked INDEX_SEMANTIC_REGION in the guarded sources, concatenated in
 * path-sorted then document order after normalizing CRLF/CR to LF; INDEX_SEMANTICS_VERSION is what
 * the runtime fingerprint hashes. A change to either must be a deliberate act - that is the point,
 * the guard exists to force the decision, not to make it.
 */
const RECORDED_SEMANTICS_VERSION = 2;
const RECORDED_SEMANTIC_DIGEST = "19bfe457904aff62876c66fdd7a35ac739fcd4ce14893a713b976958cc1aed6b";

type GuardedSource = { readonly path: string; readonly source: string };

function readGuardedSources(): GuardedSource[] {
	return [...GUARDED_SOURCE_PATHS]
		.sort((a, b) => a.localeCompare(b))
		.map((path) => ({ path, source: readFileSync(path, "utf8") }));
}

/**
 * All marked regions in one source file, in document order, with line endings normalized to LF.
 * The markers are located with plain string indexes; the guard throws when a marker is missing,
 * when start/end counts differ, when an end precedes its start, when regions overlap, or when a
 * region has no content between its markers - so removing or scrambling a marker can never quietly
 * disable the digest. Every guarded file must satisfy this, or the guard fails.
 */
function semanticRegions(source: string, sourcePath: string): string[] {
	const normalized = source.replaceAll("\r\n", "\n").replaceAll("\r", "\n");
	const starts: number[] = [];
	let from = 0;
	while (from < normalized.length) {
		const at = normalized.indexOf(REGION_START_MARKER, from);
		if (at === -1) break;
		starts.push(at);
		from = at + REGION_START_MARKER.length;
	}
	const ends: number[] = [];
	from = 0;
	while (from < normalized.length) {
		const at = normalized.indexOf(REGION_END_MARKER, from);
		if (at === -1) break;
		ends.push(at);
		from = at + REGION_END_MARKER.length;
	}
	if (starts.length === 0) {
		throw new Error(
			`Semantic guard: the start marker \`${REGION_START_MARKER}\` is missing from ${sourcePath}. ` +
				"Restore it above the artifact-producing code; a guard that silently passes after its marker " +
				"is deleted is worse than one that fails.",
		);
	}
	if (ends.length === 0) {
		throw new Error(
			`Semantic guard: the end marker \`${REGION_END_MARKER}\` is missing from ${sourcePath}. ` +
				"Restore it after the artifact-producing code; a guard that silently passes after its marker " +
				"is deleted is worse than one that fails.",
		);
	}
	if (starts.length !== ends.length) {
		throw new Error(
			`Semantic guard: expected equal marker counts in ${sourcePath}, found ${starts.length} ` +
				`start markers and ${ends.length} end markers. Every \`${REGION_START_MARKER}\` needs a ` +
				`matching \`${REGION_END_MARKER}\` after it.`,
		);
	}
	const regions: string[] = [];
	let previousEnd = -1;
	for (let i = 0; i < starts.length; i += 1) {
		const start = starts[i] as number;
		const end = ends[i] as number;
		if (end < start) {
			throw new Error(
				`Semantic guard: \`${REGION_END_MARKER}\` (marker pair ${i + 1}) precedes its ` +
					`\`${REGION_START_MARKER}\` in ${sourcePath}. Each region must span its code.`,
			);
		}
		if (start < previousEnd) {
			throw new Error(
				`Semantic guard: marked region ${i + 1} in ${sourcePath} starts inside the previous ` +
					"region. Regions must be sequential, each end marker followed by the next start marker.",
			);
		}
		const region = normalized.slice(start, end + REGION_END_MARKER.length);
		const body = region.slice(REGION_START_MARKER.length, region.length - REGION_END_MARKER.length);
		if (body.trim().length === 0) {
			throw new Error(
				`Semantic guard: marked region ${i + 1} in ${sourcePath} has no content between its ` +
					"markers; it must contain the artifact-producing code.",
			);
		}
		regions.push(region);
		previousEnd = end;
	}
	return regions;
}

function countOccurrences(value: string, needle: string): number {
	let count = 0;
	let from = 0;
	while (from < value.length) {
		const at = value.indexOf(needle, from);
		if (at === -1) break;
		count += 1;
		from = at + needle.length;
	}
	return count;
}

/**
 * Digest of every marked region across the guarded sources. Sources are sorted by path so the
 * result is deterministic regardless of the caller's order; each file is validated independently
 * (markers present and paired, artifact symbols inside regions, query-time symbols outside), and
 * any violation in any file throws.
 */
function semanticDigest(sources: readonly GuardedSource[]): string {
	const ordered = [...sources].sort((a, b) => a.path.localeCompare(b.path));
	const combined: string[] = [];
	for (const { path, source } of ordered) {
		const regions = semanticRegions(source, path);
		const regionText = regions.join("\n");
		for (const symbol of ARTIFACT_SYMBOLS[path] ?? []) {
			if (!regionText.includes(symbol)) {
				throw new Error(
					`Semantic guard: \`${symbol}\` is missing from the marked regions in ${path}. ` +
						"Move the declaration between the region markers, or update ARTIFACT_SYMBOLS in this " +
						"test and bump INDEX_SEMANTICS_VERSION.",
				);
			}
			if (countOccurrences(regionText, symbol) !== countOccurrences(source, symbol)) {
				throw new Error(
					`Semantic guard: \`${symbol}\` appears outside the marked regions in ${path}. ` +
						"Move the declaration inside a region; a copy left outside would silently change the " +
						"artifact without changing the digest.",
				);
			}
		}
		for (const symbol of QUERY_TIME_SYMBOLS) {
			if (regionText.includes(symbol)) {
				throw new Error(
					`Semantic guard: \`${symbol}\` is inside a marked region in ${path}. ` +
						"Query-time code applies to existing artifacts without a rebuild; guarding it would " +
						"force a global rebuild for every tokenizer or scoring change. Keep it outside the " +
						"markers.",
				);
			}
		}
		combined.push(regionText);
	}
	return sha256Hex(combined.join("\n"));
}

it("fails until a semantic change is acknowledged with a version bump", () => {
	const digest = semanticDigest(readGuardedSources());

	if (digest !== RECORDED_SEMANTIC_DIGEST) {
		if (INDEX_SEMANTICS_VERSION === RECORDED_SEMANTICS_VERSION) {
			throw new Error(
				"Artifact-producing code inside an INDEX_SEMANTIC_REGION changed without a version bump. " +
					"If the change alters the stored artifact (chunking, schema, serialization), bump " +
					"INDEX_SEMANTICS_VERSION in src/retrieval/methods/bm25.ts and set " +
					`RECORDED_SEMANTIC_DIGEST in this test to ${digest}. ` +
					`If it was a pure refactor, set RECORDED_SEMANTIC_DIGEST to ${digest} only.`,
			);
		}
		throw new Error(
			"Artifact-producing code changed and INDEX_SEMANTICS_VERSION was bumped. Set " +
				`RECORDED_SEMANTIC_DIGEST in this test to ${digest}; keep the bump only if the change ` +
				"really alters the stored artifact.",
		);
	}
	if (INDEX_SEMANTICS_VERSION !== RECORDED_SEMANTICS_VERSION) {
		throw new Error(
			"INDEX_SEMANTICS_VERSION changed although no artifact-producing code changed. Revert the bump, " +
				`or set RECORDED_SEMANTICS_VERSION in this test to ${INDEX_SEMANTICS_VERSION} if it was ` +
				"deliberate.",
		);
	}
});

describe("semantic region validation", () => {
	const source = readFileSync(BM25_SOURCE_PATH, "utf8");
	const allSymbols = [
		"type IndexedChunk = {",
		"type FallbackIndex = {",
		"type IndexFingerprint = {",
		"private async writeTantivyIndex() {}",
		"private writeFallbackIndex() {}",
		"private readFallbackIndex() {}",
		"private computeFingerprint() {}",
		"private reuseExistingIndex() {}",
		"private readFingerprint() {}",
		"private writeFingerprint() {}",
		"private fallbackArtifactBytes() {}",
		"function loadChunks() {}",
		"function chunkMarkdown() {}",
		"function hash() {}",
	].join("\n");

	it("throws when the start marker is missing", () => {
		const mutated = source.replaceAll(REGION_START_MARKER, "// SEMANTIC_REGION_START_REMOVED");
		expect(() => semanticRegions(mutated, BM25_SOURCE_PATH)).toThrow(/start marker/);
	});

	it("throws when the end marker is missing", () => {
		const mutated = source.replaceAll(REGION_END_MARKER, "// SEMANTIC_REGION_END_REMOVED");
		expect(() => semanticRegions(mutated, BM25_SOURCE_PATH)).toThrow(/end marker/);
	});

	it("throws when start and end marker counts differ", () => {
		const mutated = source.replace(REGION_START_MARKER, `${REGION_START_MARKER}\n${REGION_START_MARKER}`);
		expect(() => semanticRegions(mutated, BM25_SOURCE_PATH)).toThrow(/expected equal marker counts/);
	});

	it("throws when an end marker precedes its start marker", () => {
		const mutated = [
			REGION_END_MARKER,
			REGION_START_MARKER,
			"function loadChunks() {}",
			REGION_START_MARKER,
			"function chunkMarkdown() {}",
			REGION_END_MARKER,
		].join("\n");
		expect(() => semanticRegions(mutated, "synthetic.ts")).toThrow(/precedes its/);
	});

	it("throws when regions overlap", () => {
		const mutated = [
			REGION_START_MARKER,
			"function loadChunks() {}",
			REGION_START_MARKER,
			"function chunkMarkdown() {}",
			REGION_END_MARKER,
			REGION_END_MARKER,
		].join("\n");
		expect(() => semanticRegions(mutated, "synthetic.ts")).toThrow(/starts inside the previous/);
	});

	it("throws when a region has no content between its markers", () => {
		const mutated = [
			REGION_START_MARKER,
			REGION_END_MARKER,
			REGION_START_MARKER,
			"function loadChunks() {}",
			REGION_END_MARKER,
		].join("\n");
		expect(() => semanticRegions(mutated, "synthetic.ts")).toThrow(/no content between its markers/);
	});

	it("throws when an artifact symbol sits outside the regions", () => {
		const inside = allSymbols.replace("function loadChunks() {}", "");
		const mutated = [REGION_START_MARKER, inside, REGION_END_MARKER, "function loadChunks() {}"].join("\n");
		expect(() => semanticDigest([{ path: BM25_SOURCE_PATH, source: mutated }])).toThrow(
			/missing from the marked regions/,
		);
	});

	it("throws when an artifact symbol is duplicated outside the regions", () => {
		const mutated = [REGION_START_MARKER, allSymbols, REGION_END_MARKER, "function loadChunks() {}"].join("\n");
		expect(() => semanticDigest([{ path: BM25_SOURCE_PATH, source: mutated }])).toThrow(
			/appears outside the marked regions/,
		);
	});

	it("throws when a query-time function moves inside a region", () => {
		const mutated = source.replace(
			REGION_START_MARKER,
			`${REGION_START_MARKER}\nfunction tokenize(value: string): string[] { return []; }`,
		);
		expect(() => semanticDigest([{ path: BM25_SOURCE_PATH, source: mutated }])).toThrow(/is inside a marked region/);
	});

	it("digests CRLF and CR line endings identically", () => {
		const original = semanticDigest([{ path: BM25_SOURCE_PATH, source }]);
		expect(semanticDigest([{ path: BM25_SOURCE_PATH, source: source.replaceAll("\n", "\r\n") }])).toBe(original);
		expect(semanticDigest([{ path: BM25_SOURCE_PATH, source: source.replaceAll("\n", "\r") }])).toBe(original);
	});

	it("throws when text.ts loses its start marker", () => {
		const textSource = readFileSync(TEXT_SOURCE_PATH, "utf8");
		const mutated = textSource.replaceAll(REGION_START_MARKER, "// SEMANTIC_REGION_START_REMOVED");
		expect(() => semanticRegions(mutated, TEXT_SOURCE_PATH)).toThrow(/start marker/);
	});

	it("throws when text.ts has unmatched region markers", () => {
		const textSource = readFileSync(TEXT_SOURCE_PATH, "utf8");
		const mutated = textSource.replace(REGION_END_MARKER, `${REGION_END_MARKER}\n${REGION_END_MARKER}`);
		expect(() => semanticRegions(mutated, TEXT_SOURCE_PATH)).toThrow(/expected equal marker counts/);
	});

	it("throws when any guarded file loses its markers entirely", () => {
		const stripped = readGuardedSources().map(({ path, source }) =>
			path === TEXT_SOURCE_PATH
				? { path, source: source.replaceAll(REGION_START_MARKER, "").replaceAll(REGION_END_MARKER, "") }
				: { path, source },
		);
		expect(() => semanticDigest(stripped)).toThrow(/start marker/);
	});

	it("digests the guarded files in deterministic path order", () => {
		const sources = readGuardedSources();
		expect(semanticDigest(sources)).toBe(semanticDigest([...sources].reverse()));
	});
});

/**
 * Each of these replays a reproduced attack on the guard. The attack code is applied to a copy of
 * the source and must change the guarded digest - i.e. the real guard run would fail - or the
 * guard is not covering the code the attacker changed. The reverse property (query-time changes
 * must NOT change the digest) is pinned last, because a change there must never force a rebuild.
 */
describe("semantic guard coverage", () => {
	function digestWith(bm25Source: string, textSource: string): string {
		return semanticDigest([
			{ path: BM25_SOURCE_PATH, source: bm25Source },
			{ path: TEXT_SOURCE_PATH, source: textSource },
		]);
	}

	it("fails when the chunk-id hash truncation is changed", () => {
		const original = readFileSync(BM25_SOURCE_PATH, "utf8");
		const textSource = readFileSync(TEXT_SOURCE_PATH, "utf8");
		const mutated = original.replace(".slice(0, 12)", ".slice(0, 8)");
		expect(mutated).not.toBe(original);
		expect(digestWith(mutated, textSource)).not.toBe(digestWith(original, textSource));
	});

	it("fails when normalizeMarkdown normalization is changed", () => {
		const bm25Source = readFileSync(BM25_SOURCE_PATH, "utf8");
		const original = readFileSync(TEXT_SOURCE_PATH, "utf8");
		const mutated = original.replace('markdown.normalize("NFC")', 'markdown.normalize("NFD")');
		expect(mutated).not.toBe(original);
		expect(digestWith(bm25Source, mutated)).not.toBe(digestWith(bm25Source, original));
	});

	it("fails when the fingerprint material drops content digests", () => {
		const original = readFileSync(BM25_SOURCE_PATH, "utf8");
		const textSource = readFileSync(TEXT_SOURCE_PATH, "utf8");
		// This is source text being mutated, not an interpolation: the string has to match the
		// literal characters in bm25.ts for the mutation to land.
		// biome-ignore lint/suspicious/noTemplateCurlyInString: matching literal source text
		const mutated = original.replace('${entry.contentSha256 ?? "-"}\\u0000', "");
		expect(mutated).not.toBe(original);
		expect(digestWith(mutated, textSource)).not.toBe(digestWith(original, textSource));
	});
	it("fails when the fingerprint equality check in reuseExistingIndex is removed", () => {
		const original = readFileSync(BM25_SOURCE_PATH, "utf8");
		const textSource = readFileSync(TEXT_SOURCE_PATH, "utf8");
		const mutated = original.replace(
			"if (!stored || stored.fingerprint !== fingerprint) return undefined;",
			"if (!stored) return undefined;",
		);
		expect(mutated).not.toBe(original);
		expect(digestWith(mutated, textSource)).not.toBe(digestWith(original, textSource));
	});

	it("fails when reuseExistingIndex always trusts a stored fingerprint", () => {
		const original = readFileSync(BM25_SOURCE_PATH, "utf8");
		const textSource = readFileSync(TEXT_SOURCE_PATH, "utf8");
		const mutated = original.replace("stored.fingerprint !== fingerprint", "false");
		expect(mutated).not.toBe(original);
		expect(digestWith(mutated, textSource)).not.toBe(digestWith(original, textSource));
	});

	it("fails when the fallback artifact size comparison is removed", () => {
		const original = readFileSync(BM25_SOURCE_PATH, "utf8");
		const textSource = readFileSync(TEXT_SOURCE_PATH, "utf8");
		const mutated = original.replace("this.fallbackArtifactBytes() === stored.artifactBytes", "true");
		expect(mutated).not.toBe(original);
		expect(digestWith(mutated, textSource)).not.toBe(digestWith(original, textSource));
	});

	it("passes when only the query-time tokenizer regex is changed", () => {
		const original = readFileSync(BM25_SOURCE_PATH, "utf8");
		const textSource = readFileSync(TEXT_SOURCE_PATH, "utf8");
		const mutated = original.replaceAll("[\\p{Letter}\\p{Number}_]+", "[\\p{Letter}\\p{Number}]+");
		expect(mutated).not.toBe(original);
		expect(digestWith(mutated, textSource)).toBe(digestWith(original, textSource));
	});

	it("fails when a WINDOWS_949_LABELS entry is removed", () => {
		const original = readFileSync(TEXT_SOURCE_PATH, "utf8");
		const bm25Source = readFileSync(BM25_SOURCE_PATH, "utf8");
		const mutated = original.replace('"EUC-KR", "ISO-2022-KR"', '"EUC-KR"');
		expect(mutated).not.toBe(original);
		expect(digestWith(bm25Source, mutated)).not.toBe(digestWith(bm25Source, original));
	});

	it("fails when replacementCount is changed", () => {
		const original = readFileSync(TEXT_SOURCE_PATH, "utf8");
		const bm25Source = readFileSync(BM25_SOURCE_PATH, "utf8");
		const mutated = original.replace('character === "\\uFFFD"', 'character !== "\\uFFFD"');
		expect(mutated).not.toBe(original);
		expect(digestWith(bm25Source, mutated)).not.toBe(digestWith(bm25Source, original));
	});

	it("fails when the decodeText fast-path predicate is loosened", () => {
		const original = readFileSync(TEXT_SOURCE_PATH, "utf8");
		const bm25Source = readFileSync(BM25_SOURCE_PATH, "utf8");
		// Dropping the ESC exclusion would let chardet's ISO-2022-JP/CN detectors claim the buffer.
		const mutated = original.replace("byte === 0x1b", "byte === 0x1c");
		expect(mutated).not.toBe(original);
		expect(digestWith(bm25Source, mutated)).not.toBe(digestWith(bm25Source, original));
	});
});

const roots: string[] = [];

function makeWorkspace(documentCount: number): { root: string; searchPaths: string[] } {
	const root = mkdtempSync(join(tmpdir(), "autorag-semantics-"));
	roots.push(root);
	const docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	for (let i = 0; i < documentCount; i += 1) {
		writeFileSync(join(docs, `doc-${i}.md`), `# Document ${i}\n\nRefund approval policy and escalation threshold.\n`);
	}
	return { root, searchPaths: [docs] };
}

function makeAgent(root: string, searchPaths: string[]): AutoRAGAgent {
	return new AutoRAGAgent({
		// `workspacePath` is what the agent uses as its project root; passing anything else silently
		// falls back to process.cwd() and writes indexes into the current repository.
		workspacePath: root,
		searchPaths,
		bm25: { forceEngine: "typescript-fallback" },
		minSync: false,
	});
}

afterEach(() => {
	for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

function sha256Hex(value: string | Buffer): string {
	return createHash("sha256").update(value).digest("hex");
}

/** Numeric value of a top-level `const NAME = value;` declaration, read from disk. */
function moduleConstant(source: string, name: string): number {
	const pattern = new RegExp(`^const\\s+${name}\\s*=\\s*([^;]+);`, "m");
	const match = pattern.exec(source);
	if (!match) throw new Error(`Module constant \`${name}\` was not found in ${BM25_SOURCE_PATH}.`);
	const value = Number(match[1].replaceAll("_", ""));
	if (!Number.isFinite(value)) throw new Error(`Module constant \`${name}\` in ${BM25_SOURCE_PATH} is not a number.`);
	return value;
}

/**
 * Replicate `BM25Method.computeFingerprint` material from the mirror index and the constants read
 * off disk. The guard test below asserts the replication against the real sidecar, so a change to
 * the material format fails loudly instead of silently weakening the tamper test.
 */
function fingerprintMaterial(root: string, source: string): string {
	const index = loadMirrorIndex(root);
	const entries = Object.values(index.entries)
		.map((entry) => `${entry.virtualPath}\u0000${entry.contentSha256 ?? "-"}\u0000${entry.parserName}`)
		.sort();
	return [
		`semantics:${INDEX_SEMANTICS_VERSION}`,
		`chunk:${moduleConstant(source, "MAX_CHUNK_CHARS")}`,
		"engine:typescript-fallback",
		"fallback:typescript",
		`entries:${entries.length}`,
		...entries,
	].join("\n");
}

type Sidecar = {
	readonly version: 1;
	readonly fingerprint: string;
	readonly engine: string;
	readonly indexedChunks: number;
	readonly artifactBytes?: number;
};

function sidecarPath(root: string): string {
	return join(root, ".autorag", "bm25", "index-fingerprint.json");
}

function readSidecar(root: string): Sidecar {
	return JSON.parse(readFileSync(sidecarPath(root), "utf8")) as Sidecar;
}

function writeSidecar(root: string, sidecar: Sidecar): void {
	writeFileSync(sidecarPath(root), `${JSON.stringify(sidecar)}\n`);
}

describe("BM25 index semantics guard", () => {
	it("rebuilds when the stored fingerprint was built with different chunking parameters", async () => {
		const { root, searchPaths } = makeWorkspace(5);
		const agent = makeAgent(root, searchPaths);
		const first = await agent.refresh(true);
		expect(first.outcome).toBe("completed");
		expect(first.bm25?.skipped).toBeUndefined();
		expect(first.bm25?.indexedChunks).toBeGreaterThan(0);

		const source = readFileSync(BM25_SOURCE_PATH, "utf8");
		const material = fingerprintMaterial(root, source);
		const stored = readSidecar(root);
		// The replication must describe the real fingerprint, or the tamper below proves nothing.
		expect(stored.fingerprint).toBe(sha256Hex(material));

		// Forge the sidecar a developer would leave behind after tuning MAX_CHUNK_CHARS without a
		// rebuild: same corpus, same entries, different chunking material.
		const tampered = material.replace(/^chunk:.*$/m, "chunk:1200");
		expect(tampered).not.toBe(material);
		writeSidecar(root, { ...stored, fingerprint: sha256Hex(tampered) });

		const second = await agent.refresh(false);
		expect(second.outcome).toBe("completed");
		expect(second.bm25?.skipped).toBeUndefined();
		expect(second.bm25?.indexedChunks).toBe(first.bm25?.indexedChunks);
		expect(readSidecar(root).fingerprint).toBe(sha256Hex(material));
	});

	it("still skips the rebuild when nothing changed", async () => {
		const { root, searchPaths } = makeWorkspace(6);
		const agent = makeAgent(root, searchPaths);
		const first = await agent.refresh(true);
		expect(first.outcome).toBe("completed");
		expect(first.bm25?.skipped).toBeUndefined();
		expect(first.bm25?.indexedChunks).toBeGreaterThan(0);

		const artifactPath = join(root, ".autorag", "bm25", "fallback-index.json");
		const digestBefore = sha256Hex(readFileSync(artifactPath));
		const mtimeBefore = statSync(artifactPath).mtimeMs;

		const second = await agent.refresh(false);
		expect(second.outcome).toBe("completed");
		expect(second.bm25?.skipped).toBe(true);
		expect(second.bm25?.indexedChunks).toBe(first.bm25?.indexedChunks);
		expect(sha256Hex(readFileSync(artifactPath))).toBe(digestBefore);
		expect(statSync(artifactPath).mtimeMs).toBe(mtimeBefore);
	});

	it("applies a query-time tokenizer change to the stored artifact without a rebuild", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-semantics-"));
		roots.push(root);
		const docs = join(root, "docs");
		mkdirSync(docs, { recursive: true });
		writeFileSync(join(docs, "tokenizer-probe.md"), "alpha_beta gamma\n");
		const searchPaths = [docs];

		const agent = makeAgent(root, searchPaths);
		const first = await agent.refresh(true);
		expect(first.outcome).toBe("completed");
		expect(first.bm25?.skipped).toBeUndefined();
		expect(first.bm25?.indexedChunks).toBe(1);

		const artifactPath = join(root, ".autorag", "bm25", "fallback-index.json");
		const artifactBefore = readFileSync(artifactPath);
		const mtimeBeforeMs = statSync(artifactPath).mtimeMs;
		const fingerprintBefore = readSidecar(root).fingerprint;

		// The stock tokenizer keeps `_`, so `alpha_beta` is one token and `alpha` matches nothing.
		expect(await agent.retrieve("alpha", { topK: 10 })).toEqual([]);

		// A second module graph with different tokenizer semantics is needed. The real bm25.ts
		// cannot be mutated in place: vitest run-mode caches transforms by module id, so a changed
		// file would still be served stale after vi.resetModules(). Copy src/ instead and swap the
		// tokenizer regex in the copy; it resolves the same node_modules and behaves identically
		// except for the tokenizer.
		const originalSource = readFileSync(BM25_SOURCE_PATH, "utf8");
		const mutatedSource = originalSource.replaceAll("[\\p{Letter}\\p{Number}_]+", "[\\p{Letter}\\p{Number}]+");
		expect(mutatedSource).not.toBe(originalSource);
		const repoRoot = fileURLToPath(new URL("../..", import.meta.url));
		const mutationRoot = mkdtempSync(join(repoRoot, ".semantics-mutation-"));
		roots.push(mutationRoot);
		cpSync(join(repoRoot, "src"), join(mutationRoot, "src"), { recursive: true });
		writeFileSync(join(mutationRoot, "src", "retrieval", "methods", "bm25.ts"), mutatedSource);
		try {
			vi.resetModules();
			const { AutoRAGAgent: MutatedAgent } = await import(
				pathToFileURL(join(mutationRoot, "src", "agent", "agent.ts")).href
			);
			const mutated = new MutatedAgent({
				workspacePath: root,
				searchPaths,
				bm25: { forceEngine: "typescript-fallback" },
				minSync: false,
			});
			const second = await mutated.refresh(false);
			expect(second.outcome).toBe("completed");
			// Tokenization is query-time, so the fingerprint is unchanged and the artifact is reused.
			expect(second.bm25?.skipped).toBe(true);
			expect(readSidecar(root).fingerprint).toBe(fingerprintBefore);
			expect(statSync(artifactPath).mtimeMs).toBe(mtimeBeforeMs);
			expect(readFileSync(artifactPath)).toEqual(artifactBefore);
			// Dropping `_` splits `alpha_beta` into `alpha` + `beta`, so the same stored artifact now
			// answers `alpha` without any rebuild.
			const results = await mutated.retrieve("alpha", { topK: 10 });
			expect(results.length).toBeGreaterThan(0);
			expect(results.some((result: RetrievalResult) => result.content.includes("alpha_beta"))).toBe(true);
		} finally {
			rmSync(mutationRoot, { recursive: true, force: true });
		}
	});
});
/**
 * `sync()` must commit the chosen artifact and record its fingerprint exactly once. A fingerprint
 * that fails to record is a commit failure, not an engine failure: it must surface loudly and must
 * never trigger the engine fallback path, which would rewrite the artifact (and for a tantivy
 * success would write a fallback artifact that was never chosen).
 */
describe("BM25 sync commit-once behavior", () => {
	function commitCount(pathSuffix: string): number {
		return fsSpy.renamedDestinations.filter((destination) => destination.endsWith(pathSuffix)).length;
	}

	it("commits the fallback artifact exactly once when the fingerprint write fails", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-semantics-"));
		roots.push(root);
		const docs = join(root, "docs");
		mkdirSync(docs, { recursive: true });
		writeFileSync(join(docs, "doc-0.md"), "# Document 0\n\nRefund approval policy and escalation threshold.\n");
		await syncParsedMirrors({ root, searchPaths: [docs] });
		const method = new BM25Method({ root, forceEngine: "typescript-fallback" });
		await method.sync();
		expect(method.getStatus().engine).toBe("typescript-fallback");

		// Sabotage the fingerprint commit: renaming over a directory fails, so writeFingerprint
		// throws after the fallback artifact was already committed.
		const fingerprintPath = join(root, ".autorag", "bm25", "index-fingerprint.json");
		rmSync(fingerprintPath);
		mkdirSync(fingerprintPath);
		writeFileSync(join(docs, "doc-0.md"), "# Changed\n\nEntirely different chargeback content.\n");
		await syncParsedMirrors({ root, searchPaths: [docs] });

		fsSpy.renamedDestinations.length = 0;
		await expect(method.sync()).rejects.toThrow();
		// The old code mistook the fingerprint failure for an engine failure and committed the
		// artifact a second time; the new code commits exactly once and surfaces the failure.
		expect(commitCount("fallback-index.json")).toBe(1);
		// The sabotage was cleared by the fingerprint write cleanup, so the next sync retries.
		expect(existsSync(fingerprintPath)).toBe(false);
		const retried = await method.sync();
		expect(retried.skipped).toBeUndefined();
		expect((await method.retrieve("chargeback", { topK: 5 })).length).toBeGreaterThan(0);
		expect((await method.sync()).skipped).toBe(true);
	});

	it("does not write a fallback artifact when a tantivy build succeeded but its fingerprint commit failed", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-semantics-"));
		roots.push(root);
		const docs = join(root, "docs");
		mkdirSync(docs, { recursive: true });
		writeFileSync(join(docs, "doc-0.md"), "# Document 0\n\nRefund approval policy and escalation threshold.\n");
		await syncParsedMirrors({ root, searchPaths: [docs] });
		const method = new BM25Method({ root, forceEngine: "tantivy" });
		await method.sync();
		expect(method.getStatus().engine).toBe("tantivy");

		const fingerprintPath = join(root, ".autorag", "bm25", "index-fingerprint.json");
		rmSync(fingerprintPath);
		mkdirSync(fingerprintPath);
		writeFileSync(join(docs, "doc-0.md"), "# Changed\n\nEntirely different chargeback content.\n");
		await syncParsedMirrors({ root, searchPaths: [docs] });

		fsSpy.renamedDestinations.length = 0;
		await expect(method.sync()).rejects.toThrow();
		// The tantivy artifact is the chosen one, so the engine fallback path must not be entered:
		// no fallback artifact may be committed and no rename may target it.
		expect(commitCount("fallback-index.json")).toBe(0);
		expect(existsSync(join(root, ".autorag", "bm25", "fallback-index.json"))).toBe(false);
	});
});

import {
	existsSync,
	mkdirSync,
	mkdtempSync,
	readFileSync,
	realpathSync,
	rmSync,
	symlinkSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
	assertAbsoluteReadableSource,
	assertKoreanLexicalHit,
	assertLiteRetrieveHealthy,
	assertLiteRetrieveReportsSkippedSurface,
	assertLiteRetrieveWindowExcludes,
	assertServiceReady,
	buildLiveStackOptions,
	cleanupCloneState,
	isFingerprintCurrent,
	tryAcquireWorkflowLock,
} from "../../scripts/live-e2e/workflow.mjs";

const roots: string[] = [];

afterEach(() => {
	for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

function tempRoot(): string {
	const root = mkdtempSync(join(tmpdir(), "autorag-live-workflow-test-"));
	roots.push(root);
	return root;
}

describe("live-e2e core workflow invariants", () => {
	it("refuses stale fingerprints rather than treating warm state as reusable", () => {
		expect(
			isFingerprintCurrent(
				{ corpusDigest: "old", embeddingModel: "Qwen/Qwen3-Embedding-0.6B" },
				{ corpusDigest: "new", embeddingModel: "Qwen/Qwen3-Embedding-0.6B" },
			),
		).toBe(false);
	});

	it("refuses an unavailable embedding service with a stable diagnostic", () => {
		expect(() => assertServiceReady({ verdict: "refused", code: "live-e2e-embedding-unavailable" })).toThrow(
			"live-e2e-embedding-unavailable",
		);
	});

	it("requires a healthy lite retrieve run to state that nothing was skipped", () => {
		const envelope = { ok: true, results: [{ source: "/corpus/sample.txt" }], unsearched: [] };
		expect(assertLiteRetrieveHealthy(envelope, "/corpus/sample.txt")).toBe(true);
		// A pre-#1636 envelope has no `unsearched` field at all; that must fail loudly.
		const legacy = { ok: true, results: envelope.results } as unknown as typeof envelope;
		expect(() => assertLiteRetrieveHealthy(legacy, "/corpus/sample.txt")).toThrow("lite-retrieve-unsearched-missing");
		expect(() =>
			assertLiteRetrieveHealthy(
				{ ...envelope, unsearched: [{ surface: "minsync", methods: ["minsync"], reason: "down" }] },
				"/corpus/sample.txt",
			),
		).toThrow("lite-retrieve-unexpected-unsearched");
	});

	it("requires a skipped surface to carry the real error, not a placeholder", () => {
		const degraded = {
			ok: true,
			results: [],
			unsearched: [
				{
					surface: "minsync",
					methods: ["hybrid", "minsync"],
					reason: "MinSyncQueryError: connect ECONNREFUSED 127.0.0.1:9",
				},
			],
		};
		expect(assertLiteRetrieveReportsSkippedSurface(degraded, "minsync").reason).toContain("ECONNREFUSED");
		expect(() => assertLiteRetrieveReportsSkippedSurface(degraded, "discord")).toThrow(
			"lite-retrieve-surface-not-reported:discord",
		);
		const suppressed = {
			...degraded,
			unsearched: [{ ...degraded.unsearched[0], reason: "path details suppressed" }],
		};
		expect(() => assertLiteRetrieveReportsSkippedSurface(suppressed, "minsync")).toThrow(
			"lite-retrieve-reason-suppressed",
		);
		const empty = { ...degraded, unsearched: [{ ...degraded.unsearched[0], reason: "  " }] };
		expect(() => assertLiteRetrieveReportsSkippedSurface(empty, "minsync")).toThrow("lite-retrieve-reason-missing");
	});

	it("requires a time window to exclude the document that would otherwise rank first, and to say so (#1958)", () => {
		const unwindowed = {
			ok: true,
			unsearched: [],
			results: [{ source: "/c/july.md" }, { source: "/c/august.md" }],
		};
		const windowed = {
			ok: true,
			unsearched: [],
			results: [{ source: "/c/august.md" }],
			window: {
				since: "2026-08-01T00:00:00.000Z",
				examined: 2,
				matched: 1,
				untimed: 0,
				ceilingHit: false,
				fetchDepth: 50,
			},
		};
		expect(assertLiteRetrieveWindowExcludes(unwindowed, windowed, "/c/july.md", "/c/august.md")).toBe(true);
		// The excluded document must really have ranked first without the window, or the proof proves nothing.
		expect(() =>
			assertLiteRetrieveWindowExcludes(
				{ ...unwindowed, results: [{ source: "/c/august.md" }, { source: "/c/july.md" }] },
				windowed,
				"/c/july.md",
				"/c/august.md",
			),
		).toThrow("lite-retrieve-window-excluded-not-first");
		expect(() =>
			assertLiteRetrieveWindowExcludes(
				unwindowed,
				{ ...windowed, results: [{ source: "/c/july.md" }] },
				"/c/july.md",
				"/c/august.md",
			),
		).toThrow("lite-retrieve-window-leaked");
		expect(() =>
			assertLiteRetrieveWindowExcludes(unwindowed, { ...windowed, results: [] }, "/c/july.md", "/c/august.md"),
		).toThrow("lite-retrieve-window-kept-missing");
		const { window: _window, ...noReport } = windowed;
		expect(() => assertLiteRetrieveWindowExcludes(unwindowed, noReport, "/c/july.md", "/c/august.md")).toThrow(
			"lite-retrieve-window-report-missing",
		);
	});

	it("reports lock contention without allowing a second workflow", () => {
		const root = tempRoot();
		const first = tryAcquireWorkflowLock(root);
		expect(first.ok).toBe(true);
		const second = tryAcquireWorkflowLock(root);
		expect(second).toMatchObject({ ok: false, code: "live-e2e-lock-held" });
		first.release?.();
	});

	it("cleanup removes clone state but preserves the shared corpus", () => {
		const root = tempRoot();
		const shared = join(root, "corpus");
		const clone = join(root, ".autorag-e2e");
		mkdirSync(shared, { recursive: true });
		mkdirSync(clone, { recursive: true });
		writeFileSync(join(shared, "fixture.txt"), "fixture");
		writeFileSync(join(clone, "state"), "state");
		cleanupCloneState(clone, shared);
		expect(readFileSync(join(shared, "fixture.txt"), "utf8")).toBe("fixture");
		expect(() => readFileSync(join(clone, "state"))).toThrow();
	});

	it("canonicalizes readable source paths before comparing retrieval sources", () => {
		const root = tempRoot();
		const actual = join(root, "actual.txt");
		const alias = join(root, "alias.txt");
		writeFileSync(actual, "readable");
		symlinkSync(actual, alias);
		expect(assertAbsoluteReadableSource(alias)).toBe(realpathSync(actual));
	});

	it("builds the agent with clone-local workspace, shared search path, and MinSync's own embedder", () => {
		const root = tempRoot();
		const workspace = join(root, ".autorag-e2e", "workspace");
		const options = buildLiveStackOptions(root, workspace);
		expect(options.workspacePath).toBe(workspace);
		expect(options.searchPaths).toEqual([join(root, "corpus")]);
		expect(options.workspacePath).not.toBe(root);
		// The corpus root is immutable fixture state: jikji must never run against it.
		expect(options.jikji).toBe(false);
		// AutoRAG owns no embedding model: MinSync's native Qwen3 is the only local embedder, and the live
		// lane may install MinSync because the run is the one place a daemon is allowed to start.
		expect(options.minSync).toMatchObject({
			autoInstall: true,
			embedder: { id: "native:Qwen/Qwen3-Embedding-0.6B", dimension: 1024 },
		});
	});

	it("keeps mutable parsed state out of the shared corpus root", () => {
		const root = tempRoot();
		const workspace = join(root, ".autorag-e2e", "workspace");
		mkdirSync(join(workspace, ".autorag", "parsed"), { recursive: true });
		expect(existsSync(join(root, ".autorag"))).toBe(false);
		expect(existsSync(join(workspace, ".autorag", "parsed"))).toBe(true);
	});

	it("accepts only absolute existing readable source paths", () => {
		const root = tempRoot();
		const source = join(root, "source.txt");
		writeFileSync(source, "readable");
		expect(assertAbsoluteReadableSource(source)).toBe(realpathSync(source));
		expect(() => assertAbsoluteReadableSource("relative.txt")).toThrow("source-not-absolute");
	});

	it("requires a Korean stem query to reach the fixture through MinSync sparse retrieval", () => {
		const hit = JSON.stringify({
			results: [{ path: "files/corpus/korean-policy.txt.md", score: 1.2, text: "인증서를" }],
		});
		expect(assertKoreanLexicalHit(hit, "korean-policy.txt")).toBe(true);
		// The shape a retriever blind to Korean inflection produces for the stem: no hit at all.
		expect(() => assertKoreanLexicalHit(JSON.stringify({ results: [] }), "korean-policy.txt")).toThrow(
			"korean-lexical-stem-miss",
		);
		expect(() =>
			assertKoreanLexicalHit(
				JSON.stringify([{ path: "files/corpus/sample.txt.md", score: 1, text: "x" }]),
				"korean-policy.txt",
			),
		).toThrow("korean-lexical-stem-miss");
		expect(() => assertKoreanLexicalHit("not json", "korean-policy.txt")).toThrow("korean-lexical-unparseable");
	});
});

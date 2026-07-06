import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { AutoRAGRefreshResult, AutoRAGRefreshStatus } from "../../src/agent/agent.ts";
import type { SearchDocumentsResponse } from "../../src/agent/search-documents.ts";
import {
	renderError,
	renderFeedback,
	renderIndex,
	renderMemory,
	renderRefresh,
	renderSearch,
	renderStatus,
} from "../../src/cli/output.ts";
import type { MemorySchemaV4 } from "../../src/memory/memory.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-cli-output-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

const FAKE_ABSOLUTE_ROOT = "/tmp/autorag-test-fake-root-X9";

function cannedSearchResponse(): SearchDocumentsResponse {
	return {
		sessionId: "sess-123",
		query: "how to refresh",
		results: [
			{
				number: 1,
				title: "Refresh Guide",
				summary: "Run autorag refresh to rebuild the index.",
				evidence: [{ excerpt: "Run refresh to sync the parsed mirror.", lineNumber: 42 }],
				confidence: 0.87,
				feedbackId: "fb-1",
			},
		],
		answer: "Run `autorag refresh` to rebuild the index.",
		searched: 5,
		warnings: [],
		diagnostics: [
			{
				code: "bm25-degraded-fallback",
				severity: "warning",
				message: "BM25 running in degraded fallback mode.",
				source: "bm25",
			},
		],
	};
}

function cannedRefreshResult(): AutoRAGRefreshResult {
	return {
		scanned: 3,
		written: 2,
		deleted: 1,
		skipped: 0,
		indexPath: `${FAKE_ABSOLUTE_ROOT}/.autorag/parsed-mirror`,
		diagnostics: [],
		bm25: {
			indexPath: `${FAKE_ABSOLUTE_ROOT}/.autorag/bm25`,
			indexedChunks: 10,
			readiness: "ready",
			engine: "typescript-fallback",
		},
	};
}

function cannedStatus(): AutoRAGRefreshStatus {
	return {
		state: "success",
		inFlight: false,
		lastStartedAt: "2026-01-01T00:00:00.000Z",
		lastFinishedAt: "2026-01-01T00:00:01.000Z",
		counts: { scanned: 3, written: 2, deleted: 1, skipped: 0 },
		stale: false,
		diagnostics: [],
		components: { bm25: "ready", minsync: "unavailable" },
	};
}

function cannedMemory(): MemorySchemaV4 {
	return {
		version: 4,
		curatedResults: [],
		evidenceChunks: [],
		feedbackSignals: [],
		signalDefaults: {
			explicitWeight: 1,
			followupWeight: 0.25,
			retryWeight: -0.25,
			implicitCap: 0.5,
		},
		warnings: [],
		insights: [],
		pendingInsightSignals: [],
	};
}

describe("renderSearch", () => {
	it("default output contains number, title, and summary, and no temp-root path", () => {
		const out = renderSearch(cannedSearchResponse(), {});
		expect(out).toContain("1");
		expect(out).toContain("Refresh Guide");
		expect(out).toContain("Run autorag refresh to rebuild the index.");
		expect(out).toContain("Run `autorag refresh` to rebuild the index.");
		expect(out).not.toContain(root);
	});

	it("default output omits confidence, diagnostics, and sessionId", () => {
		const out = renderSearch(cannedSearchResponse(), {});
		expect(out).not.toContain("0.87");
		expect(out).not.toContain("bm25-degraded-fallback");
		expect(out).not.toContain("sess-123");
	});

	it("debug output includes confidence, evidence, diagnostics, and sessionId", () => {
		const out = renderSearch(cannedSearchResponse(), { debug: true });
		expect(out).toContain("0.87");
		expect(out).toContain("line 42");
		expect(out).toContain("Run refresh to sync the parsed mirror.");
		expect(out).toContain("bm25-degraded-fallback");
		expect(out).toContain("sess-123");
	});

	it("json output matches the documented search envelope shape", () => {
		const out = renderSearch(cannedSearchResponse(), { json: true });
		const parsed = JSON.parse(out) as Record<string, unknown>;
		expect(parsed.answer).toBe("Run `autorag refresh` to rebuild the index.");
		expect(Array.isArray(parsed.results)).toBe(true);
		const first = (parsed.results as Array<Record<string, unknown>>)[0];
		expect(first.number).toBe(1);
		expect(first.title).toBe("Refresh Guide");
		expect(first.summary).toBe("Run autorag refresh to rebuild the index.");
		expect("confidence" in first).toBe(false);
		expect("evidence" in first).toBe(false);
		expect("feedbackId" in first).toBe(false);
		expect("sessionId" in parsed).toBe(false);
	});

	it("json debug output adds confidence, evidence, diagnostics, and sessionId", () => {
		const out = renderSearch(cannedSearchResponse(), { json: true, debug: true });
		const parsed = JSON.parse(out) as Record<string, unknown>;
		expect(parsed.sessionId).toBe("sess-123");
		expect(parsed.query).toBe("how to refresh");
		expect(parsed.searched).toBe(5);
		expect(Array.isArray(parsed.diagnostics)).toBe(true);
		const first = (parsed.results as Array<Record<string, unknown>>)[0];
		expect(first.confidence).toBe(0.87);
		expect(first.feedbackId).toBe("fb-1");
		expect(Array.isArray(first.evidence)).toBe(true);
	});
});

describe("renderRefresh path opacity", () => {
	it("human output contains no indexPath, no fake absolute path, and no .autorag/bm25", () => {
		const out = renderRefresh(cannedRefreshResult(), {});
		expect(out).not.toContain("indexPath");
		expect(out).not.toContain(FAKE_ABSOLUTE_ROOT);
		expect(out).not.toContain(".autorag/bm25");
	});

	it("json output contains no indexPath, no fake absolute path, and no .autorag/bm25", () => {
		const out = renderRefresh(cannedRefreshResult(), { json: true });
		expect(out).not.toContain("indexPath");
		expect(out).not.toContain(FAKE_ABSOLUTE_ROOT);
		expect(out).not.toContain(".autorag/bm25");

		const parsed = JSON.parse(out) as Record<string, unknown>;
		expect(parsed.ok).toBe(true);
		const counts = parsed.counts as Record<string, unknown>;
		expect(counts.scanned).toBe(3);
		expect(counts.written).toBe(2);
		expect(counts.deleted).toBe(1);
		expect(counts.skipped).toBe(0);
		const bm25 = parsed.bm25 as Record<string, unknown>;
		expect(bm25.indexedChunks).toBe(10);
		expect(bm25.readiness).toBe("ready");
		expect(bm25.engine).toBe("typescript-fallback");
		expect("indexPath" in bm25).toBe(false);
	});
});

describe("renderStatus", () => {
	it("json output projects the full path-opaque status", () => {
		const out = renderStatus(cannedStatus(), { json: true });
		const parsed = JSON.parse(out) as Record<string, unknown>;
		expect(parsed.state).toBe("success");
		expect(parsed.inFlight).toBe(false);
		expect(parsed.stale).toBe(false);
		expect(parsed.counts).toEqual({ scanned: 3, written: 2, deleted: 1, skipped: 0 });
		expect(parsed.components).toEqual({ bm25: "ready", minsync: "unavailable" });
		expect(Array.isArray(parsed.diagnostics)).toBe(true);
		expect("indexPath" in parsed).toBe(false);
	});

	it("human output contains state and components without paths", () => {
		const out = renderStatus(cannedStatus(), {});
		expect(out).toContain("success");
		expect(out).toContain("bm25=ready");
		expect(out).not.toContain(root);
	});
});

describe("renderMemory", () => {
	it("json output emits the full path-opaque memory schema", () => {
		const out = renderMemory(cannedMemory(), { json: true });
		const parsed = JSON.parse(out) as Record<string, unknown>;
		expect(parsed.version).toBe(4);
		expect(Array.isArray(parsed.curatedResults)).toBe(true);
		expect(Array.isArray(parsed.feedbackSignals)).toBe(true);
		expect(Array.isArray(parsed.insights)).toBe(true);
		const sd = parsed.signalDefaults as Record<string, unknown>;
		expect(sd.explicitWeight).toBe(1);
		expect(sd.followupWeight).toBe(0.25);
		expect(sd.retryWeight).toBe(-0.25);
		expect(sd.implicitCap).toBe(0.5);
	});

	it("human output summarizes counts and signal defaults", () => {
		const out = renderMemory(cannedMemory(), { json: false });
		expect(out).toContain("counts:");
		expect(out).toContain("feedbackSignals=0");
		expect(out).toContain("signalDefaults:");
		expect(out.trim().startsWith("{")).toBe(false);
	});
});

describe("renderIndex", () => {
	it("json output has action, removed, and path-free rebuilt envelope", () => {
		const out = renderIndex(
			{
				action: "rebuild",
				removed: ["parsed", "bm25", "minsync"],
				rebuilt: cannedRefreshResult(),
			},
			{ json: true },
		);
		expect(out).not.toContain("indexPath");
		expect(out).not.toContain(FAKE_ABSOLUTE_ROOT);
		expect(out).not.toContain(".autorag/bm25");
		const parsed = JSON.parse(out) as Record<string, unknown>;
		expect(parsed.ok).toBe(true);
		expect(parsed.action).toBe("rebuild");
		expect(parsed.removed).toEqual(["parsed", "bm25", "minsync"]);
		const rebuilt = parsed.rebuilt as Record<string, unknown>;
		expect(rebuilt.ok).toBe(true);
		expect("indexPath" in rebuilt).toBe(false);
	});

	it("json output without rebuilt has only action and removed", () => {
		const out = renderIndex({ action: "reset", removed: ["parsed"] }, { json: true });
		const parsed = JSON.parse(out) as Record<string, unknown>;
		expect(parsed.action).toBe("reset");
		expect(parsed.removed).toEqual(["parsed"]);
		expect("rebuilt" in parsed).toBe(false);
	});
});

describe("renderFeedback", () => {
	it("json output has ok, applied, and sessionId", () => {
		const out = renderFeedback({ applied: true, sessionId: "sess-123" }, { json: true });
		const parsed = JSON.parse(out) as Record<string, unknown>;
		expect(parsed.ok).toBe(true);
		expect(parsed.applied).toBe(true);
		expect(parsed.sessionId).toBe("sess-123");
	});
});

describe("renderError", () => {
	it("json output has ok false and error message", () => {
		const out = renderError(new Error("boom"), { json: true });
		const parsed = JSON.parse(out) as Record<string, unknown>;
		expect(parsed.ok).toBe(false);
		expect(parsed.error).toBe("boom");
	});

	it("human output contains the error message", () => {
		const out = renderError(new Error("boom"), {});
		expect(out).toContain("boom");
	});

	it("never emits a stack trace under --debug (path-opacity)", () => {
		// A real Error carries a .stack with absolute filesystem paths and "    at"
		// frames; renderError must not surface it in human or JSON debug output.
		const err = new Error("boom");
		const human = renderError(err, { debug: true });
		expect(human).toBe("error: boom");
		expect(human).not.toContain("    at ");
		const json = JSON.parse(renderError(err, { json: true, debug: true })) as Record<string, unknown>;
		expect(json.stack).toBeUndefined();
		expect(json).toEqual({ ok: false, error: "boom" });
	});
});

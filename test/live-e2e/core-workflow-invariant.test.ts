import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
	assertAbsoluteReadableSource,
	assertServiceReady,
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
				{ corpusDigest: "old", embeddingModel: "embeddinggemma:latest" },
				{ corpusDigest: "new", embeddingModel: "embeddinggemma:latest" },
			),
		).toBe(false);
	});

	it("refuses an unavailable embedding service with a stable diagnostic", () => {
		expect(() => assertServiceReady({ verdict: "refused", code: "live-e2e-embedding-unavailable" })).toThrow(
			"live-e2e-embedding-unavailable",
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

	it("accepts only absolute existing readable source paths", () => {
		const root = tempRoot();
		const source = join(root, "source.txt");
		writeFileSync(source, "readable");
		expect(assertAbsoluteReadableSource(source)).toBe(resolve(source));
		expect(() => assertAbsoluteReadableSource("relative.txt")).toThrow("source-not-absolute");
	});
});

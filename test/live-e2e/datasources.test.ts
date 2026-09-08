import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
	buildDatasourceMatrix,
	parseDatasourceSelection,
	runDatasourceMatrix,
	sanitizeDiagnostic,
	validateNativeIdentity,
} from "../../scripts/live-e2e/datasources.mjs";

const roots: string[] = [];
const fixtureRoot = (): string => {
	const root = mkdtempSync(join(tmpdir(), "autorag-live-datasource-test-"));
	roots.push(root);
	mkdirSync(join(root, "corpus"));
	writeFileSync(join(root, "corpus", "sample.txt"), "native fixture");
	return root;
};

afterEach(() => {
	for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("live-e2e datasource matrix", () => {
	it("skips unavailable optional lanes without converting them to pass or throwing", async () => {
		const result = await runDatasourceMatrix({
			root: "/tmp/live-e2e-root",
			selection: ["missing-native"],
			which: () => false,
		});
		expect(result.lanes).toEqual([expect.objectContaining({ name: "missing-native", status: "SKIP" })]);
		expect(result.lanes[0]?.status).not.toBe("PASS");
	});

	it("reports an available native lane failure as FAIL", async () => {
		const result = await runDatasourceMatrix({
			root: "/tmp/live-e2e-root",
			selection: ["katok"],
			which: () => true,
			configured: () => true,
			run: async () => ({ ok: false, stderr: "fixture failed", stdout: "", code: 1 }),
		});
		expect(result.lanes[0]).toMatchObject({ name: "katok", status: "FAIL" });
	});

	it("accepts native identities and rejects slash-prefixed fake filesystem paths", () => {
		expect(validateNativeIdentity("kakao:chat/sender/chunk")).toBe(true);
		expect(validateNativeIdentity("/autorag/fake/chunks/1")).toBe(false);
	});

	it("redacts secrets and separates the core summary from datasource lanes", async () => {
		const diagnostic = sanitizeDiagnostic("failed token=super-secret password=hunter2 at /Users/me/private");
		expect(diagnostic).not.toContain("super-secret");
		expect(diagnostic).not.toContain("hunter2");
		const matrix = await runDatasourceMatrix({ root: fixtureRoot(), selection: ["local"], which: () => true });
		expect(matrix.summary).toMatchObject({ pass: 1, skip: 0, fail: 0 });
		expect(parseDatasourceSelection("local,configured")).toContain("local");
		expect(buildDatasourceMatrix().some((lane) => lane.name === "local")).toBe(true);
	});
});

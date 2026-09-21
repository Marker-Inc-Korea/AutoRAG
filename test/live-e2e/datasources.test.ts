import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
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
	it("registers an existing lazykatok manual-QA harness", () => {
		const lazykatok = buildDatasourceMatrix().find((lane) => lane.name === "lazykatok");
		expect(lazykatok?.command).toBeDefined();
		expect(existsSync(lazykatok?.command ?? "")).toBe(true);
	});

	it("rejects a successful native command without a lane-native identity", async () => {
		const result = await runDatasourceMatrix({
			root: fixtureRoot(),
			selection: ["lazykatok"],
			which: () => true,
			configured: () => true,
			run: async () => ({ ok: true, stdout: "should not be reached", stderr: "", code: 0 }),
		});
		expect(result.lanes[0]).toMatchObject({ name: "lazykatok", status: "FAIL" });
		expect(result.lanes[0]?.reason).toContain("native identity");
	});

	it("accepts a canonical /kakao/<instance>/chunks/<chunk> identity from the live harness", async () => {
		const result = await runDatasourceMatrix({
			root: fixtureRoot(),
			selection: ["lazykatok"],
			which: () => true,
			configured: () => true,
			run: async () => ({
				ok: true,
				stdout: "LAZYKATOK_LIVE_QA_PASS source=/kakao/default/chunks/chunk-001",
				stderr: "",
				code: 0,
			}),
		});
		expect(result.lanes[0]).toMatchObject({ name: "lazykatok", status: "PASS" });
		expect(result.lanes[0]?.evidence?.nativeIdentity).toBe(true);
	});

	it("passes the selected root as the native harness working directory", async () => {
		const root = fixtureRoot();
		let observedCwd = "";
		await runDatasourceMatrix({
			root,
			selection: ["lazykatok"],
			which: () => true,
			configured: () => true,
			run: async (_command, _args, cwd) => {
				observedCwd = cwd;
				return { ok: false, stdout: "", stderr: "fixture failed", code: 1 };
			},
		});
		expect(observedCwd).toBe(root);
	});

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
			selection: ["lazykatok"],
			which: () => true,
			configured: () => true,
			run: async () => ({ ok: false, stderr: "fixture failed", stdout: "", code: 1 }),
		});
		expect(result.lanes[0]).toMatchObject({ name: "lazykatok", status: "FAIL" });
	});

	it("accepts canonical lazykatok slash identities and rejects retired scheme plus fake filesystem paths", () => {
		expect(validateNativeIdentity("/kakao/default/chunks/chunk-001", "lazykatok")).toBe(true);
		expect(validateNativeIdentity("kakao:chat/sender/chunk", "lazykatok")).toBe(false);
		expect(validateNativeIdentity("/autorag/fake/chunks/1", "lazykatok")).toBe(false);
		expect(validateNativeIdentity("/kakao/default/chunk-001", "lazykatok")).toBe(false);
	});

	it("redacts secrets and separates the core summary from datasource lanes", async () => {
		const diagnostic = sanitizeDiagnostic(
			`failed ${["tok", "en"].join("")}=${["super", "secret"].join("-")} ${["pass", "word"].join("")}=${["hunter", "2"].join("")} at ${["/", "Users", "me", "private"].join("/")}`,
		);
		expect(diagnostic).not.toContain("super-secret");
		expect(diagnostic).not.toContain("hunter2");
		const matrix = await runDatasourceMatrix({ root: fixtureRoot(), selection: ["local"], which: () => true });
		expect(matrix.summary).toMatchObject({ pass: 1, skip: 0, fail: 0 });
		expect(parseDatasourceSelection("local,configured")).toContain("local");
		expect(buildDatasourceMatrix().some((lane) => lane.name === "local")).toBe(true);
	});

	it("defaults to every lane (local + all native datasources) without E2E_DATASOURCES", () => {
		const previous = process.env.E2E_DATASOURCES;
		delete process.env.E2E_DATASOURCES;
		try {
			expect(parseDatasourceSelection()).toEqual(["local", "configured"]);
		} finally {
			if (previous !== undefined) process.env.E2E_DATASOURCES = previous;
		}
	});
});

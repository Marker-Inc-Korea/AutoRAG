import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AutoRAGAgent, type AutoRAGAgentOptions } from "../../src/agent/agent.ts";
import type { CloudFileProbe, CloudRoot } from "../../src/filesystem/cloud-placeholder.ts";
import { JikjiClient, type JikjiDoctorResult } from "../../src/jikji/index.ts";
import type { ParsedMirrorSyncResult } from "../../src/mirror/index.ts";
import { writeFakeMinSync } from "../helpers/fake-minsync.ts";

let root: string;
let docs: string;
let minSyncBinary: string;

const prepared: JikjiDoctorResult = { state: "prepared", docsFailed: 0, errors: [], warnings: [] };

beforeEach(() => {
	root = realpathSync(mkdtempSync(join(tmpdir(), "autorag-jikji-status-")));
	docs = join(root, "docs");
	mkdirSync(docs);
	writeFileSync(join(docs, "note.txt"), "A document for indexing.\n");
	minSyncBinary = join(root, "fake-minsync.mjs");
	writeFakeMinSync(minSyncBinary);
	vi.stubEnv("JIKJI_DATA_DIR", join(root, "jikji-data"));
	vi.spyOn(JikjiClient.prototype, "prepare").mockResolvedValue({ ok: true, stdout: "{}", stderr: "", code: 0 });
});

afterEach(() => {
	vi.restoreAllMocks();
	vi.unstubAllEnvs();
	rmSync(root, { recursive: true, force: true, maxRetries: 20, retryDelay: 100 });
});

function cloudProbe(roots: readonly CloudRoot[] = []): CloudFileProbe {
	return {
		cloudRoots: async () => roots,
		datalessAmong: async () => new Set(),
		providerOf: async () => undefined,
		canRelease: () => false,
		evict: async () => ({ ok: true }),
	};
}

function makeAgent(overrides: Partial<AutoRAGAgentOptions> = {}): AutoRAGAgent {
	return new AutoRAGAgent({
		searchPaths: [docs],
		memoryPath: join(root, "memory.json"),
		workspacePath: root,
		minSync: { binaryPath: minSyncBinary, autoInstall: false },
		jikji: { autoInstall: false },
		everything: false,
		fsearch: false,
		cloudProbe: cloudProbe(),
		...overrides,
	});
}

describe("Jikji index status", () => {
	it("inspects a prepared index from a fresh agent without preparing it", async () => {
		const doctor = vi.spyOn(JikjiClient.prototype, "doctor").mockResolvedValue(prepared);

		const status = await makeAgent().getRefreshStatus();
		const freshStatus = await makeAgent().getRefreshStatus();

		expect(status.jikji).toEqual([{ ...prepared, source: "/docs" }]);
		expect(freshStatus.jikji).toEqual(status.jikji);
		expect(status.components.jikji).toBe("ready");
		expect(status.state).toBe("idle");
		expect(status.stale).toBe(true);
		expect(doctor).toHaveBeenCalledTimes(2);
		expect(doctor).toHaveBeenCalledWith(docs, { signal: undefined });
		expect(JikjiClient.prototype.prepare).not.toHaveBeenCalled();
	});

	it("observes external repairs on the same agent and removes obsolete diagnostics", async () => {
		const error = `missing required artifact row: wiki_index (${join(root, "index.sqlite3")})`;
		const doctor = vi
			.spyOn(JikjiClient.prototype, "doctor")
			.mockResolvedValueOnce({ state: "needs-reprepare", errors: [error], warnings: [] })
			.mockResolvedValueOnce(prepared);
		const agent = makeAgent();

		const broken = await agent.getRefreshStatus();
		expect(broken.components.jikji).toBe("degraded");
		expect(broken.jikji?.[0]?.state).toBe("needs-reprepare");
		expect(broken.diagnostics).toContainEqual(expect.objectContaining({ source: "/docs", message: error }));

		const repaired = await agent.getRefreshStatus();
		expect(repaired.jikji?.[0]?.state).toBe("prepared");
		expect(repaired.components.jikji).toBe("ready");
		expect(repaired.diagnostics.some((diagnostic) => diagnostic.code.startsWith("jikji-"))).toBe(false);
		expect(repaired.state).toBe(broken.state);
		expect(repaired.stale).toBe(broken.stale);
		expect(doctor).toHaveBeenCalledTimes(2);
	});

	it.each([
		{
			result: { state: "not-prepared", errors: [], warnings: [] },
			summary: "not-prepared",
			code: "jikji-not-prepared",
		},
		{
			result: { state: "unavailable", errors: ["doctor executable is unavailable"], warnings: [] },
			summary: "unavailable",
			code: "jikji-doctor-failed",
		},
		{ result: { ...prepared, docsFailed: 3 }, summary: "degraded", code: "jikji-documents-failed" },
		{
			result: { ...prepared, warnings: ["index manifest warning"] },
			summary: "degraded",
			code: "jikji-index-unhealthy",
		},
	] satisfies { result: JikjiDoctorResult; summary: string; code: string }[])(
		"reports $summary health with $code independently from refresh state",
		async ({ result, summary, code }) => {
			vi.spyOn(JikjiClient.prototype, "doctor").mockResolvedValue(result);
			const status = await makeAgent().getRefreshStatus();
			expect(status.jikji).toEqual([{ ...result, source: "/docs" }]);
			expect(status.components.jikji).toBe(summary);
			expect(status.diagnostics).toContainEqual(expect.objectContaining({ code, source: "/docs" }));
			expect(status.state).toBe("idle");
		},
	);

	it("reports each outermost root separately and aggregates mixed health", async () => {
		const other = join(root, "other");
		const nested = join(docs, "nested");
		mkdirSync(other);
		mkdirSync(nested);
		const doctor = vi
			.spyOn(JikjiClient.prototype, "doctor")
			.mockImplementation(async (path) =>
				path === docs ? prepared : { state: "not-prepared", errors: [], warnings: [] },
			);
		const status = await makeAgent({ searchPaths: [other, docs, nested, docs] }).getRefreshStatus();
		expect(status.jikji).toEqual([
			{ ...prepared, source: "/docs" },
			{ state: "not-prepared", errors: [], warnings: [], source: "/other" },
		]);
		expect(status.components.jikji).toBe("not-prepared");
		expect(doctor).toHaveBeenCalledTimes(2);
	});

	it.each(["cloud-files", "file-provider", "network"] as const)("skips doctor for a %s root", async (kind) => {
		const doctor = vi.spyOn(JikjiClient.prototype, "doctor");
		const status = await makeAgent({ cloudProbe: cloudProbe([{ path: docs, kind }]) }).getRefreshStatus();
		expect(status.jikji).toEqual([{ source: "/docs", state: "skipped", errors: [], warnings: [] }]);
		expect(status.components.jikji).toBe("skipped");
		expect(doctor).not.toHaveBeenCalled();
	});

	it("omits Jikji status when disabled", async () => {
		const doctor = vi.spyOn(JikjiClient.prototype, "doctor");
		const status = await makeAgent({ jikji: false }).getRefreshStatus();
		expect(status.jikji).toBeUndefined();
		expect(status.components.jikji).toBeUndefined();
		expect(doctor).not.toHaveBeenCalled();
	});

	it("inspects after prepare and includes current index problems in refresh diagnostics", async () => {
		const error = "missing required artifact row: wiki_index";
		const doctor = vi.spyOn(JikjiClient.prototype, "doctor").mockImplementation(async () => {
			expect(JikjiClient.prototype.prepare).toHaveBeenCalledOnce();
			return { state: "needs-reprepare", docsFailed: 2, errors: [error], warnings: [] };
		});
		const agent = makeAgent();
		const refreshed = await agent.refresh(false, { methods: ["jikji"] });
		expect(doctor).toHaveBeenCalledOnce();
		expect(refreshed.diagnostics).toEqual(
			expect.arrayContaining([
				expect.objectContaining({ code: "jikji-index-unhealthy", source: "/docs", message: error }),
				expect.objectContaining({ code: "jikji-documents-failed", source: "/docs" }),
			]),
		);
		expect(agent.refreshComponentStatus().jikji).toBe("degraded");
	});

	it("does not inspect Jikji for an unrelated refresh method", async () => {
		const doctor = vi.spyOn(JikjiClient.prototype, "doctor");
		const result = await makeAgent().refresh(false, { methods: ["datasources"] });
		expect(doctor).not.toHaveBeenCalled();
		expect(JikjiClient.prototype.prepare).not.toHaveBeenCalled();
		expect(result.diagnostics.some((diagnostic) => diagnostic.code.startsWith("jikji-"))).toBe(false);
	});

	it("keeps a running refresh separate from the current on-disk index health", async () => {
		let markStarted = () => {};
		let releasePrepare = () => {};
		const started = new Promise<void>((resolve) => {
			markStarted = resolve;
		});
		const release = new Promise<void>((resolve) => {
			releasePrepare = resolve;
		});
		vi.mocked(JikjiClient.prototype.prepare).mockImplementation(async () => {
			markStarted();
			await release;
			return { ok: true, stdout: "{}", stderr: "", code: 0 };
		});
		vi.spyOn(JikjiClient.prototype, "doctor").mockResolvedValue(prepared);
		const agent = makeAgent();
		const refresh = agent.refresh(false, { methods: ["jikji"] });
		await started;
		try {
			const status = await agent.getRefreshStatus();
			expect(status.state).toBe("indexing");
			expect(status.inFlight).toBe(true);
			expect(status.jikji?.[0]?.state).toBe("prepared");
			expect(status.components.jikji).toBe("ready");
		} finally {
			releasePrepare();
			await refresh;
		}
	});

	it("uses current doctor health rather than an earlier prepare failure", async () => {
		vi.mocked(JikjiClient.prototype.prepare).mockResolvedValue({
			ok: false,
			reason: "nonzero-exit",
			stdout: "",
			stderr: "failed",
			code: 1,
		});
		const doctor = vi
			.spyOn(JikjiClient.prototype, "doctor")
			.mockResolvedValue({ state: "not-prepared", errors: [], warnings: [] });
		const agent = makeAgent();
		const refreshed = await agent.refresh(false, { methods: ["jikji"] });
		expect(refreshed.diagnostics.some((diagnostic) => diagnostic.code === "jikji-prepare-failed")).toBe(true);
		const failed = await agent.getRefreshStatus();
		expect(failed.components.jikji).toBe("not-prepared");
		expect(failed.diagnostics.some((diagnostic) => diagnostic.code === "jikji-prepare-failed")).toBe(true);

		doctor.mockResolvedValue(prepared);
		const current = await agent.getRefreshStatus();
		expect(current.components.jikji).toBe("ready");
		expect(current.diagnostics.some((diagnostic) => diagnostic.code === "jikji-prepare-failed")).toBe(false);
		expect(current.state).toBe("success");

		doctor.mockResolvedValue({ state: "not-prepared", errors: [], warnings: [] });
		const removed = await agent.getRefreshStatus();
		expect(removed.jikji?.[0]?.state).toBe("not-prepared");
		expect(removed.diagnostics.some((diagnostic) => diagnostic.code === "jikji-prepare-failed")).toBe(false);
	});

	it("keeps concurrent status snapshots consistent with their own doctor results", async () => {
		vi.spyOn(JikjiClient.prototype, "doctor")
			.mockResolvedValueOnce(prepared)
			.mockResolvedValue({ state: "needs-reprepare", errors: ["missing artifact"], warnings: [] });
		const agent = makeAgent();
		const internals = agent as unknown as { scanMirrorStaleness(): Promise<ParsedMirrorSyncResult> };
		const scan = internals.scanMirrorStaleness.bind(agent);
		let markStarted = () => {};
		let releaseScan = () => {};
		const started = new Promise<void>((resolve) => {
			markStarted = resolve;
		});
		const release = new Promise<void>((resolve) => {
			releaseScan = resolve;
		});
		vi.spyOn(internals, "scanMirrorStaleness").mockImplementationOnce(async () => {
			markStarted();
			await release;
			return scan();
		});
		const first = agent.getRefreshStatus();
		await started;
		try {
			const second = await agent.getRefreshStatus();
			expect(second.jikji?.[0]?.state).toBe("needs-reprepare");
			expect(second.components.jikji).toBe("degraded");
		} finally {
			releaseScan();
		}
		const earlier = await first;
		expect(earlier.jikji?.[0]?.state).toBe("prepared");
		expect(earlier.components.jikji).toBe("ready");
		expect(agent.refreshComponentStatus().jikji).toBe("degraded");
	});
});

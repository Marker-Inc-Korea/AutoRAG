import { describe, expect, it } from "vitest";
import type { AutoRAGRefreshResult } from "../../src/agent/agent.ts";
import type { HealthReportV1 } from "../../src/cli/commands/health.ts";
import { renderHealth, renderRefresh } from "../../src/cli/output.ts";

const report: HealthReportV1 = {
	healthSchemaVersion: 1,
	ok: true,
	category: "ok",
	command: "health",
	probesSkipped: false,
	coverage: { modelProvider: true, retrievalTools: false, searchCuration: false, indexHealth: false },
	config: { ok: true, source: "defaults" },
	model: {
		provider: "test",
		modelId: "single-agent",
		api: "openai-completions",
		capabilities: { text: true, image: false },
		auth: { present: true, source: "env", envName: "TEST_API_KEY" },
		resolutionSource: "config",
	},
	probe: { skipped: false, ok: true, category: "ok", durationMs: 2 },
	indexHealth: { separate: true, command: "autorag status", included: false },
};

describe("renderHealth", () => {
	it("renders one model and one probe", () => {
		const output = renderHealth(report, { json: false, debug: false });
		expect(output).toContain("model: test/single-agent");
		expect(output).toContain("probe: skipped=false");
		expect(output).not.toMatch(/subagent|explorer/i);
	});

	it("renders stable JSON", () => {
		expect(JSON.parse(renderHealth(report, { json: true, debug: false }))).toEqual(report);
	});
});

describe("renderRefresh", () => {
	it("includes minsync result in the JSON envelope and human output on success", () => {
		const result: AutoRAGRefreshResult = {
			scanned: 3,
			written: 2,
			deleted: 0,
			skipped: 1,
			indexPath: "/secret/path/.autorag/parsed",
			diagnostics: [],
			minsync: {
				ok: true,
				synced: 2,
			},
		};

		const jsonStr = renderRefresh(result, { json: true });
		const parsed = JSON.parse(jsonStr);
		expect(parsed.ok).toBe(true);
		expect(parsed.counts).toEqual({ scanned: 3, written: 2, deleted: 0, skipped: 1 });
		expect(parsed.minsync).toEqual({ ok: true, synced: 2 });
		expect(jsonStr).not.toContain("/secret/path");

		const humanStr = renderRefresh(result, { json: false });
		expect(humanStr).toContain("refresh: ok");
		expect(humanStr).toContain("minsync: ok=true synced=2");
	});

	it("reports ok: false and surfaces minsync error diagnostic when minsync fails", () => {
		const result: AutoRAGRefreshResult = {
			scanned: 1,
			written: 1,
			deleted: 0,
			skipped: 0,
			indexPath: "/secret/path/.autorag/parsed",
			diagnostics: [
				{
					code: "embedder-unavailable",
					severity: "error",
					message: "TEI API error 502 Bad Gateway at /secret/cache/model",
					source: "minsync",
				},
			],
			minsync: {
				ok: false,
				synced: 0,
				reason: "check-failed: embedder unavailable at /secret/path",
				diagnostics: [
					{
						code: "embedder-unavailable",
						severity: "error",
						message: "TEI API error 502 Bad Gateway at /secret/cache/model",
						source: "minsync",
					},
				],
			},
		};

		const jsonStr = renderRefresh(result, { json: true });
		const parsed = JSON.parse(jsonStr);
		expect(parsed.ok).toBe(false);
		expect(parsed.minsync.ok).toBe(false);
		expect(parsed.minsync.synced).toBe(0);
		expect(parsed.minsync.reason).toContain("<path>");
		expect(parsed.minsync.diagnostics).toHaveLength(1);
		expect(parsed.diagnostics[0].code).toBe("embedder-unavailable");
		expect(jsonStr).not.toContain("/secret/");

		const humanStr = renderRefresh(result, { json: false });
		expect(humanStr).toContain("refresh: failed");
		expect(humanStr).toContain("minsync: ok=false synced=0");
		expect(humanStr).toContain("diagnostic: [error] embedder-unavailable");
		expect(humanStr).not.toContain("/secret/");
	});

	it("reports ok: false when a datasource index fails", () => {
		const result: AutoRAGRefreshResult = {
			scanned: 1,
			written: 1,
			deleted: 0,
			skipped: 0,
			indexPath: "/secret/path",
			diagnostics: [],
			datasources: [
				{
					ok: false,
					skill: "slack",
					instanceId: "inst-1",
					indexedAt: 1774000000000,
					diagnostics: [],
				},
			],
		};

		const parsed = JSON.parse(renderRefresh(result, { json: true }));
		expect(parsed.ok).toBe(false);
		expect(parsed.datasources[0].ok).toBe(false);
	});
});

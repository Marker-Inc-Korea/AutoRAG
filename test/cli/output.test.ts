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

describe("renderRefresh", () => {
	it("carries the MinSync staging-exclusion count in the JSON envelope", () => {
		// A file name with no canonical source-id form is excluded from the
		// MinSync index; the count must be visible where the reporter looks.
		const result: AutoRAGRefreshResult = {
			indexPath: "/tmp/workspace/.autorag/parsed/index.json",
			scanned: 2,
			written: 2,
			deleted: 0,
			skipped: 0,
			diagnostics: [],
			minsync: { ok: true, synced: 2, stagingExcludedCount: 1 },
		};
		const envelope = JSON.parse(renderRefresh(result, { json: true, debug: false })) as {
			minsync?: { stagingExcludedCount?: number };
		};
		expect(envelope.minsync?.stagingExcludedCount).toBe(1);
	});
});

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

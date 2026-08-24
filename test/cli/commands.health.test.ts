import type { Api, Model } from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import { type HealthDeps, type HealthReportV1, type ProbeOutput, runHealth } from "../../src/cli/commands/health.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";

function model(): Model<Api> {
	return {
		id: "single-agent",
		name: "Single Agent",
		api: "openai-completions",
		provider: "test",
		baseUrl: "https://example.test/v1",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 1000,
		maxTokens: 100,
	};
}

function context(flags: CommandContext["flags"] = {}) {
	const output: string[] = [];
	const ctx: CommandContext = {
		positionals: [],
		flags,
		json: true,
		debug: false,
		cwd: "/tmp",
		stdout: (line) => output.push(line),
		stderr: () => {},
	};
	return { ctx, output };
}

function deps(probe: ProbeOutput = { ok: true, category: "ok" }, auth = true): HealthDeps {
	return {
		configResolver: () => ({ searchPaths: ["."], workspacePath: ".", memoryPath: "memory.json" }),
		modelResolver: () => ({
			model: model(),
			role: {
				provider: "test",
				modelId: "single-agent",
				displayName: "Single Agent",
				api: "openai-completions",
				baseUrl: "https://example.test/v1",
				contextWindow: 1000,
				maxTokens: 100,
				capabilities: { input: ["text"], reasoning: false },
				auth: { present: auth, source: auth ? "env" : "none", envName: "TEST_API_KEY" },
				resolutionSource: "config",
			},
		}),
		probe: async () => probe,
		now: () => 1000,
	};
}

describe("runHealth single-model checks", () => {
	it("reports healthy model auth when probes are skipped", async () => {
		const { ctx, output } = context({ "skip-probes": true });
		expect(await runHealth(ctx, deps())).toBe(0);
		const report = JSON.parse(output[0]) as HealthReportV1;
		expect(report.ok).toBe(true);
		expect(report.model?.modelId).toBe("single-agent");
		expect(report.probe?.skipped).toBe(true);
		expect(JSON.stringify(report)).not.toMatch(/subagent|explorer/i);
	});

	it("runs one completion probe", async () => {
		const { ctx, output } = context();
		expect(await runHealth(ctx, deps())).toBe(0);
		const report = JSON.parse(output[0]) as HealthReportV1;
		expect(report.probe).toMatchObject({ skipped: false, ok: true, category: "ok" });
		expect(report.coverage.modelProvider).toBe(true);
	});

	it("reports missing auth without probing", async () => {
		const { ctx, output } = context();
		expect(await runHealth(ctx, deps({ ok: true, category: "ok" }, false))).toBe(1);
		expect((JSON.parse(output[0]) as HealthReportV1).category).toBe("auth_missing");
	});

	it("validates timeout input", async () => {
		const { ctx, output } = context({ "timeout-ms": "bad" });
		expect(await runHealth(ctx, deps())).toBe(2);
		expect((JSON.parse(output[0]) as HealthReportV1).category).toBe("config");
	});
});

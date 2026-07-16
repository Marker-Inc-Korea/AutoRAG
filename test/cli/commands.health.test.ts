import type { Api, Model } from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import {
	type HealthCategory,
	type HealthDeps,
	type HealthReportV1,
	type ProbeInput,
	type ProbeOutput,
	type ResolvedHealthModel,
	type ResolvedHealthRole,
	runHealth,
} from "../../src/cli/commands/health.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import type { CliConfig } from "../../src/cli/config.ts";

// ---------------------------------------------------------------------------
// Fakes
// ---------------------------------------------------------------------------

const FAKE_API_KEY = "sk-test-fake-key-DO-NOT-LEAK-1234567890";

function fakeModel(provider = "anthropic", id = "claude-haiku-4-5"): Model<Api> {
	return {
		id,
		name: `${id} (fake)`,
		api: "anthropic-messages",
		provider,
		baseUrl: "https://api.anthropic.com",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 1, output: 5, cacheRead: 0.1, cacheWrite: 1.25 },
		contextWindow: 200_000,
		maxTokens: 64_000,
	} as unknown as Model<Api>;
}

function fakeExplorerModel(): Model<Api> {
	return fakeModel("openai", "gpt-5.6-luna");
}

function fakeRole(
	provider: string,
	modelId: string,
	present: boolean,
	source: ResolvedHealthRole["auth"]["source"] = "env",
	envName = "ANTHROPIC_API_KEY",
): ResolvedHealthRole {
	return {
		provider,
		modelId,
		displayName: `${modelId} (fake)`,
		api: "anthropic-messages",
		baseUrl: "https://api.anthropic.com",
		contextWindow: 200_000,
		maxTokens: 64_000,
		capabilities: { input: ["text", "image"], reasoning: true },
		auth: { present, source, ...(present ? { envName } : {}) },
		resolutionSource: "catalog",
	};
}

function fakeResolvedModel(opts: { orchAuth?: boolean; explorerAuth?: boolean }): ResolvedHealthModel {
	const orchPresent = opts.orchAuth ?? true;
	const explorerPresent = opts.explorerAuth ?? true;
	return {
		model: fakeModel(),
		explorerModel: fakeExplorerModel(),
		apiKey: FAKE_API_KEY,
		providerApiKeys: { anthropic: FAKE_API_KEY, openai: FAKE_API_KEY },
		roles: {
			orchestrator: fakeRole("anthropic", "claude-haiku-4-5", orchPresent),
			explorer: fakeRole("openai", "gpt-5.6-luna", explorerPresent, "env", "OPENAI_API_KEY"),
		},
	};
}

function fakeConfig(): CliConfig {
	return {
		searchPaths: ["."],
		workspacePath: ".",
		memoryPath: "memory.json",
		model: { provider: "anthropic", id: "claude-haiku-4-5" },
		agents: {
			orchestrator: { provider: "anthropic", id: "claude-haiku-4-5" },
			explorer: { provider: "openai", id: "gpt-5.6-luna" },
		},
	};
}

function makeCtx(overrides: Partial<CommandContext> = {}): CommandContext {
	return {
		positionals: [],
		flags: {},
		json: true,
		debug: false,
		cwd: "/tmp/autorag-health-test",
		stdout: () => {},
		stderr: () => {},
		...overrides,
	};
}

function makeDeps(opts: {
	orchAuth?: boolean;
	explorerAuth?: boolean;
	orchProbe?: (input: ProbeInput, signal: AbortSignal) => Promise<ProbeOutput>;
	explorerProbe?: (input: ProbeInput, signal: AbortSignal) => Promise<ProbeOutput>;
	modelResolverThrow?: Error;
	configResolverThrow?: Error;
}): HealthDeps {
	return {
		configResolver: opts.configResolverThrow
			? () => {
					throw opts.configResolverThrow as Error;
				}
			: () => fakeConfig(),
		modelResolver: opts.modelResolverThrow
			? () => {
					throw opts.modelResolverThrow as Error;
				}
			: () => fakeResolvedModel({ orchAuth: opts.orchAuth, explorerAuth: opts.explorerAuth }),
		...(opts.orchProbe ? { orchestratorProbe: opts.orchProbe } : {}),
		...(opts.explorerProbe ? { explorerProbe: opts.explorerProbe } : {}),
		now: () => 1000,
	};
}

function probe(ok: boolean, category: HealthCategory, message?: string): Promise<ProbeOutput> {
	return Promise.resolve({ ok, category, ...(message !== undefined ? { message } : {}) });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("runHealth --skip-probes", () => {
	it("exits 0 / category ok when config, model, and auth are healthy", async () => {
		const out: string[] = [];
		const ctx = makeCtx({ flags: { "skip-probes": true }, stdout: (l) => out.push(l) });
		const code = await runHealth(ctx, makeDeps({}));
		expect(code).toBe(0);
		const report = JSON.parse(out[0]) as HealthReportV1;
		expect(report.healthSchemaVersion).toBe(1);
		expect(report.ok).toBe(true);
		expect(report.category).toBe("ok");
		expect(report.probesSkipped).toBe(true);
		expect(report.coverage.modelProvider).toBe(true);
		expect(report.coverage.subagentDispatch).toBe(false);
		expect(report.coverage.retrievalTools).toBe(false);
		expect(report.coverage.searchCuration).toBe(false);
		expect(report.coverage.indexHealth).toBe(false);
		expect(report.probes.orchestrator?.skipped).toBe(true);
		expect(report.probes.explorer?.skipped).toBe(true);
	});

	it("exits 1 / category auth_missing when orchestrator auth is absent (auth checked even with skip-probes)", async () => {
		const out: string[] = [];
		const ctx = makeCtx({ flags: { "skip-probes": true }, stdout: (l) => out.push(l) });
		const code = await runHealth(ctx, makeDeps({ orchAuth: false }));
		expect(code).toBe(1);
		const report = JSON.parse(out[0]) as HealthReportV1;
		expect(report.category).toBe("auth_missing");
		expect(report.ok).toBe(false);
		expect(report.probes.orchestrator?.category).toBe("auth_missing");
		expect(report.coverage.modelProvider).toBe(false);
	});

	it("exits 1 / category auth_missing when explorer auth is absent", async () => {
		const out: string[] = [];
		const ctx = makeCtx({ flags: { "skip-probes": true }, stdout: (l) => out.push(l) });
		const code = await runHealth(ctx, makeDeps({ explorerAuth: false }));
		expect(code).toBe(1);
		const report = JSON.parse(out[0]) as HealthReportV1;
		expect(report.category).toBe("auth_missing");
		expect(report.probes.explorer?.category).toBe("auth_missing");
	});
});

describe("runHealth config failure", () => {
	it("exits 2 / category config when configResolver throws", async () => {
		const out: string[] = [];
		const ctx = makeCtx({ stdout: (l) => out.push(l) });
		const code = await runHealth(
			ctx,
			makeDeps({ configResolverThrow: new Error("Config file not found: /secret/path/config.json") }),
		);
		expect(code).toBe(2);
		const report = JSON.parse(out[0]) as HealthReportV1;
		expect(report.category).toBe("config");
		expect(report.config.ok).toBe(false);
		// Sanitized: no absolute path leaked.
		expect(out[0]).not.toContain("/secret/path/config.json");
		expect(report.config.message).not.toContain("/secret");
	});
});

describe("runHealth model_resolution failure", () => {
	it("exits 2 / category model_resolution when modelResolver throws", async () => {
		const out: string[] = [];
		const ctx = makeCtx({ stdout: (l) => out.push(l) });
		const code = await runHealth(
			ctx,
			makeDeps({ modelResolverThrow: new Error("Unknown configured model: badprov/badid") }),
		);
		expect(code).toBe(2);
		const report = JSON.parse(out[0]) as HealthReportV1;
		expect(report.category).toBe("model_resolution");
		expect(report.probes.orchestrator?.category).toBe("model_resolution");
		expect(report.probes.orchestrator?.message).toContain("Unknown configured model");
		// Only one line of output (no double render).
		expect(out).toHaveLength(1);
	});
});

describe("runHealth probe categories", () => {
	it("orchestrator timeout maps to category timeout / exit 1", async () => {
		const out: string[] = [];
		const ctx = makeCtx({ stdout: (l) => out.push(l) });
		const code = await runHealth(
			ctx,
			makeDeps({
				orchProbe: () => probe(false, "timeout", "orchestrator probe timed out after 10000ms"),
			}),
		);
		expect(code).toBe(1);
		const report = JSON.parse(out[0]) as HealthReportV1;
		expect(report.category).toBe("timeout");
		expect(report.probes.orchestrator?.category).toBe("timeout");
		expect(report.probes.orchestrator?.skipped).toBe(false);
	});

	it("explorer subagent_failed maps to category subagent_failed / exit 1", async () => {
		const out: string[] = [];
		const ctx = makeCtx({ stdout: (l) => out.push(l) });
		const code = await runHealth(
			ctx,
			makeDeps({
				orchProbe: () => probe(true, "ok"),
				explorerProbe: () => probe(false, "subagent_failed", "concurrent AutoRAG session busy"),
			}),
		);
		expect(code).toBe(1);
		const report = JSON.parse(out[0]) as HealthReportV1;
		expect(report.category).toBe("subagent_failed");
		expect(report.probes.explorer?.category).toBe("subagent_failed");
		expect(report.coverage.subagentDispatch).toBe(false);
	});

	it("orchestrator completion_failed maps to category completion_failed / exit 1", async () => {
		const out: string[] = [];
		const ctx = makeCtx({ stdout: (l) => out.push(l) });
		const code = await runHealth(
			ctx,
			makeDeps({
				orchProbe: () => probe(false, "completion_failed", "non-timeout completion error"),
				explorerProbe: () => probe(true, "ok"),
			}),
		);
		expect(code).toBe(1);
		const report = JSON.parse(out[0]) as HealthReportV1;
		expect(report.category).toBe("completion_failed");
	});

	it("provider_unreachable maps correctly / exit 1", async () => {
		const out: string[] = [];
		const ctx = makeCtx({ stdout: (l) => out.push(l) });
		const code = await runHealth(
			ctx,
			makeDeps({
				orchProbe: () => probe(false, "provider_unreachable", "ENOTFOUND api.anthropic.com"),
				explorerProbe: () => probe(true, "ok"),
			}),
		);
		expect(code).toBe(1);
		const report = JSON.parse(out[0]) as HealthReportV1;
		expect(report.category).toBe("provider_unreachable");
	});

	it("all probes ok => category ok / exit 0, subagentDispatch true", async () => {
		const out: string[] = [];
		const ctx = makeCtx({ stdout: (l) => out.push(l) });
		const code = await runHealth(
			ctx,
			makeDeps({
				orchProbe: () => probe(true, "ok"),
				explorerProbe: () => probe(true, "ok"),
			}),
		);
		expect(code).toBe(0);
		const report = JSON.parse(out[0]) as HealthReportV1;
		expect(report.category).toBe("ok");
		expect(report.ok).toBe(true);
		expect(report.coverage.subagentDispatch).toBe(true);
		expect(report.probesSkipped).toBe(false);
	});
});

describe("runHealth multi-failure precedence", () => {
	it("timeout outranks completion_failed", async () => {
		const out: string[] = [];
		const ctx = makeCtx({ stdout: (l) => out.push(l) });
		const code = await runHealth(
			ctx,
			makeDeps({
				orchProbe: () => probe(false, "timeout", "timed out"),
				explorerProbe: () => probe(false, "completion_failed", "completion error"),
			}),
		);
		expect(code).toBe(1);
		const report = JSON.parse(out[0]) as HealthReportV1;
		expect(report.category).toBe("timeout");
		// Per-role categories preserved.
		expect(report.probes.orchestrator?.category).toBe("timeout");
		expect(report.probes.explorer?.category).toBe("completion_failed");
	});

	it("auth_missing outranks timeout (auth checked before probes)", async () => {
		const out: string[] = [];
		const ctx = makeCtx({ stdout: (l) => out.push(l) });
		const code = await runHealth(
			ctx,
			makeDeps({
				orchAuth: false,
				orchProbe: () => probe(false, "timeout", "timed out"),
			}),
		);
		expect(code).toBe(1);
		const report = JSON.parse(out[0]) as HealthReportV1;
		expect(report.category).toBe("auth_missing");
	});

	it("provider_unreachable outranks subagent_failed", async () => {
		const out: string[] = [];
		const ctx = makeCtx({ stdout: (l) => out.push(l) });
		const code = await runHealth(
			ctx,
			makeDeps({
				orchProbe: () => probe(false, "provider_unreachable", "ENOTFOUND"),
				explorerProbe: () => probe(false, "subagent_failed", "busy"),
			}),
		);
		expect(code).toBe(1);
		const report = JSON.parse(out[0]) as HealthReportV1;
		expect(report.category).toBe("provider_unreachable");
	});
});

describe("runHealth --timeout-ms", () => {
	it("invalid --timeout-ms => config category / exit 2", async () => {
		const out: string[] = [];
		const ctx = makeCtx({ flags: { "timeout-ms": "abc" }, stdout: (l) => out.push(l) });
		const code = await runHealth(ctx, makeDeps({}));
		expect(code).toBe(2);
		const report = JSON.parse(out[0]) as HealthReportV1;
		expect(report.category).toBe("config");
		expect(report.config.ok).toBe(false);
		expect(report.config.message).toContain("--timeout-ms");
	});

	it("zero --timeout-ms => config category / exit 2", async () => {
		const out: string[] = [];
		const ctx = makeCtx({ flags: { "timeout-ms": "0" }, stdout: (l) => out.push(l) });
		const code = await runHealth(ctx, makeDeps({}));
		expect(code).toBe(2);
		const report = JSON.parse(out[0]) as HealthReportV1;
		expect(report.category).toBe("config");
	});

	it("valid --timeout-ms is accepted and passed to probes", async () => {
		let receivedTimeout = 0;
		const out: string[] = [];
		const ctx = makeCtx({ flags: { "timeout-ms": "5000" }, stdout: (l) => out.push(l) });
		const code = await runHealth(
			ctx,
			makeDeps({
				orchProbe: (input) => {
					receivedTimeout = input.timeoutMs;
					return probe(true, "ok");
				},
				explorerProbe: () => probe(true, "ok"),
			}),
		);
		expect(code).toBe(0);
		expect(receivedTimeout).toBe(5000);
	});
});

describe("runHealth secret/path opacity", () => {
	it("never prints the API key, env values, or absolute paths in JSON output", async () => {
		const out: string[] = [];
		const ctx = makeCtx({ flags: { "skip-probes": true }, stdout: (l) => out.push(l) });
		await runHealth(ctx, makeDeps({}));
		const blob = out[0];
		expect(blob).not.toContain(FAKE_API_KEY);
		expect(blob).not.toContain("sk-test");
		// envName is allowed (the var name, not the value).
		expect(blob).not.toContain("DO-NOT-LEAK");
	});

	it("sanitizes probe messages containing secrets/paths/stacks", async () => {
		const out: string[] = [];
		const ctx = makeCtx({ stdout: (l) => out.push(l) });
		await runHealth(
			ctx,
			makeDeps({
				orchProbe: () =>
					probe(
						false,
						"completion_failed",
						`error at /Users/secret/config.json\n    at Object.<anonymous> (sk-leaked-key-1234567890abcdef)`,
					),
				explorerProbe: () => probe(true, "ok"),
			}),
		);
		const blob = out[0];
		expect(blob).not.toContain("/Users/secret/config.json");
		expect(blob).not.toContain("sk-leaked-key");
		expect(blob).not.toContain("    at ");
	});

	it("indexHealth is separate and points to autorag status", async () => {
		const out: string[] = [];
		const ctx = makeCtx({ flags: { "skip-probes": true }, stdout: (l) => out.push(l) });
		await runHealth(ctx, makeDeps({}));
		const report = JSON.parse(out[0]) as HealthReportV1;
		expect(report.indexHealth).toEqual({ separate: true, command: "autorag status", included: false });
	});
});

describe("runHealth human output", () => {
	it("renders human-readable output when json is false", async () => {
		const out: string[] = [];
		const ctx = makeCtx({ json: false, flags: { "skip-probes": true }, stdout: (l) => out.push(l) });
		const code = await runHealth(ctx, makeDeps({}));
		expect(code).toBe(0);
		const text = out[0];
		expect(text).toContain("health: ok");
		expect(text).toContain("schemaVersion: 1");
		expect(text).toContain("probesSkipped: true");
		expect(text).toContain("model orchestrator: anthropic/claude-haiku-4-5");
		expect(text).toContain("indexHealth: separate=true");
		expect(text).not.toContain(FAKE_API_KEY);
	});
});

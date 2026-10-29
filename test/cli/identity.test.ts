import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Model } from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { runIndex } from "../../src/cli/commands/index.ts";
import { runRefresh } from "../../src/cli/commands/refresh.ts";
import { runSearch } from "../../src/cli/commands/search.ts";
import { runStatus } from "../../src/cli/commands/status.ts";
import { runTui } from "../../src/cli/commands/tui.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import { runWatch } from "../../src/cli/commands/watch.ts";
import { resolveConfig } from "../../src/cli/config.ts";
import { createAutoRAGLite } from "../../src/core.ts";
import { createRetrievalToolkit } from "../../src/profiles/retrieval-tools.ts";

// These tests stop before any model, terminal, or remote bridge starts.
vi.mock("../../src/cli/config.ts", async (importOriginal) => ({
	...(await importOriginal<typeof import("../../src/cli/config.ts")>()),
	resolveTuiAgentModel: async () => undefined,
	resolveQueryDecompositionModel: async () => undefined,
}));

vi.mock("../../src/cli/whatsapp-extension.ts", () => ({
	resolveWhatsAppExtension: async () => undefined,
}));

vi.mock("../../src/convert/index.ts", async (importOriginal) => ({
	...(await importOriginal<typeof import("../../src/convert/index.ts")>()),
	warmFileeBinary: async () => undefined,
}));

let root: string;
let configPath: string;
let identityPath: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-cli-identity-"));
	const home = join(root, "home");
	const autoragHome = join(home, ".autorag");
	vi.stubEnv("HOME", home);
	vi.stubEnv("USERPROFILE", home);
	vi.stubEnv("AUTORAG_HOME", autoragHome);
	mkdirSync(autoragHome, { recursive: true });
	identityPath = join(autoragHome, "IDENTITY.md");
	configPath = join(root, "config.json");
	writeFileSync(
		configPath,
		JSON.stringify({
			searchPaths: [root],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: { autoInstall: false },
			jikji: false,
			everything: false,
			fsearch: false,
			webSearch: false,
			jev: false,
			dupey: { enabled: false },
			filee: { enabled: false },
		}),
	);
});

afterEach(() => {
	vi.restoreAllMocks();
	vi.unstubAllEnvs();
	rmSync(root, { recursive: true, force: true });
});

function context(positionals: string[] = [], flags: CommandContext["flags"] = {}, json = true) {
	const stdout: string[] = [];
	const stderr: string[] = [];
	const ctx: CommandContext = {
		positionals,
		flags: { config: configPath, ...flags },
		json,
		debug: false,
		cwd: root,
		stdout: (line) => stdout.push(line),
		stderr: (line) => stderr.push(line),
	};
	return { ctx, stdout, stderr };
}

const model: Model<"openai-responses"> = {
	id: "test",
	name: "Test",
	api: "openai-responses",
	provider: "test",
	baseUrl: "https://example.test/v1",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 1000,
	maxTokens: 100,
};

describe("search construction failures", () => {
	for (const withModelResolver of [false, true]) {
		it.each([false, true])(
			`reports the original error and releases embedders (resolved model: ${withModelResolver}, JSON: %s)`,
			async (json) => {
				const error = new Error(`EACCES: permission denied, open '${identityPath}'`);
				const releaseEmbedders = vi.fn(async () => {});
				const { ctx, stdout, stderr } = context(["query"], {}, json);
				const code = await runSearch(ctx, {
					agentFactory: () => {
						throw error;
					},
					...(withModelResolver ? { modelResolver: () => ({ model }) } : {}),
					releaseEmbedders,
				});
				expect(code).toBe(1);
				expect(stdout).toEqual([]);
				expect(stderr).toHaveLength(1);
				expect(json ? JSON.parse(stderr[0]).error : stderr[0]).toBe(
					json ? error.message : `error: ${error.message}`,
				);
				expect(releaseEmbedders).toHaveBeenCalledTimes(1);
			},
		);
	}
});

describe("identity loading boundaries", () => {
	beforeEach(() => {
		// A directory at the optional file path reliably fails reads without chmod privileges.
		mkdirSync(identityPath);
	});

	it.each(["telegram", "discord", "whatsapp"])("does not read local identity in a %s TUI", async (bridge) => {
		const runtimeError = new Error("stopped before starting the interactive runtime");
		const runtime = vi.spyOn(AutoRAGAgent.prototype, "createPiInteractiveRuntime").mockRejectedValue(runtimeError);
		const { ctx, stderr } = context([], { [bridge]: true });
		expect(await runTui(ctx)).toBe(1);
		expect(runtime).toHaveBeenCalledTimes(1);
		expect(JSON.parse(stderr[0]).error).toBe(runtimeError.message);
	});

	it("reports an unreadable identity before starting a local TUI", async () => {
		const runtime = vi
			.spyOn(AutoRAGAgent.prototype, "createPiInteractiveRuntime")
			.mockRejectedValue(new Error("unexpected interactive runtime"));
		const { ctx, stderr } = context();
		expect(await runTui(ctx)).toBe(1);
		expect(runtime).not.toHaveBeenCalled();
		expect(JSON.parse(stderr[0]).error).toContain("EISDIR");
	});

	it.each(["refresh", "watch", "index"])("keeps %s model-free when identity is unreadable", async (command) => {
		const refresh = vi.spyOn(AutoRAGAgent.prototype, "refresh").mockResolvedValue({
			scanned: 0,
			written: 0,
			deleted: 0,
			skipped: 0,
			indexPath: join(root, "index.json"),
			diagnostics: [],
		});
		const { ctx, stderr } = context(command === "index" ? ["rebuild"] : [], { once: true, yes: true });
		const run = command === "refresh" ? runRefresh : command === "watch" ? runWatch : runIndex;
		expect(await run(ctx)).toBe(0);
		expect(stderr).toEqual([]);
		expect(refresh).toHaveBeenCalledTimes(1);
	});

	it("keeps status model-free when identity is unreadable", async () => {
		const status = vi.spyOn(AutoRAGAgent.prototype, "getRefreshStatus").mockResolvedValue({
			state: "idle",
			inFlight: false,
			stale: false,
			diagnostics: [],
			components: {},
		});
		const { ctx, stderr } = context();
		expect(await runStatus(ctx)).toBe(0);
		expect(stderr).toEqual([]);
		expect(status).toHaveBeenCalledTimes(1);
	});

	it("constructs the model-free core facade without reading identity", () => {
		expect(() => createAutoRAGLite({ flags: { config: configPath }, cwd: root })).not.toThrow();
	});

	it("constructs profile retrieval tools without reading identity", async () => {
		const config = resolveConfig({ flags: { config: configPath }, cwd: root });
		const toolkit = await createRetrievalToolkit(config, { searchAll: true });
		try {
			expect(toolkit.tools).toHaveLength(1);
		} finally {
			await toolkit.dispose();
		}
	});
});

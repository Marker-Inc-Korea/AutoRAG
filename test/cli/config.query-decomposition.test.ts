import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { ConfigError, resolveConfig, resolveQueryDecompositionModel } from "../../src/cli/config.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-config-decompose-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function configWith(section: unknown) {
	const path = join(root, "config.json");
	writeFileSync(path, JSON.stringify({ queryDecomposition: section }), "utf8");
	return resolveConfig({ flags: { config: path }, env: { HOME: root }, cwd: root });
}

describe("queryDecomposition config", () => {
	it("reads a dedicated decomposition model", () => {
		const config = configWith({ model: { provider: "openrouter", id: "google/gemini-3.5-flash-lite" } });
		expect(config.queryDecomposition).toEqual({
			model: { provider: "openrouter", id: "google/gemini-3.5-flash-lite" },
		});
	});

	it("rejects unknown fields and malformed models", () => {
		expect(() => configWith({ models: {} })).toThrow(ConfigError);
		expect(() => configWith({ model: { provider: "openrouter" } })).toThrow(/queryDecomposition\.model\.id/u);
		expect(() => configWith("gpt")).toThrow(ConfigError);
	});

	it("defaults the decomposition model to OpenRouter qwen/qwen3.7-flash when unset", async () => {
		const resolved = await resolveQueryDecompositionModel(
			{ searchPaths: ["."], workspacePath: root, memoryPath: join(root, "memory.json") },
			{ configPath: join(root, "missing.toml"), agentDir: join(root, "agent"), env: {} },
		);
		expect(resolved?.model).toMatchObject({ provider: "openrouter", id: "qwen/qwen3.7-flash" });
	});

	it("decomposes with the session model when queryDecomposition is false", async () => {
		const resolved = await resolveQueryDecompositionModel(
			{ searchPaths: ["."], workspacePath: root, memoryPath: join(root, "memory.json"), queryDecomposition: false },
			{ configPath: join(root, "missing.toml"), agentDir: join(root, "agent") },
		);
		expect(resolved).toBeUndefined();
	});

	it("resolves the dedicated model and its credential independently of the agent model", async () => {
		const agentDir = join(root, "agent");
		mkdirSync(agentDir, { recursive: true });
		writeFileSync(join(agentDir, "auth.json"), JSON.stringify({ openai: { type: "api_key", key: "sk-decompose" } }));
		const resolved = await resolveQueryDecompositionModel(
			{
				searchPaths: ["."],
				workspacePath: root,
				memoryPath: join(root, "memory.json"),
				model: { provider: "anthropic", id: "claude-sonnet-5" },
				queryDecomposition: { model: { provider: "openai", id: "gpt-5.6-luna" } },
			},
			{ configPath: join(root, "missing.toml"), agentDir },
		);
		expect(resolved?.model).toMatchObject({ provider: "openai", id: "gpt-5.6-luna" });
		expect(resolved?.apiKey).toBe("sk-decompose");
	});
});

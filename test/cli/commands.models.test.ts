import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import { runModels } from "../../src/cli/commands/models.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import { buildStoreEntry, mapServerModels } from "../../src/cloud/models.ts";
import type { ProfileId } from "../../src/embedding-runtime/types.ts";

function context(positionals: string[], flags: CommandContext["flags"] = {}) {
	const stdout: string[] = [];
	const stderr: string[] = [];
	const ctx: CommandContext = {
		positionals,
		flags,
		json: flags.json === true,
		debug: false,
		cwd: "/tmp",
		stdout: (v) => stdout.push(v),
		stderr: (v) => stderr.push(v),
	};
	return { ctx, stdout, stderr };
}

describe("models commands", () => {
	it("prefetches the requested profile and emits JSON", async () => {
		const prefetchModel = vi.fn(
			async (): Promise<{ profileId: ProfileId; path: string }> => ({
				profileId: "qwen3-embedding-0.6b",
				path: "model.gguf",
			}),
		);
		const { ctx, stdout } = context(["prefetch"], { profile: "qwen3-embedding-0.6b", json: true });
		expect(await runModels(ctx, { prefetchModel })).toBe(0);
		expect(prefetchModel).toHaveBeenCalledWith("qwen3-embedding-0.6b");
		expect(JSON.parse(stdout[0]).ok).toBe(true);
	});

	it("rejects an import without a path", async () => {
		const { ctx, stderr } = context(["import"]);
		expect(await runModels(ctx, { importModel: vi.fn() })).toBe(2);
		expect(stderr.join("\n")).toContain("Usage");
	});

	it("returns a failing typed error for a bad verification", async () => {
		const { ctx, stderr } = context(["verify"]);
		expect(
			await runModels(ctx, {
				verifyModel: vi.fn(async () => {
					throw new Error("hash mismatch");
				}),
			}),
		).toBe(1);
		expect(stderr.join("\n")).toContain("hash mismatch");
	});

	it("lists chat models through the injected pi runtime lister as JSON", async () => {
		const listModels = vi.fn(async () => [
			{ provider: "openai", id: "gpt-6-luna", name: "GPT-6 Luna", api: "openai-responses", available: true },
		]);
		const { ctx, stdout } = context(["list"], { json: true });
		expect(await runModels(ctx, { listModels })).toBe(0);
		expect(listModels).toHaveBeenCalledWith({});
		const parsed = JSON.parse(stdout[0]) as { ok: boolean; action: string; count: number; models: unknown[] };
		expect(parsed).toMatchObject({ ok: true, action: "list", count: 1 });
		expect(parsed.models).toHaveLength(1);
	});

	it("forwards --provider and --available to the model lister", async () => {
		const listModels = vi.fn(async () => []);
		const { ctx } = context(["list"], { provider: "openai", available: true });
		expect(await runModels(ctx, { listModels })).toBe(0);
		expect(listModels).toHaveBeenCalledWith({ provider: "openai", available: true });
	});

	it("lists a seeded autorag catalog snapshot from a fresh agent home", async () => {
		const agentDir = mkdtempSync(join(tmpdir(), "autorag-models-"));
		const previousAgentDir = process.env.PI_CODING_AGENT_DIR;
		const previousKey = process.env.AUTORAG_API_KEY;
		process.env.PI_CODING_AGENT_DIR = agentDir;
		process.env.AUTORAG_API_KEY = "dz_test_key";
		try {
			const models = mapServerModels({
				object: "list",
				data: [
					{
						id: "anthropic/claude-haiku-5.5",
						name: "Claude Haiku 5.5",
						context_window: 200_000,
						max_output_tokens: 8_192,
					},
				],
			});
			writeFileSync(
				join(agentDir, "models-store.json"),
				JSON.stringify({ autorag: buildStoreEntry(models, "autorag", "https://api.dazziapp.com/v1") }),
			);

			const { ctx, stdout } = context(["list"], { provider: "autorag", json: true });
			expect(await runModels(ctx)).toBe(0);

			const parsed = JSON.parse(stdout[0]) as { count: number; models: unknown[] };
			expect(parsed.count).toBe(1);
			expect(parsed.models).toEqual([
				{
					provider: "autorag",
					id: "anthropic/claude-haiku-5.5",
					name: "Claude Haiku 5.5",
					api: "openai-responses",
					available: true,
				},
			]);
		} finally {
			if (previousAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
			else process.env.PI_CODING_AGENT_DIR = previousAgentDir;
			if (previousKey === undefined) delete process.env.AUTORAG_API_KEY;
			else process.env.AUTORAG_API_KEY = previousKey;
			rmSync(agentDir, { recursive: true, force: true });
		}
	});
});

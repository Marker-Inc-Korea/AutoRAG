import { describe, expect, it, vi } from "vitest";
import { runModels } from "../../src/cli/commands/models.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
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
			{ provider: "openai", id: "gpt-4o", name: "GPT-4o", api: "openai-responses", available: true },
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
});

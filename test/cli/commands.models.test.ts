import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import { runModels } from "../../src/cli/commands/models.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import { buildStoreEntry, mapServerModels } from "../../src/cloud/models.ts";

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
	it("no longer manages an embedding cache: prefetch, import and verify are usage errors", async () => {
		for (const sub of ["prefetch", "import", "verify"]) {
			const { ctx, stderr } = context([sub]);
			expect(await runModels(ctx)).toBe(2);
			expect(stderr.join("\n")).toContain("Usage");
		}
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

	it("lists a seeded dazzi catalog snapshot, available once signed in", async () => {
		const agentDir = mkdtempSync(join(tmpdir(), "autorag-models-"));
		const previousAgentDir = process.env.PI_CODING_AGENT_DIR;
		process.env.PI_CODING_AGENT_DIR = agentDir;
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
				JSON.stringify({ dazzi: buildStoreEntry(models, "dazzi", "https://api.dazziapp.com/v1") }),
			);
			writeFileSync(
				join(agentDir, "auth.json"),
				JSON.stringify({
					dazzi: { type: "oauth", access: "dz_login", refresh: "", expires: Number.MAX_SAFE_INTEGER },
				}),
			);

			const { ctx, stdout } = context(["list"], { provider: "dazzi", json: true });
			expect(await runModels(ctx)).toBe(0);

			const parsed = JSON.parse(stdout[0]) as { count: number; models: unknown[] };
			expect(parsed.count).toBe(1);
			expect(parsed.models).toEqual([
				{
					provider: "dazzi",
					id: "anthropic/claude-haiku-5.5",
					name: "Claude Haiku 5.5",
					api: "openai-responses",
					available: true,
				},
			]);
		} finally {
			if (previousAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
			else process.env.PI_CODING_AGENT_DIR = previousAgentDir;
			rmSync(agentDir, { recursive: true, force: true });
		}
	});

	it("lists decision models with --decision, GPT-6 Luna and the hosted Jev among them", async () => {
		const agentDir = mkdtempSync(join(tmpdir(), "autorag-models-decision-"));
		const previousAgentDir = process.env.PI_CODING_AGENT_DIR;
		process.env.PI_CODING_AGENT_DIR = agentDir;
		try {
			const { ctx, stdout } = context(["list"], { decision: true, json: true });
			expect(await runModels(ctx)).toBe(0);
			const parsed = JSON.parse(stdout[0]) as { type: string; models: { provider: string; id: string }[] };
			expect(parsed.type).toBe("decision");
			const ids = parsed.models.map((model) => `${model.provider}/${model.id}`);
			expect(ids).toEqual(
				expect.arrayContaining([
					"openai/gpt-6-luna",
					"openrouter/openai/gpt-6-luna-decisions",
					"openrouter/typesafe/jev-1.13",
					"vercel-ai-gateway/liquid/d1",
					"dazzi/jev-1.13",
				]),
			);
			// No chat model is a decision model.
			expect(ids).not.toContain("openrouter/qwen/qwen3.7-flash");
		} finally {
			if (previousAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
			else process.env.PI_CODING_AGENT_DIR = previousAgentDir;
			rmSync(agentDir, { recursive: true, force: true });
		}
	});
});

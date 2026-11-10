import { afterEach, describe, expect, it, vi } from "vitest";
import { liteUsage } from "../../src/cli/commands/lite.ts";
import { runLiteFindWith } from "../../src/cli/commands/lite-find.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import type { CliConfig } from "../../src/cli/config.ts";
import { main } from "../../src/cli/index.ts";
import { answer, fakeRuntime, judgeFrom, result } from "../find/fakes.ts";

function baseConfig(overrides: Partial<CliConfig> = {}): CliConfig {
	return {
		searchPaths: ["/tmp/docs"],
		workspacePath: "/tmp/ws",
		memoryPath: "/tmp/mem.json",
		...overrides,
	};
}

function context(
	positionals: string[],
	flags: CommandContext["flags"] = {},
): { readonly ctx: CommandContext; readonly out: string[]; readonly err: string[] } {
	const out: string[] = [];
	const err: string[] = [];
	const ctx: CommandContext = {
		positionals,
		flags,
		json: flags.json === true,
		debug: false,
		cwd: "/tmp",
		stdout: (line) => out.push(line),
		stderr: (line) => err.push(line),
	};
	return { ctx, out, err };
}

describe("autorag lite find dispatch", () => {
	afterEach(() => {
		vi.restoreAllMocks();
	});

	it("documents the find subcommand in the lite usage text", () => {
		expect(liteUsage()).toContain("autorag lite find <query>");
		expect(liteUsage()).toContain("--groups");
	});

	it("rejects an empty query with exit 2", async () => {
		const err = vi.spyOn(process.stderr, "write").mockReturnValue(true);
		const code = await main(["lite", "find", "--json"]);
		expect(code).toBe(2);
		expect(String(err.mock.calls.at(-1)?.[0] ?? "")).toContain("Usage: autorag lite find");
	});

	it("rejects an invalid --top-k with exit 2 and a JSON error", async () => {
		const err = vi.spyOn(process.stderr, "write").mockReturnValue(true);
		const code = await main(["lite", "find", "query", "--top-k", "0", "--json"]);
		expect(code).toBe(2);
		const parsed = JSON.parse(String(err.mock.calls.at(-1)?.[0] ?? ""));
		expect(parsed.error).toContain("positive integer");
	});
});

describe("runLiteFindWith", () => {
	it("streams NDJSON: groups first, group lines, done last", async () => {
		const { ctx, out } = context(["needle"], { groups: "files,content,source:kakao", json: true });
		const runtime = fakeRuntime({
			datasources: ["kakao"],
			searchSelected: async () =>
				answer({ results: [result({ id: "m1", source: "kakao", content: "a needle here" })] }),
		});
		const code = await runLiteFindWith(ctx, {
			config: baseConfig({ datasources: { kakao: { type: "kakao" } } }),
			runtime,
			judge: judgeFrom([0.9]),
			platform: "darwin",
		});
		expect(code).toBe(0);
		const lines = out.map((line) => JSON.parse(line));
		expect(lines[0]).toMatchObject({
			type: "groups",
			groups: [
				{ key: "files" },
				{ key: "content" },
				{ key: "source:kakao", datasourceId: "kakao", datasourceType: "kakao" },
			],
		});
		expect(lines.at(-1)?.type).toBe("done");
		expect(typeof lines.at(-1)?.elapsedMs).toBe("number");
		const groupKeys = lines
			.filter((line) => line.type === "group")
			.map((line) => line.key)
			.sort();
		expect(groupKeys).toEqual(["content", "files", "source:kakao"]);
	});

	it("narrows to the requested groups", async () => {
		const { ctx, out } = context(["needle"], { groups: "files", json: true });
		await runLiteFindWith(ctx, {
			config: baseConfig(),
			runtime: fakeRuntime(),
			platform: "darwin",
		});
		const lines = out.map((line) => JSON.parse(line));
		expect(lines[0].groups).toEqual([{ key: "files" }]);
		expect(lines.filter((line) => line.type === "group")).toHaveLength(1);
	});

	it("keeps exit 0 when a group errors", async () => {
		const { ctx, out, err } = context(["needle"], { json: true });
		const runtime = fakeRuntime({
			datasources: ["bad"],
			searchSelected: async (_query, selection) => {
				if (selection?.datasourceIds?.[0] === "bad") throw new Error("kakao missing");
				return answer({});
			},
		});
		const code = await runLiteFindWith(ctx, {
			config: baseConfig({ datasources: { bad: { type: "kakao" } } }),
			runtime,
			platform: "darwin",
		});
		expect(code).toBe(0);
		expect(err).toEqual([]);
		const bad = out.map((line) => JSON.parse(line)).find((line) => line.key === "source:bad");
		expect(bad).toMatchObject({ status: "error", error: "kakao missing" });
	});

	it("prints a human summary without --json", async () => {
		const { ctx, out } = context(["needle"], {});
		await runLiteFindWith(ctx, {
			config: baseConfig(),
			runtime: fakeRuntime(),
			platform: "darwin",
		});
		expect(out.join("\n")).toContain("files");
		expect(out.join("\n")).toContain("done");
	});
});

import { spawnSync } from "node:child_process";
import { describe, expect, it } from "vitest";
import { createAutoRAGTool } from "../../src/tool/tool.ts";

const FIXTURE_DIR = "test/fixtures/sample-project";

function runCli(args: string[]): { stdout: string; exitCode: number } {
	const result = spawnSync("node", ["--experimental-strip-types", "src/cli/cli.ts", ...args], {
		encoding: "utf-8",
		cwd: process.cwd(),
	});
	return { stdout: result.stdout ?? "", exitCode: result.status ?? 0 };
}

describe("CLI integration", () => {
	it("CLI search produces results consistent with API", async () => {
		const tool = createAutoRAGTool({ searchPaths: [FIXTURE_DIR] });
		const apiResult = await tool.execute("test", { query: "function", scope: FIXTURE_DIR });

		const { stdout, exitCode } = runCli(["search", "function", "--scope", FIXTURE_DIR]);
		expect(exitCode).toBe(0);
		expect(stdout.length).toBeGreaterThan(0);

		if (apiResult.details.resultCount > 0) {
			expect(stdout).toContain("function");
		}
	});

	it("CLI JSON output matches RetrievalResult schema", () => {
		const { stdout, exitCode } = runCli(["search", "function", "--scope", FIXTURE_DIR, "--format", "json"]);
		expect(exitCode).toBe(0);
		const parsed = JSON.parse(stdout);
		expect(parsed).toHaveProperty("results");
		expect(parsed).toHaveProperty("metadata");
		expect(parsed.metadata).toHaveProperty("resultCount");
		expect(parsed.metadata).toHaveProperty("methodsUsed");
		expect(parsed.metadata).toHaveProperty("elapsedMs");
	});
});

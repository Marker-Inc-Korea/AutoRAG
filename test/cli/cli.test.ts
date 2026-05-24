import { spawnSync } from "node:child_process";
import { describe, expect, it } from "vitest";

const FIXTURE_SCOPE = "test/fixtures/sample-project";

function runCli(args: string): { stdout: string; stderr: string; exitCode: number } {
	const result = spawnSync("node", ["--experimental-strip-types", "src/cli/cli.ts", ...args.split(" ")], {
		encoding: "utf-8",
		cwd: process.cwd(),
	});
	return {
		stdout: result.stdout ?? "",
		stderr: result.stderr ?? "",
		exitCode: result.status ?? 0,
	};
}

describe("autorag CLI", () => {
	it("--help prints usage information", () => {
		const { stdout, exitCode } = runCli("--help");
		expect(exitCode).toBe(0);
		expect(stdout).toContain("autorag");
		expect(stdout).toContain("search");
		expect(stdout).toContain("--scope");
	});

	it("--version prints version", () => {
		const { stdout, exitCode } = runCli("--version");
		expect(exitCode).toBe(0);
		expect(stdout.trim()).toMatch(/^\d+\.\d+\.\d+$/);
	});

	it("search returns results for known content", () => {
		const { stdout, exitCode } = runCli(`search function --scope ${FIXTURE_SCOPE}`);
		expect(exitCode).toBe(0);
		expect(stdout.length).toBeGreaterThan(0);
		expect(stdout).toContain("function");
	});

	it("search with --format json returns valid JSON", () => {
		const { stdout, exitCode } = runCli(`search function --scope ${FIXTURE_SCOPE} --format json`);
		expect(exitCode).toBe(0);
		const parsed = JSON.parse(stdout);
		expect(parsed).toHaveProperty("results");
		expect(parsed).toHaveProperty("metadata");
		expect(Array.isArray(parsed.results)).toBe(true);
	});

	it("search with no results exits 0 with no results message", () => {
		const { stdout, exitCode } = runCli(`search absolutely_nonexistent_xyz_12345 --scope ${FIXTURE_SCOPE}`);
		expect(exitCode).toBe(0);
		expect(stdout).toContain("No results found");
	});

	it("search with --top-k limits results", () => {
		const { stdout, exitCode } = runCli(`search function --scope ${FIXTURE_SCOPE} --top-k 1`);
		expect(exitCode).toBe(0);
		const lines = stdout
			.trim()
			.split("\n")
			.filter((l) => l.length > 0);
		expect(lines.length).toBeLessThanOrEqual(1);
	});
});

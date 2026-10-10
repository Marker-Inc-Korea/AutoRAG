import { describe, expect, it, vi } from "vitest";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import { runUpdateCheck } from "../../src/cli/commands/update-check.ts";
import { AUTORAG_PACKAGE_NAME, AUTORAG_UPDATE_COMMAND, type AutoRAGUpdateResult } from "../../src/cli/update-check.ts";

function context(flags: CommandContext["flags"] = {}) {
	const stdout: string[] = [];
	const stderr: string[] = [];
	const ctx: CommandContext = {
		positionals: [],
		flags,
		json: flags.json === true,
		debug: false,
		cwd: "/tmp",
		stdout: (v) => stdout.push(v),
		stderr: (v) => stderr.push(v),
	};
	return { ctx, stdout, stderr };
}

function result(status: AutoRAGUpdateResult["status"], latestVersion?: string): AutoRAGUpdateResult {
	return {
		status,
		packageName: AUTORAG_PACKAGE_NAME,
		currentVersion: "1.0.0",
		...(latestVersion === undefined ? {} : { latestVersion }),
		installCommand: AUTORAG_UPDATE_COMMAND,
	};
}

describe("update-check command", () => {
	it("prints the update notice and exits 0 when a newer release exists", async () => {
		const { ctx, stdout } = context();
		const check = vi.fn(async () => result("available", "2.0.0"));
		expect(await runUpdateCheck(ctx, { currentVersion: "1.0.0", check })).toBe(0);
		expect(check).toHaveBeenCalledWith("1.0.0");
		expect(stdout.join("\n")).toContain("2.0.0");
		expect(stdout.join("\n")).toContain(AUTORAG_UPDATE_COMMAND);
	});

	it("reports up-to-date and skipped states", async () => {
		const upToDate = context();
		expect(
			await runUpdateCheck(upToDate.ctx, { currentVersion: "1.0.0", check: async () => result("up-to-date") }),
		).toBe(0);
		expect(upToDate.stdout.join("\n")).toContain("up to date");

		const skipped = context();
		expect(await runUpdateCheck(skipped.ctx, { currentVersion: "1.0.0", check: async () => result("skipped") })).toBe(
			0,
		);
		expect(skipped.stdout.join("\n")).toContain("disabled");
	});

	it("emits machine-readable JSON with --json", async () => {
		const { ctx, stdout } = context({ json: true });
		expect(
			await runUpdateCheck(ctx, { currentVersion: "1.0.0", check: async () => result("available", "2.0.0") }),
		).toBe(0);
		const parsed = JSON.parse(stdout[0]) as { ok: boolean; action: string; status: string; latestVersion: string };
		expect(parsed).toMatchObject({ ok: true, action: "update-check", status: "available", latestVersion: "2.0.0" });
	});
});

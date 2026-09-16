import { describe, expect, it, vi } from "vitest";
import { runGateway } from "../../src/cli/commands/gateway.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import type { RuntimeStatus } from "../../src/embedding-runtime/index.ts";

function context(positionals: string[], flags: CommandContext["flags"] = {}) {
	const stdout: string[] = [];
	const stderr: string[] = [];
	const ctx: CommandContext = {
		positionals,
		flags,
		json: flags.json === true || flags.format === "json",
		debug: false,
		cwd: "/tmp",
		stdout: (v) => stdout.push(v),
		stderr: (v) => stderr.push(v),
	};
	return { ctx, stdout, stderr };
}

describe("gateway commands", () => {
	it("reports status in JSON format", async () => {
		const runtimeStatus = vi.fn(
			async (): Promise<RuntimeStatus> => ({
				state: "stopped",
				backend: "auto",
				model: "model.gguf",
				health: { ok: false, code: "unavailable", message: "Gateway is stopped.", retryable: true },
			}),
		);
		const { ctx, stdout } = context(["status"], { format: "json" });
		expect(await runGateway(ctx, { runtimeStatus })).toBe(0);
		expect(JSON.parse(stdout[0]).state).toBe("stopped");
	});

	it("stops the runtime and reports stopped state", async () => {
		const stopRuntime = vi.fn(async () => {});
		const { ctx, stdout } = context(["stop"]);
		expect(await runGateway(ctx, { stopRuntime })).toBe(0);
		expect(stopRuntime).toHaveBeenCalledTimes(1);
		expect(stdout[0]).toContain("stopped");
	});

	it("rejects unsupported gateway start", async () => {
		const { ctx, stderr } = context(["start"]);
		expect(await runGateway(ctx, {})).toBe(2);
		expect(stderr.join("\n")).toContain("status|stop");
	});
});

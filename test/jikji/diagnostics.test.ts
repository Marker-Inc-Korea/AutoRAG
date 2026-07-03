import { describe, expect, it } from "vitest";
import { jikjiPrepareDiagnostic } from "../../src/jikji/diagnostics.ts";
import type { JikjiPrepareResult } from "../../src/jikji/types.ts";

const ABS_BINARY = "/Users/someone/.local/bin/jikji";

describe("jikjiPrepareDiagnostic", () => {
	it("returns undefined for a successful prepare", () => {
		const ok: JikjiPrepareResult = { ok: true, stdout: "{}", stderr: "", code: 0 };
		expect(jikjiPrepareDiagnostic(ok)).toBeUndefined();
	});

	it("maps a spawn-error (missing binary) to jikji-unavailable without leaking the binary path", () => {
		const failed: JikjiPrepareResult = {
			ok: false,
			reason: "spawn-error",
			stdout: "",
			stderr: `spawn ${ABS_BINARY} ENOENT`,
			code: null,
		};
		const diag = jikjiPrepareDiagnostic(failed);
		expect(diag).toBeDefined();
		expect(diag?.code).toBe("jikji-unavailable");
		expect(diag?.severity).toBe("warning");
		expect(diag?.source).toBe("jikji");
		expect(diag?.message).not.toContain(ABS_BINARY);
		expect(diag?.message).not.toContain("/Users/");
	});

	it("maps a timeout to jikji-prepare-failed with a path-free message", () => {
		const failed: JikjiPrepareResult = {
			ok: false,
			reason: "timeout",
			stdout: "",
			stderr: `partial output from ${ABS_BINARY}`,
			code: null,
		};
		const diag = jikjiPrepareDiagnostic(failed);
		expect(diag?.code).toBe("jikji-prepare-failed");
		expect(diag?.message).not.toContain(ABS_BINARY);
		expect(diag?.message.toLowerCase()).toMatch(/timed out|timeout/);
	});

	it("maps a nonzero-exit to jikji-prepare-failed", () => {
		const failed: JikjiPrepareResult = {
			ok: false,
			reason: "nonzero-exit",
			stdout: "",
			stderr: "boom",
			code: 2,
		};
		expect(jikjiPrepareDiagnostic(failed)?.code).toBe("jikji-prepare-failed");
	});
	it("[red-team] never leaks a path for ANY failure reason even with path-laden stderr/stdout", () => {
		const reasons = [
			"aborted",
			"nonzero-exit",
			"spawn-error",
			"stderr-too-large",
			"stdout-too-large",
			"timeout",
		] as const;
		const poison = `/Users/victim/secret/${ABS_BINARY} ~/private C:\\Users\\v\\jikji.exe`;
		for (const reason of reasons) {
			const failed: JikjiPrepareResult = {
				ok: false,
				reason,
				stdout: poison,
				stderr: poison,
				code: null,
			};
			const diag = jikjiPrepareDiagnostic(failed);
			expect(diag, `reason=${reason}`).toBeDefined();
			expect(["jikji-unavailable", "jikji-prepare-failed"]).toContain(diag?.code);
			expect(diag?.severity).toBe("warning");
			expect(diag?.source).toBe("jikji");
			expect(diag?.message).not.toContain("/Users/");
			expect(diag?.message).not.toContain("~/");
			expect(diag?.message).not.toContain("C:\\");
			expect(diag?.message).not.toContain(ABS_BINARY);
		}
	});
});

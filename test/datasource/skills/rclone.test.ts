import { describe, expect, it } from "vitest";
import { RcloneConnector, type RcloneRunResult } from "../../../src/datasource/skills/gdrive/rclone-connector.ts";
import { GDriveSkill } from "../../../src/datasource/skills/gdrive/skill.ts";

const LISTING = JSON.stringify([
	{
		Path: "contracts/vendor.txt",
		Name: "vendor.txt",
		Size: 120,
		MimeType: "text/plain",
		ModTime: "2026-05-20T00:00:00.000Z",
	},
	{ Path: "logo.png", Name: "logo.png", Size: 5000, MimeType: "image/png", ModTime: "2026-05-21T00:00:00.000Z" },
	{ Path: "notes.md", Name: "notes.md", Size: 80, MimeType: "text/markdown", ModTime: "2026-05-22T00:00:00.000Z" },
]);

function runnerFrom(handler: (args: readonly string[]) => RcloneRunResult) {
	return async (args: readonly string[]): Promise<RcloneRunResult> => handler(args);
}

const ok = (stdout: string): RcloneRunResult => ({ ok: true, stdout, stderr: "", code: 0 });

describe("RcloneConnector", () => {
	it("returns not-configured without a remote", async () => {
		expect(await new RcloneConnector({}).fetch()).toMatchObject({ ok: false, reason: "not-configured" });
	});

	it("lists recursively, cats text files only, and builds folder hierarchy", async () => {
		const catted: string[] = [];
		const runner = runnerFrom((args) => {
			if (args[0] === "lsjson") return ok(LISTING);
			if (args[0] === "cat") {
				catted.push(args[1] ?? "");
				return ok(args[1]?.includes("vendor") ? "Contract renews annually with 60-day notice." : "Meeting notes.");
			}
			return { ok: false, stdout: "", stderr: "unexpected", code: 1 };
		});
		const result = await new RcloneConnector({ remote: "gdrive:", runner }).fetch();
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.documents.map((d) => d.docId).sort()).toEqual(["contracts/vendor.txt", "notes.md"]);
			const vendor = result.documents.find((d) => d.docId === "contracts/vendor.txt");
			expect(vendor).toMatchObject({ title: "vendor.txt", hierarchy: ["files", "contracts"] });
			expect(vendor?.content).toContain("renews annually");
			// PNG skipped, reported as count-only warning.
			expect(result.warnings?.some((w) => w.includes("skipped"))).toBe(true);
		}
		expect(catted).toEqual(["gdrive:contracts/vendor.txt", "gdrive:notes.md"]);
	});

	it("classifies config/auth/permission failures without leaking stderr", async () => {
		const cases: [string, string][] = [
			["didn't find section in config file", "not-configured"],
			["failed to refresh oauth token: 401 unauthorized", "auth"],
			["googleapi: Error 403: forbidden", "permission"],
		];
		for (const [stderr, reason] of cases) {
			const runner = runnerFrom(() => ({ ok: false, stdout: "", stderr, code: 1 }));
			const result = await new RcloneConnector({ remote: "gdrive:", runner }).fetch();
			expect(result).toMatchObject({ ok: false, reason });
			if (!result.ok) {
				expect(result.message).not.toContain("config file");
				expect(result.message).not.toContain("401");
			}
		}
	});

	it("degrades per-file cat failures to a count warning", async () => {
		const runner = runnerFrom((args) =>
			args[0] === "lsjson" ? ok(LISTING) : { ok: false, stdout: "", stderr: "read failed", code: 1 },
		);
		const result = await new RcloneConnector({ remote: "gdrive:", runner }).fetch();
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.documents).toHaveLength(0);
			expect(result.warnings?.some((w) => w.includes("failed to read"))).toBe(true);
		}
	});

	it("plugs into GDriveSkill for indexing and opaque-source search", async () => {
		const runner = runnerFrom((args) =>
			args[0] === "lsjson" ? ok(LISTING) : ok("Contract renews annually with 60-day cancellation notice."),
		);
		const skill = new GDriveSkill({
			instanceId: "drive-1",
			connector: new RcloneConnector({ remote: "gdrive:", runner }),
		});
		expect(await skill.index()).toMatchObject({ ok: true, chunkCount: 2 });
		const [method] = skill.retrievalMethods();
		const hits = await method?.retrieve("contract cancellation notice", { topK: 3 });
		expect(hits?.[0]?.source).toMatch(/^\/gdrive\/drive-1\/chunks\//);
	});
});

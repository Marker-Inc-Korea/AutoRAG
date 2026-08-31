import { describe, expect, it } from "vitest";
import { listMailAccounts, listRcloneRemotes } from "../../src/ui/discover.ts";

describe("UI account discovery", () => {
	it("parses rclone remotes without spawning when a runner is injected", async () => {
		const remotes = await listRcloneRemotes(async () => ({ ok: true, stdout: "gdrive:\nwork-drive:\n" }));
		expect(remotes).toEqual([
			{ value: "gdrive:", label: "gdrive:" },
			{ value: "work-drive:", label: "work-drive:" },
		]);
	});

	it("merges himalaya accounts with Gmail/Outlook/iCloud and Other", async () => {
		const accounts = await listMailAccounts(async () => ({ ok: true, stdout: "* work\n  personal\n" }));
		expect(accounts.map((item) => item.value)).toEqual(["work", "personal", "gmail", "outlook", "icloud", "other"]);
	});

	it("returns an empty remote list when rclone is missing", async () => {
		const remotes = await listRcloneRemotes(async () => ({ ok: false, stdout: "" }));
		expect(remotes).toEqual([]);
	});
});

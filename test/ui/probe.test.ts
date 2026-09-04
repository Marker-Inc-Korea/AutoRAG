import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { probeConnection } from "../../src/ui/probe.ts";

let root: string;
let previousToken: string | undefined;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-ui-probe-"));
	previousToken = process.env.GITHUB_TOKEN;
	delete process.env.GITHUB_TOKEN;
});

afterEach(() => {
	if (previousToken === undefined) delete process.env.GITHUB_TOKEN;
	else process.env.GITHUB_TOKEN = previousToken;
	rmSync(root, { recursive: true, force: true });
});

describe("datasource UI probes", () => {
	it("reports a missing GitHub token env without echoing any secret", () => {
		const result = probeConnection(
			{
				alias: "work-github",
				type: "github",
				enabled: true,
				connector: { tokenEnv: "GITHUB_TOKEN", token: "ghp_secret", repos: ["acme/repo"] },
			},
			{ env: { ...process.env, GITHUB_TOKEN: undefined }, pathExists: () => false },
		);
		expect(result.ok).toBe(false);
		expect(result.status).toBe("auth-missing");
		expect(JSON.stringify(result)).not.toContain("ghp_secret");
		expect(result.detail).toContain("GITHUB_TOKEN");
	});

	it("accepts GitHub when the env name is present and repos are set", () => {
		const result = probeConnection(
			{
				alias: "work-github",
				type: "github",
				enabled: true,
				connector: { tokenEnv: "GITHUB_TOKEN", repos: ["acme/repo"] },
			},
			{ env: { GITHUB_TOKEN: "present" } },
		);
		expect(result.ok).toBe(true);
		expect(result.status).toBe("ready");
		expect(JSON.stringify(result)).not.toContain("present");
	});

	it("checks Obsidian vault paths and CLI-backed binaries locally", () => {
		const vault = join(root, "vault");
		mkdirSync(vault);
		writeFileSync(join(vault, "note.md"), "hi");
		const missingBin = probeConnection(
			{
				alias: "notes",
				type: "obsidian",
				enabled: true,
				connector: { vaultPath: vault, binaryPath: join(root, "no-qmd") },
			},
			{ env: process.env },
		);
		expect(missingBin.ok).toBe(false);
		expect(missingBin.status).toBe("binary-missing");

		const okPath = probeConnection(
			{ alias: "notes", type: "obsidian", enabled: true, connector: { vaultPath: vault } },
			{ env: process.env, binaryExists: () => true },
		);
		expect(okPath.ok).toBe(true);

		const missingVault = probeConnection(
			{ alias: "notes", type: "obsidian", enabled: true, connector: { vaultPath: join(root, "missing") } },
			{ env: process.env, binaryExists: () => true },
		);
		expect(missingVault.ok).toBe(false);
		expect(missingVault.status).toBe("path-missing");
	});

	it("requires at least one RSS feed", () => {
		const result = probeConnection(
			{ alias: "news", type: "rss", enabled: true, connector: { feeds: [] } },
			{ env: {} },
		);
		expect(result.ok).toBe(false);
		expect(result.status).toBe("not-configured");
	});

	it("rejects the retired Gmail Himalaya backend", () => {
		const result = probeConnection({
			alias: "legacy-imap",
			type: "gmail",
			enabled: true,
			connector: { backend: "himalaya", account: "personal", folder: "INBOX" },
		});

		expect(result.ok).toBe(false);
		expect(result.status).toBe("not-configured");
		expect(result.detail).toContain("mailcrawl");
	});
});

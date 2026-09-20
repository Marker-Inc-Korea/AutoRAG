/**
 * Tests for the pull-request DCO gate (scripts/ci/check-dco.mjs).
 *
 * The decision logic is exercised directly with fabricated commit records; the git
 * plumbing is exercised against a throwaway repository so the range parsing, the
 * NUL-separated field format, and the process exit codes are covered too.
 */

import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import { type DcoCommit, evaluateCommits, isBotCommit, parseSignOffs } from "../../scripts/ci/check-dco.mjs";

const SCRIPT = fileURLToPath(new URL("../../scripts/ci/check-dco.mjs", import.meta.url));
const SHA = "a".repeat(40);

function commit(overrides: Partial<DcoCommit> = {}): DcoCommit {
	return {
		sha: SHA,
		authorName: "Alice Contributor",
		authorEmail: "alice@example.com",
		committerName: "Alice Contributor",
		committerEmail: "alice@example.com",
		parents: "b".repeat(40),
		message: "feat(agent): add a tool",
		...overrides,
	};
}

function runCheck(base: string, head: string, cwd: string): { status: number; output: string } {
	try {
		const output = execFileSync(process.execPath, [SCRIPT, base, head], {
			cwd,
			encoding: "utf8",
			stdio: ["ignore", "pipe", "pipe"],
		});
		return { status: 0, output };
	} catch (error) {
		const failed = error as { status?: number; stdout?: string; stderr?: string };
		return { status: failed.status ?? -1, output: `${failed.stdout ?? ""}${failed.stderr ?? ""}` };
	}
}

describe("parseSignOffs", () => {
	it("reads trailer lines and lower-cases the email", () => {
		expect(parseSignOffs("Subject\n\nBody\n\nSigned-off-by: Alice <ALICE@Example.com>")).toEqual([
			{ name: "Alice", email: "alice@example.com" },
		]);
	});

	it("ignores prose that merely mentions the trailer", () => {
		expect(parseSignOffs("We require a Signed-off-by: line in every commit.")).toEqual([]);
		expect(parseSignOffs("Signed-off-by Alice <alice@example.com>")).toEqual([]);
	});

	it("reads every sign-off when a commit has several", () => {
		expect(
			parseSignOffs("Subject\n\nSigned-off-by: Alice <alice@example.com>\nSigned-off-by: Bob <bob@example.com>"),
		).toEqual([
			{ name: "Alice", email: "alice@example.com" },
			{ name: "Bob", email: "bob@example.com" },
		]);
	});
});

describe("isBotCommit", () => {
	it("treats app authors as bots", () => {
		expect(
			isBotCommit(
				commit({ authorName: "dependabot[bot]", authorEmail: "49699333+dependabot[bot]@users.noreply.github.com" }),
			),
		).toBe(true);
		expect(isBotCommit(commit())).toBe(false);
	});
});

describe("evaluateCommits", () => {
	it("passes a commit signed off by its author", () => {
		const result = evaluateCommits([commit({ message: "feat: x\n\nSigned-off-by: Alice <alice@example.com>" })]);
		expect(result.failed).toEqual([]);
		expect(result.passed).toHaveLength(1);
		expect(result.passed[0]?.signer?.email).toBe("alice@example.com");
	});

	it("passes when the sign-off matches the committer instead of the author", () => {
		const result = evaluateCommits([
			commit({
				authorName: "Bob",
				authorEmail: "bob@example.com",
				committerName: "Alice Contributor",
				committerEmail: "alice@example.com",
				message: "feat: x\n\nSigned-off-by: Alice <alice@example.com>",
			}),
		]);
		expect(result.failed).toEqual([]);
	});

	it("compares emails case-insensitively", () => {
		const result = evaluateCommits([commit({ message: "feat: x\n\nSigned-off-by: Alice <Alice@Example.COM>" })]);
		expect(result.failed).toEqual([]);
	});

	it("fails a commit with no trailer", () => {
		const result = evaluateCommits([commit()]);
		expect(result.passed).toEqual([]);
		expect(result.failed).toHaveLength(1);
		expect(result.failed[0]?.reason).toBe("no Signed-off-by: trailer");
	});

	it("fails when the signer is someone other than the author", () => {
		const result = evaluateCommits([
			commit({ message: "feat: x\n\nSigned-off-by: Someone Else <someone@example.com>" }),
		]);
		expect(result.passed).toEqual([]);
		expect(result.failed[0]?.reason).toContain("someone@example.com");
		expect(result.failed[0]?.reason).toContain("alice@example.com");
	});

	it("does not accept a co-author trailer as a sign-off", () => {
		const result = evaluateCommits([commit({ message: "feat: x\n\nCo-authored-by: Alice <alice@example.com>" })]);
		expect(result.failed[0]?.reason).toBe("no Signed-off-by: trailer");
	});

	it("exempts merge commits and bot authors", () => {
		const result = evaluateCommits([
			commit({ parents: `${"b".repeat(40)} ${"c".repeat(40)}` }),
			commit({
				sha: "d".repeat(40),
				authorName: "dependabot[bot]",
				authorEmail: "49699333+dependabot[bot]@users.noreply.github.com",
			}),
		]);
		expect(result.passed).toEqual([]);
		expect(result.failed).toEqual([]);
		expect(result.exempt.map((entry) => entry.reason)).toEqual(["merge commit", "bot author (dependabot[bot])"]);
	});
});

describe("check-dco.mjs against a real repository", () => {
	let repo: string | undefined;

	afterEach(() => {
		if (repo) rmSync(repo, { recursive: true, force: true });
		repo = undefined;
	});

	function initRepo(): string {
		repo = mkdtempSync(join(tmpdir(), "autorag-dco-test-"));
		const git = (...args: string[]) =>
			execFileSync("git", args, { cwd: repo, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
		git("init", "-q");
		git("config", "user.name", "Alice Contributor");
		git("config", "user.email", "alice@example.com");
		git("config", "commit.gpgsign", "false");
		return repo;
	}

	function commitEmpty(message: string): string {
		execFileSync("git", ["commit", "-q", "--allow-empty", "-m", message], {
			cwd: repo,
			encoding: "utf8",
			stdio: ["ignore", "pipe", "pipe"],
		});
		return execFileSync("git", ["rev-parse", "HEAD"], { cwd: repo, encoding: "utf8" }).trim();
	}

	it("accepts a range whose commits are all signed off", () => {
		const root = initRepo();
		const base = commitEmpty("chore: base");
		const head = commitEmpty("feat: x\n\nSigned-off-by: Alice Contributor <alice@example.com>");

		const result = runCheck(base, head, root);

		expect(result.output).toContain("all 1 non-exempt commit(s) carry a matching Signed-off-by: trailer.");
		expect(result.status).toBe(0);
	});

	it("reports the offending commit verbatim and exits 1", () => {
		const root = initRepo();
		const base = commitEmpty("chore: base");
		const head = commitEmpty("feat: x");

		const result = runCheck(base, head, root);

		expect(result.status).toBe(1);
		expect(result.output).toContain(head);
		expect(result.output).toContain("no Signed-off-by: trailer");
		expect(result.output).toContain("git rebase --signoff");
	});

	it("exits 2 when the range cannot be read", () => {
		const root = initRepo();
		commitEmpty("chore: base");

		const result = runCheck("0".repeat(40), "1".repeat(40), root);

		expect(result.status).toBe(2);
		expect(result.output).toContain("could not read");
	});
});

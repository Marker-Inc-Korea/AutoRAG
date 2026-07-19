import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { SourceRoot } from "../../src/filesystem/source-paths.ts";
import { normalizeJikjiAnswerPath, parseJikjiAnswerPack } from "../../src/jikji/answer-pack.ts";

let tmp: string;

beforeEach(() => {
	tmp = mkdtempSync(join(tmpdir(), "autorag-answer-pack-"));
});

afterEach(() => {
	rmSync(tmp, { recursive: true, force: true });
});

function root(path: string): SourceRoot & { rootPath: string } {
	return { rootPath: path, prefix: `/${path.split("/").pop()}` };
}

function validAnswerPack(overrides: Record<string, unknown> = {}): string {
	return JSON.stringify({
		answer_paths: ["/repo/src/a.ts"],
		paths: ["/repo/src/a.ts", "/repo/src/b.ts"],
		candidates: [
			{ path: "/repo/src/a.ts", next_read: "cache", label: "A", score: 0.9 },
			{ path: "/repo/src/b.ts", next_read: "wiki" },
		],
		evidence_pack: [
			{ path: "/repo/src/a.ts", next_read: "cache" },
			{ path: "/repo/src/b.ts", next_read: "original" },
		],
		handoff_action: "direct_use",
		tool_call_policy: {
			stop_after_find: true,
			forbidden_tools: ["bash"],
			allowed_followups: ["jikji_find"],
		},
		agent_should_not_rerank: true,
		...overrides,
	});
}

describe("parseJikjiAnswerPack", () => {
	it("accepts a valid answer-pack with all fields", () => {
		const pack = parseJikjiAnswerPack(validAnswerPack());
		expect(pack).toBeDefined();
		expect(pack?.answerPaths).toEqual(["/repo/src/a.ts"]);
		expect(pack?.paths).toEqual(["/repo/src/a.ts", "/repo/src/b.ts"]);
		expect(pack?.candidates).toHaveLength(2);
		expect(pack?.candidates[0]).toMatchObject({
			path: "/repo/src/a.ts",
			nextRead: "cache",
			label: "A",
			score: 0.9,
		});
		expect(pack?.candidates[1]).toMatchObject({
			path: "/repo/src/b.ts",
			nextRead: "wiki",
		});
		expect(pack?.candidates[1]?.label).toBeUndefined();
		expect(pack?.candidates[1]?.score).toBeUndefined();
		expect(pack?.evidencePack).toHaveLength(2);
		expect(pack?.evidencePack[1]).toEqual({ path: "/repo/src/b.ts", nextRead: "original" });
		expect(pack?.handoffAction).toBe("direct_use");
		expect(pack?.toolCallPolicy).toEqual({
			stopAfterFind: true,
			forbiddenTools: ["bash"],
			allowedFollowups: ["jikji_find"],
		});
		expect(pack?.agentShouldNotRerank).toBe(true);
	});

	it("accepts all three handoff_action values", () => {
		for (const action of ["direct_use", "jikji_retry", "raw_fallback_after_retry"] as const) {
			const pack = parseJikjiAnswerPack(validAnswerPack({ handoff_action: action }));
			expect(pack?.handoffAction, `action=${action}`).toBe(action);
		}
	});

	it("accepts all next_read enum values on candidates", () => {
		for (const nextRead of ["cache", "wiki", "original", "none"] as const) {
			const pack = parseJikjiAnswerPack(
				validAnswerPack({
					candidates: [{ path: "/x", next_read: nextRead }],
					evidence_pack: [{ path: "/x", next_read: nextRead }],
				}),
			);
			expect(pack?.candidates[0]?.nextRead, `nextRead=${nextRead}`).toBe(nextRead);
			expect(pack?.evidencePack[0]?.nextRead, `nextRead=${nextRead}`).toBe(nextRead);
		}
	});

	it("tolerates extra fields", () => {
		const pack = parseJikjiAnswerPack(validAnswerPack({ extra_top: 42, nested: { foo: "bar" } }));
		expect(pack).toBeDefined();
		expect(pack?.answerPaths).toEqual(["/repo/src/a.ts"]);
	});

	it("rejects missing answer_paths", () => {
		const pack = parseJikjiAnswerPack(
			JSON.stringify({
				paths: [],
				candidates: [],
				evidence_pack: [],
				handoff_action: "direct_use",
				tool_call_policy: { stop_after_find: false, forbidden_tools: [], allowed_followups: [] },
				agent_should_not_rerank: false,
			}),
		);
		expect(pack).toBeUndefined();
	});

	it("rejects missing handoff_action", () => {
		const pack = parseJikjiAnswerPack(
			JSON.stringify({
				answer_paths: [],
				paths: [],
				candidates: [],
				evidence_pack: [],
				tool_call_policy: { stop_after_find: false, forbidden_tools: [], allowed_followups: [] },
				agent_should_not_rerank: false,
			}),
		);
		expect(pack).toBeUndefined();
	});

	it("rejects missing tool_call_policy", () => {
		const pack = parseJikjiAnswerPack(
			JSON.stringify({
				answer_paths: [],
				paths: [],
				candidates: [],
				evidence_pack: [],
				handoff_action: "direct_use",
				agent_should_not_rerank: false,
			}),
		);
		expect(pack).toBeUndefined();
	});

	it("rejects an invalid next_read enum value", () => {
		const pack = parseJikjiAnswerPack(
			validAnswerPack({
				candidates: [{ path: "/x", next_read: "banana" }],
			}),
		);
		expect(pack).toBeUndefined();
	});

	it("rejects an invalid handoff_action value", () => {
		const pack = parseJikjiAnswerPack(validAnswerPack({ handoff_action: "nope" }));
		expect(pack).toBeUndefined();
	});

	it("rejects a non-boolean agent_should_not_rerank", () => {
		const pack = parseJikjiAnswerPack(validAnswerPack({ agent_should_not_rerank: "yes" }));
		expect(pack).toBeUndefined();
	});

	it("rejects malformed JSON", () => {
		expect(parseJikjiAnswerPack("not json")).toBeUndefined();
		expect(parseJikjiAnswerPack("{")).toBeUndefined();
		expect(parseJikjiAnswerPack("")).toBeUndefined();
	});

	it("rejects a JSON array or primitive (not an object)", () => {
		expect(parseJikjiAnswerPack("[]")).toBeUndefined();
		expect(parseJikjiAnswerPack('"hello"')).toBeUndefined();
		expect(parseJikjiAnswerPack("42")).toBeUndefined();
		expect(parseJikjiAnswerPack("null")).toBeUndefined();
	});

	it("rejects wrong-typed answer_paths", () => {
		expect(parseJikjiAnswerPack(validAnswerPack({ answer_paths: "not-an-array" }))).toBeUndefined();
		expect(parseJikjiAnswerPack(validAnswerPack({ answer_paths: [1, 2, 3] }))).toBeUndefined();
	});

	it("rejects wrong-typed candidate.path", () => {
		expect(parseJikjiAnswerPack(validAnswerPack({ candidates: [{ path: 42, next_read: "cache" }] }))).toBeUndefined();
	});

	it("rejects wrong-typed tool_call_policy.stop_after_find", () => {
		expect(
			parseJikjiAnswerPack(
				validAnswerPack({
					tool_call_policy: {
						stop_after_find: "yes",
						forbidden_tools: [],
						allowed_followups: [],
					},
				}),
			),
		).toBeUndefined();
	});

	it("allows optional label and score to be absent", () => {
		const pack = parseJikjiAnswerPack(
			validAnswerPack({
				candidates: [{ path: "/x", next_read: "none" }],
			}),
		);
		expect(pack?.candidates[0]?.label).toBeUndefined();
		expect(pack?.candidates[0]?.score).toBeUndefined();
	});
});

describe("normalizeJikjiAnswerPath", () => {
	let repoRoot: string;
	let subDir: string;
	let outsideDir: string;

	beforeEach(() => {
		repoRoot = join(tmp, "repo");
		subDir = join(repoRoot, "src");
		outsideDir = join(tmp, "outside");
		mkdirSync(subDir, { recursive: true });
		mkdirSync(outsideDir, { recursive: true });
		writeFileSync(join(subDir, "a.ts"), "export const a = 1;");
		writeFileSync(join(outsideDir, "secret.txt"), "secret");
	});

	it("accepts an absolute path within a configured root", () => {
		const roots = [root(repoRoot)];
		const result = normalizeJikjiAnswerPath(join(subDir, "a.ts"), roots);
		expect(result).toBe(join(subDir, "a.ts"));
	});

	it("accepts a relative path scoped to a root", () => {
		const roots = [root(repoRoot)];
		const result = normalizeJikjiAnswerPath("src/a.ts", roots);
		expect(result).toBe(join(subDir, "a.ts"));
	});

	it("accepts a relative path at the root itself", () => {
		writeFileSync(join(repoRoot, "README.md"), "readme");
		const roots = [root(repoRoot)];
		const result = normalizeJikjiAnswerPath("README.md", roots);
		expect(result).toBe(join(repoRoot, "README.md"));
	});

	it("rejects a root escape via relative traversal", () => {
		const roots = [root(repoRoot)];
		const result = normalizeJikjiAnswerPath("../outside/secret.txt", roots);
		expect(result).toBeUndefined();
	});

	it("rejects an absolute path outside all roots", () => {
		const roots = [root(repoRoot)];
		const result = normalizeJikjiAnswerPath(join(outsideDir, "secret.txt"), roots);
		expect(result).toBeUndefined();
	});

	it("rejects Windows drive-letter paths", () => {
		const roots = [root(repoRoot)];
		expect(normalizeJikjiAnswerPath("C:\\Users\\victim\\secret", roots)).toBeUndefined();
		expect(normalizeJikjiAnswerPath("C:/Users/victim/secret", roots)).toBeUndefined();
	});

	it("rejects URL-like values", () => {
		const roots = [root(repoRoot)];
		expect(normalizeJikjiAnswerPath("https://evil.example.com/x", roots)).toBeUndefined();
		expect(normalizeJikjiAnswerPath("file:///etc/passwd", roots)).toBeUndefined();
		expect(normalizeJikjiAnswerPath("file:foo", roots)).toBeUndefined();
	});

	it("rejects empty or whitespace-only input", () => {
		const roots = [root(repoRoot)];
		expect(normalizeJikjiAnswerPath("", roots)).toBeUndefined();
		expect(normalizeJikjiAnswerPath("   ", roots)).toBeUndefined();
	});

	it("dedupes across duplicate roots (same realpath)", () => {
		const roots = [root(repoRoot), root(repoRoot)];
		const result = normalizeJikjiAnswerPath("src/a.ts", roots);
		expect(result).toBe(join(subDir, "a.ts"));
	});

	it("resolves against the first matching root when multiple roots exist", () => {
		const otherRoot = join(tmp, "other");
		mkdirSync(join(otherRoot, "src"), { recursive: true });
		writeFileSync(join(otherRoot, "src", "a.ts"), "other");
		const roots = [root(repoRoot), root(otherRoot)];
		const result = normalizeJikjiAnswerPath("src/a.ts", roots);
		expect(result).toBe(join(repoRoot, "src", "a.ts"));
	});

	it("follows a symlink that stays within the root", () => {
		const linkTarget = join(repoRoot, "linked.ts");
		writeFileSync(linkTarget, "linked");
		const symlinkPath = join(subDir, "link.ts");
		symlinkSync(linkTarget, symlinkPath);
		const roots = [root(repoRoot)];
		const result = normalizeJikjiAnswerPath("src/link.ts", roots);
		expect(result).toBe(linkTarget);
	});

	it("rejects a symlink that escapes the root", () => {
		const symlinkPath = join(subDir, "escape.ts");
		symlinkSync(join(outsideDir, "secret.txt"), symlinkPath);
		const roots = [root(repoRoot)];
		const result = normalizeJikjiAnswerPath("src/escape.ts", roots);
		expect(result).toBeUndefined();
	});

	it("returns undefined when no roots are configured", () => {
		expect(normalizeJikjiAnswerPath("src/a.ts", [])).toBeUndefined();
	});

	it("returns undefined when the path does not exist on disk", () => {
		const roots = [root(repoRoot)];
		expect(normalizeJikjiAnswerPath("src/nonexistent.ts", roots)).toBeUndefined();
	});
});

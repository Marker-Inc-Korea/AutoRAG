import { describe, expect, it } from "vitest";
import { readDevLabel, resolveClonePath } from "../src/main/dev-label";

function readerFor(files: Readonly<Record<string, string>>) {
	return (path: string): string => {
		const value = files[path];
		if (value === undefined) throw new Error(`ENOENT: ${path}`);
		return value;
	};
}

const CLONE = "/clones/autorag";

describe("resolveClonePath", () => {
	it("takes the parent of the app directory", () => {
		expect(resolveClonePath("/clones/autorag/app")).toBe(CLONE);
	});
});

describe("readDevLabel", () => {
	it("reads the branch and its loose ref commit", () => {
		const label = readDevLabel(
			CLONE,
			readerFor({
				[`${CLONE}/.git/HEAD`]: "ref: refs/heads/feat/ai-finder-dev-label\n",
				[`${CLONE}/.git/refs/heads/feat/ai-finder-dev-label`]: "612ae5f1a2b3c4d5e6f708192a3b4c5d6e7f8091\n",
			}),
		);
		expect(label).toEqual({ clonePath: CLONE, branch: "feat/ai-finder-dev-label", commit: "612ae5f" });
	});

	it("falls back to packed-refs when the loose ref is absent", () => {
		const label = readDevLabel(
			CLONE,
			readerFor({
				[`${CLONE}/.git/HEAD`]: "ref: refs/heads/main\n",
				[`${CLONE}/.git/packed-refs`]:
					"# pack-refs with: peeled fully-peeled sorted\n0bfb85ac1234567890abcdef1234567890abcdef refs/heads/main\n",
			}),
		);
		expect(label).toEqual({ clonePath: CLONE, branch: "main", commit: "0bfb85a" });
	});

	it("reports a detached HEAD with a null branch", () => {
		const label = readDevLabel(
			CLONE,
			readerFor({ [`${CLONE}/.git/HEAD`]: "612ae5f1a2b3c4d5e6f708192a3b4c5d6e7f8091\n" }),
		);
		expect(label).toEqual({ clonePath: CLONE, branch: null, commit: "612ae5f" });
	});

	it("degrades to the bare path when there is no git metadata", () => {
		expect(readDevLabel(CLONE, readerFor({}))).toEqual({ clonePath: CLONE, branch: null, commit: null });
	});

	it("keeps the branch when only the commit is unknown", () => {
		const label = readDevLabel(CLONE, readerFor({ [`${CLONE}/.git/HEAD`]: "ref: refs/heads/feature\n" }));
		expect(label).toEqual({ clonePath: CLONE, branch: "feature", commit: null });
	});
});

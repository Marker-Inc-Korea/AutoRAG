import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { linkCitations, openableTarget } from "../../src/agent/citation-links.ts";

// On Windows a root-relative path resolves onto the current drive, so its file URL carries `/D:`.
const drive = process.platform === "win32" ? `/${resolve("/").slice(0, 2)}` : "";

const sources: Record<string, string> = {
	e1: "/data/reports/q3 (final).pdf",
	e2: "https://example.com/a?b=1",
	e3: "kakao:room-12/msg-9",
};
const sourceOf = (id: string): string | undefined => sources[id];
const noFiles = (): boolean => false;

describe("openableTarget", () => {
	it("maps absolute paths to file URLs and keeps http(s) URLs", () => {
		expect(openableTarget("/a b/[최종].pdf")).toBe(`file://${drive}/a%20b/%5B%EC%B5%9C%EC%A2%85%5D.pdf`);
		expect(openableTarget("https://example.com/x")).toBe("https://example.com/x");
	});

	it("returns undefined for sources that are not a path or URL", () => {
		expect(openableTarget("kakao:room-12/msg-9")).toBeUndefined();
		expect(openableTarget("relative/file.txt")).toBeUndefined();
	});
});

describe("linkCitations", () => {
	it("links an evidence id to its recorded source and keeps the brackets", () => {
		expect(linkCitations("Cap is 28% [e1].", sourceOf, noFiles)).toBe(
			`Cap is 28% [[e1](<file://${drive}/data/reports/q3%20(final).pdf>)].`,
		);
	});

	it("links each id of a multi-id marker and leaves unopenable ones plain", () => {
		expect(linkCitations("x [e2, e3, e9]", sourceOf, noFiles)).toBe("x [[e2](<https://example.com/a?b=1>), e3, e9]");
	});

	it("leaves a marker with nothing openable exactly as written", () => {
		expect(linkCitations("see [e3] and [e9] and [4]", sourceOf, noFiles)).toBe("see [e3] and [e9] and [4]");
	});

	it("turns a real [file:] marker into a link labelled with the file name", () => {
		const path = "/tmp/x/[final] a.txt";
		expect(linkCitations(`see [file:${path}] here`, sourceOf, (candidate) => candidate === path)).toBe(
			`see [[\\[final\\] a\\.txt](<file://${drive}/tmp/x/%5Bfinal%5D%20a.txt>)] here`,
		);
	});

	it("does not touch markdown links or image embeds", () => {
		expect(linkCitations("[e1](http://x) ![p](</p/[e1].png>)", sourceOf, noFiles)).toBe(
			"[e1](http://x) ![p](</p/[e1].png>)",
		);
	});
});

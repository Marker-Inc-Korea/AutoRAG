import { describe, expect, it } from "vitest";
import { copyName, splitName, validateName } from "../src/renderer/src/state/collision";

describe("splitName", () => {
	it("splits the extension off a file name", () => {
		expect(splitName("report.pdf")).toEqual({ stem: "report", ext: ".pdf" });
		expect(splitName("archive.tar.gz")).toEqual({ stem: "archive.tar", ext: ".gz" });
	});

	it("treats a dotfile and a folder as extensionless", () => {
		expect(splitName(".env")).toEqual({ stem: ".env", ext: "" });
		expect(splitName("Finance")).toEqual({ stem: "Finance", ext: "" });
	});
});

describe("copyName", () => {
	it("appends the Finder copy suffix before the extension", () => {
		expect(copyName("report.pdf", ["report.pdf"])).toBe("report copy.pdf");
		expect(copyName("Finance", ["Finance"])).toBe("Finance copy");
	});

	it("numbers further copies", () => {
		expect(copyName("report.pdf", ["report.pdf", "report copy.pdf"])).toBe("report copy 2.pdf");
		expect(copyName("report.pdf", ["report.pdf", "report copy.pdf", "report copy 2.pdf"])).toBe("report copy 3.pdf");
	});

	it("keeps the name when nothing collides", () => {
		expect(copyName("report.pdf", ["other.pdf"])).toBe("report.pdf");
	});
});

describe("validateName", () => {
	const siblings = ["report.pdf", "Finance"];

	it("accepts a fresh name", () => {
		expect(validateName("draft.pdf", siblings)).toEqual({ ok: true });
	});

	it("accepts the unchanged current name", () => {
		expect(validateName("report.pdf", siblings, "report.pdf")).toEqual({ ok: true });
	});

	it("rejects an empty name", () => {
		expect(validateName("   ", siblings)).toEqual({ ok: false, message: "이름을 입력해 주세요" });
	});

	it("rejects a path separator", () => {
		expect(validateName("a/b.pdf", siblings)).toEqual({ ok: false, message: "이름에 / 는 쓸 수 없습니다" });
	});

	it("rejects a name already used in this folder", () => {
		expect(validateName("Finance", siblings)).toEqual({ ok: false, message: "같은 이름의 항목이 이미 있습니다" });
	});
});

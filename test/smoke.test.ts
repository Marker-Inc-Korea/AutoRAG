import { existsSync } from "node:fs";
import { describe, expect, it } from "vitest";

describe("test fixtures", () => {
	it("sample-project/src/main.ts exists", () => {
		expect(existsSync("test/fixtures/sample-project/src/main.ts")).toBe(true);
	});

	it("sample-project/src/utils.ts exists", () => {
		expect(existsSync("test/fixtures/sample-project/src/utils.ts")).toBe(true);
	});

	it("sample-project/README.md exists", () => {
		expect(existsSync("test/fixtures/sample-project/README.md")).toBe(true);
	});

	it("sample-project/data/notes.txt exists", () => {
		expect(existsSync("test/fixtures/sample-project/data/notes.txt")).toBe(true);
	});

	it("sample-project/.hidden-file exists", () => {
		expect(existsSync("test/fixtures/sample-project/.hidden-file")).toBe(true);
	});
});

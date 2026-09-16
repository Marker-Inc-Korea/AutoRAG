import { describe, expect, it } from "vitest";
import { isFilesystemAbsolutePath } from "../../src/filesystem/source-paths.ts";

describe("isFilesystemAbsolutePath", () => {
	it("accepts POSIX absolute paths", () => {
		expect(isFilesystemAbsolutePath("/Shared Docs/guide.md")).toBe(true);
		expect(isFilesystemAbsolutePath("/")).toBe(true);
	});

	it("accepts Windows drive absolute paths regardless of host OS", () => {
		expect(isFilesystemAbsolutePath("C:\\docs\\notes.md")).toBe(true);
		expect(isFilesystemAbsolutePath("D:/shared/file.txt")).toBe(true);
	});

	it("accepts Windows UNC absolute paths", () => {
		expect(isFilesystemAbsolutePath("\\\\server\\share\\file.txt")).toBe(true);
	});

	it("rejects relative and drive-relative paths", () => {
		expect(isFilesystemAbsolutePath("docs/notes.md")).toBe(false);
		expect(isFilesystemAbsolutePath("docs\\notes.md")).toBe(false);
		expect(isFilesystemAbsolutePath("C:partial")).toBe(false);
		expect(isFilesystemAbsolutePath("")).toBe(false);
	});
});

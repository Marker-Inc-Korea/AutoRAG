import { describe, expect, it } from "vitest";
import { previewCommandForPlatform } from "../src/main/fs-service";

describe("previewCommandForPlatform", () => {
	const target = "/tmp/autorag-preview.txt";

	it("uses native Quick Look on macOS", () => {
		expect(previewCommandForPlatform("darwin", target)).toEqual({
			command: "/usr/bin/qlmanage",
			args: ["-p", target],
		});
	});

	it("opens the selected path with Explorer on Windows", () => {
		expect(previewCommandForPlatform("win32", target)).toEqual({
			command: "explorer.exe",
			args: [target],
		});
	});

	it("uses the desktop opener on Linux", () => {
		expect(previewCommandForPlatform("linux", target)).toEqual({
			command: "xdg-open",
			args: [target],
		});
	});
});

import { describe, expect, it } from "vitest";
import config from "../../vitest.config.ts";

describe("Vitest Windows configuration", () => {
	it("serializes test files on Windows to avoid Bun fs-event fork crashes", () => {
		expect(config.test?.fileParallelism).toBe(process.platform !== "win32");
	});
});

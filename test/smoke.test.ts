import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { AutoRAGAgent, buildSystemPrompt, createEmitResultsTool } from "../src/index.ts";

describe("dependency hygiene", () => {
	it("declares no required Python or pip dependency", () => {
		const pkg = JSON.parse(readFileSync("package.json", "utf8")) as {
			dependencies?: Record<string, string>;
			devDependencies?: Record<string, string>;
			peerDependencies?: Record<string, string>;
		};
		const names = [
			...Object.keys(pkg.dependencies ?? {}),
			...Object.keys(pkg.devDependencies ?? {}),
			...Object.keys(pkg.peerDependencies ?? {}),
		];
		for (const name of names) {
			expect(name).not.toMatch(/python|^pip$|pyodide|pyright-python/i);
		}
	});
});

describe("test fixtures", () => {
	it("exports the public library API", () => {
		expect(AutoRAGAgent).toBeDefined();
		expect(typeof buildSystemPrompt).toBe("function");
		expect(typeof createEmitResultsTool).toBe("function");
	});

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

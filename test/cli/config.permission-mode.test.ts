import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { buildAgentOptions, ConfigError, resolveConfig } from "../../src/cli/config.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-config-permission-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function resolve(options: { file?: unknown; flags?: Record<string, string | boolean>; env?: NodeJS.ProcessEnv }) {
	const path = join(root, "config.json");
	writeFileSync(path, JSON.stringify(options.file === undefined ? {} : { permissionMode: options.file }), "utf8");
	return resolveConfig({
		flags: { config: path, ...options.flags },
		env: { HOME: root, ...options.env },
		cwd: root,
	});
}

describe("permission mode resolution", () => {
	it("defaults to auto", () => {
		expect(resolve({}).permissionMode).toBe("auto");
	});

	it("reads the config file", () => {
		expect(resolve({ file: "auto" }).permissionMode).toBe("auto");
	});

	it("prefers the environment over the file, and the flag over both", () => {
		expect(resolve({ file: "ask", env: { AUTORAG_PERMISSION_MODE: "full" } }).permissionMode).toBe("full");
		expect(
			resolve({
				file: "ask",
				env: { AUTORAG_PERMISSION_MODE: "full" },
				flags: { "permission-mode": "auto" },
			}).permissionMode,
		).toBe("auto");
	});

	it("accepts any letter case", () => {
		expect(resolve({ flags: { "permission-mode": "FULL" } }).permissionMode).toBe("full");
	});

	it("rejects an unknown mode from every source, naming the valid ones", () => {
		expect(() => resolve({ file: "yolo" })).toThrow(ConfigError);
		expect(() => resolve({ file: 3 })).toThrow(/permissionMode/u);
		expect(() => resolve({ env: { AUTORAG_PERMISSION_MODE: "yolo" } })).toThrow(/ask, auto, full/u);
		expect(() => resolve({ flags: { "permission-mode": "yolo" } })).toThrow(/--permission-mode/u);
	});

	it("hands the mode to the agent", () => {
		expect(buildAgentOptions(resolve({ flags: { "permission-mode": "auto" } })).permissionMode).toBe("auto");
	});
});

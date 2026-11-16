import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { AutoRAGLite } from "../../src/core.ts";
import { configSignature, createReloadingLite } from "../../src/mcp/reloading-lite.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-reload-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

/** A stand-in runtime whose behaviour depends on the "config" it was built from. */
function fakeRuntime(label: string): AutoRAGLite {
	const self = {
		label,
		config: { searchPaths: [label] },
		refresh() {
			return `${this.label}:refreshed`;
		},
	};
	return self as unknown as AutoRAGLite;
}

describe("createReloadingLite", () => {
	it("builds the runtime once and keeps it while the config is unchanged", () => {
		let built = 0;
		const lite = createReloadingLite({
			create: () => {
				built += 1;
				return fakeRuntime(`v${built}`);
			},
			signature: () => "a",
		});
		expect(lite.config.searchPaths).toEqual(["v1"]);
		expect(lite.config.searchPaths).toEqual(["v1"]);
		expect(built).toBe(1);
	});

	it("rebuilds when the config appears after the server started", () => {
		let signature = "absent";
		const lite = createReloadingLite({
			create: () => fakeRuntime(signature === "absent" ? "empty" : "indexed"),
			signature: () => signature,
		});
		expect(lite.config.searchPaths).toEqual(["empty"]);
		signature = "written";
		expect(lite.config.searchPaths).toEqual(["indexed"]);
	});

	it("rebuilds when the config is edited, and only once per edit", () => {
		let signature = "1";
		let built = 0;
		const lite = createReloadingLite({
			create: () => fakeRuntime(`build-${++built}`),
			signature: () => signature,
		});
		expect(lite.config.searchPaths).toEqual(["build-1"]);
		signature = "2";
		expect(lite.config.searchPaths).toEqual(["build-2"]);
		expect(lite.config.searchPaths).toEqual(["build-2"]);
		expect(built).toBe(2);
	});

	it("calls methods on the current runtime, not the one it started with", () => {
		let signature = "1";
		let built = 0;
		const lite = createReloadingLite({
			create: () => fakeRuntime(`r${++built}`),
			signature: () => signature,
		});
		expect(lite.refresh()).toBe("r1:refreshed");
		signature = "2";
		expect(lite.refresh()).toBe("r2:refreshed");
	});

	it("keeps the previous runtime when the new config cannot be built, and recovers once it can", () => {
		let signature = "good";
		let broken = false;
		let built = 0;
		const errors: string[] = [];
		const lite = createReloadingLite({
			create: () => {
				if (broken) throw new Error("Failed to parse config file");
				return fakeRuntime(`ok-${++built}`);
			},
			signature: () => signature,
			onError: (error: Error) => errors.push(error.message),
		});
		expect(lite.config.searchPaths).toEqual(["ok-1"]);

		broken = true;
		signature = "half-written";
		expect(lite.config.searchPaths).toEqual(["ok-1"]);
		expect(errors).toEqual(["Failed to parse config file"]);

		broken = false;
		signature = "fixed";
		expect(lite.config.searchPaths).toEqual(["ok-2"]);
	});

	it("reports a broken config once, not on every access", () => {
		let signature = "good";
		let broken = false;
		const errors: string[] = [];
		const lite = createReloadingLite({
			create: () => {
				if (broken) throw new Error("bad");
				return fakeRuntime("ok");
			},
			signature: () => signature,
			onError: (error: Error) => errors.push(error.message),
		});
		void lite.config;
		broken = true;
		signature = "bad";
		void lite.config;
		void lite.config;
		void lite.config;
		expect(errors).toHaveLength(1);
	});

	it("builds eagerly so a bad first config fails at start like before", () => {
		expect(() =>
			createReloadingLite({
				create: () => {
					throw new Error("Config file not found");
				},
				signature: () => "x",
			}),
		).toThrow("Config file not found");
	});
});

describe("configSignature", () => {
	it("changes when the config file is created, edited and removed", () => {
		const path = join(root, "config.json");
		const env = { AUTORAG_CONFIG: path };
		const absent = configSignature(env);
		writeFileSync(path, JSON.stringify({ searchPaths: ["a"] }));
		const first = configSignature(env);
		writeFileSync(path, JSON.stringify({ searchPaths: ["a", "bb"] }));
		const second = configSignature(env);
		rmSync(path);
		const removed = configSignature(env);

		expect(new Set([absent, first, second]).size).toBe(3);
		expect(removed).toBe(absent);
	});

	it("follows AUTORAG_HOME when no explicit config is named", () => {
		const env = { AUTORAG_HOME: root };
		const before = configSignature(env);
		writeFileSync(join(root, "config.json"), "{}");
		expect(configSignature(env)).not.toBe(before);
	});
});

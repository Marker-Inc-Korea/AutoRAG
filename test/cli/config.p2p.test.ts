import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { ConfigError, resolveConfig } from "../../src/cli/config.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-p2p-config-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function writeConfig(extra: Record<string, unknown>): string {
	const path = join(root, "config.json");
	mkdirSync(root, { recursive: true });
	writeFileSync(
		path,
		JSON.stringify({
			searchPaths: ["."],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			...extra,
		}),
	);
	return path;
}

describe("P2pConfig normalization", () => {
	it("defaults when p2p key absent (injectionClassifier true, piiNer false, searchTimeoutMs 120000)", () => {
		const path = writeConfig({});
		const config = resolveConfig({ flags: { config: path }, cwd: root, env: {} });
		expect(config.p2p).toBeDefined();
		expect(config.p2p!.enabled).toBe(false);
		expect(config.p2p!.injectionClassifier).toBe(true);
		expect(config.p2p!.piiNer).toBe(false);
		expect(config.p2p!.searchTimeoutMs).toBe(120000);
	});

	it("sets enabled to true when explicitly true in config", () => {
		const path = writeConfig({ p2p: { enabled: true } });
		const config = resolveConfig({ flags: { config: path }, cwd: root, env: {} });
		expect(config.p2p!.enabled).toBe(true);
	});

	it("accepts valid port and host", () => {
		const path = writeConfig({ p2p: { enabled: true, port: 9470, host: "0.0.0.0" } });
		const config = resolveConfig({ flags: { config: path }, cwd: root, env: {} });
		expect(config.p2p!.port).toBe(9470);
		expect(config.p2p!.host).toBe("0.0.0.0");
	});

	it("rejects port out of range 1-65535 (port 70000)", () => {
		const path = writeConfig({ p2p: { port: 70000 } });
		expect(() => resolveConfig({ flags: { config: path }, cwd: root, env: {} })).toThrow(ConfigError);
	});

	it("rejects port 0", () => {
		const path = writeConfig({ p2p: { port: 0 } });
		expect(() => resolveConfig({ flags: { config: path }, cwd: root, env: {} })).toThrow(ConfigError);
	});

	it("clamps quota values to positive integers (rejects zero)", () => {
		const path = writeConfig({ p2p: { quotas: { queriesPerHour: 0 } } });
		expect(() => resolveConfig({ flags: { config: path }, cwd: root, env: {} })).toThrow(ConfigError);
	});

	it("clamps quota values to positive integers (rejects negative)", () => {
		const path = writeConfig({ p2p: { quotas: { queriesPerHour: -1 } } });
		expect(() => resolveConfig({ flags: { config: path }, cwd: root, env: {} })).toThrow(ConfigError);
	});

	it("rejects searchTimeoutMs below 5000", () => {
		const path = writeConfig({ p2p: { searchTimeoutMs: 1000 } });
		expect(() => resolveConfig({ flags: { config: path }, cwd: root, env: {} })).toThrow(ConfigError);
	});

	it("rejects searchTimeoutMs above 600000", () => {
		const path = writeConfig({ p2p: { searchTimeoutMs: 700000 } });
		expect(() => resolveConfig({ flags: { config: path }, cwd: root, env: {} })).toThrow(ConfigError);
	});

	it("accepts searchTimeoutMs at boundary values 5000 and 600000", () => {
		const path1 = writeConfig({ p2p: { searchTimeoutMs: 5000 } });
		const config1 = resolveConfig({ flags: { config: path1 }, cwd: root, env: {} });
		expect(config1.p2p!.searchTimeoutMs).toBe(5000);

		const path2 = writeConfig({ p2p: { searchTimeoutMs: 600000 } });
		const config2 = resolveConfig({ flags: { config: path2 }, cwd: root, env: {} });
		expect(config2.p2p!.searchTimeoutMs).toBe(600000);
	});

	it("rejects unknown keys with typed error", () => {
		const path = writeConfig({ p2p: { unknownKey: "whatever" } });
		expect(() => resolveConfig({ flags: { config: path }, cwd: root, env: {} })).toThrow(ConfigError);
	});

	it("rejects non-boolean injectionClassifier", () => {
		const path = writeConfig({ p2p: { injectionClassifier: "yes" } });
		expect(() => resolveConfig({ flags: { config: path }, cwd: root, env: {} })).toThrow(ConfigError);
	});

	it("rejects non-boolean piiNer", () => {
		const path = writeConfig({ p2p: { piiNer: "yes" } });
		expect(() => resolveConfig({ flags: { config: path }, cwd: root, env: {} })).toThrow(ConfigError);
	});

	it("rejects non-integer port", () => {
		const path = writeConfig({ p2p: { port: "abc" } });
		expect(() => resolveConfig({ flags: { config: path }, cwd: root, env: {} })).toThrow(ConfigError);
	});

	it("disabled by default", () => {
		const path = writeConfig({});
		const config = resolveConfig({ flags: { config: path }, cwd: root, env: {} });
		expect(config.p2p!.enabled).toBe(false);
	});

	it("existing config.json without p2p key still loads correctly (no crash)", () => {
		const path = writeConfig({ ui: { host: "localhost" } });
		const config = resolveConfig({ flags: { config: path }, cwd: root, env: {} });
		expect(config.ui).toBeDefined();
		expect(config.ui!.host).toBe("localhost");
		expect(config.p2p).toBeDefined();
		expect(config.p2p!.enabled).toBe(false);
	});

	it("accepts optional policy as Record<string,unknown>", () => {
		const path = writeConfig({ p2p: { policy: { "docs/**": { tier: "always" } } } });
		const config = resolveConfig({ flags: { config: path }, cwd: root, env: {} });
		expect(config.p2p!.policy).toEqual({ "docs/**": { tier: "always" } });
	});

	it("accepts optional quotas object", () => {
		const path = writeConfig({ p2p: { quotas: { queriesPerHour: 10, burst: 3 } } });
		const config = resolveConfig({ flags: { config: path }, cwd: root, env: {} });
		expect(config.p2p!.quotas).toEqual({ queriesPerHour: 10, burst: 3 });
	});
});

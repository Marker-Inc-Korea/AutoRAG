import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { buildAgentOptions, resolveConfig, writeDefaultConfig } from "../../src/cli/config.ts";
import { main } from "../../src/cli/index.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-p2p-removed-"));
});

afterEach(() => {
	vi.restoreAllMocks();
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

describe("P2P/SimpleX surface is removed", () => {
	it.each(["serve", "p2p"])("`autorag %s` is an unknown command", async (command) => {
		const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
		const stdout = vi.spyOn(process.stdout, "write").mockReturnValue(true);
		expect(await main([command])).toBe(2);
		expect(stderr.mock.calls.map((call) => String(call[0])).join("")).toContain(`Unknown command: ${command}`);
		stdout.mockRestore();
	});

	it("does not list serve or p2p in the global usage", async () => {
		const stdout = vi.spyOn(process.stdout, "write").mockReturnValue(true);
		expect(await main(["--help"])).toBe(0);
		const text = stdout.mock.calls.map((call) => String(call[0])).join("");
		expect(text).not.toMatch(/simplex|p2p|autorag serve/i);
	});
});

describe("legacy `p2p` config key", () => {
	// Existing user configs (including ~/.autorag/config.json) still carry
	// `p2p: { "enabled": false }`. Removing the feature must not break them.
	it.each([
		["the default block", { enabled: false }],
		["a fully populated block", { enabled: true, port: 9470, quotas: { burst: 2 }, policy: {}, piiNer: true }],
		["an unrecognized field", { somethingNew: 1 }],
		["a non-object value", "off"],
	])("is ignored when it is %s", (_label, p2p) => {
		const path = writeConfig({ p2p });
		const config = resolveConfig({ flags: { config: path }, cwd: root, env: {} });
		expect(config).not.toHaveProperty("p2p");
		expect(buildAgentOptions(config)).not.toHaveProperty("peerQuery");
	});

	it("is not written into new configs", () => {
		const path = join(root, "written.json");
		writeDefaultConfig(path, { workspacePath: root }, { cwd: root, env: {} });
		const written = JSON.parse(readFileSync(path, "utf8")) as Record<string, unknown>;
		expect(written).not.toHaveProperty("p2p");
	});

	it("does not stop a config that carries it from loading the rest of its settings", () => {
		const path = writeConfig({ p2p: { enabled: false }, limits: { mergedEvidenceCeiling: 1000 } });
		const config = resolveConfig({ flags: { config: path }, cwd: root, env: {} });
		expect(config.limits?.mergedEvidenceCeiling).toBe(1000);
	});
});

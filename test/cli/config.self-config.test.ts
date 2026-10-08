import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { buildAgentOptions, resolveConfig } from "../../src/cli/config.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-self-config-cli-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

describe("agent self-configuration wiring", () => {
	it("points the agent at the config file the CLI actually resolved (--config)", () => {
		const path = join(root, "custom.json");
		writeFileSync(path, JSON.stringify({ searchPaths: ["."], workspacePath: root }));
		const config = resolveConfig({ flags: { config: path }, cwd: root, env: {} });
		expect(buildAgentOptions(config).selfConfig).toMatchObject({ configPath: path, validate: expect.any(Function) });
	});

	it("points the agent at AUTORAG_CONFIG when the environment selects the file", () => {
		const path = join(root, "from-env.json");
		writeFileSync(path, JSON.stringify({ searchPaths: ["."], workspacePath: root }));
		const config = resolveConfig({ flags: {}, cwd: root, env: { AUTORAG_CONFIG: path } });
		expect(buildAgentOptions(config).selfConfig).toMatchObject({ configPath: path, validate: expect.any(Function) });
	});

	it("falls back to $AUTORAG_HOME/config.json for the implicit home config", () => {
		const home = join(root, "home");
		const config = resolveConfig({ flags: {}, cwd: root, env: { AUTORAG_HOME: home }, readOnly: true });
		expect(buildAgentOptions(config).selfConfig).toMatchObject({
			configPath: join(home, "config.json"),
			validate: expect.any(Function),
		});
	});
});

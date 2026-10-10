import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { validateAgentConfigFile } from "../../src/cli/config.ts";

let root: string;
let agentDir: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-self-config-validate-"));
	agentDir = join(root, "agent");
	mkdirSync(agentDir, { recursive: true });
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function writeConfig(model: unknown): string {
	const path = join(root, "config.json");
	writeFileSync(
		path,
		JSON.stringify({ searchPaths: ["."], workspacePath: root, memoryPath: join(root, "m.json"), model }),
	);
	return path;
}

describe("validateAgentConfigFile", () => {
	it("accepts a config whose model resolves in the pi catalog", async () => {
		const path = writeConfig({ provider: "openai", id: "gpt-6-luna" });
		expect(
			await validateAgentConfigFile(path, { agentDir, configPath: join(root, "none.toml"), env: {} }),
		).toBeUndefined();
	});

	it("reports a model id the catalog does not know and no endpoint declares", async () => {
		const path = writeConfig({ provider: "openrouter", id: "glm-5.3-flash" });
		const problem = await validateAgentConfigFile(path, { agentDir, configPath: join(root, "none.toml"), env: {} });
		expect(problem).toMatch(/glm-5\.3-flash/u);
	});

	it("reports a config file that is not valid JSON", async () => {
		const path = join(root, "config.json");
		writeFileSync(path, "{ not json");
		expect(await validateAgentConfigFile(path, { agentDir, configPath: join(root, "none.toml"), env: {} })).toMatch(
			/\S/u,
		);
	});

	it("does not treat a missing credential as a broken config", async () => {
		const path = writeConfig({ provider: "openai", id: "gpt-6-luna" });
		expect(
			await validateAgentConfigFile(path, { agentDir, configPath: join(root, "none.toml"), env: {} }),
		).toBeUndefined();
	});
});

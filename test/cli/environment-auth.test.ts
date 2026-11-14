import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { resolveAgentModel } from "../../src/cli/config.ts";
import { readDazziLoginToken } from "../../src/cloud/credential.ts";

let root: string;
beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-environment-auth-"));
	vi.stubEnv("AUTORAG_AUTH_MODE", "environment");
});
afterEach(() => {
	vi.unstubAllEnvs();
	rmSync(root, { recursive: true, force: true });
});

it("does not reuse a stored Dazzi subscription login", () => {
	writeFileSync(join(root, "auth.json"), JSON.stringify({ dazzi: { type: "api_key", key: "foreign-login" } }));
	expect(readDazziLoginToken(root)).toBeUndefined();
	vi.stubEnv("AUTORAG_AUTH_MODE", "");
	expect(readDazziLoginToken(root)).toBe("foreign-login");
});

it("ignores foreign model overrides and stored credentials while accepting an explicit environment key", async () => {
	writeFileSync(join(root, "models.json"), "invalid foreign config");
	mkdirSync(join(root, "auth.json"));
	const resolved = await resolveAgentModel(
		{
			searchPaths: [root],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			model: {
				provider: "explicit",
				id: "gpt-6-luna",
				baseUrl: "http://127.0.0.1:19876/v1",
				apiKeyEnv: "EXPLICIT_KEY",
			},
		},
		{ agentDir: root, env: { EXPLICIT_KEY: "injected-key" } },
	);
	expect(resolved.apiKey).toBe("injected-key");
	expect(resolved.model.baseUrl).toBe("http://127.0.0.1:19876/v1");
});

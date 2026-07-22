import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { loadLocalAutoRAGModels } from "../../src/subagents/local-models.ts";

describe("local AutoRAG model configuration", () => {
	it("builds sol and luna models from a Codex provider configuration", () => {
		const dir = mkdtempSync(join(tmpdir(), "autorag-models-"));
		const configPath = join(dir, "config.toml");
		writeFileSync(
			configPath,
			'model = "gpt-5.6-sol"\nmodel_provider = "test-proxy"\n\n[model_providers.test-proxy]\nbase_url = "https://proxy.example/v1"\nwire_api = "responses"\nenv_key = "TEST_PROXY_KEY"\n',
		);
		const models = loadLocalAutoRAGModels({ configPath, env: { TEST_PROXY_KEY: "secret" } });
		expect(models.orchestrator.id).toBe("gpt-5.6-sol");
		expect(models.orchestrator.name).toBe("GPT-5.6 Sol");
		expect(models.explorer.id).toBe("gpt-5.6-luna");
		expect(models.explorer.name).toBe("GPT-5.6 Luna");
		expect(models.orchestrator.baseUrl).toBe("https://proxy.example/v1");
		expect(models.provider).toBe("test-proxy");
		expect(models.apiKey).toBe("secret");
	});

	it("uses custom role model ids as their display names", () => {
		const dir = mkdtempSync(join(tmpdir(), "autorag-models-"));
		const configPath = join(dir, "config.toml");
		writeFileSync(
			configPath,
			'model_provider = "test-proxy"\n[model_providers.test-proxy]\nbase_url = "https://proxy.example/v1"\nwire_api = "responses"\nenv_key = "TEST_PROXY_KEY"\n',
		);
		const models = loadLocalAutoRAGModels({
			configPath,
			env: { TEST_PROXY_KEY: "secret" },
			orchestratorModelId: "custom-orchestrator",
			explorerModelId: "custom-explorer",
		});

		expect(models.orchestrator.name).toBe("custom-orchestrator");
		expect(models.explorer.name).toBe("custom-explorer");
	});

	it("fails when the local provider is not Responses-compatible", () => {
		const dir = mkdtempSync(join(tmpdir(), "autorag-models-"));
		const configPath = join(dir, "config.toml");
		writeFileSync(
			configPath,
			'model_provider = "test-proxy"\n[model_providers.test-proxy]\nbase_url = "https://proxy.example/v1"\nwire_api = "chat"\nenv_key = "TEST_PROXY_KEY"\n',
		);
		expect(() => loadLocalAutoRAGModels({ configPath, env: { TEST_PROXY_KEY: "secret" } })).toThrow(/responses/i);
	});
});

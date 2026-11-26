import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	buildAgentOptions,
	type CliConfig,
	ConfigError,
	normalizeVisualParseConfig,
	resolveConfig,
} from "../../src/cli/config.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-config-visual-parse-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function baseConfig(visualParse?: CliConfig["visualParse"]): CliConfig {
	return {
		searchPaths: ["."],
		workspacePath: root,
		memoryPath: join(root, "memory.json"),
		minSync: { autoInstall: false },
		...(visualParse === undefined ? {} : { visualParse }),
	};
}

describe("visualParse CLI config", () => {
	it("normalizes a full VLM visualParse config", () => {
		const normalized = normalizeVisualParseConfig({
			enabled: true,
			gate: { backend: "bivia", model: "dec-1", threshold: 0.65 },
			extractor: {
				kind: "vlm",
				model: { provider: "openrouter", id: "google/gemini-3.5-flash-lite", name: "Gemini 3.5 Flash Lite" },
				timeoutMs: 45_000,
			},
			maxPagesPerFile: 40,
			maxWidthPx: 1280,
		});
		expect(normalized).toEqual({
			enabled: true,
			gate: { backend: "bivia", model: "dec-1", threshold: 0.65 },
			extractor: {
				kind: "vlm",
				model: { provider: "openrouter", id: "google/gemini-3.5-flash-lite", name: "Gemini 3.5 Flash Lite" },
				timeoutMs: 45_000,
			},
			maxPagesPerFile: 40,
			maxWidthPx: 1280,
		});
	});

	it("normalizes an OCR visualParse config without requiring a VLM model", () => {
		const normalized = normalizeVisualParseConfig({
			enabled: true,
			extractor: { kind: "ocr", languages: ["ko", "en"] },
		});
		expect(normalized).toEqual({
			enabled: true,
			extractor: { kind: "ocr", languages: ["ko", "en"] },
		});
	});

	it("accepts false and disabled objects", () => {
		expect(normalizeVisualParseConfig(false)).toBe(false);
		expect(normalizeVisualParseConfig({ enabled: false })).toEqual({ enabled: false });
	});

	it("rejects invalid thresholds, unknown extractor kinds, and VLM without a model", () => {
		expect(() => normalizeVisualParseConfig({ enabled: true, gate: { threshold: 1.5 } })).toThrow(ConfigError);
		expect(() => normalizeVisualParseConfig({ enabled: true, gate: { threshold: 0 } })).toThrow(ConfigError);
		expect(() => normalizeVisualParseConfig({ enabled: true, extractor: { kind: "magic" } })).toThrow(ConfigError);
		expect(() => normalizeVisualParseConfig({ enabled: true, extractor: { kind: "vlm" } })).toThrow(ConfigError);
		expect(() => normalizeVisualParseConfig({ enabled: true, maxPagesPerFile: 0 })).toThrow(ConfigError);
	});

	it("resolves visualParse from config.json and forwards it through buildAgentOptions", () => {
		const configPath = join(root, "config.json");
		writeFileSync(
			configPath,
			JSON.stringify({
				searchPaths: [root],
				workspacePath: root,
				memoryPath: join(root, "memory.json"),
				visualParse: {
					enabled: true,
					gate: { backend: "openrouter", model: "typesafe/jev-1.14", threshold: 0.6 },
					extractor: {
						kind: "vlm",
						model: { provider: "openrouter", id: "google/gemini-3.5-flash-lite" },
					},
				},
			}),
		);
		const resolved = resolveConfig({ flags: { config: configPath } });
		expect(resolved.visualParse).toMatchObject({
			enabled: true,
			gate: { backend: "openrouter", model: "typesafe/jev-1.14", threshold: 0.6 },
			extractor: {
				kind: "vlm",
				model: { provider: "openrouter", id: "google/gemini-3.5-flash-lite" },
			},
		});
		const opts = buildAgentOptions(resolved);
		expect(opts.visualParse).toMatchObject({
			enabled: true,
			gate: { backend: "openrouter", model: "typesafe/jev-1.14", threshold: 0.6 },
		});
		expect(buildAgentOptions(baseConfig(false)).visualParse).toBe(false);
	});
});

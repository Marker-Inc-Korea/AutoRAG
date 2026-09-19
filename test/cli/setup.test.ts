import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { runSetup } from "../../src/cli/setup.ts";

const roots: string[] = [];
afterEach(() => {
	for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});
function fakeLock() {
	return { path: "setup.lock", contents: "", assertOwned() {}, release() {} };
}

describe("setup orchestration", () => {
	it("keeps probes independent, preserves operator embedder config, and degrades to BM25", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-setup-"));
		roots.push(root);
		const configPath = join(root, "config.json");
		writeFileSync(
			configPath,
			JSON.stringify({
				searchPaths: ["/docs"],
				workspacePath: root,
				minSync: {
					embedder: { id: "operator-model", baseUrl: "https://remote.invalid/v1", apiKeyEnv: "SECRET_KEY" },
				},
				datasources: { discord: { connector: { binaryPath: "/bin/sh" } } },
			}),
		);
		const report = await runSetup({
			configPath,
			workspacePath: root,
			deps: {
				env: { PATH: "/bin", HOME: root },
				executable: (name) => (name === "/bin/sh" ? name : undefined),
				pathExists: () => true,
				acquireLock: fakeLock,
				runtime: {
					runtimeStatus: async () => {
						throw new Error("health /Users/alice/.secret/token=abc");
					},
					verifyModel: async () => {
						throw new Error("offline");
					},
				},
			},
		});
		expect(report.mode).toBe("bm25");
		expect(report.datasources.find((d) => d.name === "discord")?.state).toBe("blocked");
		expect(JSON.stringify(report)).not.toContain("/Users/alice");
		const config = JSON.parse(readFileSync(configPath, "utf8"));
		expect(config.minSync.embedder).toEqual({
			id: "operator-model",
			baseUrl: "https://remote.invalid/v1",
			apiKeyEnv: "SECRET_KEY",
		});
	});

	it("reports semantic mode when the runtime answers a live health probe", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-setup-"));
		roots.push(root);
		const configPath = join(root, "config.json");
		writeFileSync(configPath, JSON.stringify({ searchPaths: [join(root, "docs")], workspacePath: root }));
		const report = await runSetup({
			configPath,
			workspacePath: root,
			deps: {
				env: { PATH: "/bin", HOME: root },
				executable: () => undefined,
				pathExists: () => true,
				acquireLock: fakeLock,
				runtime: {
					runtimeStatus: async () => ({
						state: "ready",
						backend: "auto",
						model: "Qwen3-Embedding-0.6B-Q8_0.gguf",
						health: { ok: true, profileId: "qwen3-embedding-0.6b", dimension: 1024, runtimeBuild: "b10951" },
					}),
					verifyModel: async () => ({
						profileId: "qwen3-embedding-0.6b",
						path: "/cache/model.gguf",
						hash: "sha",
					}),
				},
			},
		});
		expect(report.mode).toBe("semantic");
		expect(report.ok).toBe(true);
		expect(report.remediation).toBeUndefined();
		expect(report.runtime).toEqual({
			state: "ready",
			health: true,
			model: "Qwen3-Embedding-0.6B-Q8_0.gguf",
		});
	});

	it("surfaces a runtime startup failure instead of discarding it", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-setup-"));
		roots.push(root);
		const configPath = join(root, "config.json");
		writeFileSync(configPath, JSON.stringify({ searchPaths: [join(root, "docs")], workspacePath: root }));
		const report = await runSetup({
			configPath,
			workspacePath: root,
			deps: {
				env: { PATH: "/bin", HOME: root },
				executable: () => undefined,
				pathExists: () => true,
				acquireLock: fakeLock,
				runtime: {
					runtimeStatus: async () => ({
						state: "stopped",
						backend: "auto",
						model: "Qwen3-Embedding-0.6B-Q8_0.gguf",
						health: { ok: false, code: "unavailable", message: "Gateway is stopped.", retryable: true },
					}),
					verifyModel: async () => ({
						profileId: "qwen3-embedding-0.6b",
						path: "/cache/model.gguf",
						hash: "sha",
					}),
					ensureRuntime: async () => {
						throw new Error("spawn failed at /Users/alice/secret token=abc");
					},
				},
			},
		});
		expect(report.mode).toBe("bm25");
		expect(report.runtime.reason).toContain("spawn failed");
		expect(JSON.stringify(report)).not.toContain("/Users/alice");
	});
});

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
});

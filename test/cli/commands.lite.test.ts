import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { main } from "../../src/cli/index.ts";

function writeConfig(root: string, configPath: string, model = false): void {
	const docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	writeFileSync(join(docs, "note.md"), "Lifecycle fixture\n");
	writeFileSync(
		configPath,
		JSON.stringify({
			searchPaths: [docs],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			...(model ? { model: { provider: "invalid-provider", id: "invalid-model" } } : {}),
			minSync: false,
			jikji: false,
		}),
	);
}

describe("autorag lite lifecycle dispatch", () => {
	afterEach(() => {
		vi.restoreAllMocks();
	});

	it("routes init and status without model resolution", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-lite-lifecycle-"));
		const configPath = join(root, "config.json");
		const docs = join(root, "docs");
		mkdirSync(docs, { recursive: true });
		writeFileSync(join(docs, "note.md"), "Lifecycle fixture\n");
		const out = vi.spyOn(process.stdout, "write").mockReturnValue(true);
		try {
			expect(await main(["lite", "init", "--config", configPath, "--search-paths", docs])).toBe(0);
			writeConfig(root, configPath, true);
			expect(await main(["lite", "status", "--config", configPath, "--json"])).toBe(0);
			expect(String(out.mock.calls.at(-1)?.[0] ?? "")).toContain('"state"');
		} finally {
			rmSync(root, { recursive: true, force: true });
		}
	});

	it("returns exit 2 for lite config resolution failures", async () => {
		const err = vi.spyOn(process.stderr, "write").mockReturnValue(true);
		const code = await main([
			"lite",
			"retrieve",
			"query",
			"--config",
			"/definitely/missing/autorag-config.json",
			"--json",
		]);
		expect(code).toBe(2);
		expect(String(err.mock.calls.at(-1)?.[0] ?? "")).toContain("Config file not found");
	});

	it("rejects fractional top-k values instead of truncating them", async () => {
		const err = vi.spyOn(process.stderr, "write").mockReturnValue(true);
		const code = await main(["lite", "retrieve", "query", "--top-k", "1.5", "--json"]);
		expect(code).toBe(2);
		expect(String(err.mock.calls.at(-1)?.[0] ?? "")).toContain("positive integer");
	});

	it("does not treat a datasource-only refresh as a parsed index refresh", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-lite-readiness-"));
		const configPath = join(root, "config.json");
		writeConfig(root, configPath);
		const out = vi.spyOn(process.stdout, "write").mockReturnValue(true);
		try {
			expect(await main(["lite", "refresh", "--method", "datasources", "--config", configPath, "--json"])).toBe(0);
			expect(await main(["lite", "retrieve", "query", "--config", configPath, "--json"])).toBe(2);
			expect(String(out.mock.calls.at(-1)?.[0] ?? "")).toContain('"index-not-ready"');
		} finally {
			rmSync(root, { recursive: true, force: true });
		}
	});

	it("rejects retrieval after the parsed index becomes stale", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-lite-stale-"));
		const configPath = join(root, "config.json");
		writeConfig(root, configPath);
		const out = vi.spyOn(process.stdout, "write").mockReturnValue(true);
		try {
			expect(await main(["lite", "refresh", "--config", configPath, "--json"])).toBe(0);
			writeFileSync(join(root, "docs", "note.md"), "Changed after refresh\n");
			expect(await main(["lite", "retrieve", "query", "--config", configPath, "--json"])).toBe(2);
			expect(String(out.mock.calls.at(-1)?.[0] ?? "")).toContain("Index is stale");
		} finally {
			rmSync(root, { recursive: true, force: true });
		}
	});

	it("routes refresh, watch, and index with path-opaque output", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-lite-lifecycle-"));
		const configPath = join(root, "config.json");
		writeConfig(root, configPath);
		const out = vi.spyOn(process.stdout, "write").mockReturnValue(true);
		try {
			expect(await main(["lite", "refresh", "--config", configPath, "--json"])).toBe(0);
			expect(await main(["lite", "watch", "--once", "--config", configPath, "--json"])).toBe(0);
			mkdirSync(join(root, ".autorag", "parsed"), { recursive: true });
			writeFileSync(join(root, ".autorag", "parsed", "index.json"), "{}\n");
			expect(await main(["lite", "index", "reset", "--yes", "--config", configPath, "--json"])).toBe(0);
			expect(existsSync(join(root, ".autorag", "parsed"))).toBe(false);
			expect(out.mock.calls.map((call) => String(call[0] ?? "")).join("\n")).not.toContain("indexPath");
		} finally {
			rmSync(root, { recursive: true, force: true });
		}
	});

	it("routes UI, duplicates, and model-free health", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-lite-lifecycle-"));
		const configPath = join(root, "config.json");
		writeConfig(root, configPath, true);
		const out = vi.spyOn(process.stdout, "write").mockReturnValue(true);
		const err = vi.spyOn(process.stderr, "write").mockReturnValue(true);
		try {
			expect(await main(["lite", "ui", "--config", configPath, "--host", "0.0.0.0", "--no-open"])).toBe(2);
			expect(String(err.mock.calls.at(-1)?.[0] ?? "")).toContain("loopback");
			expect(await main(["lite", "duplicates", "--config", join(root, "missing.json"), "--json"])).toBe(2);
			expect(String(err.mock.calls.at(-1)?.[0] ?? "")).toContain("Config file not found");
			expect(await main(["lite", "health", "--config", configPath, "--json"])).toBe(0);
			const health = String(out.mock.calls.at(-1)?.[0] ?? "");
			expect(health).toContain('"state"');
			expect(health).not.toContain("model_resolution");
		} finally {
			rmSync(root, { recursive: true, force: true });
		}
	});
});

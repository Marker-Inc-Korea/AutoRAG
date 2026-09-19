import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
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

	it("reports staleness without blocking retrieval, and --strict keeps the hard failure", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-lite-stale-"));
		const configPath = join(root, "config.json");
		writeConfig(root, configPath);
		const out = vi.spyOn(process.stdout, "write").mockReturnValue(true);
		try {
			expect(await main(["lite", "refresh", "--config", configPath, "--json"])).toBe(0);
			writeFileSync(join(root, "docs", "note.md"), "Changed after refresh\n");

			// Default: answer from the index and report the staleness it is answering past.
			expect(await main(["lite", "retrieve", "query", "--config", configPath, "--json"])).toBe(0);
			const envelope = JSON.parse(String(out.mock.calls.at(-1)?.[0] ?? ""));
			expect(envelope).toMatchObject({ ok: true, query: "query", stale: true });
			expect(envelope.diagnostics).toContainEqual({
				code: "stale-index",
				severity: "warning",
				message: expect.any(String),
				source: "/docs/note.md",
				reason: "mtime-and-size-changed",
				action: "refresh",
			});

			// --strict keeps the previous fail-closed contract for callers that need it.
			expect(await main(["lite", "retrieve", "query", "--config", configPath, "--strict", "--json"])).toBe(2);
			const strict = JSON.parse(String(out.mock.calls.at(-1)?.[0] ?? ""));
			expect(strict.ok).toBe(false);
			expect(strict.diagnostics[0]).toMatchObject({
				code: "index-not-ready",
				source: "/docs/note.md",
				reason: "mtime-and-size-changed",
				action: "refresh",
			});
		} finally {
			rmSync(root, { recursive: true, force: true });
		}
	});

	it("refreshes first on request and then reports the corpus current", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-lite-refresh-first-"));
		const configPath = join(root, "config.json");
		writeConfig(root, configPath);
		const out = vi.spyOn(process.stdout, "write").mockReturnValue(true);
		try {
			expect(await main(["lite", "refresh", "--config", configPath, "--json"])).toBe(0);
			writeFileSync(join(root, "docs", "note.md"), "Changed after refresh\n");

			expect(await main(["lite", "retrieve", "query", "--config", configPath, "--refresh", "--json"])).toBe(0);
			const envelope = JSON.parse(String(out.mock.calls.at(-1)?.[0] ?? ""));
			expect(envelope).toMatchObject({ ok: true, stale: false });
			expect(envelope.diagnostics).not.toContainEqual(expect.objectContaining({ code: "stale-index" }));
		} finally {
			rmSync(root, { recursive: true, force: true });
		}
	});

	it("reports a clean corpus as current, including a product artifact in the root", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-lite-clean-"));
		const configPath = join(root, "config.json");
		writeConfig(root, configPath);
		writeFileSync(join(root, "docs", ".jikji_agent_map.md"), "# Jikji Agent Map\n");
		const out = vi.spyOn(process.stdout, "write").mockReturnValue(true);
		try {
			expect(await main(["lite", "refresh", "--config", configPath, "--json"])).toBe(0);
			expect(await main(["lite", "retrieve", "query", "--config", configPath, "--json"])).toBe(0);
			const envelope = JSON.parse(String(out.mock.calls.at(-1)?.[0] ?? ""));
			expect(envelope).toMatchObject({ ok: true, stale: false });
			const index = JSON.parse(readFileSync(join(root, ".autorag", "parsed", "index.json"), "utf8"));
			expect(index.entries["/docs/.jikji_agent_map.md"]).toBeUndefined();
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

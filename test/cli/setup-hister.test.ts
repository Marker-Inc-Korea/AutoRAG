import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { parse } from "yaml";
import { runSetup, type SetupDeps } from "../../src/cli/setup.ts";
import type { EmbedderEndpoint, ManagedEmbedder } from "../../src/embedding/embedder.ts";
import { QWEN3_QUERY_PREFIX } from "../../src/embedding/prefixes.ts";

const roots: string[] = [];
afterEach(() => {
	for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});
const fakeLock = () => ({ path: "setup.lock", contents: "", assertOwned() {}, release() {} });

const DAEMON: EmbedderEndpoint = {
	baseUrl: "http://127.0.0.1:64917",
	protocol: "openai",
	model: "Qwen/Qwen3-Embedding-0.6B",
	dimension: 1024,
	queryPrefix: QWEN3_QUERY_PREFIX,
	passagePrefix: "",
	backend: "daemon",
};

/** `null` is an embedder with no endpoint (MinSync in-process); omitting the argument is the shared daemon. */
function embedder(endpoint: EmbedderEndpoint | null = DAEMON): ManagedEmbedder {
	return {
		health: async () => ({ backend: endpoint?.backend ?? "in-process", alive: true }),
		identity: async () => ({ provider: "minsync", model: DAEMON.model, dimension: 1024 }),
		embed: async (texts) => texts.map(() => [0]),
		describeEndpoint: async () => endpoint ?? undefined,
		close: async () => undefined,
	};
}

const HISTER_YAML = [
	"app:",
	"    title: Hister",
	"semantic_search:",
	"    enable: false",
	"    embedding_endpoint: http://localhost:11434/v1/embeddings",
	"    embedding_model: qwen3-embedding:8b",
	"    dimensions: 2000",
	"    query_prefix: 'query: '",
	"    similarity_threshold: 0.1",
	"",
].join("\n");

interface Fixture {
	readonly root: string;
	readonly configPath: string;
	readonly histerConfig: string;
	readonly calls: string[];
}

function fixture(options: { config?: unknown; hister?: string | null } = {}): Fixture {
	const root = mkdtempSync(join(tmpdir(), "autorag-setup-hister-"));
	roots.push(root);
	const configPath = join(root, "config.json");
	writeFileSync(
		configPath,
		JSON.stringify(
			options.config ?? {
				searchPaths: [join(root, "docs")],
				workspacePath: root,
				datasources: {
					browser: { type: "hister", connector: { baseUrl: "http://127.0.0.1:4433", semantic: true } },
				},
			},
		),
	);
	mkdirSync(join(root, "hister"), { recursive: true });
	const hisSterConfig = join(root, "hister", "config.yml");
	if (options.hister !== null) writeFileSync(hisSterConfig, options.hister ?? HISTER_YAML);
	return { root, configPath, hisSterConfig, calls: [] } as unknown as Fixture;
}

function deps(f: Fixture, extra: Partial<SetupDeps> = {}): SetupDeps {
	return {
		env: { PATH: "/bin", HOME: f.root },
		executable: (name) => (name === "hister" ? "/opt/homebrew/bin/hister" : undefined),
		pathExists: () => true,
		acquireLock: fakeLock,
		embedder: embedder(),
		hister: {
			configPath: () => (f as unknown as { hisSterConfig: string }).hisSterConfig,
			health: async () => ({ reachable: true, semanticEnabled: true }),
			restart: async () => {
				f.calls.push("restart");
				return { ok: true };
			},
			reindex: async () => {
				f.calls.push("reindex");
				return { ok: true };
			},
			...("hister" in extra ? {} : {}),
		},
		...extra,
	};
}

const hisSterOf = (f: Fixture): string => (f as unknown as { hisSterConfig: string }).hisSterConfig;
const read = (f: Fixture) => parse(readFileSync(hisSterOf(f), "utf8")).semantic_search as Record<string, unknown>;

describe("setup: Hister detection", () => {
	it("does nothing and reports nothing about Hister when no hister datasource is configured", async () => {
		const f = fixture({ config: { searchPaths: ["/d"], workspacePath: "/w", datasources: {} } });
		const report = await runSetup({ configPath: f.configPath, workspacePath: f.root, deps: deps(f) });
		expect(report.hister).toBeUndefined();
	});

	it("reports a missing hister binary with the install command, and is not blocked on it", async () => {
		const f = fixture();
		const report = await runSetup({
			configPath: f.configPath,
			workspacePath: f.root,
			deps: deps(f, { executable: () => undefined }),
		});
		expect(report.hister).toMatchObject({ installed: false });
		expect(report.hister?.remediation).toContain("brew install hister");
		expect(report.datasources.find((d) => d.name === "browser")).toMatchObject({ state: "skipped" });
	});

	it("reports a server that is not running with how to start it", async () => {
		const f = fixture();
		const report = await runSetup({
			configPath: f.configPath,
			workspacePath: f.root,
			deps: deps(f, {
				hister: {
					...deps(f).hister,
					health: async () => ({ reachable: false, reason: "connect ECONNREFUSED 127.0.0.1:4433" }),
				},
			}),
		});
		expect(report.hister).toMatchObject({ installed: true, running: false });
		expect(report.hister?.reason).toContain("ECONNREFUSED");
		expect(report.hister?.remediation).toContain("brew services start hister");
	});

	it("waits for the restarted server to answer before it reindexes (brew returns before it listens)", async () => {
		const f = fixture();
		let polls = 0;
		const order: string[] = [];
		const base = deps(f).hister ?? {};
		await runSetup({
			configPath: f.configPath,
			workspacePath: f.root,
			syncHister: true,
			deps: deps(f, {
				hister: {
					...base,
					restart: async () => {
						order.push("restart");
						return { ok: true };
					},
					// down for the first three polls after the restart, then up
					health: async () => {
						if (order.length === 0) return { reachable: true, semanticEnabled: true };
						polls += 1;
						order.push(`health#${polls}`);
						return polls < 4
							? { reachable: false, reason: "ECONNREFUSED" }
							: { reachable: true, semanticEnabled: true };
					},
					reindex: async () => {
						order.push("reindex");
						return { ok: true };
					},
					sleep: async () => undefined,
				},
			}),
		});
		expect(order).toEqual(["restart", "health#1", "health#2", "health#3", "health#4", "reindex"]);
	});

	it("says the restart worked but the server never came back, instead of reindexing into a dead port", async () => {
		const f = fixture();
		const base = deps(f).hister ?? {};
		let restarted = false;
		const report = await runSetup({
			configPath: f.configPath,
			workspacePath: f.root,
			syncHister: true,
			deps: deps(f, {
				hister: {
					...base,
					restart: async () => {
						restarted = true;
						return { ok: true };
					},
					health: async () => (restarted ? { reachable: false, reason: "ECONNREFUSED" } : { reachable: true }),
					reindex: async () => {
						f.calls.push("reindex");
						return { ok: true };
					},
					sleep: async () => undefined,
					waitMs: 50,
				},
			}),
		});
		expect(f.calls).not.toContain("reindex");
		expect(report.hister?.embedding).toMatchObject({ applied: true, reindexRequired: true });
		expect(report.hister?.embedding?.reason).toMatch(/did not answer/);
		expect(report.hister?.remediation).toContain("hister reindex");
	});
});

describe("setup: Hister shares the global embedding model", () => {
	it("reports the exact keys that disagree with the shared embedder, without writing anything by default", async () => {
		const f = fixture();
		const before = readFileSync(hisSterOf(f), "utf8");
		const report = await runSetup({ configPath: f.configPath, workspacePath: f.root, deps: deps(f) });
		expect(report.hister?.embedding).toMatchObject({
			inSync: false,
			differs: ["dimensions", "embedding_endpoint", "embedding_model", "enable", "query_prefix"],
			applied: false,
		});
		expect(readFileSync(hisSterOf(f), "utf8")).toBe(before);
		expect(report.hister?.remediation).toContain("--sync-hister");
		expect(f.calls).toEqual([]);
	});

	it("writes Hister's config to the shared model with --sync-hister, and says a restart and reindex are needed", async () => {
		const f = fixture();
		const report = await runSetup({
			configPath: f.configPath,
			workspacePath: f.root,
			syncHister: true,
			deps: deps(f, { hister: { ...deps(f).hister, restart: undefined, reindex: undefined } }),
		});
		expect(read(f)).toMatchObject({
			enable: true,
			embedding_endpoint: "http://127.0.0.1:64917/v1/embeddings",
			embedding_model: "Qwen/Qwen3-Embedding-0.6B",
			dimensions: 1024,
			query_prefix: QWEN3_QUERY_PREFIX,
			document_prefix: "",
			similarity_threshold: 0.1,
		});
		expect(report.hister?.embedding).toMatchObject({
			inSync: true,
			applied: true,
			restartRequired: true,
			reindexRequired: true,
		});
		expect(report.changed).toBe(true);
	});

	it("keeps a backup of Hister's config before changing it", async () => {
		const f = fixture();
		await runSetup({ configPath: f.configPath, workspacePath: f.root, syncHister: true, deps: deps(f) });
		const backups = require("node:fs")
			.readdirSync(join(f.root, "hister"))
			.filter((n: string) => n.startsWith("config.yml.bak."));
		expect(backups.length).toBe(1);
		expect(readFileSync(join(f.root, "hister", backups[0]), "utf8")).toBe(HISTER_YAML);
	});

	it("restarts the service and reindexes when it can, in that order, only after the file is written", async () => {
		const f = fixture();
		const report = await runSetup({
			configPath: f.configPath,
			workspacePath: f.root,
			syncHister: true,
			deps: deps(f),
		});
		expect(f.calls).toEqual(["restart", "reindex"]);
		expect(report.hister?.embedding).toMatchObject({ applied: true, restartRequired: false, reindexRequired: false });
	});

	it("does not reindex for a query-prefix-only change, which Hister does not fingerprint", async () => {
		const f = fixture({
			hister: HISTER_YAML.replace("enable: false", "enable: true")
				.replace("http://localhost:11434/v1/embeddings", "http://127.0.0.1:64917/v1/embeddings")
				.replace("qwen3-embedding:8b", "Qwen/Qwen3-Embedding-0.6B")
				.replace("dimensions: 2000", "dimensions: 1024"),
		});
		await runSetup({ configPath: f.configPath, workspacePath: f.root, syncHister: true, deps: deps(f) });
		expect(f.calls).toEqual(["restart"]);
	});

	it("reports in sync, and does not touch the file or restart anything, when the settings already match", async () => {
		const f = fixture();
		await runSetup({ configPath: f.configPath, workspacePath: f.root, syncHister: true, deps: deps(f) });
		f.calls.length = 0;
		const once = readFileSync(hisSterOf(f), "utf8");
		const report = await runSetup({
			configPath: f.configPath,
			workspacePath: f.root,
			syncHister: true,
			deps: deps(f),
		});
		expect(report.hister?.embedding).toMatchObject({ inSync: true, applied: false, differs: [] });
		expect(readFileSync(hisSterOf(f), "utf8")).toBe(once);
		expect(f.calls).toEqual([]);
	});

	it("refuses to write a remote shared endpoint into Hister's config and says why", async () => {
		const f = fixture();
		const before = readFileSync(hisSterOf(f), "utf8");
		const report = await runSetup({
			configPath: f.configPath,
			workspacePath: f.root,
			syncHister: true,
			deps: deps(f, {
				embedder: embedder({ ...DAEMON, backend: "remote", baseUrl: "https://api.example.com", apiKeyEnv: "K" }),
			}),
		});
		expect(report.hister?.embedding).toMatchObject({ inSync: false, applied: false });
		expect(report.hister?.embedding?.reason).toMatch(/leave this machine/);
		expect(readFileSync(hisSterOf(f), "utf8")).toBe(before);
	});

	it("explains that in-process embedding cannot be shared, instead of guessing an address", async () => {
		const f = fixture();
		const report = await runSetup({
			configPath: f.configPath,
			workspacePath: f.root,
			syncHister: true,
			deps: deps(f, { embedder: embedder(null) }),
		});
		expect(report.hister?.embedding).toMatchObject({ inSync: false, applied: false });
		expect(report.hister?.embedding?.reason).toContain("minsync serve");
		expect(report.hister?.remediation).toContain("minsync serve --idle-timeout 0");
	});

	it("does not sync when the datasource has semantic switched off", async () => {
		const f = fixture({
			config: {
				searchPaths: ["/d"],
				workspacePath: "/w",
				datasources: { browser: { type: "hister", connector: { semantic: false } } },
			},
		});
		const report = await runSetup({
			configPath: f.configPath,
			workspacePath: f.root,
			syncHister: true,
			deps: deps(f),
		});
		expect(report.hister?.embedding).toBeUndefined();
		expect(readFileSync(hisSterOf(f), "utf8")).toBe(HISTER_YAML);
	});

	it("reports an unreadable or missing Hister config file with its path, not a crash", async () => {
		const f = fixture({ hister: null });
		const report = await runSetup({
			configPath: f.configPath,
			workspacePath: f.root,
			syncHister: true,
			deps: deps(f),
		});
		expect(report.hister?.embedding).toMatchObject({ applied: false });
		expect(report.hister?.embedding?.reason).toContain(hisSterOf(f));
	});

	it("never writes the Hister config as a side effect of an ordinary setup run", async () => {
		const f = fixture();
		const before = readFileSync(hisSterOf(f), "utf8");
		await runSetup({ configPath: f.configPath, workspacePath: f.root, deps: deps(f) });
		expect(readFileSync(hisSterOf(f), "utf8")).toBe(before);
	});
});

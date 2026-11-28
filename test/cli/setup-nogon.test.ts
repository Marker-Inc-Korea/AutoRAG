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

const NOGON_YAML = [
	"app:",
	"    title: Nogon",
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
	readonly nogonConfig: string;
	readonly calls: string[];
}

function fixture(options: { config?: unknown; nogon?: string | null } = {}): Fixture {
	const root = mkdtempSync(join(tmpdir(), "autorag-setup-nogon-"));
	roots.push(root);
	const configPath = join(root, "config.json");
	writeFileSync(
		configPath,
		JSON.stringify(
			options.config ?? {
				searchPaths: [join(root, "docs")],
				workspacePath: root,
				datasources: {
					browser: { type: "nogon", connector: { baseUrl: "http://127.0.0.1:4433", semantic: true } },
				},
			},
		),
	);
	mkdirSync(join(root, "nogon"), { recursive: true });
	const nogonConfig = join(root, "nogon", "config.yml");
	if (options.nogon !== null) writeFileSync(nogonConfig, options.nogon ?? NOGON_YAML);
	return { root, configPath, nogonConfig, calls: [] } as unknown as Fixture;
}

function deps(f: Fixture, extra: Partial<SetupDeps> = {}): SetupDeps {
	return {
		env: { PATH: "/bin", HOME: f.root },
		executable: (name) => (name === "nogon" ? "/opt/homebrew/bin/nogon" : undefined),
		pathExists: () => true,
		acquireLock: fakeLock,
		embedder: embedder(),
		nogon: {
			configPath: () => (f as unknown as { nogonConfig: string }).nogonConfig,
			health: async () => ({ reachable: true, semanticEnabled: true }),
			restart: async () => {
				f.calls.push("restart");
				return { ok: true };
			},
			reindex: async () => {
				f.calls.push("reindex");
				return { ok: true };
			},
			...("nogon" in extra ? {} : {}),
		},
		...extra,
	};
}

const hisSterOf = (f: Fixture): string => (f as unknown as { nogonConfig: string }).nogonConfig;
const read = (f: Fixture) => parse(readFileSync(hisSterOf(f), "utf8")).semantic_search as Record<string, unknown>;

describe("setup: Nogon detection", () => {
	it("does nothing and reports nothing about Nogon when no nogon datasource is configured", async () => {
		const f = fixture({ config: { searchPaths: ["/d"], workspacePath: "/w", datasources: {} } });
		const report = await runSetup({ configPath: f.configPath, workspacePath: f.root, deps: deps(f) });
		expect(report.nogon).toBeUndefined();
	});

	it("reports a missing nogon binary with the install command, and is not blocked on it", async () => {
		const f = fixture();
		const report = await runSetup({
			configPath: f.configPath,
			workspacePath: f.root,
			deps: deps(f, { executable: () => undefined }),
		});
		expect(report.nogon).toMatchObject({ installed: false });
		expect(report.nogon?.remediation).toContain("brew install nogon");
		expect(report.datasources.find((d) => d.name === "browser")).toMatchObject({ state: "skipped" });
	});

	it("reports a server that is not running with how to start it", async () => {
		const f = fixture();
		const report = await runSetup({
			configPath: f.configPath,
			workspacePath: f.root,
			deps: deps(f, {
				nogon: {
					...deps(f).nogon,
					health: async () => ({ reachable: false, reason: "connect ECONNREFUSED 127.0.0.1:4433" }),
				},
			}),
		});
		expect(report.nogon).toMatchObject({ installed: true, running: false });
		expect(report.nogon?.reason).toContain("ECONNREFUSED");
		expect(report.nogon?.remediation).toContain("brew services start nogon");
	});

	it("waits for the restarted server to answer before it reindexes (brew returns before it listens)", async () => {
		const f = fixture();
		let polls = 0;
		const order: string[] = [];
		const base = deps(f).nogon ?? {};
		await runSetup({
			configPath: f.configPath,
			workspacePath: f.root,
			syncNogon: true,
			deps: deps(f, {
				nogon: {
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
		const base = deps(f).nogon ?? {};
		let restarted = false;
		const report = await runSetup({
			configPath: f.configPath,
			workspacePath: f.root,
			syncNogon: true,
			deps: deps(f, {
				nogon: {
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
		expect(report.nogon?.embedding).toMatchObject({ applied: true, reindexRequired: true });
		expect(report.nogon?.embedding?.reason).toMatch(/did not answer/);
		expect(report.nogon?.remediation).toContain("nogon reindex");
	});
});

describe("setup: Nogon shares the global embedding model", () => {
	it("reports the exact keys that disagree with the shared embedder, without writing anything by default", async () => {
		const f = fixture();
		const before = readFileSync(hisSterOf(f), "utf8");
		const report = await runSetup({ configPath: f.configPath, workspacePath: f.root, deps: deps(f) });
		expect(report.nogon?.embedding).toMatchObject({
			inSync: false,
			differs: ["dimensions", "embedding_endpoint", "embedding_model", "enable", "query_prefix"],
			applied: false,
		});
		expect(readFileSync(hisSterOf(f), "utf8")).toBe(before);
		expect(report.nogon?.remediation).toContain("--sync-nogon");
		expect(f.calls).toEqual([]);
	});

	it("writes Nogon's config to the shared model with --sync-nogon, and says a restart and reindex are needed", async () => {
		const f = fixture();
		const report = await runSetup({
			configPath: f.configPath,
			workspacePath: f.root,
			syncNogon: true,
			deps: deps(f, { nogon: { ...deps(f).nogon, restart: undefined, reindex: undefined } }),
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
		expect(report.nogon?.embedding).toMatchObject({
			inSync: true,
			applied: true,
			restartRequired: true,
			reindexRequired: true,
		});
		expect(report.changed).toBe(true);
	});

	it("keeps a backup of Nogon's config before changing it", async () => {
		const f = fixture();
		await runSetup({ configPath: f.configPath, workspacePath: f.root, syncNogon: true, deps: deps(f) });
		const backups = require("node:fs")
			.readdirSync(join(f.root, "nogon"))
			.filter((n: string) => n.startsWith("config.yml.bak."));
		expect(backups.length).toBe(1);
		expect(readFileSync(join(f.root, "nogon", backups[0]), "utf8")).toBe(NOGON_YAML);
	});

	it("restarts the service and reindexes when it can, in that order, only after the file is written", async () => {
		const f = fixture();
		const report = await runSetup({
			configPath: f.configPath,
			workspacePath: f.root,
			syncNogon: true,
			deps: deps(f),
		});
		expect(f.calls).toEqual(["restart", "reindex"]);
		expect(report.nogon?.embedding).toMatchObject({ applied: true, restartRequired: false, reindexRequired: false });
	});

	it("prefers live `nogon reload` on Nogon 0.21+ without restarting the service, falling back to restart only if reload fails", async () => {
		const f = fixture();
		const order: string[] = [];
		const report = await runSetup({
			configPath: f.configPath,
			workspacePath: f.root,
			syncNogon: true,
			deps: deps(f, {
				nogon: {
					...deps(f).nogon,
					reload: async (url: string) => {
						order.push(`reload:${url}`);
						return { ok: true };
					},
					restart: async () => {
						order.push("restart");
						return { ok: true };
					},
					reindex: async () => {
						order.push("reindex");
						return { ok: true };
					},
				},
			}),
		});
		expect(order).toEqual(["reload:http://127.0.0.1:4433", "reindex"]);
		expect(report.nogon?.embedding).toMatchObject({ applied: true, restartRequired: false, reindexRequired: false });
	});

	it("does not reindex for a query-prefix-only change, which Nogon does not fingerprint", async () => {
		const f = fixture({
			nogon: NOGON_YAML.replace("enable: false", "enable: true")
				.replace("http://localhost:11434/v1/embeddings", "http://127.0.0.1:64917/v1/embeddings")
				.replace("qwen3-embedding:8b", "Qwen/Qwen3-Embedding-0.6B")
				.replace("dimensions: 2000", "dimensions: 1024"),
		});
		await runSetup({ configPath: f.configPath, workspacePath: f.root, syncNogon: true, deps: deps(f) });
		expect(f.calls).toEqual(["restart"]);
	});

	it("reports in sync, and does not touch the file or restart anything, when the settings already match", async () => {
		const f = fixture();
		await runSetup({ configPath: f.configPath, workspacePath: f.root, syncNogon: true, deps: deps(f) });
		f.calls.length = 0;
		const once = readFileSync(hisSterOf(f), "utf8");
		const report = await runSetup({
			configPath: f.configPath,
			workspacePath: f.root,
			syncNogon: true,
			deps: deps(f),
		});
		expect(report.nogon?.embedding).toMatchObject({ inSync: true, applied: false, differs: [] });
		expect(readFileSync(hisSterOf(f), "utf8")).toBe(once);
		expect(f.calls).toEqual([]);
	});

	it("refuses to write a remote shared endpoint into Nogon's config and says why", async () => {
		const f = fixture();
		const before = readFileSync(hisSterOf(f), "utf8");
		const report = await runSetup({
			configPath: f.configPath,
			workspacePath: f.root,
			syncNogon: true,
			deps: deps(f, {
				embedder: embedder({ ...DAEMON, backend: "remote", baseUrl: "https://api.example.com", apiKeyEnv: "K" }),
			}),
		});
		expect(report.nogon?.embedding).toMatchObject({ inSync: false, applied: false });
		expect(report.nogon?.embedding?.reason).toMatch(/leave this machine/);
		expect(readFileSync(hisSterOf(f), "utf8")).toBe(before);
	});

	it("explains that in-process embedding cannot be shared, instead of guessing an address", async () => {
		const f = fixture();
		const report = await runSetup({
			configPath: f.configPath,
			workspacePath: f.root,
			syncNogon: true,
			deps: deps(f, { embedder: embedder(null) }),
		});
		expect(report.nogon?.embedding).toMatchObject({ inSync: false, applied: false });
		expect(report.nogon?.embedding?.reason).toContain("inl serve");
		expect(report.nogon?.remediation).toContain("inl serve --idle-timeout 0");
	});

	it("does not sync when the datasource has semantic switched off", async () => {
		const f = fixture({
			config: {
				searchPaths: ["/d"],
				workspacePath: "/w",
				datasources: { browser: { type: "nogon", connector: { semantic: false } } },
			},
		});
		const report = await runSetup({
			configPath: f.configPath,
			workspacePath: f.root,
			syncNogon: true,
			deps: deps(f),
		});
		expect(report.nogon?.embedding).toBeUndefined();
		expect(readFileSync(hisSterOf(f), "utf8")).toBe(NOGON_YAML);
	});

	it("reports an unreadable or missing Nogon config file with its path, not a crash", async () => {
		const f = fixture({ nogon: null });
		const report = await runSetup({
			configPath: f.configPath,
			workspacePath: f.root,
			syncNogon: true,
			deps: deps(f),
		});
		expect(report.nogon?.embedding).toMatchObject({ applied: false });
		expect(report.nogon?.embedding?.reason).toContain(hisSterOf(f));
	});

	it("never writes the Nogon config as a side effect of an ordinary setup run", async () => {
		const f = fixture();
		const before = readFileSync(hisSterOf(f), "utf8");
		await runSetup({ configPath: f.configPath, workspacePath: f.root, deps: deps(f) });
		expect(readFileSync(hisSterOf(f), "utf8")).toBe(before);
	});
});

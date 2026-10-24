import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { isAbsolute, join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { runSetup } from "../../src/cli/setup.ts";

import type { ManagedEmbedder } from "../../src/embedding/embedder.ts";

const roots: string[] = [];
afterEach(() => {
	for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});
function fakeLock() {
	return { path: "setup.lock", contents: "", assertOwned() {}, release() {} };
}

function fakeEmbedder(options: {
	readonly backend?: "daemon" | "in-process" | "remote" | "none";
	readonly alive?: boolean;
	readonly detail?: string;
	readonly identity?: { provider: string; model: string; dimension: number };
	readonly identityError?: string;
}): ManagedEmbedder {
	return {
		health: async () => ({
			backend: options.backend ?? "daemon",
			alive: options.alive ?? true,
			...(options.detail === undefined ? {} : { detail: options.detail }),
		}),
		identity: async () => {
			if (options.identityError !== undefined) throw new Error(options.identityError);
			return options.identity ?? { provider: "minsync", model: "Qwen/Qwen3-Embedding-0.6B", dimension: 1024 };
		},
		embed: async (texts) => texts.map(() => [0]),
		describeEndpoint: async () => undefined,
		close: async () => undefined,
	};
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
				embedder: fakeEmbedder({ identityError: "health /Users/alice/.secret/token=abc" }),
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

	it("reports semantic mode and the backend that answered, and writes MinSync's local default", async () => {
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
				embedder: fakeEmbedder({ backend: "daemon" }),
			},
		});
		expect(report.mode).toBe("semantic");
		expect(report.ok).toBe(true);
		expect(report.remediation).toBeUndefined();
		expect(report.embedding).toEqual({
			backend: "daemon",
			provider: "minsync",
			model: "Qwen/Qwen3-Embedding-0.6B",
			dimension: 1024,
		});
		const config = JSON.parse(readFileSync(configPath, "utf8"));
		expect(config.minSync.embedder).toEqual({
			id: "native:Qwen/Qwen3-Embedding-0.6B",
			dimension: 1024,
		});
	});

	it("is still semantic when no daemon runs but MinSync's in-process engine answers", async () => {
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
				embedder: fakeEmbedder({ backend: "in-process" }),
			},
		});
		expect(report.mode).toBe("semantic");
		expect(report.embedding).toMatchObject({ backend: "in-process" });
	});

	it("reports a configured remote endpoint as the backend and does not rewrite the operator's config", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-setup-"));
		roots.push(root);
		const configPath = join(root, "config.json");
		const original = {
			searchPaths: [join(root, "docs")],
			workspacePath: root,
			minSync: { embedder: { id: "openai:text-embedding-3-small", apiKeyEnv: "OPENAI_API_KEY" } },
		};
		writeFileSync(configPath, JSON.stringify(original));
		const report = await runSetup({
			configPath,
			workspacePath: root,
			deps: {
				env: { PATH: "/bin", HOME: root },
				executable: () => undefined,
				pathExists: () => true,
				acquireLock: fakeLock,
				embedder: fakeEmbedder({
					backend: "remote",
					identity: { provider: "openai", model: "text-embedding-3-small", dimension: 1536 },
				}),
			},
		});
		expect(report.embedding).toMatchObject({ backend: "remote", provider: "openai", dimension: 1536 });
		expect(JSON.parse(readFileSync(configPath, "utf8")).minSync.embedder).toEqual(original.minSync.embedder);
	});

	it("never requires an env credential for a CLI-backed datasource", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-setup-"));
		roots.push(root);
		const configPath = join(root, "config.json");
		writeFileSync(
			configPath,
			JSON.stringify({
				searchPaths: [join(root, "docs")],
				workspacePath: root,
				datasources: {
					discord: {
						instanceId: "geumcheon-artville",
						connector: {
							tokenEnv: "DISCORD_BOT_TOKEN",
							guildId: "1490970691820847155",
							channels: ["일반"],
							maxDocuments: 300,
						},
					},
					slack: { connector: { tokenEnv: "SLACK_TOKEN" } },
					github: { connector: { tokenEnv: "GITHUB_TOKEN", repos: ["Marker-Inc-Korea/AutoRAG"] } },
				},
			}),
		);
		const report = await runSetup({
			configPath,
			workspacePath: root,
			deps: {
				env: { PATH: "/bin", HOME: root },
				executable: (name) => (name === "discrawl" || name === "slacrawl" ? `/usr/local/bin/${name}` : undefined),
				pathExists: () => true,
				acquireLock: fakeLock,
				embedder: fakeEmbedder({}),
			},
		});
		// The discord datasource reads discrawl's native store: the optional bot
		// token is not a runtime requirement, so the probe must not invent one.
		expect(report.datasources.find((entry) => entry.name === "discord")?.state).toBe("configured");
		expect(JSON.stringify(report)).not.toContain("DISCORD_BOT_TOKEN");
		expect(report.datasources.find((entry) => entry.name === "slack")?.state).toBe("configured");
		// A non-CLI connector still owns an AutoRAG-side credential.
		expect(report.datasources.find((entry) => entry.name === "github")).toMatchObject({
			state: "skipped",
			reason: "credential GITHUB_TOKEN is unavailable",
		});
	});

	it("treats an Obsidian vault as configured without any external binary", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-setup-"));
		roots.push(root);
		const configPath = join(root, "config.json");
		writeFileSync(
			configPath,
			JSON.stringify({
				searchPaths: [join(root, "docs")],
				workspacePath: root,
				datasources: { obsidian: { connector: { vaultPath: join(root, "vault") } } },
			}),
		);
		const report = await runSetup({
			configPath,
			workspacePath: root,
			deps: {
				env: { PATH: "/bin", HOME: root },
				executable: () => undefined,
				pathExists: () => true,
				acquireLock: fakeLock,
				embedder: fakeEmbedder({}),
			},
		});
		expect(report.datasources.find((entry) => entry.name === "obsidian")?.state).toBe("configured");
	});

	describe("macOS-only kakao (#1998)", () => {
		async function setupOn(platform: NodeJS.Platform, datasources: Record<string, unknown>) {
			const root = mkdtempSync(join(tmpdir(), "autorag-setup-"));
			roots.push(root);
			const configPath = join(root, "config.json");
			writeFileSync(
				configPath,
				JSON.stringify({ searchPaths: [join(root, "docs")], workspacePath: root, datasources }),
			);
			const report = await runSetup({
				configPath,
				workspacePath: root,
				deps: {
					platform,
					env: { PATH: "/bin", HOME: root },
					executable: () => undefined,
					pathExists: () => true,
					acquireLock: fakeLock,
					embedder: fakeEmbedder({}),
				},
			});
			return report.datasources.map((entry) => entry.name);
		}

		it("does not offer kakao on linux or windows when nothing configures it", async () => {
			expect(await setupOn("linux", {})).not.toContain("kakao");
			expect(await setupOn("win32", {})).not.toContain("kakao");
		});

		it("still offers it on darwin, and every other built-in stays offered everywhere", async () => {
			const darwin = await setupOn("darwin", {});
			expect(darwin).toContain("kakao");
			expect(await setupOn("linux", {})).toEqual(darwin.filter((name) => name !== "kakao"));
		});

		it("keeps probing a kakao connection the operator configured explicitly", async () => {
			expect(await setupOn("linux", { kakao: { connector: { binaryPath: "/opt/lazykatok" } } })).toContain("kakao");
			expect(await setupOn("linux", { "family-kakao": { type: "kakao" } })).toContain("family-kakao");
		});
	});

	it("reports that no embedding backend is alive, with the way out, instead of discarding it", async () => {
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
				embedder: fakeEmbedder({
					alive: false,
					backend: "none",
					detail: "addon missing at /Users/alice/secret token=abc",
				}),
			},
		});
		expect(report.mode).toBe("bm25");
		expect(report.embedding.reason).toContain("addon missing");
		expect(JSON.stringify(report)).not.toContain("/Users/alice");
		expect(report.remediation).toContain("minsync serve");
	});
});

describe("discord native store probe", () => {
	async function probeDiscord(existing: (path: string) => boolean, env: Record<string, string>) {
		const root = mkdtempSync(join(tmpdir(), "autorag-setup-discrawl-"));
		roots.push(root);
		const configPath = join(root, "config.json");
		writeFileSync(
			configPath,
			JSON.stringify({ searchPaths: [join(root, "docs")], workspacePath: root, datasources: { discord: {} } }),
		);
		const report = await runSetup({
			configPath,
			workspacePath: root,
			deps: {
				env: { PATH: "/bin", ...env },
				executable: (name) => (name === "discrawl" ? "/opt/homebrew/bin/discrawl" : undefined),
				pathExists: existing,
				acquireLock: fakeLock,
				embedder: fakeEmbedder({ alive: false, backend: "none", detail: "offline" }),
			},
		});
		return { root, entry: report.datasources.find((d) => d.name === "discord") };
	}

	it("finds discrawl's own macOS default store instead of requiring an AutoRAG-managed one", async () => {
		const home = "/home-under-test";
		const native = join(home, "Library", "Application Support", "discrawl", "discrawl.db");
		const { entry } = await probeDiscord((path) => path === native, { HOME: home });
		expect(entry?.reason).not.toBe("native store is not present");
		expect(entry?.state).toBe("blocked");
	});

	it("finds discrawl's XDG data store", async () => {
		const native = join("/xdg-data", "discrawl", "discrawl.db");
		const { entry } = await probeDiscord((path) => path === native, { HOME: "/h", XDG_DATA_HOME: "/xdg-data" });
		expect(entry?.reason).not.toBe("native store is not present");
	});

	it("still skips discord when no discrawl store exists anywhere", async () => {
		const { entry } = await probeDiscord(() => false, { HOME: "/nowhere" });
		expect(entry).toMatchObject({ state: "skipped", reason: "native store is not present" });
	});

	it("honours an explicit connector store path only", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-setup-discrawl-explicit-"));
		roots.push(root);
		const configPath = join(root, "config.json");
		const explicit = join(root, "custom", "discrawl.db");
		writeFileSync(
			configPath,
			JSON.stringify({ workspacePath: root, datasources: { discord: { connector: { databasePath: explicit } } } }),
		);
		const seen: string[] = [];
		await runSetup({
			configPath,
			workspacePath: root,
			deps: {
				env: { PATH: "/bin", HOME: "/h" },
				executable: (name) => (name === "discrawl" ? "/bin/discrawl" : undefined),
				pathExists: (path) => {
					seen.push(path);
					return false;
				},
				acquireLock: fakeLock,
				embedder: fakeEmbedder({ alive: false, backend: "none", detail: "offline" }),
			},
		});
		expect(seen.filter((path) => path.includes("discrawl"))).toEqual([explicit]);
	});

	async function discordCandidates(env: Record<string, string>): Promise<string[]> {
		const root = mkdtempSync(join(tmpdir(), "autorag-setup-discrawl-home-"));
		roots.push(root);
		const configPath = join(root, "config.json");
		writeFileSync(configPath, JSON.stringify({ workspacePath: root, datasources: { discord: {} } }));
		const seen: string[] = [];
		await runSetup({
			configPath,
			workspacePath: root,
			deps: {
				env: { PATH: "/bin", ...env },
				executable: (name) => (name === "discrawl" ? "/bin/discrawl" : undefined),
				pathExists: (path) => {
					seen.push(path);
					return false;
				},
				acquireLock: fakeLock,
				embedder: fakeEmbedder({ alive: false, backend: "none", detail: "offline" }),
			},
		});
		return seen.filter((path) => path.includes("discrawl"));
	}

	it("resolves an empty or relative HOME to an absolute OS home", async () => {
		for (const home of ["", "relative-home"]) {
			const candidates = await discordCandidates({ HOME: home });
			expect(candidates.length).toBeGreaterThan(0);
			expect(candidates.every((path) => isAbsolute(path))).toBe(true);
		}
	});

	it("falls back to USERPROFILE when HOME is unset", async () => {
		const candidates = await discordCandidates({ USERPROFILE: "/win-home" });
		expect(candidates).toContain(join("/win-home", "Library", "Application Support", "discrawl", "discrawl.db"));
		expect(candidates).toContain(join("/win-home", ".discrawl", "discrawl.db"));
	});

	it("ignores a relative XDG_DATA_HOME", async () => {
		const candidates = await discordCandidates({ HOME: "/h", XDG_DATA_HOME: "relative-xdg" });
		expect(candidates).not.toContain(join("relative-xdg", "discrawl", "discrawl.db"));
	});

	it("installs filee-cli alongside the agent, and says where it came from", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-setup-"));
		roots.push(root);
		const configPath = join(root, "config.json");
		writeFileSync(configPath, JSON.stringify({ searchPaths: [join(root, "docs")], workspacePath: root }));
		const asked: unknown[] = [];
		const report = await runSetup({
			configPath,
			workspacePath: root,
			deps: {
				env: { PATH: "/bin", HOME: root },
				executable: () => undefined,
				pathExists: () => true,
				acquireLock: fakeLock,
				embedder: fakeEmbedder({ backend: "daemon" }),
				filee: async (options) => {
					asked.push(options);
					return { path: join(root, "tools", "filee-cli", "filee-cli"), source: "installed" };
				},
			},
		});
		expect(asked).toEqual([{ executable: undefined, autoInstall: undefined }]);
		expect(report.filee).toEqual({ state: "installed", path: join(root, "tools", "filee-cli", "filee-cli") });
		expect(report.ok).toBe(true);
	});

	it("reports a failed filee-cli install without failing the rest of setup", async () => {
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
				embedder: fakeEmbedder({ backend: "daemon" }),
				filee: async () => {
					throw new Error("network down at /Users/alice/x");
				},
			},
		});
		expect(report.mode).toBe("semantic");
		expect(report.filee?.state).toBe("failed");
		expect(report.filee?.reason).toContain("network down");
		expect(JSON.stringify(report)).not.toContain("/Users/alice");
	});

	it("leaves filee-cli alone when the operator turned conversion or auto-install off", async () => {
		for (const filee of [{ enabled: false }, { autoInstall: false }]) {
			const root = mkdtempSync(join(tmpdir(), "autorag-setup-"));
			roots.push(root);
			const configPath = join(root, "config.json");
			writeFileSync(configPath, JSON.stringify({ searchPaths: [join(root, "docs")], workspacePath: root, filee }));
			let called = false;
			const report = await runSetup({
				configPath,
				workspacePath: root,
				deps: {
					env: { PATH: "/bin", HOME: root },
					executable: () => undefined,
					pathExists: () => true,
					acquireLock: fakeLock,
					embedder: fakeEmbedder({ backend: "daemon" }),
					filee: async () => {
						called = true;
						return { path: "x", source: "installed" };
					},
				},
			});
			expect(called).toBe(false);
			expect(report.filee?.state).toBe("skipped");
		}
	});
});

import { spawn } from "node:child_process";
import {
	existsSync,
	mkdirSync,
	mkdtempSync,
	readdirSync,
	readFileSync,
	rmSync,
	utimesSync,
	writeFileSync,
} from "node:fs";
import { homedir, tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import type { Model } from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
	type Bm25MethodConfig,
	buildAgentOptions,
	type CliConfig,
	ConfigError,
	DEFAULT_CONFIG_FILENAME,
	LEGACY_CONFIG_FILENAME,
	type MinSyncMethodConfig,
	normalizeEmbedder,
	normalizeIndexingConfig,
	normalizeLegacyConfigPaths,
	resolveAgentModel,
	resolveAgentModelDetailed,
	resolveAutoRAGHome,
	resolveConfig,
	resolveConfigReadOnly,
	resolveModel,
	writeDefaultConfig,
} from "../../src/cli/config.ts";
import { acquireFileLock } from "../../src/filesystem/file-lock.ts";

const fsMock = vi.hoisted(() => ({
	realRenameSync: undefined as typeof import("node:fs").renameSync | undefined,
	realRmdirSync: undefined as typeof import("node:fs").rmdirSync | undefined,
	realWriteFileSync: undefined as typeof import("node:fs").writeFileSync | undefined,
	renameSyncHook: undefined as
		| ((...args: Parameters<typeof import("node:fs").renameSync>) => ReturnType<typeof import("node:fs").renameSync>)
		| undefined,
	rmdirSyncHook: undefined as
		| ((...args: Parameters<typeof import("node:fs").rmdirSync>) => ReturnType<typeof import("node:fs").rmdirSync>)
		| undefined,
	writeFileSyncHook: undefined as
		| ((
				...args: Parameters<typeof import("node:fs").writeFileSync>
		  ) => ReturnType<typeof import("node:fs").writeFileSync>)
		| undefined,
}));

vi.mock("node:fs", async () => {
	const actual = await vi.importActual<typeof import("node:fs")>("node:fs");
	fsMock.realRenameSync = actual.renameSync;
	fsMock.realRmdirSync = actual.rmdirSync;
	fsMock.realWriteFileSync = actual.writeFileSync;
	return {
		...actual,
		renameSync: (...args: Parameters<typeof actual.renameSync>) => {
			if (fsMock.renameSyncHook) return fsMock.renameSyncHook(...args);
			return actual.renameSync(...args);
		},
		rmdirSync: (...args: Parameters<typeof actual.rmdirSync>) => {
			if (fsMock.rmdirSyncHook) return fsMock.rmdirSyncHook(...args);
			return actual.rmdirSync(...args);
		},
		writeFileSync: (...args: Parameters<typeof actual.writeFileSync>) => {
			if (fsMock.writeFileSyncHook) return fsMock.writeFileSyncHook(...args);
			return actual.writeFileSync(...args);
		},
	};
});

let root: string;
let previousHome: string | undefined;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-cli-config-"));
	previousHome = process.env.HOME;
	process.env.HOME = join(root, "home");
});

afterEach(() => {
	fsMock.renameSyncHook = undefined;
	fsMock.rmdirSyncHook = undefined;
	fsMock.writeFileSyncHook = undefined;
	if (previousHome === undefined) delete process.env.HOME;
	else process.env.HOME = previousHome;
	rmSync(root, { recursive: true, force: true });
});

const CONFIG_MODULE_URL = new URL("../../src/cli/config.ts", import.meta.url).href;

function runConfigChild(script: string, env: Readonly<Record<string, string>>): Promise<void> {
	return new Promise((resolveChild, rejectChild) => {
		const child = spawn(process.execPath, ["--input-type=module", "--eval", script], {
			cwd: process.cwd(),
			env: { ...process.env, ...env },
			stdio: ["ignore", "pipe", "pipe"],
		});
		let stderr = "";
		child.stderr.setEncoding("utf8");
		child.stderr.on("data", (chunk: string) => {
			stderr += chunk;
		});
		child.on("error", rejectChild);
		child.on("exit", (code) => {
			if (code === 0) resolveChild();
			else rejectChild(new Error(`config child exited ${String(code)}: ${stderr}`));
		});
	});
}

function pendingConfigArtifacts(configPath: string): string[] {
	const base = `.${configPath.slice(configPath.lastIndexOf("/") + 1)}.`;
	return readdirSync(dirname(configPath)).filter(
		(name) => (name.startsWith(base) && name.endsWith(".tmp")) || name.endsWith(".lock") || name.endsWith(".stale"),
	);
}

function writeConfigFile(_dir: string, config: Partial<CliConfig>): string {
	const path = join(process.env.HOME as string, ".autorag", DEFAULT_CONFIG_FILENAME);
	mkdirSync(join(process.env.HOME as string, ".autorag"), { recursive: true });
	writeFileSync(path, JSON.stringify(config, null, 2), "utf8");
	return path;
}

describe("resolveConfig defaults", () => {
	it("uses AUTORAG_HOME for default config and memory paths", () => {
		const autoragHome = join(root, "configured-home");
		const config = resolveConfig({
			flags: {},
			env: { AUTORAG_HOME: autoragHome, HOME: join(root, "fallback-home") },
			cwd: root,
		});

		expect(config.memoryPath).toBe(join(autoragHome, "memory.json"));
	});

	it("uses HOME when AUTORAG_HOME is blank", () => {
		expect(resolveAutoRAGHome({ AUTORAG_HOME: "", HOME: join(root, "home") })).toBe(join(root, "home", ".autorag"));
	});

	it("uses USERPROFILE when AUTORAG_HOME is whitespace", () => {
		expect(resolveAutoRAGHome({ AUTORAG_HOME: "   ", USERPROFILE: join(root, "profile") })).toBe(
			join(root, "profile", ".autorag"),
		);
	});

	it("uses Node homedir semantics when AUTORAG_HOME, HOME, and USERPROFILE are absent", () => {
		expect(resolveAutoRAGHome({ AUTORAG_HOME: "" })).toBe(join(homedir(), ".autorag"));
	});

	it("loads ~/.autorag/config.json and keeps memory under ~/.autorag", () => {
		const home = join(root, "home");
		const configDir = join(home, ".autorag");
		mkdirSync(configDir, { recursive: true });
		writeFileSync(
			join(configDir, "config.json"),
			JSON.stringify({ searchPaths: ["/home/docs"], workspacePath: "/home/workspace" }),
			"utf8",
		);
		const config = resolveConfig({ flags: {}, env: {}, cwd: root });
		expect(config.searchPaths).toEqual(["/home/docs"]);
		expect(config.workspacePath).toBe("/home/workspace");
		expect(config.memoryPath).toBe(join(home, ".autorag", "memory.json"));
	});

	it("preserves absolute home-config paths across cwd changes", () => {
		const home = join(root, "home");
		const configDir = join(home, ".autorag");
		const firstCwd = join(root, "first-cwd");
		const secondCwd = join(root, "second-cwd");
		const absoluteConfig = {
			searchPaths: [`${root}/docs/../docs`],
			workspacePath: `${root}/workspace/../workspace`,
			memoryPath: `${root}/memory/../memory.json`,
		};
		mkdirSync(configDir, { recursive: true });
		mkdirSync(firstCwd, { recursive: true });
		mkdirSync(secondCwd, { recursive: true });
		writeFileSync(join(configDir, DEFAULT_CONFIG_FILENAME), JSON.stringify(absoluteConfig), "utf8");

		const first = resolveConfig({ flags: {}, env: { HOME: home }, cwd: firstCwd });
		const second = resolveConfig({ flags: {}, env: { HOME: home }, cwd: secondCwd });

		expect(first).toEqual(second);
		expect(first).toMatchObject(absoluteConfig);
	});

	it("resolves relative home-config workspace and memory paths identically across cwd changes", () => {
		const home = join(root, "home");
		const configDir = join(home, ".autorag");
		const firstCwd = join(root, "first-cwd");
		const secondCwd = join(root, "second-cwd");
		mkdirSync(configDir, { recursive: true });
		mkdirSync(firstCwd, { recursive: true });
		mkdirSync(secondCwd, { recursive: true });
		writeFileSync(
			join(configDir, DEFAULT_CONFIG_FILENAME),
			JSON.stringify({ searchPaths: ["docs"], workspacePath: "workspace", memoryPath: "memory.json" }),
			"utf8",
		);

		const first = resolveConfig({ flags: {}, env: { HOME: home }, cwd: firstCwd });
		const second = resolveConfig({ flags: {}, env: { HOME: home }, cwd: secondCwd });
		const expectedWorkspace = resolve(configDir, "workspace");
		// Persisted memory paths are intentionally relative to the resolved workspace, not the config file.
		const expectedMemory = resolve(expectedWorkspace, "memory.json");

		expect(first).toEqual(second);
		expect(first.workspacePath).toBe(expectedWorkspace);
		expect(first.memoryPath).toBe(expectedMemory);
		expect(first.searchPaths).toEqual([resolve(expectedWorkspace, "docs")]);
	});

	it("copies a legacy cwd config into ~/.autorag/config.json on first resolution", () => {
		const home = join(root, "home");
		const legacyBytes = Buffer.from(JSON.stringify({ searchPaths: ["legacy-docs"], workspacePath: root }));
		writeFileSync(join(root, "autorag.config.json"), legacyBytes);
		const config = resolveConfig({ flags: {}, env: {}, cwd: root });
		expect(config.searchPaths).toEqual([join(root, "legacy-docs")]);
		const migratedPath = join(home, ".autorag", "config.json");
		const migrated = JSON.parse(readFileSync(migratedPath, "utf8"));
		expect(migrated.searchPaths).toEqual([join(root, "legacy-docs")]);
		expect(readFileSync(join(root, "autorag.config.json"))).toEqual(legacyBytes);
		expect(existsSync(join(root, "autorag.config.json"))).toBe(true);
	});

	it("anchors a no-workspace legacy relative search path to its original cwd across invocations", () => {
		const home = join(root, "home");
		const otherCwd = join(root, "other");
		const legacyPath = join(root, "autorag.config.json");
		const migratedPath = join(home, ".autorag", "config.json");
		mkdirSync(otherCwd, { recursive: true });
		const legacyBytes = Buffer.from(JSON.stringify({ searchPaths: ["legacy-docs"] }));
		writeFileSync(legacyPath, legacyBytes);

		const first = resolveConfig({ flags: {}, env: { HOME: home }, cwd: root });
		const migrated = JSON.parse(readFileSync(migratedPath, "utf8"));
		const later = resolveConfig({ flags: {}, env: { HOME: home }, cwd: otherCwd });

		expect(first.searchPaths).toEqual([join(root, "legacy-docs")]);
		expect(first.workspacePath).toBe(root);
		expect(migrated).toMatchObject({ searchPaths: [join(root, "legacy-docs")], workspacePath: root });
		expect(readFileSync(legacyPath)).toEqual(legacyBytes);
		expect(later.searchPaths).toEqual([join(root, "legacy-docs")]);
		expect(later.workspacePath).toBe(root);
	});

	it("normalizes a relative legacy workspace against the legacy config directory across invocations", () => {
		const home = join(root, "home");
		const legacyWorkspace = join(root, "legacy-workspace");
		const otherCwd = join(root, "other");
		const legacyPath = join(legacyWorkspace, "autorag.config.json");
		const migratedPath = join(home, ".autorag", "config.json");
		mkdirSync(legacyWorkspace, { recursive: true });
		mkdirSync(otherCwd, { recursive: true });
		writeFileSync(legacyPath, JSON.stringify({ workspacePath: ".", searchPaths: ["docs"] }), "utf8");

		const first = resolveConfig({ flags: {}, env: { HOME: home }, cwd: legacyWorkspace });
		const migrated = JSON.parse(readFileSync(migratedPath, "utf8"));
		const later = resolveConfig({ flags: {}, env: { HOME: home }, cwd: otherCwd });

		expect(first.workspacePath).toBe(legacyWorkspace);
		expect(first.searchPaths).toEqual([join(legacyWorkspace, "docs")]);
		expect(migrated).toMatchObject({ workspacePath: legacyWorkspace, searchPaths: [join(legacyWorkspace, "docs")] });
		expect(later.workspacePath).toBe(legacyWorkspace);
		expect(later.searchPaths).toEqual([join(legacyWorkspace, "docs")]);
	});

	it("keeps a migrated legacy relative memory path stable across cwd changes", () => {
		const home = join(root, "home");
		const legacyDir = join(root, "legacy");
		const otherCwd = join(root, "other");
		const migratedPath = join(home, ".autorag", DEFAULT_CONFIG_FILENAME);
		mkdirSync(legacyDir, { recursive: true });
		mkdirSync(otherCwd, { recursive: true });
		writeFileSync(
			join(legacyDir, "autorag.config.json"),
			JSON.stringify({ searchPaths: ["docs"], workspacePath: "workspace", memoryPath: "memory.json" }),
			"utf8",
		);

		const first = resolveConfig({ flags: {}, env: { HOME: home }, cwd: legacyDir });
		const second = resolveConfig({ flags: {}, env: { HOME: home }, cwd: otherCwd });
		const expectedWorkspace = resolve(legacyDir, "workspace");
		// Legacy relative memory paths follow the same workspace-relative rule after migration.
		const expectedMemory = resolve(expectedWorkspace, "memory.json");

		expect(first).toEqual(second);
		expect(first.workspacePath).toBe(expectedWorkspace);
		expect(first.memoryPath).toBe(expectedMemory);
		expect(first.searchPaths).toEqual([resolve(expectedWorkspace, "docs")]);
		expect(JSON.parse(readFileSync(migratedPath, "utf8"))).toMatchObject({ workspacePath: expectedWorkspace });
		expect(JSON.parse(readFileSync(migratedPath, "utf8"))).toMatchObject({
			searchPaths: [resolve(expectedWorkspace, "docs")],
			memoryPath: expectedMemory,
		});
	});

	it("normalizes inherited legacy paths relative to the legacy workspace", () => {
		const legacyCwd = join(root, "legacy-cwd");
		const expectedWorkspace = join(legacyCwd, "workspace");

		expect(
			normalizeLegacyConfigPaths(
				{ workspacePath: "workspace", searchPaths: ["docs"], memoryPath: "memory.json" },
				legacyCwd,
			),
		).toMatchObject({
			workspacePath: expectedWorkspace,
			searchPaths: [join(expectedWorkspace, "docs")],
			memoryPath: join(expectedWorkspace, "memory.json"),
		});
	});

	it("keeps a migrated legacy search root anchored to its original workspace across cwd changes", () => {
		const home = join(root, "home");
		const otherCwd = join(root, "other");
		mkdirSync(otherCwd, { recursive: true });
		writeFileSync(
			join(root, "autorag.config.json"),
			JSON.stringify({ searchPaths: ["legacy-docs"], workspacePath: root }),
			"utf8",
		);

		resolveConfig({ flags: {}, env: { HOME: home }, cwd: root });
		const fromOtherCwd = resolveConfig({ flags: {}, env: { HOME: home }, cwd: otherCwd });

		expect(fromOtherCwd.searchPaths).toEqual([join(root, "legacy-docs")]);
	});

	it("preserves legacy bytes and removes the same-directory temp when migration rename fails", () => {
		const home = join(root, "home");
		const legacyPath = join(root, "autorag.config.json");
		const configPath = join(home, ".autorag", DEFAULT_CONFIG_FILENAME);
		const legacyBytes = Buffer.from(
			`${JSON.stringify({ searchPaths: ["legacy-docs"], workspacePath: root }, null, 2)}\n`,
		);
		writeFileSync(legacyPath, legacyBytes);
		let renameSource: string | undefined;
		fsMock.renameSyncHook = (...args) => {
			const [source, destination] = args.map(String);
			if (destination === configPath) {
				renameSource = source;
				throw Object.assign(new Error("simulated migration rename failure"), { code: "EIO" });
			}
			return fsMock.realRenameSync?.(...args);
		};

		expect(() => resolveConfig({ flags: {}, env: { HOME: home }, cwd: root })).toThrow(
			/simulated migration rename failure/,
		);
		expect(readFileSync(legacyPath)).toEqual(legacyBytes);
		expect(existsSync(configPath)).toBe(false);
		expect(renameSource).toBeDefined();
		expect(dirname(renameSource as string)).toBe(dirname(configPath));
		expect(existsSync(renameSource as string)).toBe(false);
		expect(pendingConfigArtifacts(configPath)).toEqual([]);
	});

	it("coordinates concurrent legacy migrations without changing the source or leaving partial files", async () => {
		const home = join(root, "home");
		const legacyPath = join(root, "autorag.config.json");
		const configPath = join(home, ".autorag", DEFAULT_CONFIG_FILENAME);
		const legacyBytes = Buffer.from(
			`${JSON.stringify({ searchPaths: ["legacy-docs"], workspacePath: root }, null, 2)}\n`,
		);
		writeFileSync(legacyPath, legacyBytes);
		const script = `
			const { resolveConfig } = await import(process.env.CONFIG_MODULE_URL);
			resolveConfig({ flags: {}, env: { HOME: process.env.TEST_HOME }, cwd: process.env.TEST_CWD });
		`;

		await Promise.all(
			Array.from({ length: 6 }, () =>
				runConfigChild(script, {
					CONFIG_MODULE_URL,
					TEST_CWD: root,
					TEST_HOME: home,
				}),
			),
		);

		expect(readFileSync(legacyPath)).toEqual(legacyBytes);
		expect(JSON.parse(readFileSync(configPath, "utf8"))).toMatchObject({
			searchPaths: [join(root, "legacy-docs")],
			workspacePath: root,
		});
		expect(pendingConfigArtifacts(configPath)).toEqual([]);
	});

	it("uses the winner when exclusive legacy migration loses with EEXIST", () => {
		const home = join(root, "home");
		const legacyPath = join(root, "autorag.config.json");
		const configPath = join(home, ".autorag", DEFAULT_CONFIG_FILENAME);
		const legacyBytes = Buffer.from(`${JSON.stringify({ searchPaths: ["legacy"] })}\n`);
		const winnerBytes = Buffer.from(`${JSON.stringify({ searchPaths: ["winner"], workspacePath: root })}\n`);
		writeFileSync(legacyPath, legacyBytes);

		let writeOptions: unknown;
		fsMock.writeFileSyncHook = (...args) => {
			writeOptions = args[2];
			fsMock.realWriteFileSync?.(configPath, winnerBytes);
			const error = new Error("config already exists");
			Object.assign(error, { code: "EEXIST" });
			throw error;
		};

		try {
			const config = resolveConfig({ flags: {}, env: { HOME: home }, cwd: root });

			expect(config.searchPaths).toEqual([join(root, "winner")]);
			expect(config.workspacePath).toBe(root);
			expect(readFileSync(legacyPath)).toEqual(legacyBytes);
			expect(readFileSync(configPath)).toEqual(winnerBytes);
			expect(writeOptions).toEqual(expect.objectContaining({ flag: "wx" }));
		} finally {
			fsMock.writeFileSyncHook = undefined;
		}
	});

	it("reclaims an abandoned stale dead-owner lock without leaving cleanup artifacts", () => {
		const path = join(root, DEFAULT_CONFIG_FILENAME);
		const lockPath = `${path}.lock`;
		writeFileSync(lockPath, `${JSON.stringify({ token: "abandoned", pid: 999_999, createdAt: 0 })}\n`, "utf8");
		const staleTime = new Date(Date.now() - 60_000);
		utimesSync(lockPath, staleTime, staleTime);

		writeDefaultConfig(path, { searchPaths: ["docs"] }, { force: true, cwd: root });

		expect(JSON.parse(readFileSync(path, "utf8"))).toMatchObject({ searchPaths: [resolve(root, "docs")] });
		expect(pendingConfigArtifacts(path)).toEqual([]);
	});

	it("preserves a fresh config winner during stale-lock turnover", () => {
		const path = join(root, DEFAULT_CONFIG_FILENAME);
		const lockPath = `${path}.lock`;
		const staleContents = `${JSON.stringify({ token: "stale-owner", pid: 999_999, createdAt: 0 })}\n`;
		const winnerBytes = Buffer.from(`${JSON.stringify({ winner: true, workspacePath: root })}\n`);
		mkdirSync(lockPath, { mode: 0o700 });
		const staleMarkerPath = join(lockPath, "owner-stale-owner.json");
		writeFileSync(staleMarkerPath, staleContents, "utf8");
		const staleTime = new Date(Date.now() - 60_000);
		utimesSync(staleMarkerPath, staleTime, staleTime);
		let turnoverInjected = false;
		let freshOwnerAssertions = 0;
		let freshOwnerCommitted = false;
		let staleReaperBlocked = false;
		let competingOwnerRejected = false;
		fsMock.rmdirSyncHook = (...args) => {
			if (!turnoverInjected && String(args[0]) === lockPath) {
				turnoverInjected = true;
				if (!fsMock.realRmdirSync) throw new Error("real rmdirSync is unavailable");
				fsMock.realRmdirSync(...args);
				const freshOwner = acquireFileLock(lockPath, {
					timeoutMs: 1_000,
					staleMs: 30_000,
					retryMs: 1,
					timeoutError: () => new Error("fresh turnover owner could not acquire the config lock"),
				});
				let delayedReaperError: unknown;
				try {
					freshOwner.assertOwned();
					freshOwnerAssertions++;
					try {
						fsMock.realRmdirSync(lockPath);
					} catch (error) {
						if (
							!(
								error instanceof Error &&
								"code" in error &&
								(error.code === "ENOTEMPTY" || error.code === "EEXIST")
							)
						) {
							throw error;
						}
						staleReaperBlocked = true;
						delayedReaperError = error;
					}
					freshOwner.assertOwned();
					freshOwnerAssertions++;
					expect(() =>
						acquireFileLock(lockPath, {
							timeoutMs: 0,
							staleMs: 30_000,
							retryMs: 1,
							timeoutError: () => {
								competingOwnerRejected = true;
								return new Error("turnover competitor could not acquire the config lock");
							},
						}),
					).toThrow("turnover competitor could not acquire the config lock");
					writeFileSync(path, winnerBytes);
					freshOwnerCommitted = true;
				} finally {
					freshOwner.release();
				}
				if (delayedReaperError !== undefined) throw delayedReaperError;
				throw new Error("delayed stale reaper unexpectedly removed the fresh config lock");
			}
			return fsMock.realRmdirSync?.(...args);
		};

		expect(() => writeDefaultConfig(path, {}, { cwd: root })).toThrow(ConfigError);

		expect(turnoverInjected).toBe(true);
		expect(freshOwnerAssertions).toBe(2);
		expect(freshOwnerCommitted).toBe(true);
		expect(staleReaperBlocked).toBe(true);
		expect(competingOwnerRejected).toBe(true);
		expect(readFileSync(path)).toEqual(winnerBytes);
		expect(pendingConfigArtifacts(path)).toEqual([]);
	});

	it("times out on a live current-owner lock without deleting it", () => {
		const path = join(root, DEFAULT_CONFIG_FILENAME);
		const lockPath = `${path}.lock`;
		const lockContents = `${JSON.stringify({ token: "live", pid: process.pid, createdAt: Date.now() })}\n`;
		writeFileSync(lockPath, lockContents, "utf8");
		let clock = Date.now();
		const now = vi.spyOn(Date, "now").mockImplementation(() => {
			clock += 20_000;
			return clock;
		});

		try {
			expect(() => writeDefaultConfig(path, {}, { force: true, cwd: root })).toThrow(
				/Timed out waiting to write config file/,
			);
			expect(existsSync(lockPath)).toBe(true);
			expect(readFileSync(lockPath, "utf8")).toBe(lockContents);
			expect(pendingConfigArtifacts(path)).toEqual([`${DEFAULT_CONFIG_FILENAME}.lock`]);
		} finally {
			now.mockRestore();
		}
	});

	it("uses default searchPaths, workspacePath, and memoryPath when nothing is provided", () => {
		const config = resolveConfig({ flags: {}, cwd: root });
		expect(config.searchPaths).toEqual(["."]);
		expect(config.workspacePath).toBe(root);
		expect(config.memoryPath).toBe(join(process.env.HOME as string, ".autorag", "memory.json"));
		expect(config.model).toBeUndefined();
	});

	it("does not require a config file to exist at cwd", () => {
		expect(existsSync(join(root, DEFAULT_CONFIG_FILENAME))).toBe(false);
		const config = resolveConfig({ flags: {}, cwd: root });
		expect(config.searchPaths).toEqual(["."]);
	});
});

describe("resolveConfig precedence", () => {
	it("flag overrides env overrides file overrides defaults", () => {
		writeConfigFile(root, {
			searchPaths: ["./file"],
			workspacePath: "/file/workspace",
			memoryPath: "/file/memory.json",
			model: { provider: "fileprov", id: "fileid" },
		});

		const fileOnly = resolveConfig({
			flags: {},
			env: {},
			cwd: root,
		});
		expect(fileOnly.searchPaths).toEqual(["/file/workspace/file"]);
		expect(fileOnly.workspacePath).toBe("/file/workspace");
		expect(fileOnly.memoryPath).toBe("/file/memory.json");
		expect(fileOnly.model).toEqual({ provider: "fileprov", id: "fileid" });

		const envOnly = resolveConfig({
			flags: {},
			env: {
				AUTORAG_SEARCH_PATHS: "./env1,./env2",
				AUTORAG_WORKSPACE: "/env/workspace",
				AUTORAG_MEMORY_PATH: "/env/memory.json",
				AUTORAG_MODEL_PROVIDER: "envprov",
				AUTORAG_MODEL_ID: "envid",
			},
			cwd: root,
		});
		expect(envOnly.searchPaths).toEqual(["./env1", "./env2"]);
		expect(envOnly.workspacePath).toBe("/env/workspace");
		expect(envOnly.memoryPath).toBe("/env/memory.json");
		expect(envOnly.model).toEqual({ provider: "envprov", id: "envid" });

		const flagOnly = resolveConfig({
			flags: {
				"search-paths": "./flag1,./flag2",
				workspace: "/flag/workspace",
				"memory-path": "/flag/memory.json",
				"model-provider": "flagprov",
				"model-id": "flagid",
			},
			env: {
				AUTORAG_SEARCH_PATHS: "./env1,./env2",
				AUTORAG_WORKSPACE: "/env/workspace",
				AUTORAG_MEMORY_PATH: "/env/memory.json",
				AUTORAG_MODEL_PROVIDER: "envprov",
				AUTORAG_MODEL_ID: "envid",
			},
			cwd: root,
		});
		expect(flagOnly.searchPaths).toEqual(["./flag1", "./flag2"]);
		expect(flagOnly.workspacePath).toBe("/flag/workspace");
		expect(flagOnly.memoryPath).toBe("/flag/memory.json");
		expect(flagOnly.model).toEqual({ provider: "flagprov", id: "flagid" });
	});

	it("flag overrides env even when file is absent", () => {
		const config = resolveConfig({
			flags: { workspace: "/flag/workspace" },
			env: { AUTORAG_WORKSPACE: "/env/workspace" },
			cwd: root,
		});
		expect(config.workspacePath).toBe("/flag/workspace");
	});

	it("env overrides file when flag is absent", () => {
		writeConfigFile(root, { workspacePath: "/file/workspace" });
		const config = resolveConfig({
			flags: {},
			env: { AUTORAG_WORKSPACE: "/env/workspace" },
			cwd: root,
		});
		expect(config.workspacePath).toBe("/env/workspace");
	});

	it("file overrides defaults when flag and env are absent", () => {
		writeConfigFile(root, { workspacePath: "/file/workspace" });
		const config = resolveConfig({ flags: {}, env: {}, cwd: root });
		expect(config.workspacePath).toBe("/file/workspace");
	});
});

describe("resolveConfig env var mapping", () => {
	it("maps AUTORAG_SEARCH_PATHS csv, AUTORAG_MODEL_PROVIDER, and AUTORAG_MODEL_ID", () => {
		const config = resolveConfig({
			flags: {},
			env: {
				AUTORAG_SEARCH_PATHS: "src,test,docs",
				AUTORAG_MODEL_PROVIDER: "openai",
				AUTORAG_MODEL_ID: "gpt-4o",
			},
			cwd: root,
		});
		expect(config.searchPaths).toEqual(["src", "test", "docs"]);
		expect(config.model).toEqual({ provider: "openai", id: "gpt-4o" });
	});

	it("maps AUTORAG_CONFIG to an explicit config file path", () => {
		const alt = join(root, "alt.config.json");
		writeFileSync(alt, JSON.stringify({ workspacePath: "/alt/workspace" }), "utf8");
		const config = resolveConfig({
			flags: {},
			env: { AUTORAG_CONFIG: alt },
			cwd: root,
		});
		expect(config.workspacePath).toBe("/alt/workspace");
	});

	it("maps flags.config to an explicit config file path", () => {
		const alt = join(root, "alt.config.json");
		writeFileSync(alt, JSON.stringify({ workspacePath: "/alt/workspace" }), "utf8");
		const config = resolveConfig({
			flags: { config: alt },
			env: {},
			cwd: root,
		});
		expect(config.workspacePath).toBe("/alt/workspace");
	});

	it("throws ConfigError when an explicit config file does not exist", () => {
		expect(() =>
			resolveConfig({
				flags: { config: join(root, "missing.json") },
				env: {},
				cwd: root,
			}),
		).toThrow(ConfigError);
	});
});

describe("resolveConfig model partial handling", () => {
	it("rejects an orchestrator provider without an id", () => {
		expect(() =>
			resolveConfig({
				flags: { "model-provider": "openai" },
				env: {},
				cwd: root,
			}),
		).toThrow(/orchestrator requires both provider and id/i);
	});

	it("rejects an orchestrator id without a provider", () => {
		expect(() =>
			resolveConfig({
				flags: { "model-id": "gpt-4o" },
				env: {},
				cwd: root,
			}),
		).toThrow(/orchestrator requires both provider and id/i);
	});
});

describe("buildAgentOptions", () => {
	it("always includes searchPaths and only includes optional keys when present", () => {
		const minimal = buildAgentOptions({
			searchPaths: ["."],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
		});
		expect(minimal.searchPaths).toEqual(["."]);
		expect(minimal.workspacePath).toBe(root);
		expect(minimal.memoryPath).toBe(join(root, "memory.json"));
		expect(minimal.minSync).toBe(false);
		expect(minimal.bm25).toBe(false);
		expect("jikji" in minimal).toBe(false);
		expect("parserOptions" in minimal).toBe(false);
		expect("model" in minimal).toBe(false);
	});

	it("includes optional keys when present and enabled, stripping the enabled flag", () => {
		const opts = buildAgentOptions({
			searchPaths: ["."],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: { enabled: true, autoInstall: false, binaryPath: "/bin/minsync" },
			bm25: { enabled: true, forceEngine: "typescript-fallback" },
			jikji: { baz: 3 },
			parserOptions: { qux: 4 },
		});
		expect(opts.minSync).toEqual({ autoInstall: false, binaryPath: "/bin/minsync" });
		expect(opts.bm25).toEqual({ forceEngine: "typescript-fallback" });
		expect("enabled" in (opts.minSync as object)).toBe(false);
		expect("enabled" in (opts.bm25 as object)).toBe(false);
		expect(opts.jikji).toEqual({ baz: 3 });
		expect(opts.parserOptions).toEqual({ qux: 4 });
	});

	it("passes false for minSync and bm25 when enabled is false", () => {
		const opts = buildAgentOptions({
			searchPaths: ["."],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: { enabled: false },
			bm25: { enabled: false },
		});
		expect(opts.minSync).toBe(false);
		expect(opts.bm25).toBe(false);
	});

	it("passes false for minSync and bm25 when absent from config", () => {
		const opts = buildAgentOptions({
			searchPaths: ["."],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
		});
		expect(opts.minSync).toBe(false);
		expect(opts.bm25).toBe(false);
	});
});

describe("resolveModel", () => {
	it("throws ConfigError when model is absent and message names --model-provider and the model config key", () => {
		const config: CliConfig = {
			searchPaths: ["."],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
		};
		expect(() => resolveModel(config)).toThrow(ConfigError);
		expect(() => resolveModel(config)).toThrow(/--model-provider/);
		expect(() => resolveModel(config)).toThrow(/model/);
	});

	it("throws ConfigError naming a configured model missing from getModel", () => {
		const config: CliConfig = {
			searchPaths: [root],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			model: { provider: "openai", id: "missing-autorag-model" },
		};

		expect(() => resolveModel(config)).toThrow(ConfigError);
		expect(() => resolveModel(config)).toThrow(/openai\/missing-autorag-model/);
	});
});

describe("resolveAgentModel", () => {
	it.each([
		["orchestrator", "missing-orchestrator-model"],
		["explorer", "missing-explorer-model"],
	] as const)("rejects an unknown configured %s model before returning agent options", (role, missingId) => {
		const known = { provider: "openai", id: "gpt-4o" };
		const config: CliConfig = {
			searchPaths: [root],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			agents: {
				orchestrator: role === "orchestrator" ? { provider: "openai", id: missingId } : known,
				explorer: role === "explorer" ? { provider: "openai", id: missingId } : known,
			},
		};

		expect(() => resolveAgentModel(config)).toThrow(ConfigError);
		expect(() => resolveAgentModel(config)).toThrow(new RegExp(`openai/${missingId}`));
	});

	it("resolves built-in role models without loading local Codex configuration", () => {
		const config: CliConfig = {
			searchPaths: [root],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			agents: {
				orchestrator: { provider: "openai", id: "gpt-4o" },
				explorer: { provider: "openai", id: "gpt-4o" },
			},
		};

		const resolved = resolveAgentModel(config, { configPath: join(root, "missing-codex-config.toml") });

		expect(resolved.model).toMatchObject({ provider: "openai", id: "gpt-4o" });
		expect(resolved.explorerModel).toMatchObject({ provider: "openai", id: "gpt-4o" });
		expect(resolved.apiKey).toBeUndefined();
		expect(resolved.providerApiKeys).toBeUndefined();
	});

	it("resolves same-provider roles from the active corp-proxy Codex provider", () => {
		const codexConfigPath = join(root, "config.toml");
		writeFileSync(
			codexConfigPath,
			'model_provider = "corp-proxy"\n[model_providers.corp-proxy]\nbase_url = "https://corp.example/v1"\nwire_api = "responses"\nenv_key = "CORP_PROXY_KEY"\n',
			"utf8",
		);
		const config: CliConfig = {
			searchPaths: [root],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			agents: {
				orchestrator: { provider: "corp-proxy", id: "gpt-5.6-sol-corp" },
				explorer: { provider: "corp-proxy", id: "gpt-5.6-luna-corp" },
			},
		};

		const resolved = resolveAgentModel(config, {
			configPath: codexConfigPath,
			env: { CORP_PROXY_KEY: "corp-secret" },
		});

		expect(resolved.model).toMatchObject({ provider: "corp-proxy", id: "gpt-5.6-sol-corp" });
		expect(resolved.explorerModel).toMatchObject({ provider: "corp-proxy", id: "gpt-5.6-luna-corp" });
		expect(resolved.apiKey).toBe("corp-secret");
		expect(resolved.providerApiKeys).toEqual({ "corp-proxy": "corp-secret" });
	});

	it("resolves a built-in orchestrator with an active corp-proxy explorer", () => {
		const codexConfigPath = join(root, "config.toml");
		writeFileSync(
			codexConfigPath,
			'model_provider = "corp-proxy"\n[model_providers.corp-proxy]\nbase_url = "https://corp.example/v1"\nwire_api = "responses"\nenv_key = "CORP_PROXY_KEY"\n',
			"utf8",
		);
		const config: CliConfig = {
			searchPaths: [root],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			agents: {
				orchestrator: { provider: "openai", id: "gpt-4o" },
				explorer: { provider: "corp-proxy", id: "gpt-5.6-luna-corp" },
			},
		};

		const resolved = resolveAgentModel(config, {
			configPath: codexConfigPath,
			env: { CORP_PROXY_KEY: "corp-secret" },
		});

		expect(resolved.model).toMatchObject({ provider: "openai", id: "gpt-4o" });
		expect(resolved.explorerModel).toMatchObject({ provider: "corp-proxy", id: "gpt-5.6-luna-corp" });
		expect(resolved.apiKey).toBeUndefined();
		expect(resolved.providerApiKeys).toEqual({ "corp-proxy": "corp-secret" });
	});

	it("resolves independently configured orchestrator and explorer models", () => {
		const codexConfigPath = join(root, "config.toml");
		writeFileSync(
			codexConfigPath,
			'model_provider = "test-proxy"\n[model_providers.test-proxy]\nbase_url = "https://proxy.example/v1"\nwire_api = "responses"\nenv_key = "TEST_PROXY_KEY"\n',
			"utf8",
		);
		writeConfigFile(root, {
			searchPaths: ["."],
			workspacePath: root,
			memoryPath: join(process.env.HOME as string, ".autorag", "memory.json"),
			agents: {
				orchestrator: { provider: "test-proxy", id: "gpt-5.6-sol-custom" },
				explorer: { provider: "test-proxy", id: "gpt-5.6-luna-custom" },
			},
		} as Partial<CliConfig>);
		const config = resolveConfig({ flags: {}, env: {}, cwd: root });
		expect(config).toMatchObject({
			agents: {
				orchestrator: { provider: "test-proxy", id: "gpt-5.6-sol-custom" },
				explorer: { provider: "test-proxy", id: "gpt-5.6-luna-custom" },
			},
		});
		const resolved = resolveAgentModel(config, {
			configPath: codexConfigPath,
			env: { TEST_PROXY_KEY: "secret" },
		});
		expect(resolved.model.id).toBe("gpt-5.6-sol-custom");
		expect((resolved as typeof resolved & { explorerModel: Model<"openai-responses"> }).explorerModel.id).toBe(
			"gpt-5.6-luna-custom",
		);
	});

	it("loads Sol and its API key from the active Codex provider when CLI model is omitted", () => {
		const codexConfigPath = join(root, "config.toml");
		writeFileSync(
			codexConfigPath,
			'model_provider = "test-proxy"\n[model_providers.test-proxy]\nbase_url = "https://proxy.example/v1"\nwire_api = "responses"\nenv_key = "TEST_PROXY_KEY"\n',
			"utf8",
		);
		const config: CliConfig = {
			searchPaths: ["."],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
		};

		const resolved = resolveAgentModel(config, {
			configPath: codexConfigPath,
			env: { TEST_PROXY_KEY: "secret" },
		});

		expect(resolved.model.id).toBe("gpt-5.6-sol");
		expect(resolved.model.provider).toBe("test-proxy");
		expect(resolved.apiKey).toBe("secret");
		expect(resolved.providerApiKeys).toEqual({ "test-proxy": "secret" });
	});

	it("returns a provider-scoped explorer credential without setting the orchestrator apiKey", () => {
		const codexConfigPath = join(root, "config.toml");
		writeFileSync(
			codexConfigPath,
			' model_provider = "test-proxy"\n[model_providers.test-proxy]\nbase_url = "https://proxy.example/v1"\nwire_api = "responses"\nenv_key = "TEST_PROXY_KEY"\n'.trimStart(),
			"utf8",
		);
		const config: CliConfig = {
			searchPaths: ["."],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			agents: {
				orchestrator: { provider: "openai", id: "gpt-4o" },
				explorer: { provider: "test-proxy", id: "gpt-5.6-luna" },
			},
		};

		const resolved = resolveAgentModel(config, {
			configPath: codexConfigPath,
			env: { TEST_PROXY_KEY: "secret" },
		});

		expect(resolved.model.provider).toBe("openai");
		expect(resolved.explorerModel.provider).toBe("test-proxy");
		expect(resolved.apiKey).toBeUndefined();
		expect(resolved.providerApiKeys).toEqual({ "test-proxy": "secret" });
	});
});

describe("writeDefaultConfig", () => {
	it("writes autorag.config.json with defaults when partial is empty", () => {
		const path = join(root, DEFAULT_CONFIG_FILENAME);
		writeDefaultConfig(path, {});
		expect(existsSync(path)).toBe(true);
		const written = JSON.parse(readFileSync(path, "utf8")) as CliConfig;
		expect(written.searchPaths).toEqual([process.cwd()]);
		expect(typeof written.workspacePath).toBe("string");
		expect(typeof written.memoryPath).toBe("string");
		expect(written.agents).toBeUndefined();
	});

	it("throws ConfigError when the file already exists and force is not set", () => {
		const path = join(root, DEFAULT_CONFIG_FILENAME);
		writeDefaultConfig(path, {});
		expect(() => writeDefaultConfig(path, {})).toThrow(ConfigError);
	});

	it("uses exclusive create when a concurrent no-force writer wins", () => {
		const path = join(root, DEFAULT_CONFIG_FILENAME);
		const winnerBytes = Buffer.from(`${JSON.stringify({ winner: true })}\n`);
		let writeOptions: unknown;
		fsMock.writeFileSyncHook = (...args) => {
			writeOptions = args[2];
			fsMock.realWriteFileSync?.(path, winnerBytes);
			const error = new Error("config already exists");
			Object.assign(error, { code: "EEXIST" });
			throw error;
		};

		try {
			expect(() => writeDefaultConfig(path, {})).toThrow(ConfigError);
			expect(readFileSync(path)).toEqual(winnerBytes);
			expect(writeOptions).toEqual(expect.objectContaining({ encoding: "utf8", flag: "wx" }));
		} finally {
			fsMock.writeFileSyncHook = undefined;
		}
	});

	it("overwrites the existing file when force is true", () => {
		const path = join(root, DEFAULT_CONFIG_FILENAME);
		writeDefaultConfig(path, {});
		writeDefaultConfig(path, { searchPaths: ["src"] }, { force: true });
		const written = JSON.parse(readFileSync(path, "utf8")) as CliConfig;
		expect(written.searchPaths).toEqual([resolve(process.cwd(), "src")]);
	});

	it("preserves the existing config and cleans the same-directory temp when forced rename fails", () => {
		const path = join(root, DEFAULT_CONFIG_FILENAME);
		const originalBytes = Buffer.from(`${JSON.stringify({ original: true })}\n`);
		writeFileSync(path, originalBytes);
		let renameSource: string | undefined;
		fsMock.renameSyncHook = (...args) => {
			const [source, destination] = args.map(String);
			if (destination === path) {
				renameSource = source;
				throw Object.assign(new Error("simulated force rename failure"), { code: "EIO" });
			}
			return fsMock.realRenameSync?.(...args);
		};

		expect(() => writeDefaultConfig(path, { searchPaths: ["replacement"] }, { force: true, cwd: root })).toThrow(
			/simulated force rename failure/,
		);
		expect(readFileSync(path)).toEqual(originalBytes);
		expect(renameSource).toBeDefined();
		expect(dirname(renameSource as string)).toBe(dirname(path));
		expect(existsSync(renameSource as string)).toBe(false);
		expect(pendingConfigArtifacts(path)).toEqual([]);
	});

	it("keeps concurrent forced writers crash-atomic and leaves one complete config", async () => {
		const path = join(root, DEFAULT_CONFIG_FILENAME);
		writeFileSync(path, `${JSON.stringify({ original: true })}\n`);
		const markers = Array.from({ length: 6 }, (_, index) => `writer-${index}`);
		const script = `
			const { writeDefaultConfig } = await import(process.env.CONFIG_MODULE_URL);
			const marker = process.env.TEST_MARKER;
			writeDefaultConfig(
				process.env.TEST_CONFIG_PATH,
				{
					searchPaths: [marker],
					workspacePath: process.env.TEST_CWD,
					memoryPath: process.env.TEST_MEMORY_PATH,
					parserOptions: { marker, payload: marker.repeat(200000) },
				},
				{ force: true, cwd: process.env.TEST_CWD },
			);
		`;

		await Promise.all(
			markers.map((marker) =>
				runConfigChild(script, {
					CONFIG_MODULE_URL,
					TEST_CONFIG_PATH: path,
					TEST_CWD: root,
					TEST_MARKER: marker,
					TEST_MEMORY_PATH: join(root, "memory.json"),
				}),
			),
		);

		const written = JSON.parse(readFileSync(path, "utf8")) as CliConfig & {
			parserOptions: { marker: string; payload: string };
		};
		expect(markers).toContain(written.parserOptions.marker);
		expect(written.parserOptions.payload).toBe(written.parserOptions.marker.repeat(200000));
		expect(written.searchPaths).toEqual([resolve(root, written.parserOptions.marker)]);
		expect(pendingConfigArtifacts(path)).toEqual([]);
	});

	it("preserves provided partial values", () => {
		const path = join(root, DEFAULT_CONFIG_FILENAME);
		const absoluteSearchPath = `${root}/docs/../docs`;
		const absoluteWorkspacePath = `${root}/workspace/../workspace`;
		const absoluteMemoryPath = `${root}/memory/../memory.json`;
		writeDefaultConfig(path, {
			searchPaths: [absoluteSearchPath],
			workspacePath: absoluteWorkspacePath,
			memoryPath: absoluteMemoryPath,
			model: { provider: "openai", id: "gpt-4o" },
		});
		const written = JSON.parse(readFileSync(path, "utf8")) as CliConfig;
		expect(written.searchPaths).toEqual([absoluteSearchPath]);
		expect(written.workspacePath).toBe(absoluteWorkspacePath);
		expect(written.memoryPath).toBe(absoluteMemoryPath);
		expect(written.model).toEqual({ provider: "openai", id: "gpt-4o" });
	});

	it("persists relative workspace and memory inputs as absolute stable paths", () => {
		const path = join(root, "written", DEFAULT_CONFIG_FILENAME);
		const cwd = join(root, "caller");
		mkdirSync(cwd, { recursive: true });
		writeDefaultConfig(
			path,
			{ searchPaths: ["docs"], workspacePath: "workspace", memoryPath: "memory.json" },
			{ cwd, env: { HOME: join(root, "home") } },
		);

		const written = JSON.parse(readFileSync(path, "utf8")) as CliConfig;
		const expectedWorkspace = resolve(cwd, "workspace");

		expect(written.workspacePath).toBe(expectedWorkspace);
		expect(written.memoryPath).toBe(resolve(expectedWorkspace, "memory.json"));
		expect(written.searchPaths).toEqual([resolve(cwd, "docs")]);
	});
});

describe("normalizeIndexingConfig", () => {
	it("resolves missing keys to enabled defaults with minSync autoInstall false", () => {
		const result = normalizeIndexingConfig({});
		expect(result.bm25).toEqual({ enabled: true });
		expect(result.minSync).toEqual({ enabled: true, autoInstall: false });
	});

	it("resolves false to disabled markers", () => {
		const result = normalizeIndexingConfig({ bm25: false, minSync: false });
		expect(result.bm25).toEqual({ enabled: false });
		expect(result.minSync).toEqual({ enabled: false });
	});

	it("merges an object with enabled:true default", () => {
		const result = normalizeIndexingConfig({
			bm25: { forceEngine: "typescript-fallback" },
			minSync: { autoInstall: true, binaryPath: "/bin/minsync" },
		});
		expect(result.bm25).toEqual({ enabled: true, forceEngine: "typescript-fallback" });
		expect(result.minSync).toEqual({ enabled: true, autoInstall: true, binaryPath: "/bin/minsync" });
	});

	it("preserves explicit enabled:false on an object", () => {
		const result = normalizeIndexingConfig({
			bm25: { enabled: false, forceEngine: "typescript-fallback" },
			minSync: { enabled: false, autoInstall: true },
		});
		expect(result.bm25.enabled).toBe(false);
		expect(result.minSync.enabled).toBe(false);
	});

	it("rejects unknown bm25 fields", () => {
		expect(() => normalizeIndexingConfig({ bm25: { unknownField: 1 } as unknown as Bm25MethodConfig })).toThrow(
			ConfigError,
		);
		expect(() => normalizeIndexingConfig({ bm25: { unknownField: 1 } as unknown as Bm25MethodConfig })).toThrow(
			/bm25\.unknownField/,
		);
	});

	it("rejects unknown minSync fields", () => {
		expect(() => normalizeIndexingConfig({ minSync: { unknownField: 1 } as unknown as MinSyncMethodConfig })).toThrow(
			ConfigError,
		);
		expect(() => normalizeIndexingConfig({ minSync: { unknownField: 1 } as unknown as MinSyncMethodConfig })).toThrow(
			/minSync\.unknownField/,
		);
	});

	it("drops invalid fallback enum values silently", () => {
		const result = normalizeIndexingConfig({
			bm25: { fallback: "invalid" as unknown as Bm25MethodConfig["fallback"] },
		});
		expect(result.bm25.fallback).toBeUndefined();
	});
});

describe("normalizeEmbedder", () => {
	it("returns empty object for undefined", () => {
		expect(normalizeEmbedder(undefined, "minSync.embedder")).toEqual({});
	});

	it("validates apiKeyEnv regex", () => {
		expect(() => normalizeEmbedder({ apiKeyEnv: "1invalid" }, "minSync.embedder")).toThrow(ConfigError);
		expect(() => normalizeEmbedder({ apiKeyEnv: "1invalid" }, "minSync.embedder")).toThrow(/apiKeyEnv/);
		expect(() => normalizeEmbedder({ apiKeyEnv: "OPENAI_API_KEY" }, "minSync.embedder")).not.toThrow();
	});

	it("rejects extraArgs as an unknown field", () => {
		expect(() =>
			normalizeEmbedder({ extraArgs: ["--foo"] } as unknown as Record<string, unknown>, "minSync.embedder"),
		).toThrow(ConfigError);
		expect(() =>
			normalizeEmbedder({ extraArgs: ["--foo"] } as unknown as Record<string, unknown>, "minSync.embedder"),
		).toThrow(/extraArgs/);
	});

	it("rejects non-positive dimension", () => {
		expect(() => normalizeEmbedder({ dimension: 0 }, "minSync.embedder")).toThrow(ConfigError);
		expect(() => normalizeEmbedder({ dimension: -1 }, "minSync.embedder")).toThrow(ConfigError);
		expect(() => normalizeEmbedder({ dimension: 1.5 }, "minSync.embedder")).toThrow(ConfigError);
	});

	it("rejects non-positive timeoutMs", () => {
		expect(() => normalizeEmbedder({ timeoutMs: 0 }, "minSync.embedder")).toThrow(ConfigError);
	});

	it("accepts a valid embedder config", () => {
		const result = normalizeEmbedder(
			{ id: "openai:text-embedding-3-small", apiKeyEnv: "OPENAI_API_KEY", dimension: 1536, batchSize: 64 },
			"minSync.embedder",
		);
		expect(result).toEqual({
			id: "openai:text-embedding-3-small",
			apiKeyEnv: "OPENAI_API_KEY",
			dimension: 1536,
			batchSize: 64,
		});
	});

	it("validates embedder inside minSync via normalizeIndexingConfig", () => {
		expect(() =>
			normalizeIndexingConfig({
				minSync: { embedder: { apiKeyEnv: "1bad" } } as unknown as MinSyncMethodConfig,
			}),
		).toThrow(ConfigError);
		expect(() =>
			normalizeIndexingConfig({
				minSync: { embedder: { extraArgs: [] } } as unknown as MinSyncMethodConfig,
			}),
		).toThrow(/extraArgs/);
	});
});

describe("resolveConfig indexing defaults", () => {
	it("resolves missing minSync and bm25 to enabled defaults", () => {
		const config = resolveConfig({ flags: {}, cwd: root });
		expect(config.bm25).toEqual({ enabled: true });
		expect(config.minSync).toEqual({ enabled: true, autoInstall: false });
	});

	it("resolves minSync:false to a disabled marker", () => {
		writeConfigFile(root, { minSync: false as unknown as MinSyncMethodConfig });
		const config = resolveConfig({ flags: {}, cwd: root });
		expect(config.minSync).toEqual({ enabled: false });
	});

	it("resolves bm25:false to a disabled marker", () => {
		writeConfigFile(root, { bm25: false as unknown as Bm25MethodConfig });
		const config = resolveConfig({ flags: {}, cwd: root });
		expect(config.bm25).toEqual({ enabled: false });
	});

	it("normalizes a file with typed method config and preserves fields", () => {
		writeConfigFile(root, {
			bm25: { forceEngine: "typescript-fallback" } as unknown as Bm25MethodConfig,
			minSync: { autoInstall: true, binaryPath: "/bin/minsync" } as unknown as MinSyncMethodConfig,
		});
		const config = resolveConfig({ flags: {}, cwd: root });
		expect(config.bm25).toEqual({ enabled: true, forceEngine: "typescript-fallback" });
		expect(config.minSync).toEqual({ enabled: true, autoInstall: true, binaryPath: "/bin/minsync" });
	});

	it("throws ConfigError for invalid apiKeyEnv in file minSync.embedder", () => {
		writeConfigFile(root, {
			minSync: { embedder: { apiKeyEnv: "1invalid" } } as unknown as MinSyncMethodConfig,
		});
		expect(() => resolveConfig({ flags: {}, cwd: root })).toThrow(ConfigError);
		expect(() => resolveConfig({ flags: {}, cwd: root })).toThrow(/apiKeyEnv/);
	});

	it("throws ConfigError for unknown embedder field in file minSync.embedder", () => {
		writeConfigFile(root, {
			minSync: { embedder: { extraArgs: ["--foo"] } } as unknown as MinSyncMethodConfig,
		});
		expect(() => resolveConfig({ flags: {}, cwd: root })).toThrow(ConfigError);
		expect(() => resolveConfig({ flags: {}, cwd: root })).toThrow(/extraArgs/);
	});
});

describe("writeDefaultConfig indexing defaults", () => {
	it("writes bm25 and minSync enabled defaults when not explicitly provided", () => {
		const path = join(root, DEFAULT_CONFIG_FILENAME);
		writeDefaultConfig(path, {});
		const written = JSON.parse(readFileSync(path, "utf8")) as CliConfig;
		expect(written.bm25).toEqual({ enabled: true });
		expect(written.minSync).toEqual({ enabled: true, autoInstall: false });
	});

	it("never injects an embedder id default", () => {
		const path = join(root, DEFAULT_CONFIG_FILENAME);
		writeDefaultConfig(path, {});
		const written = JSON.parse(readFileSync(path, "utf8")) as CliConfig;
		expect(written.minSync?.embedder).toBeUndefined();
	});

	it("preserves a partial minSync embedder config without injecting id", () => {
		const path = join(root, DEFAULT_CONFIG_FILENAME);
		writeDefaultConfig(path, {
			minSync: { embedder: { apiKeyEnv: "OPENAI_API_KEY", dimension: 1536 } } as MinSyncMethodConfig,
		});
		const written = JSON.parse(readFileSync(path, "utf8")) as CliConfig;
		expect(written.minSync?.enabled).toBe(true);
		expect(written.minSync?.autoInstall).toBe(false);
		expect(written.minSync?.embedder).toEqual({ apiKeyEnv: "OPENAI_API_KEY", dimension: 1536 });
		expect(written.minSync?.embedder?.id).toBeUndefined();
	});

	it("preserves explicit bm25 disabled", () => {
		const path = join(root, DEFAULT_CONFIG_FILENAME);
		writeDefaultConfig(path, { bm25: false as unknown as Bm25MethodConfig });
		const written = JSON.parse(readFileSync(path, "utf8")) as CliConfig;
		expect(written.bm25).toEqual({ enabled: false });
	});

	it("preserves explicit minSync disabled", () => {
		const path = join(root, DEFAULT_CONFIG_FILENAME);
		writeDefaultConfig(path, { minSync: false as unknown as MinSyncMethodConfig });
		const written = JSON.parse(readFileSync(path, "utf8")) as CliConfig;
		expect(written.minSync).toEqual({ enabled: false });
	});
});

describe("resolveConfigReadOnly", () => {
	it("reads a legacy cwd config in memory without creating the home config or locks", () => {
		const home = join(root, "home");
		const homeConfigDir = join(home, ".autorag");
		const legacyPath = join(root, LEGACY_CONFIG_FILENAME);
		writeFileSync(legacyPath, JSON.stringify({ searchPaths: ["legacy-docs"], workspacePath: root }), "utf8");

		const config = resolveConfigReadOnly({ flags: {}, env: { HOME: home }, cwd: root });

		expect(config.searchPaths).toEqual([join(root, "legacy-docs")]);
		expect(config.workspacePath).toBe(root);
		// No migration write, no home config dir, and the legacy source is untouched.
		expect(existsSync(join(homeConfigDir, DEFAULT_CONFIG_FILENAME))).toBe(false);
		expect(existsSync(homeConfigDir)).toBe(false);
		expect(existsSync(legacyPath)).toBe(true);
	});

	it("still loads an explicit home config read-only without writing", () => {
		const home = join(root, "home");
		const homeConfigDir = join(home, ".autorag");
		mkdirSync(homeConfigDir, { recursive: true });
		writeFileSync(
			join(homeConfigDir, DEFAULT_CONFIG_FILENAME),
			JSON.stringify({ searchPaths: ["/home/docs"], workspacePath: "/home/workspace" }),
			"utf8",
		);

		const config = resolveConfigReadOnly({ flags: {}, env: { HOME: home }, cwd: root });

		expect(config.searchPaths).toEqual(["/home/docs"]);
		expect(config.workspacePath).toBe("/home/workspace");
		expect(pendingConfigArtifacts(join(homeConfigDir, DEFAULT_CONFIG_FILENAME))).toEqual([]);
	});

	it("falls back to defaults when neither home nor legacy config exists", () => {
		const home = join(root, "home");

		const config = resolveConfigReadOnly({ flags: {}, env: { HOME: home }, cwd: root });

		expect(config.searchPaths).toEqual(["."]);
		expect(config.workspacePath).toBe(root);
		expect(existsSync(join(home, ".autorag"))).toBe(false);
	});
});

describe("resolveAgentModelDetailed", () => {
	const openRouterEndpoint = {
		provider: "openrouter",
		id: "x-ai/grok-4.5",
		name: "Grok 4.5 (OpenRouter)",
		api: "openai-completions" as const,
		baseUrl: "https://openrouter.ai/api/v1",
		apiKeyEnv: "OPENROUTER_API_KEY",
		reasoning: true,
		input: ["text", "image"] as Array<"text" | "image">,
		contextWindow: 256_000,
		maxTokens: 32_768,
	};
	const fireworksEndpoint = {
		provider: "fireworks",
		id: "accounts/fireworks/routers/glm-5p2-fast",
		name: "GLM-5.2 Fast (Fireworks)",
		api: "openai-completions" as const,
		baseUrl: "https://api.fireworks.ai/inference/v1",
		apiKeyEnv: "FIREWORKS_API_KEY",
		reasoning: false,
		input: ["text"] as Array<"text" | "image">,
		contextWindow: 202_752,
		maxTokens: 131_072,
	};

	it("resolves config-declared OpenAI-compatible endpoints without catalog hardcoding", () => {
		const config: CliConfig = {
			searchPaths: ["."],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			agents: {
				orchestrator: openRouterEndpoint,
				explorer: fireworksEndpoint,
			},
		};

		const detailed = resolveAgentModelDetailed(config, {
			env: { OPENROUTER_API_KEY: "test-or", FIREWORKS_API_KEY: "test-fw" },
		});

		expect(detailed.model.id).toBe("x-ai/grok-4.5");
		expect(detailed.explorerModel.id).toBe("accounts/fireworks/routers/glm-5p2-fast");
		expect(detailed.roles.orchestrator.modelId).toBe("x-ai/grok-4.5");
		expect(detailed.roles.explorer.modelId).toBe("accounts/fireworks/routers/glm-5p2-fast");
		expect(detailed.roles.orchestrator.provider).toBe("openrouter");
		expect(detailed.roles.explorer.provider).toBe("fireworks");
		expect(detailed.roles.orchestrator.api).toBe("openai-completions");
		expect(detailed.roles.orchestrator.baseUrl).toBe("https://openrouter.ai/api/v1");
		expect(detailed.roles.explorer.baseUrl).toBe("https://api.fireworks.ai/inference/v1");
		expect(detailed.roles.orchestrator.capabilities).toEqual({ input: ["text", "image"], reasoning: true });
		expect(detailed.roles.explorer.capabilities).toEqual({ input: ["text"], reasoning: false });
		expect(detailed.roles.orchestrator.contextWindow).toBe(256_000);
		expect(detailed.roles.explorer.contextWindow).toBe(202_752);
		expect(detailed.roles.orchestrator.resolutionSource).toBe("config");
		expect(detailed.roles.explorer.resolutionSource).toBe("config");
		// Auth is env-backed via the configured apiKeyEnv names.
		expect(detailed.roles.orchestrator.auth).toEqual({
			present: true,
			source: "env",
			envName: "OPENROUTER_API_KEY",
		});
		expect(detailed.roles.explorer.auth).toEqual({
			present: true,
			source: "env",
			envName: "FIREWORKS_API_KEY",
		});
		expect(detailed.providerApiKeys).toEqual({ openrouter: "test-or", fireworks: "test-fw" });
	});

	it("reports missing auth as present:false with the expected env var name", () => {
		const config: CliConfig = {
			searchPaths: ["."],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			agents: {
				orchestrator: openRouterEndpoint,
				explorer: fireworksEndpoint,
			},
		};
		const previousOpenRouter = process.env.OPENROUTER_API_KEY;
		const previousFireworks = process.env.FIREWORKS_API_KEY;
		delete process.env.OPENROUTER_API_KEY;
		delete process.env.FIREWORKS_API_KEY;
		try {
			const detailed = resolveAgentModelDetailed(config, { env: {} });

			expect(detailed.roles.orchestrator.auth.present).toBe(false);
			expect(detailed.roles.orchestrator.auth.source).toBe("none");
			expect(detailed.roles.orchestrator.auth.envName).toBe("OPENROUTER_API_KEY");
			expect(detailed.roles.explorer.auth.present).toBe(false);
			expect(detailed.roles.explorer.auth.source).toBe("none");
			expect(detailed.roles.explorer.auth.envName).toBe("FIREWORKS_API_KEY");
			expect(detailed.apiKey).toBeUndefined();
			expect(detailed.providerApiKeys).toBeUndefined();
		} finally {
			if (previousOpenRouter !== undefined) process.env.OPENROUTER_API_KEY = previousOpenRouter;
			if (previousFireworks !== undefined) process.env.FIREWORKS_API_KEY = previousFireworks;
		}
	});

	it("mirrors resolveAgentModel for built-in catalog models and exposes local_runtime auth", () => {
		const codexConfigPath = join(root, "config.toml");
		writeFileSync(
			codexConfigPath,
			'model_provider = "test-proxy"\n[model_providers.test-proxy]\nbase_url = "https://proxy.example/v1"\nwire_api = "responses"\nenv_key = "TEST_PROXY_KEY"\n',
			"utf8",
		);
		const config: CliConfig = {
			searchPaths: ["."],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			agents: {
				orchestrator: { provider: "openai", id: "gpt-4o" },
				explorer: { provider: "test-proxy", id: "gpt-5.6-luna" },
			},
		};
		const localOptions = { configPath: codexConfigPath, env: { TEST_PROXY_KEY: "secret" } } as const;
		const previousOpenAi = process.env.OPENAI_API_KEY;
		process.env.OPENAI_API_KEY = "catalog-openai-key";
		try {
			const plain = resolveAgentModel(config, localOptions);
			const detailed = resolveAgentModelDetailed(config, localOptions);

			expect(detailed.model).toEqual(plain.model);
			expect(detailed.explorerModel).toEqual(plain.explorerModel);
			expect(detailed.apiKey).toBe(plain.apiKey);
			expect(detailed.providerApiKeys).toEqual(plain.providerApiKeys);
			expect(detailed.roles.orchestrator.modelId).toBe("gpt-4o");
			expect(detailed.roles.explorer.modelId).toBe("gpt-5.6-luna");
			expect(detailed.roles.explorer.auth).toEqual({ present: true, source: "local_runtime" });
			expect(detailed.roles.orchestrator.auth).toEqual({
				present: true,
				source: "catalog",
				envName: "OPENAI_API_KEY",
			});
		} finally {
			if (previousOpenAi === undefined) delete process.env.OPENAI_API_KEY;
			else process.env.OPENAI_API_KEY = previousOpenAi;
		}
	});

	it("never includes credential values in role metadata", () => {
		const config: CliConfig = {
			searchPaths: ["."],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			agents: {
				orchestrator: openRouterEndpoint,
				explorer: fireworksEndpoint,
			},
		};

		const detailed = resolveAgentModelDetailed(config, {
			env: { OPENROUTER_API_KEY: "super-secret-value-do-not-leak", FIREWORKS_API_KEY: "another-secret" },
		});
		const serialized = JSON.stringify(detailed.roles);

		expect(serialized).not.toContain("super-secret-value-do-not-leak");
		expect(serialized).not.toContain("another-secret");
		expect(detailed.roles.orchestrator.auth.envName).toBe("OPENROUTER_API_KEY");
	});
});

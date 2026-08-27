import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { ConfigError } from "../../src/cli/config.ts";
import {
	listUiState,
	removeConnection,
	setSearchPaths,
	toggleConnection,
	upsertConnection,
} from "../../src/ui/config-store.ts";

let root: string;
let configPath: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-ui-store-"));
	configPath = join(root, "config.json");
	writeFileSync(
		configPath,
		JSON.stringify(
			{
				searchPaths: [join(root, "docs")],
				workspacePath: root,
				memoryPath: join(root, "memory.json"),
				dupey: { enabled: true },
			},
			null,
			2,
		),
	);
	mkdirSync(join(root, "docs"), { recursive: true });
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function readConfig(): Record<string, unknown> {
	return JSON.parse(readFileSync(configPath, "utf8")) as Record<string, unknown>;
}

describe("datasource UI config store", () => {
	it("adds a GitHub connection without persisting the token value", () => {
		upsertConnection(configPath, {
			alias: "work-github",
			type: "github",
			enabled: true,
			connector: {
				tokenEnv: "GITHUB_TOKEN",
				token: "ghp_should-never-be-saved",
				repos: ["Marker-Inc-Korea/AutoRAG"],
			},
		});

		const saved = readConfig();
		const datasources = saved.datasources as Record<string, Record<string, unknown>>;
		expect(JSON.stringify(saved)).not.toContain("ghp_should-never-be-saved");
		expect(datasources["work-github"]).toMatchObject({
			type: "github",
			enabled: true,
			connector: { tokenEnv: "GITHUB_TOKEN", repos: ["Marker-Inc-Korea/AutoRAG"] },
		});
		const savedConnector = datasources["work-github"]?.connector as Record<string, unknown> | undefined;
		expect(savedConnector?.token).toBeUndefined();
		expect(saved.datasourceAccess).toEqual({
			allowedTags: ["github", "issues"],
			allowedScopes: ["/work-github/**"],
		});
		expect(saved.dupey).toEqual({ enabled: true });
	});

	it("rejects an invalid alias or unknown type", () => {
		expect(() =>
			upsertConnection(configPath, { alias: "../etc", type: "github", enabled: true, connector: {} }),
		).toThrow(ConfigError);
		expect(() =>
			upsertConnection(configPath, { alias: "dropbox", type: "dropbox", enabled: true, connector: {} }),
		).toThrow(ConfigError);
	});

	it("toggles, lists, and removes connections while recomputing trusted access", () => {
		upsertConnection(configPath, {
			alias: "news",
			type: "rss",
			enabled: true,
			connector: { feeds: [{ url: "https://feeds.example.com/a.xml" }] },
		});
		upsertConnection(configPath, {
			alias: "notes",
			type: "obsidian",
			enabled: true,
			connector: { vaultPath: join(root, "vault") },
		});

		toggleConnection(configPath, "news", false);
		let state = listUiState(configPath);
		expect(state.connections.find((item) => item.alias === "news")?.enabled).toBe(false);
		expect(state.access.allowedTags).toEqual(["obsidian", "notes"]);
		expect(state.access.allowedScopes).toEqual(["/notes/**"]);

		removeConnection(configPath, "notes");
		state = listUiState(configPath);
		expect(state.connections.map((item) => item.alias).sort()).toEqual(["news"]);
		expect(state.access.allowedTags).toEqual([]);
		expect(state.access.allowedScopes).toEqual([]);
		expect(readConfig().datasourceAccess).toBeUndefined();
	});

	it("updates local search folders without touching datasources", () => {
		const docs = join(root, "docs");
		const extra = join(root, "extra");
		mkdirSync(extra);
		upsertConnection(configPath, {
			alias: "news",
			type: "rss",
			enabled: true,
			connector: { feeds: [{ url: "https://feeds.example.com/a.xml" }] },
		});
		setSearchPaths(configPath, [docs, extra]);
		const state = listUiState(configPath);
		expect(state.searchPaths).toEqual([docs, extra]);
		expect(state.connections).toHaveLength(1);
	});
});

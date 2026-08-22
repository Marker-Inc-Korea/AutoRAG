import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { QmdClient } from "../../../src/datasource/skills/obsidian/client.ts";
import { ObsidianBm25Method, ObsidianSemanticMethod } from "../../../src/datasource/skills/obsidian/methods.ts";
import { toQmdCollectionName } from "../../../src/datasource/skills/obsidian/paths.ts";
import { ObsidianSkill, type ObsidianSkillClient } from "../../../src/datasource/skills/obsidian/skill.ts";
import type {
	QmdEmbedResult,
	QmdEnsureResult,
	QmdFailureReason,
	QmdSearchHit,
	QmdSearchResult,
	QmdUpdateResult,
} from "../../../src/datasource/skills/obsidian/types.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-obsidian-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

class StubClient implements ObsidianSkillClient {
	public ensureResult: QmdEnsureResult = {
		ok: true,
		data: { collectionName: "vault-1", vaultPath: "/vault", configDir: "/cfg" },
		stdout: "",
		stderr: "",
		code: 0,
	};
	public updateResult: QmdUpdateResult = {
		ok: true,
		data: { indexed: 1, updated: 0, unchanged: 0, removed: 0 },
		stdout: "Indexed: 1 new, 0 updated, 0 unchanged, 0 removed",
		stderr: "",
		code: 0,
	};
	public embedResult: QmdEmbedResult = {
		ok: true,
		data: { embedded: true },
		stdout: "",
		stderr: "",
		code: 0,
	};
	public searchResult: QmdSearchResult = {
		ok: true,
		hits: [
			{
				chunkId: "abc123",
				score: 0.91,
				content: "Ship the beta in June",
				title: "Roadmap 2024",
				file: "projects/roadmap.md",
			},
		],
		data: {
			hits: [
				{
					chunkId: "abc123",
					score: 0.91,
					content: "Ship the beta in June",
					title: "Roadmap 2024",
					file: "projects/roadmap.md",
				},
			],
		},
		stdout: "",
		stderr: "",
		code: 0,
	};

	async ensureCollection(): Promise<QmdEnsureResult> {
		return this.ensureResult;
	}
	async update(): Promise<QmdUpdateResult> {
		return this.updateResult;
	}
	async embed(): Promise<QmdEmbedResult> {
		return this.embedResult;
	}
	async search(): Promise<QmdSearchResult> {
		return this.searchResult;
	}
}

function fail(
	reason: QmdFailureReason,
	stderr = "failed",
): {
	readonly ok: false;
	readonly reason: QmdFailureReason;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number;
} {
	return { ok: false, reason, stdout: "", stderr, code: 1 };
}

describe("ObsidianSkill", () => {
	it("describes qmd-backed capabilities and indexes via update+embed", async () => {
		const client = new StubClient();
		const skill = new ObsidianSkill({ client, instanceId: "vault-1", vaultPath: "/vault" });
		expect(skill.describe()).toMatchObject({
			name: "obsidian",
			datasourceId: "obsidian",
			requiresExternalCli: true,
			capabilities: expect.arrayContaining(["bm25", "semantic", "incremental", "external-cli"]),
		});
		expect(await skill.index()).toMatchObject({ ok: true, chunkCount: 1 });
		const methods = skill.retrievalMethods();
		expect(methods.map((method) => method.describe().name)).toEqual(["obsidian-bm25", "obsidian-semantic"]);
		const hits = await methods[0]?.retrieve("ship beta June", { topK: 5 });
		expect(hits?.[0]?.source).toBe("/obsidian/vault-1/chunks/abc123");
		expect(hits?.[0]?.metadata?.path).toBe("projects/roadmap.md");
		expect(skill.describeSources().map((source) => source.source)).toContain("/obsidian/vault-1");
	});

	it("soft-fails embed while keeping a successful index", async () => {
		const client = new StubClient();
		client.embedResult = fail("nonzero-exit", "embed model missing");
		const skill = new ObsidianSkill({ client, instanceId: "vault-1" });
		const result = await skill.index();
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.diagnostics.some((item) => item.severity === "warning")).toBe(true);
		}
	});

	it("fails index when update fails", async () => {
		const client = new StubClient();
		client.updateResult = fail("binary-missing", "qmd binary not found");
		const skill = new ObsidianSkill({ client });
		expect(await skill.index()).toMatchObject({ ok: false, code: "datasource-unavailable" });
	});

	it("manifest stays free of absolute home paths", () => {
		const skill = new ObsidianSkill({
			client: new StubClient(),
			instanceId: "work",
			vaultPath: "/Users/someone/Notes",
		});
		const manifest = skill.skillManifest();
		expect(manifest.content).toContain("/obsidian/work");
		expect(manifest.content).toContain("qmd");
		expect(manifest.content).not.toContain("/Users/someone");
	});
});

describe("Obsidian retrieval methods", () => {
	it("maps lexical and semantic hits to opaque sources", async () => {
		const hits: readonly QmdSearchHit[] = [{ chunkId: "n1", score: 1.2, content: "beta ship", file: "a.md" }];
		const client = {
			async search(): Promise<QmdSearchResult> {
				return { ok: true, hits, data: { hits }, stdout: "", stderr: "", code: 0 };
			},
		};
		const lexical = await new ObsidianBm25Method({ client, instanceId: "v1" }).retrieve("beta", { topK: 3 });
		const semantic = await new ObsidianSemanticMethod({ client, instanceId: "v1" }).retrieve("beta", { topK: 3 });
		expect(lexical[0]?.source).toBe("/obsidian/v1/chunks/n1");
		expect(semantic[0]?.metadata?.mode).toBe("vsearch");
	});

	it("returns empty on client failure", async () => {
		const client = {
			async search(): Promise<QmdSearchResult> {
				return { ok: false, reason: "nonzero-exit", stdout: "", stderr: "nope", code: 1 };
			},
		};
		expect(await new ObsidianBm25Method({ client, instanceId: "v1" }).retrieve("x", { topK: 3 })).toEqual([]);
	});
});

describe("QmdClient", () => {
	it("writes isolated index.yml and parses search JSON from a fake binary", async () => {
		const binDir = join(root, "bin");
		const binaryPath = join(binDir, "qmd");
		const logPath = join(root, "calls.jsonl");
		mkdirSync(binDir, { recursive: true });
		writeFileSync(
			binaryPath,
			`#!/usr/bin/env node
import { appendFileSync } from "node:fs";
const args = process.argv.slice(2);
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args, configDir: process.env.QMD_CONFIG_DIR, cache: process.env.XDG_CACHE_HOME }) + "\\n");
if (args[0] === "search" || args[0] === "vsearch") {
  process.stdout.write(JSON.stringify([{ docid: "#deadbe", score: 0.8, file: "notes/a.md", snippet: "hello vault" }]));
} else if (args[0] === "update") {
  process.stdout.write("Indexed: 2 new, 1 updated, 3 unchanged, 0 removed\\n");
} else {
  process.stdout.write("{}\\n");
}
process.exit(0);
`,
		);
		chmodSync(binaryPath, 0o755);

		const vault = join(root, "vault");
		mkdirSync(vault, { recursive: true });
		writeFileSync(join(vault, "note.md"), "# Note\nhello");

		const client = new QmdClient({
			binaryPath,
			vaultPath: vault,
			workspaceRoot: root,
			instanceId: "vault-1",
			timeoutMs: 10_000,
		});

		const ensure = await client.ensureCollection();
		expect(ensure.ok).toBe(true);
		const configPath = join(root, ".autorag", "datasources", "obsidian", "vault-1", "config", "index.yml");
		expect(existsSync(configPath)).toBe(true);
		expect(readFileSync(configPath, "utf8")).toContain(JSON.stringify(vault));
		expect(readFileSync(configPath, "utf8")).toContain(".obsidian/**");

		const update = await client.update();
		expect(update.ok).toBe(true);
		if (update.ok) {
			expect(update.data).toMatchObject({ indexed: 2, updated: 1, unchanged: 3, removed: 0 });
		}

		const search = await client.search("search", "hello", { topK: 5 });
		expect(search.ok).toBe(true);
		if (search.ok) {
			expect(search.hits[0]).toMatchObject({ chunkId: "deadbe", content: "hello vault", score: 0.8 });
		}

		const calls = readFileSync(logPath, "utf8")
			.trim()
			.split("\n")
			.map((line) => JSON.parse(line) as { args: string[]; configDir: string });
		expect(calls.some((call) => call.args[0] === "update")).toBe(true);
		expect(calls.every((call) => call.configDir.includes("obsidian/vault-1/config"))).toBe(true);
	});

	it("returns binary-missing without throwing", async () => {
		const client = new QmdClient({
			binaryPath: join(root, "no-such-qmd"),
			vaultPath: join(root, "vault"),
			workspaceRoot: root,
			instanceId: "x",
		});
		mkdirSync(join(root, "vault"), { recursive: true });
		const result = await client.update();
		expect(result.ok).toBe(false);
		if (!result.ok) expect(result.reason).toBe("binary-missing");
	});
});

describe("toQmdCollectionName", () => {
	it("collapses punctuation and strips edge dashes without a quadratic regex", () => {
		expect(toQmdCollectionName("Vault One")).toBe("vault-one");
		expect(toQmdCollectionName("---Work---")).toBe("work");
		const padded = `${"-".repeat(10_000)}ok${"-".repeat(10_000)}`;
		expect(toQmdCollectionName(padded)).toBe("ok");
		expect(toQmdCollectionName("---")).toBe("default");
	});
});

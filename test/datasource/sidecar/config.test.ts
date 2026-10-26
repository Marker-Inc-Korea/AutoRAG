import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join, relative, sep } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { resolveSidecarOptions } from "../../../src/datasource/sidecar/config.ts";
import { CRAWLER_DESCRIPTORS } from "../../../src/datasource/sidecar/descriptors.ts";
import { buildDatasourceSkills } from "../../../src/datasource/skills/factory.ts";
import { withPlatform } from "../../helpers/platform.ts";

const base = {
	descriptor: CRAWLER_DESCRIPTORS.telegram,
	workspaceRoot: "/ws",
	registrationName: "telegram",
	instanceId: "main",
};

describe("resolveSidecarOptions: on by default", () => {
	it("is on with no sidecar setting at all, so every connected chat datasource gets hybrid search", () => {
		expect(resolveSidecarOptions({ ...base, connector: undefined, defaultCrawlerDb: "/x.db" })).toBeDefined();
		expect(resolveSidecarOptions({ ...base, connector: {}, defaultCrawlerDb: "/x.db" })).toBeDefined();
	});

	it("can be switched off with `false` or `{ enabled: false }`", () => {
		expect(
			resolveSidecarOptions({ ...base, connector: { sidecar: false }, defaultCrawlerDb: "/x.db" }),
		).toBeUndefined();
		expect(
			resolveSidecarOptions({ ...base, connector: { sidecar: { enabled: false } }, defaultCrawlerDb: "/x.db" }),
		).toBeUndefined();
	});

	it("needs a workspace root to place its own database, and never falls back to the crawler directory", () => {
		expect(
			resolveSidecarOptions({ ...base, workspaceRoot: undefined, connector: undefined, defaultCrawlerDb: "/x.db" }),
		).toBeUndefined();
	});

	it("uses the crawler's default database and a per-instance file under the workspace", () => {
		const options = resolveSidecarOptions({ ...base, connector: {}, defaultCrawlerDb: "/crawler/telecrawl.db" });
		expect(options).toMatchObject({
			descriptor: CRAWLER_DESCRIPTORS.telegram,
			crawlerDbPath: "/crawler/telecrawl.db",
			sidecarDbPath: join("/ws", ".autorag", "datasources", "telegram", "main", "sidecar.sqlite"),
		});
		expect(options?.workspaceId).toBeUndefined();
	});

	it("uses the shared embedder it is given, and is lexical-only when it is given none", () => {
		const embedder = { identity: async () => ({ provider: "p", model: "m", dimension: 1 }), embed: async () => [] };
		const input = { ...base, connector: {}, defaultCrawlerDb: "/x.db" };
		expect(resolveSidecarOptions({ ...input, embedder })?.embedder).toBe(embedder);
		expect(resolveSidecarOptions(input)?.embedder).toBeUndefined();
	});

	it("prefers an explicit crawlerDbPath, then connector.databasePath, over the default, and expands ~", () => {
		expect(
			resolveSidecarOptions({ ...base, connector: { databasePath: "/db/a.db" }, defaultCrawlerDb: "/d.db" })
				?.crawlerDbPath,
		).toBe("/db/a.db");
		expect(
			resolveSidecarOptions({
				...base,
				connector: { databasePath: "/db/a.db", sidecar: { crawlerDbPath: "~/b.db" } },
				defaultCrawlerDb: "/d.db",
			})?.crawlerDbPath,
		).toBe(join(homedir(), "b.db"));
	});

	it("passes the workspace id through and can switch semantic search off", () => {
		const options = resolveSidecarOptions({
			...base,
			descriptor: CRAWLER_DESCRIPTORS.slack,
			connector: { workspace: "T9", sidecar: { semantic: false } },
			defaultCrawlerDb: "/s.db",
			embedder: { identity: async () => ({ provider: "p", model: "m", dimension: 1 }), embed: async () => [] },
		});
		expect(options?.workspaceId).toBe("T9");
		expect(options?.embedder).toBeUndefined();
	});

	it("keeps instances apart and never lets an instance id escape the datasources directory", () => {
		const datasources = join("/ws", ".autorag", "datasources");
		const a = resolveSidecarOptions({ ...base, instanceId: "a", connector: {}, defaultCrawlerDb: "/x.db" });
		const b = resolveSidecarOptions({ ...base, instanceId: "b", connector: {}, defaultCrawlerDb: "/x.db" });
		expect(a?.sidecarDbPath).not.toBe(b?.sidecarDbPath);
		for (const hostile of ["b/../c", "..", "../..", "../../etc/passwd", "a/b"]) {
			const path =
				resolveSidecarOptions({ ...base, instanceId: hostile, connector: {}, defaultCrawlerDb: "/x.db" })
					?.sidecarDbPath ?? "";
			const rel = relative(datasources, path);
			expect(rel.startsWith("..")).toBe(false);
			expect(rel.split(sep)).toHaveLength(3);
		}
	});
});

describe("slack: the archive follows slacrawl's own configuration", () => {
	let root: string;
	beforeEach(() => {
		root = mkdtempSync(join(tmpdir(), "autorag-sidecar-cfg-"));
	});
	afterEach(() => {
		rmSync(root, { recursive: true, force: true });
	});

	const slack = (connector: Record<string, unknown>) =>
		resolveSidecarOptions({
			descriptor: CRAWLER_DESCRIPTORS.slack,
			workspaceRoot: "/ws",
			registrationName: "slack",
			instanceId: "main",
			connector,
			defaultCrawlerDb: "/home/.slacrawl/slacrawl.db",
		});

	it("reads db_path from the slacrawl config that connector.configPath points at", () => {
		const config = join(root, "slacrawl.toml");
		writeFileSync(config, `version = 1\ndb_path = '${join(root, "custom.db")}'\n`);
		expect(slack({ configPath: config })?.crawlerDbPath).toBe(join(root, "custom.db"));
	});

	it("expands ~ in db_path and resolves a relative db_path against the config's directory", () => {
		const config = join(root, "slacrawl.toml");
		writeFileSync(config, "db_path = '~/archive.db'\n");
		expect(slack({ configPath: config })?.crawlerDbPath).toBe(join(homedir(), "archive.db"));
		mkdirSync(join(root, "sub"));
		const nested = join(root, "sub", "slacrawl.toml");
		writeFileSync(nested, "db_path = 'data/s.db'\n");
		expect(slack({ configPath: nested })?.crawlerDbPath).toBe(join(root, "sub", "data", "s.db"));
	});

	it("still lets an explicit sidecar.crawlerDbPath or connector.databasePath win over the config", () => {
		const config = join(root, "slacrawl.toml");
		writeFileSync(config, "db_path = '/from/config.db'\n");
		expect(slack({ configPath: config, sidecar: { crawlerDbPath: "/explicit.db" } })?.crawlerDbPath).toBe(
			"/explicit.db",
		);
		expect(slack({ configPath: config, databasePath: "/db-path.db" })?.crawlerDbPath).toBe("/db-path.db");
	});

	it("falls back to the default archive when the config is missing, unreadable, or has no db_path", () => {
		expect(slack({ configPath: join(root, "absent.toml") })?.crawlerDbPath).toBe("/home/.slacrawl/slacrawl.db");
		const broken = join(root, "broken.toml");
		writeFileSync(broken, "this is = = not toml [");
		expect(slack({ configPath: broken })?.crawlerDbPath).toBe("/home/.slacrawl/slacrawl.db");
		const empty = join(root, "empty.toml");
		writeFileSync(empty, "version = 1\n");
		expect(slack({ configPath: empty })?.crawlerDbPath).toBe("/home/.slacrawl/slacrawl.db");
	});

	it("does not read slacrawl config for crawlers that have no such file", () => {
		const config = join(root, "x.toml");
		writeFileSync(config, "db_path = '/should/not/be/used.db'\n");
		const options = resolveSidecarOptions({
			...base,
			connector: { configPath: config },
			defaultCrawlerDb: "/telecrawl.db",
		});
		expect(options?.crawlerDbPath).toBe("/telecrawl.db");
	});
});

const stubEmbedder = {
	identity: async () => ({ provider: "minsync", model: "Qwen/Qwen3-Embedding-0.6B", dimension: 2 }),
	embed: async (texts: readonly string[]) => texts.map(() => [1, 0]),
};

describe("buildDatasourceSkills: sidecar by default", () => {
	it.each(["slack", "telegram", "whatsapp", "kakao"] as const)(
		"%s: a plain connection registers one hybrid method",
		(name) => {
			const { skills } = withPlatform("darwin", () =>
				buildDatasourceSkills({ [name]: {} }, "/tmp/ws-never-created", { embedder: stubEmbedder }),
			);
			const methods = skills.flatMap((skill) => skill.retrievalMethods().map((method) => method.describe().name));
			expect(methods).toEqual([`${name}-hybrid`]);
		},
	);

	it.each(["slack", "telegram", "whatsapp", "kakao"] as const)(
		"%s: connector.sidecar=false restores the CLI-only behavior",
		(name) => {
			const { skills } = withPlatform("darwin", () =>
				buildDatasourceSkills({ [name]: { connector: { sidecar: false } } }, "/tmp/ws-never-created"),
			);
			const expected = name === "kakao" ? ["kakao-hybrid"] : [`${name}-fts`];
			expect(skills.flatMap((skill) => skill.retrievalMethods().map((method) => method.describe().name))).toEqual(
				expected,
			);
		},
	);

	it("without a workspace there is nowhere to keep the sidecar, so behavior is the CLI-only one", () => {
		const { skills } = buildDatasourceSkills({ telegram: {} });
		expect(skills.flatMap((skill) => skill.retrievalMethods().map((method) => method.describe().name))).toEqual([
			"telegram-fts",
		]);
	});

	it("building a skill does not create the crawler archive or any directory outside the workspace", () => {
		const { skills } = buildDatasourceSkills(
			{ telegram: { connector: { databasePath: "/definitely/not/here.db" } } },
			"/tmp/ws-never-created",
		);
		expect(skills).toHaveLength(1);
	});
});

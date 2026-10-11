import { homedir } from "node:os";
import { join, relative } from "node:path";
import { describe, expect, it } from "vitest";
import { resolveSidecarOptions } from "../../../src/datasource/sidecar/config.ts";
import { CRAWLER_DESCRIPTORS } from "../../../src/datasource/sidecar/descriptors.ts";
import { buildDatasourceSkills } from "../../../src/datasource/skills/factory.ts";

const base = {
	descriptor: CRAWLER_DESCRIPTORS.telegram,
	workspaceRoot: "/ws",
	registrationName: "telegram",
	instanceId: "main",
};

describe("resolveSidecarOptions", () => {
	it("is off unless connector.sidecar is enabled", () => {
		expect(resolveSidecarOptions({ ...base, connector: undefined, defaultCrawlerDb: "/x.db" })).toBeUndefined();
		expect(resolveSidecarOptions({ ...base, connector: {}, defaultCrawlerDb: "/x.db" })).toBeUndefined();
		expect(
			resolveSidecarOptions({ ...base, connector: { sidecar: false }, defaultCrawlerDb: "/x.db" }),
		).toBeUndefined();
		expect(
			resolveSidecarOptions({ ...base, connector: { sidecar: { enabled: false } }, defaultCrawlerDb: "/x.db" }),
		).toBeUndefined();
	});

	it("needs a workspace root to place its own database, and never falls back to the crawler directory", () => {
		expect(
			resolveSidecarOptions({
				...base,
				workspaceRoot: undefined,
				connector: { sidecar: true },
				defaultCrawlerDb: "/x.db",
			}),
		).toBeUndefined();
	});

	it("`sidecar: true` uses the crawler's default database and a per-instance file under the workspace", () => {
		const options = resolveSidecarOptions({
			...base,
			connector: { sidecar: true },
			defaultCrawlerDb: "/crawler/telecrawl.db",
		});
		expect(options).toMatchObject({
			descriptor: CRAWLER_DESCRIPTORS.telegram,
			crawlerDbPath: "/crawler/telecrawl.db",
			sidecarDbPath: join("/ws", ".autorag", "datasources", "telegram", "main", "sidecar.sqlite"),
		});
		expect(options?.embedder).toBeDefined();
		expect(options?.workspaceId).toBeUndefined();
	});

	it("prefers an explicit crawlerDbPath, then connector.databasePath, over the default, and expands ~", () => {
		expect(
			resolveSidecarOptions({
				...base,
				connector: { databasePath: "/db/a.db", sidecar: true },
				defaultCrawlerDb: "/d.db",
			})?.crawlerDbPath,
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
		});
		expect(options?.workspaceId).toBe("T9");
		expect(options?.embedder).toBeUndefined();
	});

	it("keeps instances apart and never lets an instance id escape the datasources directory", () => {
		const datasources = join("/ws", ".autorag", "datasources");
		const one = resolveSidecarOptions({
			...base,
			instanceId: "a",
			connector: { sidecar: true },
			defaultCrawlerDb: "/x.db",
		});
		expect(one?.sidecarDbPath).not.toBe(
			resolveSidecarOptions({ ...base, instanceId: "b", connector: { sidecar: true }, defaultCrawlerDb: "/x.db" })
				?.sidecarDbPath,
		);
		for (const hostile of ["b/../c", "..", "../..", "../../etc/passwd", "a/b"]) {
			const path =
				resolveSidecarOptions({
					...base,
					instanceId: hostile,
					connector: { sidecar: true },
					defaultCrawlerDb: "/x.db",
				})?.sidecarDbPath ?? "";
			const rel = relative(datasources, path);
			expect(rel.startsWith("..")).toBe(false);
			expect(rel.split("/")).toHaveLength(3); // <registration>/<instance>/sidecar.sqlite: one segment each, no nesting
		}
	});
});

describe("buildDatasourceSkills sidecar wiring", () => {
	it.each(["slack", "telegram", "whatsapp"] as const)(
		"%s: connector.sidecar adds the BM25, semantic and hybrid methods",
		(name) => {
			const { skills } = buildDatasourceSkills(
				{ [name]: { connector: { sidecar: { crawlerDbPath: "/does/not/matter.db" } } } },
				"/tmp/ws-never-created",
			);
			const methods = skills.flatMap((skill) => skill.retrievalMethods().map((method) => method.describe().name));
			expect(methods).toEqual([`${name}-fts`, `${name}-sidecar-bm25`, `${name}-semantic`, `${name}-hybrid`]);
		},
	);

	it.each(["slack", "telegram", "whatsapp"] as const)(
		"%s: without connector.sidecar behavior is unchanged",
		(name) => {
			const { skills } = buildDatasourceSkills({ [name]: {} }, "/tmp/ws-never-created");
			expect(skills.flatMap((skill) => skill.retrievalMethods().map((method) => method.describe().name))).toEqual([
				`${name}-fts`,
			]);
		},
	);
});

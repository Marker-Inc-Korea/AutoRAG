import { describe, expect, it } from "vitest";
import { buildDatasourceSkills } from "../../src/datasource/skills/factory.ts";
import type { RetrievalMethod } from "../../src/retrieval/types.ts";
import { withPlatform } from "../helpers/platform.ts";

const embedder = {
	identity: async () => ({ provider: "minsync", model: "Qwen/Qwen3-Embedding-0.6B", dimension: 2 }),
	embed: async (texts: readonly string[]) => texts.map(() => [1, 0]),
};

/** Every built-in datasource that has a native hybrid search, built the way a default install wires it. */
const HYBRID_DATASOURCES = ["discord", "mailcrawl", "kakao", "slack", "telegram", "whatsapp", "clawgallery"] as const;

function methodsByDatasource(sidecar: boolean): Map<string, RetrievalMethod[]> {
	const config = Object.fromEntries(HYBRID_DATASOURCES.map((name) => [name, {}]));
	// kakao is built on macOS only; the other datasources are platform-independent.
	const { skills } = withPlatform("darwin", () =>
		sidecar ? buildDatasourceSkills(config, "/tmp/ws-never-created", { embedder }) : buildDatasourceSkills(config),
	);
	const grouped = new Map<string, RetrievalMethod[]>();
	for (const skill of skills) {
		const id = skill.describe().datasourceId ?? skill.describe().name;
		grouped.set(id, [...(grouped.get(id) ?? []), ...skill.retrievalMethods()]);
	}
	return grouped;
}

describe("hybrid-first datasource retrieval", () => {
	it.each([
		["with a sidecar and embedder", true],
		["CLI-only", false],
	])("a datasource with a hybrid method registers nothing next to it (%s)", (_label, sidecar) => {
		const grouped = methodsByDatasource(sidecar);
		expect([...grouped.keys()].sort()).toEqual([...HYBRID_DATASOURCES].sort());
		for (const [id, methods] of grouped) {
			const types = methods.map((method) => method.describe().type);
			if (types.includes("hybrid")) {
				expect(types, `${id} registers its hybrid method next to a component`).toEqual(["hybrid"]);
			}
		}
	});

	it("every built-in datasource exposes exactly one method per query surface when sidecar-backed", () => {
		const grouped = methodsByDatasource(true);
		for (const [id, methods] of grouped) {
			expect(methods, `${id} fans out to several methods`).toHaveLength(1);
		}
	});
});

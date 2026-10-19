import { describe, expect, it } from "vitest";
import { buildDatasourceSkills } from "../../src/datasource/skills/factory.ts";
import type { ManagedEmbedder } from "../../src/embedding/embedder.ts";

function recordingEmbedder(): { embedder: ManagedEmbedder; calls: string[][] } {
	const calls: string[][] = [];
	return {
		calls,
		embedder: {
			identity: async () => ({ provider: "minsync", model: "Qwen/Qwen3-Embedding-0.6B", dimension: 2 }),
			embed: async (texts) => {
				calls.push([...texts]);
				return texts.map(() => [1, 0]);
			},
			ensureReady: async () => ({ backend: "daemon", daemonStarted: false }),
			describeEndpoint: async () => undefined,
			close: async () => undefined,
		},
	};
}

describe("buildDatasourceSkills shared embedder", () => {
	it("hands the one shared embedder to every chat sidecar instead of building its own", async () => {
		const { embedder, calls } = recordingEmbedder();
		const { skills } = buildDatasourceSkills({ telegram: {} }, "/tmp/ws-never-created", { embedder });
		const methods = skills.flatMap((skill) => skill.retrievalMethods());
		expect(methods.map((method) => method.describe().name)).toContain("telegram-semantic");
		expect(calls).toEqual([]);
	});

	it("keeps a sidecar lexical-only when its connector turns semantic off", () => {
		const { embedder } = recordingEmbedder();
		const { skills } = buildDatasourceSkills(
			{ telegram: { connector: { sidecar: { semantic: false } } } },
			"/tmp/ws-never-created",
			{ embedder },
		);
		const names = skills.flatMap((skill) => skill.retrievalMethods().map((method) => method.describe().name));
		expect(names).not.toContain("telegram-semantic-vector");
	});

	it("builds no sidecar embedder at all when no embedder is supplied", () => {
		const { skills } = buildDatasourceSkills({ telegram: {} }, "/tmp/ws-never-created");
		expect(skills).toHaveLength(1);
	});
});

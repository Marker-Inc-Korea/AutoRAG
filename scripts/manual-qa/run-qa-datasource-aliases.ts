import { AliasedDatasourceSkill } from "../../src/datasource/aliased-skill.ts";
import type { DatasourceSkill } from "../../src/datasource/types.ts";

let failures = 0;
const check = (name: string, pass: boolean): void => {
	if (!pass) failures += 1;
	console.log(`${pass ? "PASS" : "FAIL"}  ${name}`);
};

const baseSkill: DatasourceSkill = {
	describe: () => ({
		name: "discord",
		id: "discord",
		type: "discord-archive",
		description: "Discord archive",
		capabilities: ["chat"],
		tags: ["discord"],
		status: "active",
		datasourceId: "discord",
		instanceId: "default",
		instances: ["default"],
	}),
	polling: () => ({ mode: "none" }),
	index: async () => ({ ok: true, instanceId: "default", skill: "discord", chunkCount: 2, indexedAt: Date.now(), diagnostics: [] }),
	retrievalMethods: () => [
		{
			describe: () => ({
				name: "discord-fts",
				type: "bm25",
				description: "discord",
				status: "active",
				capabilities: ["chat"],
				datasourceId: "discord",
				tags: ["discord"],
			}),
			retrieve: async () => [
				{ id: "a", source: "/discord/default/a", content: "release", score: 1, metadata: { channelName: "release-engineering" } },
				{ id: "b", source: "/discord/default/b", content: "random", score: 0.5, metadata: { channelName: "random" } },
			],
		},
	],
	describeSources: () => [
		{ source: "/discord/default", datasourceId: "discord", skill: "discord", instanceId: "default", metadata: {} },
	],
	skillManifest: () => ({ name: "datasource-discord", description: "Discord", content: "all channels" }),
};

const all = new AliasedDatasourceSkill(baseSkill, { alias: "all-discord" });
const restricted = new AliasedDatasourceSkill(baseSkill, {
	alias: "release-channel",
	channelNames: ["release-engineering"],
});

try {
	const allResults = await all.retrievalMethods()[0]?.retrieve("release", { topK: 10 });
	const restrictedResults = await restricted.retrievalMethods()[0]?.retrieve("release", { topK: 10 });
	check("default alias searches all channels", allResults?.length === 2);
	check("restricted alias searches only named channel", restrictedResults?.length === 1);
	check("alias source roots are independent", allResults?.[0]?.source.startsWith("/all-discord/") === true);
	check("restricted manifest explains channel selection", restricted.skillManifest().content.includes("release-engineering"));
	check("default manifest remains all-channel", all.skillManifest().content.includes("all channels"));
} finally {
	console.log(failures === 0 ? "\nDATASOURCE ALIAS QA PASSED" : `\nDATASOURCE ALIAS QA: ${failures} failure(s)`);
	if (failures > 0) process.exitCode = 1;
}

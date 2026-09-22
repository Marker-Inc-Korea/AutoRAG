/**
 * Live manual QA for the Lark/Feishu remote-search datasource (issue #1672).
 *
 * Requires `lark-cli` on PATH and a completed `lark-cli auth login`.
 * Prints opaque `/lark/...` identities. Does not write tenant content into
 * the repository. Exits non-zero when the CLI is missing, logged out, or
 * returns no hits — an empty search is not a pass.
 *
 * Run: bun scripts/manual-qa/run-qa-lark-live.ts "<query>"
 * Or:  E2E_LARK_QUERY="<query>" bun scripts/manual-qa/run-qa-lark-live.ts
 */

import { buildDatasourceSkills } from "../../src/datasource/skills/factory.ts";

const query = process.env.E2E_LARK_QUERY ?? process.argv[2];

async function main(): Promise<number> {
	if (query === undefined || query.trim().length === 0) {
		console.error("lark live QA needs a query argument or E2E_LARK_QUERY");
		return 1;
	}
	const { skills, unknown } = buildDatasourceSkills({ lark: true });
	const skill = skills[0];
	if (unknown.length > 0 || skill === undefined) {
		console.error("lark skill did not build");
		return 1;
	}
	const indexed = await skill.index();
	const probeError = skill.polling().lastError;
	if (probeError !== undefined) {
		console.error(probeError);
		return 1;
	}
	const chunkCount = indexed.ok ? indexed.chunkCount : -1;
	console.log(JSON.stringify({ ok: indexed.ok, chunkCount }));
	const sources: string[] = [];
	for (const method of skill.retrievalMethods()) {
		const hits = await method.retrieve(query, { topK: 5 });
		for (const hit of hits) {
			console.log(hit.source);
			sources.push(hit.source);
		}
	}
	if (!indexed.ok || chunkCount !== 0 || sources.length === 0) {
		console.error("lark-cli returned no /lark identities; not claiming a tenant match");
		return 1;
	}
	return 0;
}

main()
	.then((code) => {
		process.exit(code);
	})
	.catch((error: unknown) => {
		console.error(error instanceof Error ? error.message : String(error));
		process.exit(1);
	});

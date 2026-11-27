import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createSingleDatasourceSearchTools } from "../../src/agent/search-single-datasource-tool.ts";

let dir: string;
beforeEach(() => {
	dir = mkdtempSync(join(tmpdir(), "autorag-attach-tool-"));
	mkdirSync(join(dir, "media"));
	writeFileSync(join(dir, "media", "a.png"), "x");
});
afterEach(() => {
	rmSync(dir, { recursive: true, force: true });
});

function tool(hits: number) {
	const provider = {
		searchSingleDatasourceDocuments: async () => ({
			results: Array.from({ length: hits }, (_, i) => ({
				id: `c${i}`,
				source: `/telegram/default/chunks/c${i}`,
				content: "photo from Gajae",
				score: 1,
				metadata: {},
			})),
			diagnostics: [],
		}),
	};
	return createSingleDatasourceSearchTools(provider, [
		{
			datasourceId: "telegram",
			description: "Telegram",
			instanceScopes: [],
			attachmentRoots: () => [{ id: "media", path: join(dir, "media"), description: "Cached media." }],
		},
	])[0];
}

const text = (r: { content: readonly { type: string; text?: string }[] }) =>
	r.content.map((p) => p.text ?? "").join("");

describe("search_datasource_<id> attachment context", () => {
	it("appends the attachment folders when there are results", async () => {
		const out = text(await (tool(1) as NonNullable<ReturnType<typeof tool>>).execute("c", { query: "photo" }));
		expect(out).toContain("Attachment folders for this datasource");
		expect(out).toContain(`${join(dir, "media")} (has files)`);
	});

	it("stays silent when nothing matched", async () => {
		const out = text(await (tool(0) as NonNullable<ReturnType<typeof tool>>).execute("c", { query: "photo" }));
		expect(out).not.toContain("Attachment folders");
	});
});

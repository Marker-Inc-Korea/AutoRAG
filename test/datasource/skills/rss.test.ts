import { describe, expect, it } from "vitest";
import { RssConnector } from "../../../src/datasource/skills/rss/connector.ts";
import { RssSkill } from "../../../src/datasource/skills/rss/skill.ts";
import { createMockFetch } from "../../fixtures/mock-fetch.ts";

const RSS_XML = `<?xml version="1.0"?>
<rss version="2.0">
	<channel>
		<title>Tech Daily</title>
		<item>
			<title>New chip released</title>
			<link>https://tech.example.com/chip</link>
			<guid>chip-42</guid>
			<pubDate>Mon, 11 Mar 2024 10:00:00 GMT</pubDate>
			<category>hardware</category>
			<description><![CDATA[<p>The new chip is <b>twice as fast</b>.</p>]]></description>
		</item>
		<item>
			<title>Cloud outage postmortem</title>
			<guid>outage-7</guid>
			<pubDate>Tue, 12 Mar 2024 10:00:00 GMT</pubDate>
			<description>Root cause was a bad config push.</description>
		</item>
	</channel>
</rss>`;

const ATOM_XML = `<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
	<title>Research Blog</title>
	<entry>
		<title>Retrieval papers roundup</title>
		<id>urn:uuid:entry-1</id>
		<updated>2024-03-13T10:00:00Z</updated>
		<summary>Five new papers on retrieval augmentation.</summary>
	</entry>
</feed>`;

describe("RssConnector", () => {
	it("returns not-configured without feeds", async () => {
		expect(await new RssConnector({}).fetch()).toMatchObject({ ok: false, reason: "not-configured" });
	});

	it("parses RSS 2.0 and Atom feeds into documents with feed hierarchy", async () => {
		const mock = createMockFetch([
			{ match: "rss.example.com", text: RSS_XML },
			{ match: "atom.example.com", text: ATOM_XML },
		]);
		const connector = new RssConnector({
			feeds: [
				{ url: "https://rss.example.com/feed.xml", category: "tech" },
				{ url: "https://atom.example.com/feed" },
			],
			fetchImpl: mock.fetchImpl,
		});
		const result = await connector.fetch();
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.documents).toHaveLength(3);
			const chip = result.documents.find((d) => d.docId === "chip-42");
			expect(chip).toMatchObject({
				title: "New chip released",
				hierarchy: ["feeds", "tech", "Tech Daily"],
				metadata: { feedIndex: 0, feedTitle: "Tech Daily", categories: ["hardware"] },
			});
			expect(chip?.content).toContain("twice as fast");
			expect(chip?.content).not.toContain("<p>");
			const atom = result.documents.find((d) => d.docId === "urn:uuid:entry-1");
			expect(atom).toMatchObject({ hierarchy: ["feeds", "Research Blog"] });
		}
	});

	it("degrades per-feed failures to index-based warnings and fails only when all feeds fail", async () => {
		const mock = createMockFetch([
			{ match: "down.example.com", status: 404 },
			{ match: "rss.example.com", text: RSS_XML },
		]);
		const partial = await new RssConnector({
			feeds: [{ url: "https://down.example.com/f" }, { url: "https://rss.example.com/feed.xml" }],
			fetchImpl: mock.fetchImpl,
		}).fetch();
		expect(partial.ok).toBe(true);
		if (partial.ok) {
			expect(partial.warnings).toEqual(["feed 1 failed: http-404"]);
			expect(JSON.stringify(partial.warnings)).not.toContain("down.example.com");
		}

		const allDown = await new RssConnector({
			feeds: [{ url: "https://down.example.com/f" }],
			fetchImpl: mock.fetchImpl,
		}).fetch();
		expect(allDown).toMatchObject({ ok: false, reason: "unavailable" });
	});

	it("rejects unparseable XML per feed", async () => {
		const mock = createMockFetch([{ match: "bad.example.com", text: "{json: true}" }]);
		const result = await new RssConnector({
			feeds: [{ url: "https://bad.example.com/f" }],
			fetchImpl: mock.fetchImpl,
		}).fetch();
		expect(result).toMatchObject({ ok: false, reason: "unavailable" });
	});
});

describe("RssSkill", () => {
	it("indexes with a dedupe window and searches with opaque /rss sources", async () => {
		const mock = createMockFetch([{ match: "rss.example.com", text: RSS_XML }]);
		const skill = new RssSkill({
			instanceId: "feeds",
			connectorOptions: { feeds: [{ url: "https://rss.example.com/feed.xml" }], fetchImpl: mock.fetchImpl },
		});
		expect(skill.describe()).toMatchObject({ name: "rss", datasourceId: "rss" });
		expect(await skill.index()).toMatchObject({ ok: true, chunkCount: 2 });
		const [method] = skill.retrievalMethods();
		const hits = await method?.retrieve("outage root cause config", { topK: 5 });
		expect(hits?.[0]?.source).toMatch(/^\/rss\/feeds\/chunks\//);
		expect(hits?.[0]?.metadata).toMatchObject({ feedTitle: "Tech Daily" });
	});
});

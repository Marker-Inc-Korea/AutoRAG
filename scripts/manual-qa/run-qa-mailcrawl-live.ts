/**
 * Live mailcrawl 0.2.0 QA (#1496 / #1499, refreshed for the 0.2.0 contract).
 *
 * Prerequisites:
 *   npm install -g @nomadamas/mailcrawl@0.2.0
 *   Node.js 24+
 *
 * Fixture sync needs no Himalaya account or credentials. Since 0.2.0 the
 * default embedder is the in-process native Qwen3-Embedding-0.6B model, so a
 * cold Hugging Face cache makes the first `index` download ONNX weights and
 * re-embed the archive; that run legitimately outlives an interactive timeout.
 * Set `MAILCRAWL_EMBEDDER=mock` to exercise the same CLI contract with a hash
 * embedder when only the interface is under test. 0.1.3 and earlier fail the
 * second index after a no-op sync with `text array must be non-empty`.
 *
 * 0.2.0 also moved semantic vectors from JSON-in-SQLite to a LanceDB table
 * (`<data-dir>/semantic.lance` + `semantic.identity.json`) and replaced the
 * index report with `{ embedded, reused, archiveRevision, rebuilt, embedder }`.
 *
 * Usage:
 *   bun scripts/manual-qa/run-qa-mailcrawl-live.ts
 *   MAILCRAWL_BINARY=/path/to/mailcrawl bun scripts/manual-qa/run-qa-mailcrawl-live.ts
 */
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { MailcrawlClient, MailcrawlSkill } from "../../src/datasource/skills/mailcrawl/index.ts";

const binaryPath = process.env.MAILCRAWL_BINARY ?? "mailcrawl";
const root = mkdtempSync(join(tmpdir(), "autorag-mailcrawl-016-"));
const dataDir = join(root, "data");
const fixture = join(root, "messages.json");
const timeoutMs = 900_000;

writeFileSync(
	fixture,
	`${JSON.stringify(
		[
			{
				accountId: "fixture",
				mailbox: "INBOX",
				providerKey: "refund-policy-1",
				messageId: "<refund-policy-1@example.com>",
				threadId: "refund-thread-1",
				subject: "Refund policy",
				from: "finance@example.com",
				to: ["support@example.com"],
				cc: [],
				date: "2026-08-31T10:00:00Z",
				text: "Refund exceptions require director approval before payout.",
			},
		],
		null,
		2,
	)}\n`,
);
mkdirSync(dataDir, { recursive: true });

const client = new MailcrawlClient({
	binaryPath,
	dataDir,
	source: "fixture",
	fixture,
	timeoutMs,
});

try {
	const firstSync = await client.sync();
	if (!firstSync.ok) throw new Error(`first sync failed: ${firstSync.reason}`);
	if ((firstSync.data.added ?? 0) < 1 && (firstSync.data.chunksAdded ?? 0) < 1) {
		throw new Error(`first sync added no messages: ${JSON.stringify(firstSync.data)}`);
	}

	const firstIndex = await client.index();
	if (!firstIndex.ok) throw new Error(`first index failed: ${firstIndex.reason}`);
	if (firstIndex.data.archiveRevision === undefined) {
		throw new Error(`0.2.0 index report is missing archiveRevision: ${JSON.stringify(firstIndex.data)}`);
	}
	if (!firstIndex.data.embedder) {
		throw new Error(`0.2.0 index report is missing the embedder identity: ${JSON.stringify(firstIndex.data)}`);
	}
	for (const entry of ["semantic.lance", "semantic.identity.json"]) {
		if (!existsSync(join(dataDir, entry))) {
			throw new Error(`0.2.0 LanceDB semantic store is missing ${entry} under the data directory`);
		}
	}

	const secondSync = await client.sync();
	if (!secondSync.ok) throw new Error(`no-op sync failed: ${secondSync.reason}`);
	if ((secondSync.data.unchanged ?? 0) < 1) {
		throw new Error(`expected unchanged fixture message, got ${JSON.stringify(secondSync.data)}`);
	}

	const secondIndex = await client.index();
	if (!secondIndex.ok) {
		throw new Error(`0.2.0 no-op reindex failed: ${secondIndex.reason}`);
	}
	if ((secondIndex.data.reused ?? 0) < 1) {
		throw new Error(`expected reused vectors after no-op sync, got ${JSON.stringify(secondIndex.data)}`);
	}
	if (secondIndex.data.rebuilt === true) {
		throw new Error(`a no-op reindex must not rebuild the vector store: ${JSON.stringify(secondIndex.data)}`);
	}
	if (secondIndex.data.embedder !== firstIndex.data.embedder) {
		throw new Error(
			`embedder identity changed across a no-op reindex: ${firstIndex.data.embedder} -> ${secondIndex.data.embedder}`,
		);
	}

	const bm25 = await client.search("bm25", "refund", { topK: 5 });
	if (!bm25.ok) throw new Error(`bm25 search failed: ${bm25.reason}`);
	if (!bm25.hits.some((hit) => hit.subject === "Refund policy" && hit.accountId === "fixture")) {
		throw new Error(`bm25 search missed the fixture refund policy: ${JSON.stringify(bm25.hits)}`);
	}

	const semantic = await client.search("semantic", "who must approve a refund exception", { topK: 5 });
	if (!semantic.ok) throw new Error(`semantic search failed: ${semantic.reason}`);
	if (!semantic.hits.some((hit) => hit.subject === "Refund policy")) {
		throw new Error(`semantic search missed the fixture refund policy: ${JSON.stringify(semantic.hits)}`);
	}

	const hybrid = await client.search("hybrid", "refund approval", { topK: 5 });
	if (!hybrid.ok) throw new Error(`hybrid search failed: ${hybrid.reason}`);
	if (!hybrid.hits.some((hit) => hit.subject === "Refund policy")) {
		throw new Error(`hybrid search missed the fixture refund policy: ${JSON.stringify(hybrid.hits)}`);
	}

	const skill = new MailcrawlSkill({ client, instanceId: "personal", account: "fixture", mailbox: "INBOX" });
	const indexed = await skill.index();
	if (!indexed.ok) throw new Error(`skill index failed: ${indexed.error ?? "unknown"}`);
	const retrieved = await skill
		.retrievalMethods()[0]
		?.retrieve("refund", { topK: 5, allowedScopes: ["/mailcrawl/personal/**"] });
	if (retrieved?.[0]?.source === undefined || !retrieved[0].source.startsWith("/mailcrawl/personal/chunks/")) {
		throw new Error(`skill source mapping failed: ${retrieved?.[0]?.source ?? "none"}`);
	}

	console.log("MAILCRAWL_LIVE_QA_PASS");
	console.log(
		JSON.stringify({
			firstSync: firstSync.data,
			firstIndex: firstIndex.data,
			secondSync: secondSync.data,
			secondIndex: secondIndex.data,
			embedder: firstIndex.data.embedder,
			archiveRevision: firstIndex.data.archiveRevision,
			semanticStore: join(dataDir, "semantic.lance"),
			bm25Hits: bm25.hits.length,
			semanticHits: semantic.hits.length,
			hybridHits: hybrid.hits.length,
			source: retrieved[0].source,
		}),
	);
} finally {
	rmSync(root, { recursive: true, force: true });
}

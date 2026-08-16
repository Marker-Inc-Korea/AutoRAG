/**
 * Live Discord QA (#1413): index a REAL Discord archive through the external
 * `discrawl` CLI and search it with FTS, semantic, and hybrid retrieval.
 *
 * Setup (wiretap — no token, reads the local Discord Desktop cache):
 *   1. brew install openclaw/tap/discrawl
 *   2. Have the Discord desktop app installed and signed in at least once.
 *   3. (semantic) brew install ollama && ollama serve && ollama pull bge-m3
 *
 * Setup (bot API — ToS-sanctioned automation):
 *   1. https://discord.com/developers/applications -> New Application -> Bot
 *   2. Enable the MESSAGE CONTENT privileged intent.
 *   3. Invite the bot with READ_MESSAGE_HISTORY to the target guild.
 *   4. export DISCORD_BOT_TOKEN=...
 *
 * Usage:
 *   bun scripts/manual-qa/run-qa-discrawl-live.ts                      # wiretap + sample queries
 *   bun scripts/manual-qa/run-qa-discrawl-live.ts "질의" ["질의2" ...]  # custom queries
 *   DISCRAWL_SOURCE=discord bun scripts/manual-qa/run-qa-discrawl-live.ts
 */

import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DiscrawlClient, DiscrawlSkill } from "../../src/datasource/skills/discrawl/index.ts";

const source = (process.env.DISCRAWL_SOURCE ?? "wiretap") as "wiretap" | "discord" | "both";
if (source !== "wiretap" && process.env.DISCORD_BOT_TOKEN === undefined) {
	console.error(`DISCRAWL_SOURCE=${source} requires DISCORD_BOT_TOKEN. See this file's header.`);
	process.exit(1);
}

const workspace = mkdtempSync(join(tmpdir(), "discrawl-live-qa-"));
const client = new DiscrawlClient({
	source,
	root: workspace,
	timeoutMs: 600_000,
	...(process.env.DISCRAWL_GUILD_ID !== undefined ? { guildId: process.env.DISCRAWL_GUILD_ID } : {}),
});

const doctor = await client.doctor();
if (!doctor.ok) {
	console.error(`discrawl doctor failed: ${doctor.reason}`);
	console.error("Is the discrawl binary installed? brew install openclaw/tap/discrawl");
	process.exit(1);
}
console.log(
	`doctor: db=${doctor.data.databaseOk} fts=${doctor.data.ftsOk} embeddings=${doctor.data.embeddingsOk}` +
		(doctor.data.embeddingModel !== undefined ? ` model=${doctor.data.embeddingModel}` : ""),
);

const skill = new DiscrawlSkill({
	client,
	instanceId: "live",
	...(doctor.data.embeddingModel !== undefined ? { embeddingModel: doctor.data.embeddingModel } : {}),
});

console.log(`\nIndexing (source=${source})...`);
const indexed = await skill.index();
if (!indexed.ok) {
	console.error(`index failed: ${indexed.code} ${indexed.message}`);
	process.exit(1);
}
console.log(`indexed: ${indexed.chunkCount} message(s) synced`);
for (const diagnostic of indexed.diagnostics) {
	console.log(`  [${diagnostic.severity}] ${diagnostic.code}: ${diagnostic.message}`);
}

const status = await client.status();
if (status.ok) {
	console.log(`archive: ${status.data.messages} messages / ${status.data.channels} channels / ${status.data.guilds} guilds`);
}

const queries = process.argv.slice(2);
const effectiveQueries = queries.length > 0 ? queries : ["meeting schedule", "deploy failed"];

for (const method of skill.retrievalMethods()) {
	const name = method.describe().name;
	for (const query of effectiveQueries) {
		const results = await method.retrieve(query, { topK: 3 });
		console.log(`\n[${name}] "${query}" -> ${results.length} hit(s)`);
		for (const [index, result] of results.entries()) {
			const channel = result.metadata?.channelName ?? "?";
			const author = result.metadata?.authorName ?? "?";
			const line = result.content.replace(/\s+/gu, " ").slice(0, 100);
			console.log(`  ${index + 1}. #${channel} <${author}> ${line}`);
			console.log(`     source=${result.source} score=${result.score.toFixed(4)}`);
		}
	}
}

console.log("\nIncremental check: re-running index should sync 0 new messages.");
const second = await skill.index();
console.log(second.ok ? `  second pass: ${second.chunkCount} new message(s)` : `  second pass failed: ${second.message}`);

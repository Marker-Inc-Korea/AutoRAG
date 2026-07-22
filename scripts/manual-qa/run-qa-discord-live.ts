/**
 * Live Discord QA (#1305): index a REAL Discord server through the real
 * Discord REST API and search its conversation history.
 *
 * Prerequisites (one-time, ~3 minutes):
 *  1. https://discord.com/developers/applications → New Application
 *  2. Bot tab → Reset Token → copy it. Enable "MESSAGE CONTENT INTENT".
 *  3. OAuth2 → URL Generator → scope `bot`, permissions "View Channels" +
 *     "Read Message History" → open the generated URL → invite the bot to
 *     your server.
 *  4. export DISCORD_BOT_TOKEN=...
 *
 * Run:
 *   bun scripts/manual-qa/run-qa-discord-live.ts                     # guild auto-detect + sample queries
 *   bun scripts/manual-qa/run-qa-discord-live.ts "질의" ["질의2" ...] # custom queries
 *   DISCORD_GUILD_ID=... bun scripts/manual-qa/run-qa-discord-live.ts  # explicit guild
 */

import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { createSearchDatasourceDocumentsTool } from "../../src/agent/search-datasource-tool.ts";
import { DiscordSkill } from "../../src/datasource/skills/discord/index.ts";

const token = process.env.DISCORD_BOT_TOKEN;
if (token === undefined || token.length === 0) {
	console.error("DISCORD_BOT_TOKEN is not set. See the setup steps in this file's header.");
	process.exit(1);
}

// --- 1. Resolve the guild: explicit env var, or auto-detect the bot's guilds ---
let guildId = process.env.DISCORD_GUILD_ID;
if (guildId === undefined || guildId.length === 0) {
	const response = await fetch("https://discord.com/api/v10/users/@me/guilds", {
		headers: { Authorization: `Bot ${token}` },
	});
	if (!response.ok) {
		console.error(`Failed to list guilds (http-${response.status}). Is the token valid?`);
		process.exit(1);
	}
	const guilds = (await response.json()) as { id: string; name: string }[];
	if (guilds.length === 0) {
		console.error("The bot is not in any server yet. Invite it first (see header).");
		process.exit(1);
	}
	console.log("Bot is a member of:");
	for (const guild of guilds) console.log(`  ${guild.id}  ${guild.name}`);
	guildId = guilds[0]?.id;
	console.log(`Using first guild: ${guilds[0]?.name}\n(Set DISCORD_GUILD_ID to pick another.)\n`);
}

// --- 2. Build the skill + agent exactly as production config would ---
const ws = mkdtempSync(join(tmpdir(), "discord-live-qa-"));
const docs = join(ws, "docs");
mkdirSync(docs, { recursive: true });
writeFileSync(join(docs, "placeholder.txt"), "placeholder");

const skill = new DiscordSkill({
	instanceId: "live",
	workspaceRoot: ws,
	connectorOptions: { token, guildId, maxPages: 5, maxDocuments: 500 },
});
const agent = new AutoRAGAgent({
	searchPaths: [docs],
	workspacePath: ws,
	minSync: false,
	bm25: false,
	datasourceSkills: [skill],
	datasourceAccess: { allowedTags: ["discord"], allowedScopes: ["/discord/**"] },
});

// --- 3. Index real conversation history through agent.refresh() ---
console.log("Indexing real Discord history via agent.refresh() ...");
const refresh = await agent.refresh(true, { methods: ["datasources"] });
const result = refresh.datasources?.[0];
if (result?.ok !== true) {
	console.error(`Index failed: ${result?.code} — ${result?.message}`);
	console.error("Common causes: MESSAGE CONTENT INTENT disabled (empty contents), missing Read Message History.");
	process.exit(1);
}
console.log(`Indexed ${result.chunkCount} message chunk(s) from the live server.\n`);

// --- 4. Search through the same tool the LLM uses ---
const tool = createSearchDatasourceDocumentsTool(agent);
const queries = process.argv.slice(2);
const effectiveQueries = queries.length > 0 ? queries : ["hello", "meeting schedule", "링크 공유"];
for (const query of effectiveQueries) {
	const response = await tool.execute(`live-${query}`, { query, topK: 5 });
	console.log(`Q: "${query}" → ${response.details.resultCount} hit(s)`);
	const text = response.content.map((part) => (part.type === "text" ? part.text : "")).join("");
	console.log(text.split("\n").slice(0, 12).join("\n"), "\n");
}

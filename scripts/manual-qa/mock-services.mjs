/**
 * Manual-QA mock server emulating the external APIs the datasource skills
 * talk to: Slack, Discord, Notion, GitHub, Google Drive, Gmail, and an RSS
 * feed. Started by run-qa.ts on an ephemeral port.
 *
 * Auth behavior mirrors the real services closely enough for QA:
 *  - Slack expects  Authorization: Bearer qa-slack-token
 *  - Discord expects Authorization: Bot qa-discord-token
 *  - Notion/GDrive/Gmail expect their Bearer tokens
 *  - wrong/missing tokens produce the service's native auth failure shape
 */

import { createServer } from "node:http";

const b64url = (text) => Buffer.from(text, "utf8").toString("base64url");

export function startMockServices() {
	const server = createServer((req, res) => {
		const url = new URL(req.url, "http://localhost");
		const path = url.pathname;
		const auth = req.headers.authorization ?? "";
		const json = (body, status = 200) => {
			res.writeHead(status, { "content-type": "application/json" });
			res.end(JSON.stringify(body));
		};
		const text = (body, status = 200, type = "text/plain") => {
			res.writeHead(status, { "content-type": type });
			res.end(body);
		};

		// ---- Slack ----
		if (path === "/slack/conversations.list") {
			if (auth !== "Bearer qa-slack-token") return json({ ok: false, error: "invalid_auth" });
			return json({
				ok: true,
				channels: [
					{ id: "C-ENG", name: "engineering" },
					{ id: "C-SEC", name: "secret-ops" },
				],
				response_metadata: { next_cursor: "" },
			});
		}
		if (path === "/slack/conversations.history") {
			if (auth !== "Bearer qa-slack-token") return json({ ok: false, error: "invalid_auth" });
			const channel = url.searchParams.get("channel");
			if (channel === "C-SEC") return json({ ok: false, error: "not_in_channel" });
			return json({
				ok: true,
				messages: [
					{ ts: "1719999999.000100", user: "alice", text: "The payment gateway migration finishes on July 12." },
					{ ts: "1719999998.000100", user: "bob", text: "Reminder: rotate the staging certificates this week." },
				],
				response_metadata: { next_cursor: "" },
			});
		}

		// ---- Discord ----
		if (path === "/discord/guilds/qa-guild/channels") {
			if (auth !== "Bot qa-discord-token") return text("unauthorized", 401);
			return json([
				{ id: "D-GEN", name: "general", type: 0 },
				{ id: "D-VOICE", name: "voice", type: 2 },
			]);
		}
		if (path === "/discord/channels/D-GEN/messages") {
			if (auth !== "Bot qa-discord-token") return text("unauthorized", 401);
			if (url.searchParams.has("before")) return json([]);
			return json([
				{
					id: "88001",
					content: "Community meetup scheduled for August 3rd in Seoul.",
					timestamp: "2024-07-01T09:00:00.000Z",
					author: { username: "organizer" },
				},
			]);
		}

		// ---- Notion ----
		if (path === "/notion/search") {
			if (auth !== "Bearer qa-notion-token") return text("unauthorized", 401);
			return json({
				results: [
					{
						object: "page",
						id: "qa-page-1",
						last_edited_time: "2024-06-01T00:00:00.000Z",
						parent: { type: "database_id", database_id: "qa-db" },
						properties: { Name: { type: "title", title: [{ plain_text: "Incident runbook" }] } },
					},
				],
				has_more: false,
			});
		}
		if (path === "/notion/blocks/qa-page-1/children") {
			return json({
				results: [
					{
						type: "paragraph",
						paragraph: { rich_text: [{ plain_text: "Page the on-call engineer before restarting the cluster." }] },
					},
				],
			});
		}

		// ---- GitHub ----
		if (path === "/github/repos/qa-org/qa-repo/issues") {
			return json([
				{
					number: 101,
					title: "Search results ignore locale",
					body: "Korean queries are tokenized incorrectly in the ranking pass.",
					state: "open",
					updated_at: "2024-06-15T00:00:00.000Z",
					labels: [{ name: "bug" }],
				},
			]);
		}

		// ---- Google Drive ----
		if (path === "/gdrive/files") {
			if (auth !== "Bearer qa-gdrive-token") return text("unauthorized", 401);
			return json({
				files: [
					{
						id: "qa-doc-1",
						name: "Vendor contract summary",
						mimeType: "application/vnd.google-apps.document",
						modifiedTime: "2024-05-20T00:00:00.000Z",
					},
				],
			});
		}
		if (path === "/gdrive/files/qa-doc-1/export") {
			return text("The vendor contract renews annually with a 60-day cancellation notice.");
		}

		// ---- Gmail ----
		if (path === "/gmail/users/me/messages") {
			if (auth !== "Bearer qa-gmail-token") return text("unauthorized", 401);
			return json({ messages: [{ id: "qa-mail-1" }] });
		}
		if (path === "/gmail/users/me/messages/qa-mail-1") {
			return json({
				id: "qa-mail-1",
				threadId: "qa-thread-1",
				labelIds: ["INBOX"],
				internalDate: "1718000000000",
				payload: {
					mimeType: "multipart/alternative",
					headers: [
						{ name: "Subject", value: "Office relocation" },
						{ name: "From", value: "facilities@example.com" },
						{ name: "To", value: "all@example.com" },
					],
					parts: [
						{ mimeType: "text/plain", body: { data: b64url("We move to the new Gangnam office on September 2.") } },
					],
				},
			});
		}

		// ---- RSS ----
		if (path === "/rss/feed.xml") {
			return text(
				`<?xml version="1.0"?><rss version="2.0"><channel><title>QA Times</title>` +
					`<item><title>Framework 3.0 released</title><guid>qa-rss-1</guid>` +
					`<pubDate>Mon, 01 Jul 2024 10:00:00 GMT</pubDate>` +
					`<description>The new release ships incremental indexing.</description></item>` +
					`</channel></rss>`,
				200,
				"application/rss+xml",
			);
		}

		return text("not found", 404);
	});

	return new Promise((resolvePromise) => {
		server.listen(0, "127.0.0.1", () => {
			resolvePromise({ server, port: server.address().port });
		});
	});
}

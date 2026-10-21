/**
 * Live E2E: the Apple Calendar/Reminders path documented in docs/eventkit-mcp.md.
 *
 * AutoRAG does not ship a Calendar datasource. On macOS the supported way to
 * read Apple Calendar and Reminders (including Google/Exchange/iCloud accounts
 * already added to Calendar.app) is to run `mcp-server-apple-events` next to
 * `autorag-mcp` in the same MCP client. This test drives that server over
 * stdio with the repo's own MCP client, read-only.
 *
 * Gated on AUTORAG_EVENTKIT_LIVE=1 (macOS only; needs network for `npx` and
 * Calendar/Reminders permission granted to the process that launches it):
 *
 *   AUTORAG_EVENTKIT_LIVE=1 bun run vitest run test/live-e2e/eventkit-mcp.test.ts
 *
 * Assertions are structural only: no calendar or reminder content is pinned,
 * logged, or written to disk.
 */
import { Client } from "@modelcontextprotocol/client";
import { StdioClientTransport } from "@modelcontextprotocol/client/stdio";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

const LIVE = process.env.AUTORAG_EVENTKIT_LIVE === "1" && process.platform === "darwin";

interface Envelope {
	readonly provenance?: { readonly source?: string; readonly trust?: string };
	readonly records?: readonly Record<string, unknown>[];
}

const DAY_MS = 86_400_000;

describe.skipIf(!LIVE)("EventKit MCP live e2e (macOS Calendar + Reminders, read-only)", () => {
	let client: Client;

	beforeAll(async () => {
		client = new Client({ name: "autorag-eventkit-live-e2e", version: "0.0.0" });
		await client.connect(
			new StdioClientTransport({
				command: "npx",
				args: ["-y", "mcp-server-apple-events", "--require-write-confirmation"],
			}),
		);
	}, 120_000);

	afterAll(async () => {
		await client?.close();
	});

	async function read(name: string, args: Record<string, unknown>): Promise<Envelope> {
		const result = await client.callTool({ name, arguments: args });
		expect(result.isError, `${name} returned an error`).not.toBe(true);
		return result.structuredContent as Envelope;
	}

	it("exposes the calendar and reminders tools the docs describe", async () => {
		const names = (await client.listTools()).tools.map((tool) => tool.name);
		expect(names).toEqual(
			expect.arrayContaining([
				"calendar_calendars",
				"calendar_events",
				"reminders_lists",
				"reminders_tasks",
				"reminders_subtasks",
			]),
		);
	});

	it("lists every calendar with its account source", async () => {
		const envelope = await read("calendar_calendars", { action: "read" });
		expect(envelope.provenance).toMatchObject({ source: "apple-calendar-reminders", trust: "untrusted" });
		expect(envelope.records?.length ?? 0).toBeGreaterThan(0);
	});

	it("reads events across calendars for a bounded date range", async () => {
		const envelope = await read("calendar_events", {
			action: "read",
			startDate: new Date(Date.now() - 7 * DAY_MS).toISOString().slice(0, 10),
			endDate: new Date(Date.now() + 7 * DAY_MS).toISOString().slice(0, 10),
		});
		expect(Array.isArray(envelope.records)).toBe(true);
	});

	it("reads reminder lists and tasks", async () => {
		const lists = await read("reminders_lists", { action: "read" });
		expect(lists.records?.length ?? 0).toBeGreaterThan(0);
		const tasks = await read("reminders_tasks", { action: "read" });
		expect(Array.isArray(tasks.records)).toBe(true);
	});

	it("refuses writes when --require-write-confirmation is set and the client cannot elicit", async () => {
		const title = `autorag-live-e2e-must-not-exist-${Date.now()}`;
		const result = await client.callTool({
			name: "reminders_tasks",
			arguments: { action: "create", title },
		});
		expect(result.isError).toBe(true);
		const tasks = await read("reminders_tasks", { action: "read", search: title });
		const leaked = (tasks.records ?? []).filter((record) => record.title === title);
		expect(leaked).toEqual([]);
	});
});

import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { afterEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import type { SimplexIncomingMessage, SimplexTransport } from "../../src/p2p/simplex-transport.ts";
import type { PeerQueryResponse } from "../../src/p2p/wire.ts";

const FIXTURE_DIR = "test/fixtures/sample-project";
const workspaces: string[] = [];

afterEach(() => {
	for (const workspace of workspaces.splice(0)) rmSync(workspace, { recursive: true, force: true });
});

function workspaceWithPeers(): string {
	const workspace = mkdtempSync(join(tmpdir(), "autorag-peer-query-"));
	workspaces.push(workspace);
	mkdirSync(join(workspace, ".autorag", "p2p"), { recursive: true });
	writeFileSync(
		join(workspace, ".autorag", "p2p", "simplex-peers.json"),
		JSON.stringify({
			alice: {
				contactId: 42,
				addedAt: "2026-01-01T00:00:00.000Z",
				displayName: "Alice",
				description: "Finance and budget specialist",
				role: "finance lead",
			},
			carol: {
				contactId: 44,
				addedAt: "2026-01-02T00:00:00.000Z",
				displayName: "Carol",
			},
		}),
	);
	return workspace;
}

function registeredTools(agent: AutoRAGAgent): readonly AgentTool[] {
	return (
		agent as unknown as {
			readonly innerAgent: { readonly state: { readonly tools: readonly AgentTool[] } };
		}
	).innerAgent.state.tools;
}

function toolNamed(agent: AutoRAGAgent, name: string): AgentTool | undefined {
	return registeredTools(agent).find((candidate) => candidate.name === name);
}

function localAgent(workspace: string, openTransport?: () => Promise<SimplexTransport>): AutoRAGAgent {
	return new AutoRAGAgent({
		searchPaths: [FIXTURE_DIR],
		workspacePath: workspace,
		memoryPath: join(workspace, "memory.json"),
		minSync: false,
		jikji: false,
		...(openTransport !== undefined ? { peerQuery: { openTransport } } : {}),
	});
}

interface ScriptedTransport extends SimplexTransport {
	readonly sent: { contactId: number; text: string }[];
}

function scriptedTransport(response: PeerQueryResponse): ScriptedTransport {
	const handlers: ((message: SimplexIncomingMessage) => void)[] = [];
	const sent: { contactId: number; text: string }[] = [];
	return {
		dbPrefix: "test",
		displayName: "test",
		sent,
		async getUserId() {
			return 1;
		},
		async getOrCreateAddress() {
			return "simplex://test";
		},
		async createInvitation() {
			return "simplex://invite";
		},
		async connect() {},
		async listContacts() {
			return [];
		},
		async sendMessage(contactId, text) {
			sent.push({ contactId, text });
			const envelope = JSON.parse(text) as { id?: string };
			const reply = JSON.stringify({ v: 1, kind: "response", id: envelope.id, payload: response });
			for (const handler of handlers) {
				handler({ contactId, contactName: "carol", text: reply, chatItemId: 1 });
			}
		},
		onMessage(handler) {
			handlers.push(handler);
			return () => {
				const index = handlers.indexOf(handler);
				if (index >= 0) handlers.splice(index, 1);
			};
		},
		async close() {},
	};
}

describe("peer contact descriptions", () => {
	it("lists every contact and marks a missing background description", async () => {
		const workspace = workspaceWithPeers();
		const agent = localAgent(workspace);
		const tool = toolNamed(agent, "list_peer_contacts");
		expect(tool).toBeDefined();
		const result = await tool?.execute("list", {});
		expect(result?.details).toEqual({
			method: "list_peer_contacts",
			resultCount: 2,
			contacts: [
				{
					alias: "alice",
					contactId: 42,
					addedAt: "2026-01-01T00:00:00.000Z",
					displayName: "Alice",
					description: "Finance and budget specialist",
					role: "finance lead",
					descriptionMissing: false,
				},
				{
					alias: "carol",
					contactId: 44,
					addedAt: "2026-01-02T00:00:00.000Z",
					displayName: "Carol",
					descriptionMissing: true,
				},
			],
		});
	});

	it("writes a background description without changing trust or other contacts", async () => {
		const workspace = workspaceWithPeers();
		const agent = localAgent(workspace);
		const tool = toolNamed(agent, "update_peer_contact_description");
		expect(tool).toBeDefined();
		const result = await tool?.execute("update", {
			alias: "carol",
			description: "Owns the refund policy corpus",
		});
		expect(result?.details).toMatchObject({ ok: true, alias: "carol", description: "Owns the refund policy corpus" });
		const registry = JSON.parse(readFileSync(join(workspace, ".autorag", "p2p", "simplex-peers.json"), "utf8"));
		expect(registry.carol).toEqual({
			contactId: 44,
			addedAt: "2026-01-02T00:00:00.000Z",
			displayName: "Carol",
			description: "Owns the refund policy corpus",
		});
		expect(registry.alice).toMatchObject({
			contactId: 42,
			role: "finance lead",
			description: "Finance and budget specialist",
		});
	});

	it("clears a description with an empty string", async () => {
		const workspace = workspaceWithPeers();
		const agent = localAgent(workspace);
		const tool = toolNamed(agent, "update_peer_contact_description");
		await tool?.execute("clear", { alias: "alice", description: "   " });
		const registry = JSON.parse(readFileSync(join(workspace, ".autorag", "p2p", "simplex-peers.json"), "utf8"));
		expect(registry.alice.description).toBeUndefined();
		expect(registry.alice.contactId).toBe(42);
	});

	it("does not create an unknown alias or accept an overlong description", async () => {
		const workspace = workspaceWithPeers();
		const before = readFileSync(join(workspace, ".autorag", "p2p", "simplex-peers.json"), "utf8");
		const agent = localAgent(workspace);
		const tool = toolNamed(agent, "update_peer_contact_description");
		const unknown = await tool?.execute("missing", { alias: "mallory", description: "no such contact" });
		const overlong = await tool?.execute("long", { alias: "alice", description: "a".repeat(2001) });
		expect(unknown?.details).toMatchObject({ ok: false });
		expect(overlong?.details).toMatchObject({ ok: false });
		expect(readFileSync(join(workspace, ".autorag", "p2p", "simplex-peers.json"), "utf8")).toBe(before);
	});
});

describe("query_peer_agent", () => {
	it("sends a v1 query to the alias contact and omits file bytes", async () => {
		const workspace = workspaceWithPeers();
		const transport = scriptedTransport({
			v: 1,
			status: "ok",
			answer: "Refunds need director approval.",
			results: [
				{
					number: 1,
					title: "Refund policy",
					summary: "Director approval is required.",
					source: "/docs/refund.md",
					excerpt: "Director approval before payout.",
				},
			],
			files: [{ source: "/docs/refund.md", contentBase64: "U0VDUkVU", redacted: false }],
			diagnostics: [],
		});
		const agent = localAgent(workspace, async () => transport);
		const tool = toolNamed(agent, "query_peer_agent");
		expect(tool).toBeDefined();
		const result = await tool?.execute("query", { alias: "alice", query: "refund approval", topK: 3 });
		expect(transport.sent).toHaveLength(1);
		expect(transport.sent[0]?.contactId).toBe(42);
		const envelope = JSON.parse(transport.sent[0]?.text ?? "{}") as {
			v: number;
			kind: string;
			payload: { v: number; query: string; topK?: number };
		};
		expect(envelope).toMatchObject({
			v: 1,
			kind: "query",
			payload: { v: 1, query: "refund approval", topK: 3 },
		});
		expect(result?.details).toMatchObject({
			method: "query_peer_agent",
			ok: true,
			alias: "alice",
			contactId: 42,
			status: "ok",
			answer: "Refunds need director approval.",
			results: [
				{
					number: 1,
					title: "Refund policy",
					summary: "Director approval is required.",
					source: "/docs/refund.md",
					excerpt: "Director approval before payout.",
				},
			],
		});
		expect(JSON.stringify(result)).not.toContain("U0VDUkVU");
		expect(JSON.stringify(result?.content)).toContain("untrusted");
	});

	it("does not send when the alias is unknown and returns the transport error verbatim", async () => {
		const workspace = workspaceWithPeers();
		const transport = scriptedTransport({
			v: 1,
			status: "rejected",
			answer: "",
			results: [],
			files: [],
			diagnostics: [],
		});
		let opened = 0;
		const agent = localAgent(workspace, async () => {
			opened += 1;
			throw new Error("simplex-chat failed to start: db is locked");
		});
		const missing = toolNamed(agent, "query_peer_agent");
		const unknown = await missing?.execute("missing", { alias: "mallory", query: "hello" });
		expect(unknown?.details).toMatchObject({ ok: false, alias: "mallory" });
		expect(opened).toBe(0);
		expect(transport.sent).toHaveLength(0);
		const failed = await missing?.execute("down", { alias: "alice", query: "hello" });
		expect(failed?.details).toMatchObject({ ok: false });
		expect(JSON.stringify(failed?.details)).toContain("simplex-chat failed to start: db is locked");
	});
});

describe("peer query tool registration", () => {
	it("registers the contact and query tools on a local agent only", () => {
		const workspace = workspaceWithPeers();
		const local = localAgent(workspace);
		const remote = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			workspacePath: workspace,
			memoryPath: join(workspace, "memory.json"),
			remoteSession: true,
			minSync: false,
			jikji: false,
		});
		const names = ["list_peer_contacts", "update_peer_contact_description", "query_peer_agent"];
		for (const name of names) expect(registeredTools(local).map((tool) => tool.name)).toContain(name);
		for (const name of names) expect(registeredTools(remote).map((tool) => tool.name)).not.toContain(name);
	});
});

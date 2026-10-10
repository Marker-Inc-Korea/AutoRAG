import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { EvidenceLedger } from "../../src/agent/evidence-ledger.ts";
import { createQueryPeerAgentTool } from "../../src/agent/query-peer-tool.ts";
import { loadSimplexQueryState } from "../../src/p2p/simplex-query-store.ts";
import type { SimplexContact, SimplexIncomingMessage, SimplexTransport } from "../../src/p2p/simplex-transport.ts";

const roots: string[] = [];

afterEach(() => {
	for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

function workspace(): string {
	const root = mkdtempSync(join(tmpdir(), "autorag-query-peer-tool-"));
	roots.push(root);
	return root;
}

class SilentTransport implements SimplexTransport {
	readonly dbPrefix = "test";
	readonly displayName = "test";
	readonly sent: { contactId: number; text: string }[] = [];
	private readonly handlers: ((message: SimplexIncomingMessage) => void)[] = [];

	async getUserId(): Promise<number> {
		return 1;
	}
	async getOrCreateAddress(): Promise<string> {
		return "simplex:/test";
	}
	async createInvitation(): Promise<string> {
		return "simplex:/invite";
	}
	async connect(): Promise<void> {}
	async listContacts(): Promise<SimplexContact[]> {
		return [];
	}
	async sendMessage(contactId: number, text: string): Promise<void> {
		this.sent.push({ contactId, text });
	}
	onMessage(handler: (message: SimplexIncomingMessage) => void): () => void {
		this.handlers.push(handler);
		return () => {
			const index = this.handlers.indexOf(handler);
			if (index >= 0) this.handlers.splice(index, 1);
		};
	}
	emit(message: SimplexIncomingMessage): void {
		for (const handler of this.handlers) handler(message);
	}
	async close(): Promise<void> {}
}

function peerRegistry(root: string): void {
	const directory = join(root, ".autorag", "p2p");
	mkdirSync(directory, { recursive: true });
	writeFileSync(
		join(directory, "simplex-peers.json"),
		JSON.stringify({
			alice: { contactId: 42, addedAt: "2026-01-01T00:00:00.000Z" },
		}),
	);
}

const response = {
	v: 1 as const,
	status: "ok" as const,
	answer: "late peer answer",
	results: [],
	files: [],
	diagnostics: [],
};

describe("query_peer_agent", () => {
	it("returns a non-blocking pending result and persists the request", async () => {
		const root = workspace();
		peerRegistry(root);
		const transport = new SilentTransport();
		const tool = createQueryPeerAgentTool({
			workspacePath: root,
			fastTimeoutMs: 1,
			openTransport: async () => transport,
		});

		const result = await tool.execute("call-1", { alias: "alice", query: "refund approval" });

		expect(result.details).toMatchObject({
			method: "query_peer_agent",
			ok: true,
			alias: "alice",
			contactId: 42,
			status: "pending",
		});
		const pendingId = result.details.pendingId;
		expect(pendingId).toBeTypeOf("string");
		expect(loadSimplexQueryState(root, pendingId as string)).toMatchObject({
			status: "pending",
			request: { query: "refund approval" },
		});
	});

	it("resumes a late response through the event callback after tool recreation", async () => {
		const root = workspace();
		peerRegistry(root);
		const firstTransport = new SilentTransport();
		const firstTool = createQueryPeerAgentTool({
			workspacePath: root,
			fastTimeoutMs: 1,
			autoStart: true,
			openTransport: async () => firstTransport,
		});
		await firstTool.ready;
		const first = await firstTool.execute("call-2", { alias: "alice", query: "late question" });
		const pendingId = first.details.pendingId;
		expect(pendingId).toBeTypeOf("string");
		await firstTool.close();

		const resumed: string[] = [];
		const secondTransport = new SilentTransport();
		const secondTool = createQueryPeerAgentTool({
			workspacePath: root,
			fastTimeoutMs: 1,
			autoStart: true,
			openTransport: async () => secondTransport,
			onResponse: (_state, lateResponse) => resumed.push(lateResponse.answer),
		});
		await secondTool.ready;
		secondTransport.emit({
			contactId: 42,
			contactName: "alice",
			chatItemId: 1,
			text: JSON.stringify({ v: 1, kind: "response", id: pendingId, payload: response }),
		});
		secondTransport.emit({
			contactId: 42,
			contactName: "alice",
			chatItemId: 2,
			text: JSON.stringify({ v: 1, kind: "response", id: pendingId, payload: response }),
		});

		expect(resumed).toEqual(["late peer answer"]);
		expect(loadSimplexQueryState(root, pendingId as string)?.status).toBe("completed");
		await secondTool.close();
	});

	it("registers each peer result as citable evidence carrying the peer's source", async () => {
		const root = workspace();
		peerRegistry(root);
		const transport = new SilentTransport();
		transport.sendMessage = async (contactId, text) => {
			const { id } = JSON.parse(text) as { id: string };
			const payload = {
				...response,
				answer: "peer answer",
				results: [
					{
						number: 1,
						title: "Refund",
						summary: "Director approval",
						source: "peer:doc-7",
						excerpt: "needs approval",
					},
				],
			};
			queueMicrotask(() =>
				transport.emit({
					contactId,
					contactName: "alice",
					chatItemId: 3,
					text: JSON.stringify({ v: 1, kind: "response", id, payload }),
				}),
			);
		};
		const ledger = new EvidenceLedger();
		const tool = createQueryPeerAgentTool({
			workspacePath: root,
			fastTimeoutMs: 5_000,
			openTransport: async () => transport,
			ledger,
		});

		const result = await tool.execute("call-3", { alias: "alice", query: "refund approval" });

		const text = result.content[0]?.type === "text" ? result.content[0].text : "";
		expect(text).toContain('"evidenceId":"e1"');
		const [ref] = ledger.resolve(["e1"], { label: "emit", number: 1, fallbackContent: "", allowLocalFiles: false });
		expect(ref).toMatchObject({ method: "query_peer_agent", source: "peer:doc-7", content: "needs approval" });
		await tool.close();
	});
});

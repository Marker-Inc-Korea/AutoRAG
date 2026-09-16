/**
 * Manual QA harness for issue #1539 (P2P path standard unification).
 *
 * Drives the REAL serve/server/agent/egress pipeline (no production-code
 * mocks; only the SimpleX transport and the model provider are faked, per the
 * repo's test conventions) and captures three observable gates:
 *
 *   A. `autorag serve` startup prints a MinSync readiness warning on stderr.
 *   B. A peer query whose remote session emits nothing receives a structured
 *      `no-verified-results` response (never `internal-error`).
 *   C. A peer query whose results carry OS-absolute local sources allowed by
 *      a virtual policy glob receives an `ok` response with results.
 *
 * Run: bun scripts/manual-qa/run-qa-p2p-path-standard.ts
 * Evidence: .omo/evidence/p2p-path-standard-qa.json
 */
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { type FauxProviderRegistration, fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { EMIT_AUTORAG_RESULTS_TOOL_NAME } from "../../src/agent/emit-results-tool.ts";
import { runServe } from "../../src/cli/commands/serve.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import { planSourceRoots } from "../../src/filesystem/source-paths.ts";
import { listPendingPeerRequests, writePeerRequestDecision } from "../../src/p2p/approval-store.ts";
import { PolicyStore } from "../../src/p2p/policy.ts";
import {
	type SimplexPeerServer,
	startSimplexPeerServer,
} from "../../src/p2p/simplex-server.ts";
import type { SimplexIncomingMessage, SimplexTransport } from "../../src/p2p/simplex-transport.ts";
import type { PeerQueryResponse } from "../../src/p2p/wire.ts";

const EVIDENCE_PATH = ".omo/evidence/p2p-path-standard-qa.json";

interface GateResult {
	readonly gate: string;
	readonly pass: boolean;
	readonly detail: string;
}

class FakeTransport implements SimplexTransport {
	readonly dbPrefix = "fake";
	peer: FakeTransport | undefined;
	readonly sent: { contactId: number; text: string }[] = [];
	readonly received: string[] = [];
	private readonly handlers: ((message: SimplexIncomingMessage) => void)[] = [];

	constructor(
		readonly contactId: number,
		readonly displayName: string,
	) { }

	async getUserId(): Promise<number> {
		return 1;
	}
	async getOrCreateAddress(): Promise<string> {
		return "simplex:/contact#fake";
	}
	async createInvitation(): Promise<string> {
		return "simplex:/invitation#fake";
	}
	async connect(): Promise<void> { }
	async listContacts(): Promise<{ contactId: number; localDisplayName: string }[]> {
		return this.peer !== undefined
			? [{ contactId: this.peer.contactId, localDisplayName: this.peer.displayName }]
			: [];
	}
	async sendMessage(contactId: number, text: string): Promise<void> {
		this.sent.push({ contactId, text });
		const target = this.peer;
		if (target !== undefined && target.contactId === contactId) {
			queueMicrotask(() => {
				target.emit({ contactId: this.contactId, contactName: this.displayName, text, chatItemId: Date.now() });
			});
		}
	}
	onMessage(handler: (message: SimplexIncomingMessage) => void): () => void {
		this.handlers.push(handler);
		return () => {
			const index = this.handlers.indexOf(handler);
			if (index !== -1) this.handlers.splice(index, 1);
		};
	}
	emit(message: SimplexIncomingMessage): void {
		for (const handler of this.handlers) handler(message);
	}
	captureReplies(): void {
		this.onMessage((message) => this.received.push(message.text));
	}
	async close(): Promise<void> { }
}

function transportPair(): { client: FakeTransport; server: FakeTransport } {
	const client = new FakeTransport(2, "qa-client");
	const server = new FakeTransport(1, "qa-server");
	client.peer = server;
	server.peer = client;
	client.captureReplies();
	return { client, server };
}

async function lastResponse(transport: FakeTransport): Promise<PeerQueryResponse> {
	const deadline = Date.now() + 5000;
	for (; ;) {
		const responses = transport.received
			.map((raw) => {
				try {
					return JSON.parse(raw) as { kind: string; payload: PeerQueryResponse };
				} catch {
					return undefined;
				}
			})
			.filter((envelope) => envelope?.kind === "response")
			.map((envelope) => envelope?.payload as PeerQueryResponse);
		if (responses.length > 0) return responses[responses.length - 1]!;
		if (Date.now() > deadline) throw new Error("timed out waiting for peer response");
		await new Promise((resolve) => setTimeout(resolve, 10));
	}
}

async function gateA(root: string): Promise<GateResult> {
	const docsDir = join(root, "docs-a");
	mkdirSync(docsDir, { recursive: true });
	const configPath = join(root, "config.json");
	writeFileSync(
		configPath,
		JSON.stringify({
			searchPaths: [docsDir],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			p2p: { injectionClassifier: false },
		}),
	);
	const stderr: string[] = [];
	const ctx: CommandContext = {
		positionals: [],
		flags: { config: configPath, force: true },
		json: false,
		debug: false,
		cwd: root,
		stdout: () => { },
		stderr: (line) => stderr.push(line),
	};
	const transport = new FakeTransport(1, "qa-serve");
	const code = await runServe(ctx, {
		startSimplexChat: async () => transport,
		startSimplexPeerServer: async () => ({ close: async () => undefined }) as SimplexPeerServer,
		waitUntilStopped: async (server) => {
			await server.close();
		},
	});
	const text = stderr.join("\n");
	const pass = code === 0 && /minsync/i.test(text) && /not ready|degraded|unavailable|refresh/i.test(text);
	return {
		gate: "A: serve startup readiness warning",
		pass,
		detail: pass ? `stderr: ${text.trim()}` : `exit=${code} stderr=${JSON.stringify(stderr)}`,
	};
}

async function gateB(root: string, registrations: FauxProviderRegistration[]): Promise<GateResult> {
	const docsDir = join(root, "docs-b");
	mkdirSync(docsDir, { recursive: true });
	const registration = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "text-only" }] });
	registration.setResponses([
		() => fauxAssistantMessage([{ type: "text", text: "I could not find anything relevant." }], {
			stopReason: "stop",
		}),
	]);
	registrations.push(registration);
	const agent = new AutoRAGAgent({
		model: registration.getModel(),
		searchPaths: [docsDir],
		workspacePath: root,
		memoryPath: join(root, "memory-b.json"),
		remoteSession: true,
		minSync: false,
		jikji: false,
		thinking: false,
	});
	const { client, server } = transportPair();
	const handle = await startSimplexPeerServer({
		transport: server,
		agent,
		peers: { "qa-client": { contactId: 2, addedAt: new Date().toISOString() } },
		workspacePath: root,
		injectionClassifier: false,
	});
	try {
		await client.sendMessage(
			server.contactId,
			JSON.stringify({ v: 1, kind: "query", id: "qa-empty", payload: { v: 1, query: "nothing exists for this" } }),
		);
		const response = await lastResponse(client);
		const codes = response.diagnostics.map((d) => d.code);
		const pass =
			response.status === "rejected" &&
			codes.includes("no-verified-results") &&
			!codes.includes("internal-error");
		return {
			gate: "B: empty remote session -> structured no-verified-results",
			pass,
			detail: `status=${response.status} diagnostics=${JSON.stringify(codes)}`,
		};
	} finally {
		await handle.close();
	}
}

async function gateC(root: string, registrations: FauxProviderRegistration[]): Promise<GateResult> {
	const docsDir = join(root, "docs-c");
	mkdirSync(docsDir, { recursive: true });
	const sourceFile = join(docsDir, "refund-policy.txt");
	writeFileSync(sourceFile, "Refund exceptions require director approval before payout.\n");
	mkdirSync(join(root, ".autorag", "p2p"), { recursive: true });
	writeFileSync(
		join(root, ".autorag", "p2p", "policy.toml"),
		'newFilesPublic = true\n\n[policy."/docs-c/**"]\ntier = "always"\n',
	);
	const registration = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "emit-abs" }] });
	registration.setResponses([
		fauxAssistantMessage(
			[
				fauxToolCall(EMIT_AUTORAG_RESULTS_TOOL_NAME, {
					answer: "[1] Refund exceptions require director approval.",
					results: [
						{
							number: 1,
							title: "Refund policy",
							summary: "Refund exceptions require director approval.",
							evidence: [{ excerpt: "Refund exceptions require director approval before payout." }],
							confidence: 0.95,
						},
					],
					mapping: [
						{
							number: 1,
							source: sourceFile,
							method: "bash",
							content: "Refund exceptions require director approval before payout.",
						},
					],
				}),
			],
			{ stopReason: "toolUse" },
		),
	]);
	registrations.push(registration);
	const agent = new AutoRAGAgent({
		model: registration.getModel(),
		searchPaths: [docsDir],
		workspacePath: root,
		memoryPath: join(root, "memory-c.json"),
		remoteSession: true,
		minSync: false,
		jikji: false,
		thinking: false,
	});
	const policyStore = new PolicyStore({
		workspacePath: root,
		globalConfigPath: join(root, "missing-global-config.json"),
		sourceRoots: planSourceRoots([docsDir]),
	});
	const { client, server } = transportPair();
	const handle = await startSimplexPeerServer({
		transport: server,
		agent,
		peers: { "qa-client": { contactId: 2, addedAt: new Date().toISOString() } },
		workspacePath: root,
		injectionClassifier: false,
		resolvePolicy: policyStore.resolvePolicy.bind(policyStore),
	});
	try {
		await client.sendMessage(
			server.contactId,
			JSON.stringify({ v: 1, kind: "query", id: "qa-abs", payload: { v: 1, query: "refund approval" } }),
		);
		const deadline = Date.now() + 5000;
		while (listPendingPeerRequests(root).length === 0) {
			if (Date.now() > deadline) throw new Error("no pending peer request appeared");
			await new Promise((resolve) => setTimeout(resolve, 10));
		}
		writePeerRequestDecision(root, listPendingPeerRequests(root)[0]!.id, "approve");
		const response = await lastResponse(client);
		const pass = response.status === "ok" && response.results.length === 1;
		return {
			gate: "C: absolute-path source allowed by virtual glob -> ok",
			pass,
			detail: `status=${response.status} results=${response.results.length} source=${response.results[0]?.source ?? "none"}`,
		};
	} finally {
		await handle.close();
	}
}

const root = mkdtempSync(join(tmpdir(), "autorag-qa-p2p-path-standard-"));
const registrations: FauxProviderRegistration[] = [];
const results: GateResult[] = [];
try {
	results.push(await gateA(root));
	results.push(await gateB(root, registrations));
	results.push(await gateC(root, registrations));
} finally {
	for (const registration of registrations) registration.unregister();
}
const pass = results.every((result) => result.pass);
const evidence = { qa: "p2p-path-standard", issue: 1539, ranAt: new Date().toISOString(), pass, results };
mkdirSync(".omo/evidence", { recursive: true });
writeFileSync(EVIDENCE_PATH, `${JSON.stringify(evidence, null, 2)}\n`);
for (const result of results) console.log(`${result.pass ? "PASS" : "FAIL"} ${result.gate} — ${result.detail}`);
console.log(`evidence: ${EVIDENCE_PATH}`);
rmSync(root, { recursive: true, force: true });
console.log(`cleanup: removed ${root}`);
process.exit(pass ? 0 : 1);

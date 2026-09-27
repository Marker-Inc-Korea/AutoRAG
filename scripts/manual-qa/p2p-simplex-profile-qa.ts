#!/usr/bin/env bun
/**
 * Live manual QA for the profile-based P2P contact model (no mocks).
 *
 * Drives the REAL simplex-chat bot API with two local profiles: peer B
 * publishes a profile, A connects, and the harness checks the user-visible
 * contract end to end.
 *
 *   gate transport-profile : A.listContacts() carries the profile B shared.
 *   gate registry-sync     : syncSimplexPeers stores it on the trusted record
 *                            and keeps your local contact name and note.
 *   gate no-auto-trust     : a SimpleX contact with no trusted record is
 *                            reported and never written.
 *   gate cli-surface       : `autorag p2p peers` shows the profile plus your
 *                            local name and note.
 *
 * Run: bun scripts/manual-qa/p2p-simplex-profile-qa.ts
 * Evidence: .omo/evidence/p2p-profile-qa.json
 */
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadSimplexPeerRegistry, syncSimplexPeers } from "../../src/p2p/simplex-server.ts";
import { type SimplexPeerProfile, type SimplexTransport, startSimplexChat } from "../../src/p2p/simplex-transport.ts";

const repoRoot = fileURLToPath(new URL("../..", import.meta.url));
const EVIDENCE_PATH = join(repoRoot, ".omo/evidence/p2p-profile-qa.json");
const PORT_A = 25_911;
const PORT_B = 25_912;

/** The profile peer B publishes over the real SimpleX protocol. */
const PROFILE_B: SimplexPeerProfile = {
	displayName: "qa-b-profile",
	fullName: "QA Peer B",
	shortDescr: "refund policy owner",
	description: "Handles refund approvals",
};
const LOCAL_ALIAS = "refund-owner";
const LOCAL_NOTE = "내가 붙인 메모: 환불 담당";

interface GateResult {
	readonly gate: string;
	readonly pass: boolean;
	readonly detail: string;
}
const gates: GateResult[] = [];
function gate(name: string, pass: boolean, detail: string): void {
	gates.push({ gate: name, pass, detail });
	console.log(`${pass ? "PASS" : "FAIL"} ${name}: ${detail}`);
}

const dirA = mkdtempSync(join(tmpdir(), "simplex-profile-a-"));
const dirB = mkdtempSync(join(tmpdir(), "simplex-profile-b-"));
const workspace = mkdtempSync(join(tmpdir(), "simplex-profile-ws-"));
const configPath = join(workspace, "config.json");
writeFileSync(
	configPath,
	JSON.stringify({ searchPaths: [workspace], workspacePath: workspace, memoryPath: join(workspace, "memory.json") }),
);

let a: SimplexTransport | undefined;
let b: SimplexTransport | undefined;
const cleanup: string[] = [];
let registryFile = "";
let cliJson = "";
let cliPlain = "";
let untrustedDetail = "";

/** Publish a profile through the real bot API, on its own WebSocket connection. */
async function publishProfile(port: number, userId: number, profile: SimplexPeerProfile): Promise<unknown> {
	const ws = new WebSocket(`ws://127.0.0.1:${port}`);
	await new Promise<void>((resolve, reject) => {
		ws.addEventListener("open", () => resolve());
		ws.addEventListener("error", () => reject(new Error(`profile publish socket failed on port ${port}`)));
	});
	const corrId = `qa-profile-${Date.now()}`;
	const resp = await new Promise<unknown>((resolve, reject) => {
		const timer = setTimeout(() => reject(new Error("profile publish timed out")), 15_000);
		ws.addEventListener("message", (event) => {
			const data = typeof event.data === "string" ? event.data : Buffer.from(event.data as ArrayBuffer).toString();
			let parsed: { corrId?: unknown; resp?: unknown };
			try {
				parsed = JSON.parse(data) as typeof parsed;
			} catch {
				return;
			}
			if (parsed.corrId !== corrId) return;
			clearTimeout(timer);
			resolve(parsed.resp);
		});
		ws.send(JSON.stringify({ corrId, cmd: `/_profile ${userId} ${JSON.stringify(profile)}` }));
	});
	ws.close();
	return resp;
}

function responseTypeOf(payload: unknown): string {
	if (typeof payload !== "object" || payload === null) return "unknown";
	const type = (payload as { type?: unknown }).type;
	return typeof type === "string" ? type : "unknown";
}

function runCli(args: string[]): { status: number | null; stdout: string; stderr: string } {
	const result = spawnSync(process.execPath, [join(repoRoot, "src/cli/index.ts"), ...args], {
		cwd: workspace,
		encoding: "utf8",
	});
	return { status: result.status, stdout: result.stdout ?? "", stderr: result.stderr ?? "" };
}

/** Wait for the connection to establish, bounded. */
async function waitForContacts(transport: SimplexTransport): Promise<Awaited<ReturnType<SimplexTransport["listContacts"]>>> {
	const deadline = Date.now() + 30_000;
	let last: Awaited<ReturnType<SimplexTransport["listContacts"]>> = [];
	for (;;) {
		last = await transport.listContacts();
		if (last.length > 0) return last;
		if (Date.now() > deadline) return last;
		await new Promise((resolve) => setTimeout(resolve, 250));
	}
}

try {
	a = await startSimplexChat({ dbPrefix: join(dirA, "a"), displayName: "qa-a", port: PORT_A });
	b = await startSimplexChat({ dbPrefix: join(dirB, "b"), displayName: "qa-b", port: PORT_B });

	// 1. Peer B publishes its profile through the real SimpleX protocol.
	const userIdB = await b.getUserId();
	const published = await publishProfile(PORT_B, userIdB, PROFILE_B);
	const publishedType = responseTypeOf(published);
	gate(
		"peer-publishes-profile",
		publishedType === "userProfileUpdated" || publishedType === "userProfileNoChange",
		`simplex-chat response type: ${publishedType} ${JSON.stringify(published).slice(0, 200)}`,
	);

	// 2. Connect the two real profiles.
	const invitation = await a.createInvitation();
	await b.connect(invitation);
	const contacts = await waitForContacts(a);
	const peer = contacts[0];
	const received = peer?.profile;
	gate(
		"transport-profile",
		received !== undefined &&
			received.displayName === PROFILE_B.displayName &&
			received.shortDescr === PROFILE_B.shortDescr &&
			received.description === PROFILE_B.description,
		`listContacts() -> ${JSON.stringify(peer)}`,
	);

	// 3. Trust the contact locally: your name + your note only.
	const contactId = peer?.contactId ?? -1;
	const added = runCli([
		"p2p",
		"peers",
		"--add",
		LOCAL_ALIAS,
		"--contact-id",
		String(contactId),
		"--description",
		LOCAL_NOTE,
		"--config",
		configPath,
	]);
	gate("cli-add", added.status === 0 && loadSimplexPeerRegistry(workspace)[LOCAL_ALIAS]?.contactId === contactId, `exit ${added.status}: ${added.stdout.trim()} ${added.stderr.trim()}`);

	// 4. Merge the SimpleX profile into the trusted record.
	const synced = syncSimplexPeers(workspace, contacts, "2026-01-01T00:00:00.000Z");
	const record = loadSimplexPeerRegistry(workspace)[LOCAL_ALIAS];
	registryFile = readFileSync(join(workspace, ".autorag", "p2p", "simplex-peers.json"), "utf8");
	gate(
		"registry-sync",
		synced.updated.includes(LOCAL_ALIAS) &&
			record?.description === LOCAL_NOTE &&
			record?.profile?.displayName === PROFILE_B.displayName &&
			record?.profile?.shortDescr === PROFILE_B.shortDescr &&
			record?.profile?.description === PROFILE_B.description &&
			record?.profileSyncedAt === "2026-01-01T00:00:00.000Z",
		`sync=${JSON.stringify(synced.updated)} record=${JSON.stringify(record)}`,
	);

	// 5. A brand-new SimpleX contact is reported, never trusted or written.
	const before = readFileSync(join(workspace, ".autorag", "p2p", "simplex-peers.json"), "utf8");
	const stranger = syncSimplexPeers(
		workspace,
		[{ contactId: 987_654, localDisplayName: "stranger", profile: { displayName: "Stranger" } }],
		"2026-01-01T00:00:00.000Z",
	);
	const after = readFileSync(join(workspace, ".autorag", "p2p", "simplex-peers.json"), "utf8");
	untrustedDetail = `untrusted=${JSON.stringify(stranger.untrusted)} keys=${JSON.stringify(Object.keys(loadSimplexPeerRegistry(workspace)))}`;
	gate(
		"no-auto-trust",
		stranger.untrusted.some((contact) => contact.contactId === 987_654) &&
			stranger.updated.length === 0 &&
			before === after &&
			!after.includes("Stranger"),
		untrustedDetail,
	);

	// 6. The user surface: `autorag p2p peers` shows the profile and your note.
	const json = runCli(["p2p", "peers", "--json", "--config", configPath]);
	cliJson = json.stdout.trim();
	const plain = runCli(["p2p", "peers", "--config", configPath]);
	cliPlain = plain.stdout.trim();
	const listed = JSON.parse(cliJson) as {
		peers: Array<{ alias: string; contactId: number; description?: string; profile?: SimplexPeerProfile }>;
	};
	const entry = listed.peers[0];
	gate(
		"cli-surface",
		json.status === 0 &&
			plain.status === 0 &&
			entry?.alias === LOCAL_ALIAS &&
			entry?.contactId === contactId &&
			entry?.description === LOCAL_NOTE &&
			entry?.profile?.displayName === PROFILE_B.displayName &&
			cliPlain.includes(PROFILE_B.displayName) &&
			cliPlain.includes(LOCAL_NOTE),
		`json=${cliJson} plain=${JSON.stringify(cliPlain)}`,
	);
} catch (error) {
	gate("harness", false, error instanceof Error ? error.message : String(error));
} finally {
	await a?.close().catch(() => {});
	await b?.close().catch(() => {});
	cleanup.push(`closed simplex-chat transports A/B (ports ${PORT_A}/${PORT_B}); children SIGTERM'd by transport.close()`);
	rmSync(dirA, { recursive: true, force: true });
	rmSync(dirB, { recursive: true, force: true });
	rmSync(workspace, { recursive: true, force: true });
	cleanup.push(`rm -rf ${dirA} ${dirB} ${workspace}`);
	const listeners = spawnSync("bash", ["-lc", `lsof -nP -i TCP:${PORT_A} -i TCP:${PORT_B} 2>/dev/null | wc -l`], {
		encoding: "utf8",
	});
	cleanup.push(`ports ${PORT_A}/${PORT_B} listeners after teardown: ${(listeners.stdout ?? "").trim()} (1 = header only)`);

	const version = spawnSync("simplex-chat", ["--version"], { encoding: "utf8" });
	const payload = {
		ok: gates.every((entry) => entry.pass),
		ranAt: new Date().toISOString(),
		simplexChat: (version.stdout ?? "").trim().split("\n")[0] ?? "",
		gates,
		registryFile,
		cliPlain,
		cliJson,
		untrustedDetail,
		cleanupReceipt: cleanup,
	};
	writeFileSync(EVIDENCE_PATH, `${JSON.stringify(payload, null, 2)}\n`);
	console.log(`\nEvidence: ${EVIDENCE_PATH}`);
	for (const line of cleanup) console.log(`cleanup: ${line}`);
	if (!payload.ok) process.exitCode = 1;
}

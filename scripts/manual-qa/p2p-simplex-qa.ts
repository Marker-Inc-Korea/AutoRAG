#!/usr/bin/env bun
/**
 * Live QA: send/receive round-trip between two local SimpleX profiles
 * through src/p2p/simplex-transport.ts. No phone number or external account
 * needed — both profiles live on this machine.
 */
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { startSimplexChat, type SimplexIncomingMessage } from "../../src/p2p/simplex-transport.ts";

function fail(message: string): never {
	console.error(`FAIL ${message}`);
	process.exit(1);
}

const dirA = mkdtempSync(join(tmpdir(), "simplex-qa-a-"));
const dirB = mkdtempSync(join(tmpdir(), "simplex-qa-b-"));
const payload = `simplex-transport QA ${Date.now()} ${Math.random().toString(36).slice(2)}`;
const started = Date.now();

let a: Awaited<ReturnType<typeof startSimplexChat>> | undefined;
let b: Awaited<ReturnType<typeof startSimplexChat>> | undefined;

try {
	a = await startSimplexChat({ dbPrefix: join(dirA, "a"), displayName: "qa-a", port: 25_901 });
	b = await startSimplexChat({ dbPrefix: join(dirB, "b"), displayName: "qa-b", port: 25_902 });

	// Connect: A creates an invitation, B accepts it.
	const invitation = await a.createInvitation();
	await b.connect(invitation);
	// Wait for the connection to establish on both sides.
	await new Promise((resolve) => setTimeout(resolve, 3000));
	const contactsA = await a.listContacts();
	const contactsB = await b.listContacts();
	if (contactsA.length === 0) fail("A has no contacts after invitation");
	if (contactsB.length === 0) fail("B has no contacts after invitation");

	const received = new Promise<SimplexIncomingMessage>((resolve, reject) => {
		const timeout = setTimeout(() => reject(new Error("timed out waiting for delivery (30s)")), 30_000);
		b!.onMessage((message) => {
			if (message.text === payload) {
				clearTimeout(timeout);
				resolve(message);
			}
		});
	});

	await a.sendMessage(contactsA[0]!.contactId, payload);
	const message = await received;
	const elapsedMs = Date.now() - started;
	if (message.text !== payload) fail("payload mismatch");
	console.log(
		JSON.stringify({
			ok: true,
			profileA: "qa-a",
			profileB: "qa-b",
			payloadBytes: Buffer.byteLength(payload),
			elapsedMs,
			fromContact: message.contactName,
		}),
	);
} catch (error) {
	fail(error instanceof Error ? error.message : String(error));
} finally {
	await a?.close().catch(() => { });
	await b?.close().catch(() => { });
	rmSync(dirA, { recursive: true, force: true });
	rmSync(dirB, { recursive: true, force: true });
}

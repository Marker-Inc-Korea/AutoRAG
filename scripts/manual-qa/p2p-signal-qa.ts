#!/usr/bin/env bun
/**
 * Live QA: send/receive round-trip between two signal-cli accounts (or one
 * account note-to-self) through src/p2p/signal-transport.ts.
 *
 * Usage:
 *   SENDER=+8210AAAAAAAA RECIPIENT=+8210BBBBBBBB bun scripts/manual-qa/p2p-signal-qa.ts
 *   # note-to-self loopback: set only SENDER
 *
 * Both accounts must be registered (autorag p2p register/verify). The script
 * spawns one daemon per distinct data dir, sends a payload SENDER→RECIPIENT,
 * and asserts the recipient daemon receives the exact payload.
 */
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { startSignalDaemon, type SignalIncomingMessage } from "../../src/p2p/signal-transport.ts";

const SENDER = process.env.SENDER;
const RECIPIENT = process.env.RECIPIENT ?? process.env.SENDER;

function fail(message: string): never {
	console.error(`FAIL ${message}`);
	process.exit(1);
}

if (!SENDER || !/^\+[1-9][0-9]{6,14}$/.test(SENDER)) fail("SENDER env (E.164) is required");
if (!RECIPIENT || !/^\+[1-9][0-9]{6,14}$/.test(RECIPIENT)) fail("RECIPIENT env (E.164) is required");

const senderDir = process.env.SENDER_DATA_DIR ?? mkdtempSync(join(tmpdir(), "signal-qa-sender-"));
const recipientDir =
	process.env.RECIPIENT_DATA_DIR ??
	(RECIPIENT === SENDER ? senderDir : mkdtempSync(join(tmpdir(), "signal-qa-recipient-")));

const payload = `signal-transport QA ${Date.now()} ${Math.random().toString(36).slice(2)}`;
const started = Date.now();
let senderTransport: Awaited<ReturnType<typeof startSignalDaemon>> | undefined;
let recipientTransport: Awaited<ReturnType<typeof startSignalDaemon>> | undefined;

try {
	senderTransport = await startSignalDaemon({ account: SENDER, dataDir: senderDir, host: "127.0.0.1", port: 17583 });
	if (RECIPIENT === SENDER) {
		recipientTransport = senderTransport;
	} else {
		recipientTransport = await startSignalDaemon({
			account: RECIPIENT,
			dataDir: recipientDir,
			host: "127.0.0.1",
			port: 17584,
		});
	}

	const received = new Promise<SignalIncomingMessage>((resolve, reject) => {
		const timeout = setTimeout(() => reject(new Error("timed out waiting for delivery (30s)")), 30_000);
		recipientTransport!.onMessage((message) => {
			if (message.message === payload) {
				clearTimeout(timeout);
				resolve(message);
			}
		});
	});

	await senderTransport.sendMessage(RECIPIENT, payload);
	const message = await received;
	const elapsedMs = Date.now() - started;
	if (message.message !== payload) fail("payload mismatch");
	console.log(
		JSON.stringify({
			ok: true,
			sender: SENDER,
			recipient: RECIPIENT,
			loopback: RECIPIENT === SENDER,
			payloadBytes: Buffer.byteLength(payload),
			elapsedMs,
			source: message.source,
			sourceUuid: message.sourceUuid,
		}),
	);
} catch (error) {
	fail(error instanceof Error ? error.message : String(error));
} finally {
	await senderTransport?.close().catch(() => { });
	if (recipientTransport !== undefined && recipientTransport !== senderTransport) {
		await recipientTransport.close().catch(() => { });
	}
	if (process.env.SENDER_DATA_DIR === undefined) rmSync(senderDir, { recursive: true, force: true });
	if (process.env.RECIPIENT_DATA_DIR === undefined && recipientDir !== senderDir) {
		rmSync(recipientDir, { recursive: true, force: true });
	}
}

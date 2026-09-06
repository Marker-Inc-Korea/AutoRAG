import { createHash, generateKeyPairSync } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
	decodePairingCode,
	encodePairingCode,
	generateIdentity,
	IdentityError,
	loadIdentity,
	loadPeerRegistry,
	PairingCodeError,
	type PeerRecord,
	PeerRegistryError,
	registerPeer,
	signRequest,
	verifyRequest,
} from "../../src/p2p/identity.ts";

const workspaces: string[] = [];

function workspace(): string {
	const root = mkdtempSync(join(tmpdir(), "autorag-p2p-identity-"));
	workspaces.push(root);
	return root;
}

function bodyHash(body: string): string {
	return createHash("sha256").update(body).digest("hex");
}

function peerFromIdentity(identity: ReturnType<typeof generateIdentity>, endpoint = "127.0.0.1:9470"): PeerRecord {
	return {
		fingerprint: identity.fingerprint,
		endpoint,
		pubkey: identity.pubkey,
		addedAt: new Date().toISOString(),
	};
}

afterEach(() => {
	for (const root of workspaces.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("p2p peer identity", () => {
	it("generates and round-trips an installation keypair with a private 0600 identity file", () => {
		const root = workspace();
		const generated = generateIdentity(root);
		const loaded = loadIdentity(root);
		const identityPath = join(root, ".autorag", "p2p", "identity.json");

		expect(loaded.fingerprint).toBe(generated.fingerprint);
		expect(loaded.pubkey).toBe(generated.pubkey);
		expect(loaded.privateKey.export({ type: "pkcs8", format: "der" })).toEqual(
			generated.privateKey.export({ type: "pkcs8", format: "der" }),
		);
		expect(statSync(identityPath).mode & 0o777).toBe(0o600);
		expect(statSync(join(root, ".autorag", "p2p")).mode & 0o777).toBe(0o700);
		expect(statSync(join(root, ".autorag", "p2p", "peers.json")).mode & 0o777).toBe(0o600);
		expect(readFileSync(identityPath, "utf8")).not.toContain("BEGIN PRIVATE KEY");
	});

	it("signs and verifies a fixed body while rejecting tampering, wrong keys, skew, and replay", () => {
		const root = workspace();
		const identity = generateIdentity(root);
		const wrongKey = generateKeyPairSync("ed25519");
		const peer = peerFromIdentity(identity);
		const timestamp = Math.floor(Date.now() / 1000);
		const body = bodyHash('{"query":"hello"}');
		const signature = signRequest(identity.privateKey, timestamp, body);

		expect(verifyRequest(peer, timestamp, body, signature)).toBe(true);
		expect(verifyRequest(peer, timestamp, bodyHash('{"query":"tampered"}'), signature)).toBe(false);
		expect(verifyRequest(peer, timestamp, body, signRequest(wrongKey.privateKey, timestamp, body))).toBe(false);
		expect(verifyRequest(peer, timestamp - 301, body, signRequest(identity.privateKey, timestamp - 301, body))).toBe(
			false,
		);
		expect(verifyRequest(peer, timestamp, body, signature)).toBe(false);
	});

	it("persists a per-peer high-water timestamp and rejects an equal timestamp", () => {
		const root = workspace();
		const identity = generateIdentity(root);
		const peer = registerPeer(root, "friend", peerFromIdentity(identity));
		const timestamp = Math.floor(Date.now() / 1000);
		const body = bodyHash("high-water");
		const signature = signRequest(identity.privateKey, timestamp, body);

		expect(verifyRequest(peer, timestamp, body, signature)).toBe(true);
		expect(loadPeerRegistry(root).friend?.highWaterTimestamp).toBe(timestamp);

		const reloadedPeer = loadPeerRegistry(root).friend;
		expect(reloadedPeer).toBeDefined();
		const differentBody = bodyHash("different-body");
		expect(
			verifyRequest(
				reloadedPeer as PeerRecord,
				timestamp,
				differentBody,
				signRequest(identity.privateKey, timestamp, differentBody),
			),
		).toBe(false);
	});

	it("round-trips pairing codes and stores only public peer material", () => {
		const root = workspace();
		const identity = generateIdentity(root);
		const pairingCode = encodePairingCode({ endpoint: "127.0.0.1:9470", pubkey: identity.pubkey, alias: "friend" });
		const decoded = decodePairingCode(pairingCode);
		const peer = registerPeer(root, decoded.alias, {
			endpoint: decoded.endpoint,
			pubkey: decoded.pubkey,
			addedAt: new Date().toISOString(),
		});

		expect(decoded).toEqual({ endpoint: "127.0.0.1:9470", pubkey: identity.pubkey, alias: "friend" });
		expect(peer).toMatchObject({ endpoint: decoded.endpoint, pubkey: decoded.pubkey });
		expect(readFileSync(join(root, ".autorag", "p2p", "peers.json"), "utf8")).not.toContain("PRIVATE KEY");
	});

	it("raises typed errors for malformed pairing codes and peer registries", () => {
		const root = workspace();
		generateIdentity(root);
		writeFileSync(join(root, ".autorag", "p2p", "peers.json"), "{not-json", { mode: 0o600 });

		expect(() => decodePairingCode("not-a-pairing-code")).toThrow(PairingCodeError);
		expect(() => loadPeerRegistry(root)).toThrow(PeerRegistryError);
		expect(() => loadIdentity(join(root, "missing"))).toThrow(IdentityError);
	});

	it("refuses registry updates while another process holds the update lock", () => {
		const root = workspace();
		const identity = generateIdentity(root);
		const peersPath = join(root, ".autorag", "p2p", "peers.json");
		const lockPath = `${peersPath}.lock`;
		const before = readFileSync(peersPath, "utf8");
		writeFileSync(lockPath, "held", { mode: 0o600 });

		expect(() => registerPeer(root, "friend", peerFromIdentity(identity))).toThrow(PeerRegistryError);
		expect(readFileSync(peersPath, "utf8")).toBe(before);
	});
});

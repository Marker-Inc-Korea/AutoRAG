import {
	createHash,
	createPrivateKey,
	createPublicKey,
	generateKeyPairSync,
	type KeyObject,
	randomUUID,
	sign,
	timingSafeEqual,
	verify,
} from "node:crypto";
import {
	chmodSync,
	closeSync,
	mkdirSync,
	openSync,
	readFileSync,
	renameSync,
	unlinkSync,
	writeFileSync,
} from "node:fs";
import { resolve } from "node:path";

const IDENTITY_DIRECTORY = ".autorag/p2p";
const IDENTITY_FILENAME = "identity.json";
const PEERS_FILENAME = "peers.json";
const FILE_MODE = 0o600;
const DIRECTORY_MODE = 0o700;
const MAX_SEEN_REQUESTS = 1024;
const MAX_CLOCK_SKEW_SECONDS = 300;
const BODY_HASH_PATTERN = /^[a-f0-9]{64}$/i;
const FINGERPRINT_PATTERN = /^[a-f0-9]{64}$/i;
const PEER_ALIAS_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$/;
const RESERVED_ALIASES = new Set(["__proto__", "constructor", "prototype"]);

const seenRequests = new Map<string, true>();
const registryContexts = new WeakMap<object, RegistryContext>();

export class IdentityError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "IdentityError";
	}
}

export class PairingCodeError extends IdentityError {
	constructor(message: string) {
		super(`Invalid pairing code: ${message}`);
		this.name = "PairingCodeError";
	}
}

export class PeerRegistryError extends IdentityError {
	constructor(message: string) {
		super(`Invalid peer registry: ${message}`);
		this.name = "PeerRegistryError";
	}
}

export interface PairingCode {
	readonly endpoint: string;
	readonly pubkey: string;
	readonly alias: string;
}

export interface PeerRecord {
	readonly fingerprint: string;
	readonly endpoint: string;
	readonly pubkey: string;
	readonly addedAt: string;
	readonly highWaterTimestamp?: number;
}

export interface PeerInput {
	readonly fingerprint?: string;
	readonly endpoint: string;
	readonly pubkey: string;
	readonly addedAt?: string;
	readonly highWaterTimestamp?: number;
}

export type PeerRegistry = Record<string, PeerRecord>;

export interface PeerIdentity {
	readonly privateKey: KeyObject;
	readonly publicKey: KeyObject;
	readonly fingerprint: string;
	readonly pubkey: string;
	readonly pairingCode: (endpoint: string, alias: string) => string;
}

interface PersistedIdentity {
	readonly version: 1;
	readonly privateKey: string;
	readonly publicKey: string;
	readonly fingerprint: string;
}

interface RegistryContext {
	readonly path: string;
	readonly alias: string;
}

interface RegistrySnapshot {
	readonly raw: string | null;
	readonly registry: PeerRegistry;
}

export function identityPath(workspace: string): string {
	return resolve(workspace, IDENTITY_DIRECTORY, IDENTITY_FILENAME);
}

export const getIdentityPath = identityPath;

export function peersPath(workspace: string): string {
	return resolve(workspace, IDENTITY_DIRECTORY, PEERS_FILENAME);
}

export const getPeersPath = peersPath;

export function sha256Body(body: string | Uint8Array): string {
	return createHash("sha256").update(body).digest("hex");
}

export function publicKeyFingerprint(publicKey: KeyObject | string): string {
	const key = typeof publicKey === "string" ? publicKeyFromSpki(publicKey) : publicKey;
	if (key.type !== "public" || key.asymmetricKeyType !== "ed25519") {
		throw new IdentityError("Peer public key must be an Ed25519 public key.");
	}
	return createHash("sha256")
		.update(key.export({ type: "spki", format: "der" }))
		.digest("hex");
}

export function generateIdentity(workspace: string): PeerIdentity {
	const root = resolve(workspace);
	const path = identityPath(root);
	ensureDirectory(path);
	const { privateKey, publicKey } = generateKeyPairSync("ed25519");
	const pubkey = publicKey.export({ type: "spki", format: "der" }).toString("base64");
	const identity: PersistedIdentity = {
		version: 1,
		privateKey: privateKey.export({ type: "pkcs8", format: "der" }).toString("base64"),
		publicKey: pubkey,
		fingerprint: publicKeyFingerprint(publicKey),
	};
	writeJsonAtomically(path, identity);
	ensureEmptyPeerRegistry(root);
	return makeIdentity(privateKey, publicKey, identity.fingerprint);
}

export function loadIdentity(workspace: string): PeerIdentity {
	const path = identityPath(resolve(workspace));
	let persisted: unknown;
	try {
		persisted = JSON.parse(readFileSync(path, "utf8")) as unknown;
	} catch (error) {
		if (isMissingFile(error)) throw new IdentityError("Installation identity does not exist.");
		throw new IdentityError("Installation identity is not valid JSON.");
	}
	if (!isRecord(persisted)) throw new IdentityError("Installation identity must be an object.");
	if (persisted.version !== 1) throw new IdentityError("Unsupported installation identity version.");
	let privateKey: KeyObject;
	let publicKey: KeyObject;
	try {
		privateKey = privateKeyFromPkcs8(readString(persisted.privateKey, "private key"));
		publicKey = publicKeyFromSpki(readString(persisted.publicKey, "public key"));
	} catch {
		throw new IdentityError("Installation identity keys are not valid Ed25519 keys.");
	}
	const fingerprint = readFingerprint(persisted.fingerprint, "identity fingerprint");
	const derivedPublicKey = createPublicKey(privateKey);
	const derivedPubkey = derivedPublicKey.export({ type: "spki", format: "der" }).toString("base64");
	if (
		!buffersEqual(
			Buffer.from(derivedPubkey),
			Buffer.from(publicKey.export({ type: "spki", format: "der" }).toString("base64")),
		)
	) {
		throw new IdentityError("Installation identity public and private keys do not match.");
	}
	if (!fingerprintsEqual(fingerprint, publicKeyFingerprint(publicKey))) {
		throw new IdentityError("Installation identity fingerprint does not match its public key.");
	}
	try {
		chmodSync(identityPath(resolve(workspace)), FILE_MODE);
	} catch {
		throw new IdentityError("Installation identity permissions could not be secured.");
	}
	return makeIdentity(privateKey, publicKey, fingerprint);
}

export function loadOrCreateIdentity(workspace: string): PeerIdentity {
	try {
		return loadIdentity(workspace);
	} catch (error) {
		if (error instanceof IdentityError && error.message === "Installation identity does not exist.") {
			return generateIdentity(workspace);
		}
		throw error;
	}
}

export const getOrCreateIdentity = loadOrCreateIdentity;
export const loadOrCreatePeerIdentity = loadOrCreateIdentity;
export const loadPeerIdentity = loadIdentity;

export function encodePairingCode(code: PairingCode): string;
export function encodePairingCode(endpoint: string, pubkey: string, alias: string): string;
export function encodePairingCode(
	codeOrEndpoint: PairingCode | string,
	pubkeyArgument?: string,
	aliasArgument?: string,
): string {
	const code =
		typeof codeOrEndpoint === "string"
			? { endpoint: codeOrEndpoint, pubkey: pubkeyArgument, alias: aliasArgument }
			: codeOrEndpoint;
	const endpoint = validateEndpoint(code.endpoint, PairingCodeError);
	const alias = validateAlias(code.alias, PairingCodeError);
	const pubkey = validatePubkey(code.pubkey, PairingCodeError);
	return Buffer.from(JSON.stringify({ endpoint, pubkey, alias }), "utf8").toString("base64url");
}

export function decodePairingCode(encoded: string): PairingCode {
	if (typeof encoded !== "string" || encoded.length === 0 || !/^[A-Za-z0-9_-]+$/.test(encoded)) {
		throw new PairingCodeError("expected a base64url value.");
	}
	let decoded: string;
	try {
		const bytes = Buffer.from(encoded, "base64url");
		if (bytes.length === 0 || bytes.toString("base64url") !== encoded) throw new Error("non-canonical encoding");
		decoded = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
	} catch {
		throw new PairingCodeError("could not decode base64url JSON.");
	}
	let value: unknown;
	try {
		value = JSON.parse(decoded) as unknown;
	} catch {
		throw new PairingCodeError("payload is not valid JSON.");
	}
	if (!isRecord(value)) throw new PairingCodeError("payload must be an object.");
	return {
		endpoint: validateEndpoint(value.endpoint, PairingCodeError),
		pubkey: validatePubkey(value.pubkey, PairingCodeError),
		alias: validateAlias(value.alias, PairingCodeError),
	};
}

export function createPairingCode(identity: PeerIdentity, endpoint: string, alias: string): string {
	return encodePairingCode({ endpoint, pubkey: identity.pubkey, alias });
}

export function loadPeerRegistry(workspace: string): PeerRegistry {
	const root = resolve(workspace);
	const path = peersPath(root);
	const snapshot = readRegistrySnapshot(path);
	for (const [alias, peer] of Object.entries(snapshot.registry)) attachRegistryContext(peer, { path, alias });
	return snapshot.registry;
}

export function registerPeer(workspace: string, alias: string, peer: PeerInput): PeerRecord {
	const root = resolve(workspace);
	const path = peersPath(root);
	const normalizedAlias = validateAlias(alias, PeerRegistryError);
	const normalizedPeer = normalizePeer(peer, PeerRegistryError);
	return withRegistryLock(path, () => {
		const snapshot = readRegistrySnapshot(path);
		const samePeer = Object.values(snapshot.registry).filter((item) =>
			fingerprintsEqual(item.fingerprint, normalizedPeer.fingerprint),
		);
		const knownHighWater = samePeer.reduce<number | undefined>(
			(highWater, item) =>
				item.highWaterTimestamp === undefined
					? highWater
					: Math.max(highWater ?? item.highWaterTimestamp, item.highWaterTimestamp),
			undefined,
		);
		const highWaterTimestamp = knownHighWater ?? normalizedPeer.highWaterTimestamp;
		const savedPeer: PeerRecord = {
			...normalizedPeer,
			...(highWaterTimestamp === undefined ? {} : { highWaterTimestamp }),
		};
		const next: PeerRegistry = { ...snapshot.registry, [normalizedAlias]: savedPeer };
		writeRegistryIfUnchanged(path, snapshot.raw, next);
		attachRegistryContext(savedPeer, { path, alias: normalizedAlias });
		return savedPeer;
	});
}

export const addPeer = registerPeer;
export const loadPeers = loadPeerRegistry;
export const readPeerRegistry = loadPeerRegistry;

export function removePeer(workspace: string, alias: string): boolean {
	const path = peersPath(resolve(workspace));
	const normalizedAlias = validateAlias(alias, PeerRegistryError);
	return withRegistryLock(path, () => {
		const snapshot = readRegistrySnapshot(path);
		if (!snapshot.registry[normalizedAlias]) return false;
		const next = { ...snapshot.registry };
		delete next[normalizedAlias];
		writeRegistryIfUnchanged(path, snapshot.raw, next);
		return true;
	});
}

export function signRequest(privateKey: KeyObject, timestamp: number, bodySha256: string): string {
	validateTimestamp(timestamp);
	const normalizedBodyHash = validateBodyHash(bodySha256);
	if (privateKey.type !== "private" || privateKey.asymmetricKeyType !== "ed25519") {
		throw new IdentityError("Request signing requires an Ed25519 private key.");
	}
	return sign(null, requestSigningPayload(timestamp, normalizedBodyHash), privateKey).toString("base64url");
}

export function verifyRequest(
	peer: PeerRecord,
	timestamp: number,
	bodySha256: string,
	signature: string | Uint8Array,
): boolean {
	try {
		validateTimestamp(timestamp);
		const normalizedBodyHash = validateBodyHash(bodySha256);
		const normalizedPeer = normalizePeerForVerification(peer);
		if (Math.abs(Math.floor(Date.now() / 1000) - timestamp) > MAX_CLOCK_SKEW_SECONDS) return false;
		const publicKey = publicKeyFromSpki(normalizedPeer.pubkey);
		if (!fingerprintsEqual(normalizedPeer.fingerprint, publicKeyFingerprint(publicKey))) return false;
		const signatureBytes = decodeSignature(signature);
		if (!verify(null, requestSigningPayload(timestamp, normalizedBodyHash), publicKey, signatureBytes)) return false;
		const signatureHash = createHash("sha256").update(signatureBytes).digest("hex");
		const replayKey = `${normalizedPeer.fingerprint}:${timestamp}:${signatureHash}`;
		if (seenRequests.has(replayKey)) return false;
		const context = resolveRegistryContext(peer, normalizedPeer.fingerprint);
		if (!persistHighWaterTimestamp(peer, normalizedPeer, timestamp, context)) return false;
		rememberRequest(replayKey);
		return true;
	} catch {
		return false;
	}
}

function makeIdentity(privateKey: KeyObject, publicKey: KeyObject, fingerprint: string): PeerIdentity {
	const pubkey = publicKey.export({ type: "spki", format: "der" }).toString("base64");
	return {
		privateKey,
		publicKey,
		fingerprint,
		pubkey,
		pairingCode: (endpoint, alias) => encodePairingCode({ endpoint, pubkey, alias }),
	};
}

function requestSigningPayload(timestamp: number, bodySha256: string): Buffer {
	return Buffer.from(`${timestamp}:${bodySha256}`, "utf8");
}

function validateTimestamp(timestamp: number): void {
	if (!Number.isSafeInteger(timestamp) || timestamp < 0)
		throw new IdentityError("Request timestamp must be a non-negative integer.");
}

function validateBodyHash(value: string): string {
	if (typeof value !== "string" || !BODY_HASH_PATTERN.test(value))
		throw new IdentityError("Request body hash must be a SHA-256 hex digest.");
	return value.toLowerCase();
}

function normalizePeerForVerification(peer: PeerRecord): PeerRecord {
	if (!isRecord(peer)) throw new IdentityError("Peer record must be an object.");
	const fingerprint = readFingerprint(peer.fingerprint, "peer fingerprint");
	const pubkey = validatePubkey(peer.pubkey, IdentityError);
	const highWaterTimestamp = peer.highWaterTimestamp;
	if (highWaterTimestamp !== undefined && (!Number.isSafeInteger(highWaterTimestamp) || highWaterTimestamp < 0)) {
		throw new IdentityError("Peer high-water timestamp must be a non-negative integer.");
	}
	return {
		fingerprint,
		endpoint: typeof peer.endpoint === "string" ? peer.endpoint : "",
		pubkey,
		addedAt: typeof peer.addedAt === "string" ? peer.addedAt : "",
		...(highWaterTimestamp === undefined ? {} : { highWaterTimestamp }),
	};
}

function normalizePeer(peer: PeerInput, ErrorType: typeof IdentityError): PeerRecord {
	if (!isRecord(peer)) throw new ErrorType("peer must be an object.");
	const pubkey = validatePubkey(peer.pubkey, ErrorType);
	const fingerprint =
		peer.fingerprint === undefined
			? publicKeyFingerprint(pubkey)
			: readFingerprint(peer.fingerprint, "peer fingerprint", ErrorType);
	const endpoint = validateEndpoint(peer.endpoint, ErrorType);
	const addedAt = peer.addedAt ?? new Date().toISOString();
	if (typeof addedAt !== "string" || !Number.isFinite(Date.parse(addedAt))) {
		throw new ErrorType("peer addedAt must be a valid date string.");
	}
	if (!fingerprintsEqual(fingerprint, publicKeyFingerprint(pubkey))) {
		throw new ErrorType("peer fingerprint does not match its public key.");
	}
	const highWaterTimestamp = peer.highWaterTimestamp;
	if (highWaterTimestamp !== undefined && (!Number.isSafeInteger(highWaterTimestamp) || highWaterTimestamp < 0)) {
		throw new ErrorType("peer high-water timestamp must be a non-negative integer.");
	}
	return {
		fingerprint,
		endpoint,
		pubkey,
		addedAt,
		...(highWaterTimestamp === undefined ? {} : { highWaterTimestamp }),
	};
}

function readFingerprint(value: unknown, label: string, ErrorType: typeof IdentityError = IdentityError): string {
	if (typeof value !== "string" || !FINGERPRINT_PATTERN.test(value))
		throw new ErrorType(`${label} must be a SHA-256 hex digest.`);
	return value.toLowerCase();
}

function validateEndpoint(value: unknown, ErrorType: typeof IdentityError): string {
	if (typeof value !== "string") throw new ErrorType("endpoint must be a host:port string.");
	const endpoint = value.trim();
	if (endpoint.length === 0 || /[\s/\\\u0000-\u001f\u007f]/.test(endpoint))
		throw new ErrorType("endpoint must be a host:port string.");
	let host: string;
	let portText: string;
	if (endpoint.startsWith("[")) {
		const close = endpoint.indexOf("]");
		if (close < 2 || endpoint[close + 1] !== ":") throw new ErrorType("endpoint must be a host:port string.");
		host = endpoint.slice(1, close);
		portText = endpoint.slice(close + 2);
	} else {
		const separator = endpoint.lastIndexOf(":");
		if (separator <= 0 || endpoint.indexOf(":") !== separator)
			throw new ErrorType("endpoint must be a host:port string.");
		host = endpoint.slice(0, separator);
		portText = endpoint.slice(separator + 1);
	}
	const port = Number(portText);
	if (host.length === 0 || !Number.isInteger(port) || port < 1 || port > 65535) {
		throw new ErrorType("endpoint must be a host:port string.");
	}
	return endpoint;
}

function validateAlias(value: unknown, ErrorType: typeof IdentityError): string {
	if (typeof value !== "string" || !PEER_ALIAS_PATTERN.test(value) || RESERVED_ALIASES.has(value.toLowerCase()))
		throw new ErrorType("peer alias has invalid characters.");
	return value;
}

function validatePubkey(value: unknown, ErrorType: typeof IdentityError): string {
	if (typeof value !== "string") throw new ErrorType("peer public key must be base64 SPKI.");
	try {
		const key = publicKeyFromSpki(value);
		const canonical = key.export({ type: "spki", format: "der" }).toString("base64");
		if (canonical !== value) throw new Error("non-canonical key encoding");
		return value;
	} catch {
		throw new ErrorType("peer public key must be base64 Ed25519 SPKI.");
	}
}

function publicKeyFromSpki(value: string): KeyObject {
	if (!/^[A-Za-z0-9+/]+={0,2}$/.test(value) || value.length % 4 !== 0)
		throw new IdentityError("Public key is not valid base64.");
	const key = createPublicKey({ key: Buffer.from(value, "base64"), format: "der", type: "spki" });
	if (key.asymmetricKeyType !== "ed25519") throw new IdentityError("Public key must be Ed25519.");
	return key;
}

function privateKeyFromPkcs8(value: string): KeyObject {
	if (!/^[A-Za-z0-9+/]+={0,2}$/.test(value) || value.length % 4 !== 0)
		throw new IdentityError("Private key is not valid base64.");
	const key = createPrivateKey({ key: Buffer.from(value, "base64"), format: "der", type: "pkcs8" });
	if (key.asymmetricKeyType !== "ed25519") throw new IdentityError("Installation private key must be Ed25519.");
	return key;
}

function decodeSignature(value: string | Uint8Array): Buffer {
	if (typeof value !== "string") {
		const bytes = Buffer.from(value);
		if (bytes.length !== 64) throw new IdentityError("Request signature must be 64 bytes.");
		return bytes;
	}
	if (!/^[A-Za-z0-9+/_-]+={0,2}$/.test(value)) throw new IdentityError("Request signature is not base64.");
	const bytes = Buffer.from(value, "base64");
	if (bytes.length !== 64) {
		const urlBytes = Buffer.from(value, "base64url");
		if (urlBytes.length !== 64) throw new IdentityError("Request signature must be 64 bytes.");
		return urlBytes;
	}
	return bytes;
}

function readRegistrySnapshot(path: string): RegistrySnapshot {
	let raw: string;
	try {
		raw = readFileSync(path, "utf8");
	} catch (error) {
		if (isMissingFile(error)) return { raw: null, registry: {} };
		throw new PeerRegistryError("could not be read.");
	}
	let value: unknown;
	try {
		value = JSON.parse(raw) as unknown;
	} catch {
		throw new PeerRegistryError("JSON is malformed.");
	}
	if (!isRecord(value) || Array.isArray(value)) throw new PeerRegistryError("root must be an object.");
	const registry: PeerRegistry = {};
	for (const [alias, peer] of Object.entries(value)) {
		try {
			validateAlias(alias, PeerRegistryError);
			registry[alias] = normalizePeer(peer as PeerRecord, PeerRegistryError);
		} catch (error) {
			if (error instanceof PeerRegistryError) throw error;
			throw new PeerRegistryError("contains an invalid peer.");
		}
	}
	return { raw, registry };
}

function persistHighWaterTimestamp(
	peer: PeerRecord,
	normalizedPeer: PeerRecord,
	timestamp: number,
	context: RegistryContext | undefined,
): boolean {
	if (context === undefined) {
		if (normalizedPeer.highWaterTimestamp !== undefined && timestamp <= normalizedPeer.highWaterTimestamp)
			return false;
		try {
			(peer as { highWaterTimestamp?: number }).highWaterTimestamp = timestamp;
		} catch {
			return false;
		}
		return true;
	}
	const path = context.path;
	try {
		return withRegistryLock(path, () => {
			const snapshot = readRegistrySnapshot(path);
			const current = snapshot.registry[context.alias];
			if (!current || !fingerprintsEqual(current.fingerprint, normalizedPeer.fingerprint)) return false;
			if (current.highWaterTimestamp !== undefined && timestamp <= current.highWaterTimestamp) return false;
			const updated: PeerRecord = { ...current, highWaterTimestamp: timestamp };
			const next: PeerRegistry = { ...snapshot.registry, [context.alias]: updated };
			writeRegistryIfUnchanged(path, snapshot.raw, next);
			(peer as { highWaterTimestamp?: number }).highWaterTimestamp = timestamp;
			attachRegistryContext(peer, context);
			return true;
		});
	} catch {
		return false;
	}
}

function resolveRegistryContext(peer: PeerRecord, _fingerprint: string): RegistryContext | undefined {
	return registryContexts.get(peer);
}

function attachRegistryContext(peer: PeerRecord, context: RegistryContext): void {
	registryContexts.set(peer, context);
}

function rememberRequest(key: string): void {
	seenRequests.set(key, true);
	while (seenRequests.size > MAX_SEEN_REQUESTS) {
		const oldest = seenRequests.keys().next().value;
		if (oldest === undefined) break;
		seenRequests.delete(oldest);
	}
}

function ensureDirectory(filePath: string): void {
	try {
		mkdirSync(resolve(filePath, ".."), { recursive: true, mode: DIRECTORY_MODE });
		chmodSync(resolve(filePath, ".."), DIRECTORY_MODE);
	} catch {
		throw new IdentityError("P2P identity directory could not be secured.");
	}
}

function ensureEmptyPeerRegistry(workspace: string): void {
	const path = peersPath(workspace);
	try {
		readFileSync(path);
		chmodSync(path, FILE_MODE);
	} catch (error) {
		if (!isMissingFile(error)) throw new PeerRegistryError("could not be initialized.");
		writeJsonAtomically(path, {});
	}
}

function writeRegistryIfUnchanged(path: string, expectedRaw: string | null, registry: PeerRegistry): void {
	let actualRaw: string | null;
	try {
		actualRaw = readFileSync(path, "utf8");
	} catch (error) {
		if (!isMissingFile(error)) throw new PeerRegistryError("could not be checked before writing.");
		actualRaw = null;
	}
	if (actualRaw !== expectedRaw)
		throw new PeerRegistryError("changed while it was being updated; no changes were written.");
	writeJsonAtomically(path, registry);
	try {
		actualRaw = readFileSync(path, "utf8");
	} catch {
		throw new PeerRegistryError("could not be verified after writing.");
	}
	const expectedAfter = `${JSON.stringify(registry, null, 2)}\n`;
	if (actualRaw !== expectedAfter)
		throw new PeerRegistryError("changed during the update; the write was not trusted.");
}

function writeJsonAtomically(path: string, value: unknown): void {
	ensureDirectory(path);
	const temporaryPath = `${path}.${randomUUID()}.tmp`;
	try {
		writeFileSync(temporaryPath, `${JSON.stringify(value, null, 2)}\n`, { encoding: "utf8", mode: FILE_MODE });
		chmodSync(temporaryPath, FILE_MODE);
		renameSync(temporaryPath, path);
		chmodSync(path, FILE_MODE);
	} catch {
		try {
			unlinkSync(temporaryPath);
		} catch {
			// The original write error is the useful failure for callers.
		}
		throw new IdentityError("P2P identity state could not be written securely.");
	}
}

function withRegistryLock<T>(path: string, operation: () => T): T {
	ensureDirectory(path);
	const lockPath = `${path}.lock`;
	let descriptor: number;
	try {
		descriptor = openSync(lockPath, "wx", FILE_MODE);
	} catch {
		throw new PeerRegistryError("is already being updated by another process.");
	}
	try {
		return operation();
	} finally {
		try {
			closeSync(descriptor);
		} finally {
			try {
				unlinkSync(lockPath);
			} catch {
				// Keep the operation result/error; a later operation will report the lock.
			}
		}
	}
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}

function readString(value: unknown, label: string): string {
	if (typeof value !== "string" || value.length === 0) throw new IdentityError(`Installation ${label} is missing.`);
	return value;
}

function isMissingFile(error: unknown): boolean {
	return isRecord(error) && error.code === "ENOENT";
}

function fingerprintsEqual(left: string, right: string): boolean {
	const leftBytes = Buffer.from(left.toLowerCase(), "utf8");
	const rightBytes = Buffer.from(right.toLowerCase(), "utf8");
	if (leftBytes.length !== rightBytes.length) return false;
	return timingSafeEqual(leftBytes, rightBytes);
}

function buffersEqual(left: Uint8Array, right: Uint8Array): boolean {
	if (left.length !== right.length) return false;
	return timingSafeEqual(Buffer.from(left), Buffer.from(right));
}

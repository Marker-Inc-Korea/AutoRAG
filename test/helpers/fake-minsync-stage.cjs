"use strict";

// Shared, test-only MinSync `stage` protocol simulator.
//
// It speaks the narrow host protocol of MinSync v0.5.0
// (src/incremental/protocol.rs) over stdin/stdout and reproduces the resume
// planner (src/incremental/plan.rs), the durable journal (journal.rs), the
// committed snapshot (snapshot.rs) and the PID lock (state.rs) on top of the
// plain filesystem. Nothing here talks to a real `minsync`, the network, or a
// vector store: a test writes the output files itself and this process only
// validates and records them.
//
// The generated `minsync` stand-ins (`test/helpers/fake-minsync.ts` and
// `test/setup/fake-minsync-on-path.ts`) route `stage` invocations here and
// leave every other subcommand alone. The module is CommonJS so a no-extension
// shebang entry can `require` it, and it stays importable from ESM through
// `createRequire` / the default export.
//
// State layout under `--state-dir`, one stage and one corpus per directory:
//   lock           held for the whole run, released by the OS on any exit;
//                  the file text is the owning PID, so a dead holder's lock
//                  is stale and taken over.
//   txn.json       run marker while a run is open.
//   progress.jsonl header line + one record per committed/failed/deleted item.
//   snapshot.json  the committed state, rewritten once by `finish`.
//
// Exit codes match MinSync: 0 after `finish`, 3 when another run holds the
// lock, 1 when input ends mid-run (the run is left resumable).

const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const readline = require("node:readline");

const JOURNAL_FILE = "progress.jsonl";
const SNAPSHOT_FILE = "snapshot.json";
const TXN_FILE = "txn.json";
const LOCK_FILE = "lock";
const RUN_FORMAT_VERSION = 2;
const SNAPSHOT_FORMAT = 1;
const MAX_ERROR_CHARS = 2000;
const MAX_KEY_BYTES = 4096;

const LOCK_WAIT_POLL_MS = 20;

class StageError extends Error {
	constructor(code, message) {
		super(message);
		this.name = "StageError";
		this.code = code;
	}
}

function stageError(code, message) {
	return new StageError(code, message);
}

function invalidRequest(message) {
	return stageError("invalid_request", message);
}

function nowIso() {
	return new Date().toISOString();
}

function newRunToken() {
	return crypto.randomUUID().replace(/-/g, "");
}

function byteCompare(a, b) {
	return Buffer.compare(Buffer.from(a, "utf8"), Buffer.from(b, "utf8"));
}

function sha256(data) {
	return `sha256:${crypto.createHash("sha256").update(data).digest("hex")}`;
}

function mtimeNs(stat) {
	if (typeof stat.mtimeNs === "bigint") return stat.mtimeNs;
	return BigInt(Math.round(stat.mtimeMs * 1e6));
}

function atomicWrite(filePath, content) {
	const tmp = `${filePath}.${process.pid}.${crypto.randomBytes(4).toString("hex")}.tmp`;
	fs.writeFileSync(tmp, content);
	fs.renameSync(tmp, filePath);
}

// --- item identity and path hygiene (src/incremental/item.rs) ---------------

function relativeProblem(value) {
	if (typeof value !== "string") return "is not text";
	if (value.length === 0) return "is empty";
	if (Buffer.byteLength(value, "utf8") > MAX_KEY_BYTES) return "is longer than 4096 bytes";
	if (/[\u0000-\u001f\u007f]/.test(value)) return "contains a control character";
	if (value.includes("\\")) return "contains a backslash";
	if (value.startsWith("/")) return "is absolute";
	if (value.split("/").some((part) => part === "" || part === "." || part === "..")) {
		return "has an empty, `.` or `..` component";
	}
	return null;
}

function validateItemKey(key) {
	const problem = relativeProblem(key);
	if (problem) throw stageError("invalid_item_key", `item key ${JSON.stringify(key)} ${problem}`);
}

function validateOutputPath(value) {
	let problem = relativeProblem(value);
	// A string that looks relative can still be a drive prefix on Windows.
	if (!problem && /^[A-Za-z]:/.test(value)) problem = "is not a plain relative path";
	if (problem) throw stageError("invalid_output_path", `output path ${JSON.stringify(value)} ${problem}`);
}

// --- snapshot (src/incremental/snapshot.rs) --------------------------------

function emptySnapshot(stage, sourceId) {
	return {
		format: SNAPSHOT_FORMAT,
		source_id: sourceId,
		stage,
		updated_at: nowIso(),
		items: {},
		failed: {},
	};
}

function loadSnapshot(stateDir) {
	const snapshotPath = path.join(stateDir, SNAPSHOT_FILE);
	let bytes;
	try {
		bytes = fs.readFileSync(snapshotPath);
	} catch (error) {
		if (error.code === "ENOENT") return null;
		throw error;
	}
	let snapshot;
	try {
		snapshot = JSON.parse(bytes.toString("utf8"));
	} catch (error) {
		throw stageError(
			"corrupt_state",
			`${snapshotPath} is unusable (${error.message}); move it aside to rebuild this stage from scratch`,
		);
	}
	if (typeof snapshot.format === "number" && snapshot.format > SNAPSHOT_FORMAT) {
		throw stageError(
			"corrupt_state",
			`${snapshotPath} is unusable (format ${snapshot.format} is newer than supported format ${SNAPSHOT_FORMAT})`,
		);
	}
	return snapshot;
}

/** Identity of the committed state, for a run header's `baseline`. */
function snapshotHash(snapshot) {
	const hasher = crypto.createHash("sha256");
	for (const key of Object.keys(snapshot.items).sort(byteCompare)) {
		const item = snapshot.items[key];
		for (const part of [
			"item",
			key,
			item.version,
			item.fingerprint,
			item.output.path,
			String(item.output.size),
			item.output.digest,
			String(item.output.mtime_ns),
		]) {
			hasher.update(part);
			hasher.update(Buffer.from([0]));
		}
		hasher.update("\n");
	}
	for (const key of Object.keys(snapshot.failed).sort(byteCompare)) {
		const failure = snapshot.failed[key];
		const version = failure.version === null || failure.version === undefined ? "-" : `+${failure.version}`;
		for (const part of ["failed", key, version, failure.error]) {
			hasher.update(part);
			hasher.update(Buffer.from([0]));
		}
		hasher.update("\n");
	}
	return `sha256:${hasher.digest("hex")}`;
}

// --- journal / ledger (src/incremental/journal.rs) -------------------------

function newLedger(header) {
	return { header, done: new Map(), failed: new Map() };
}

function recordToJson(record) {
	const out = { item: record.item };
	if (record.version !== null && record.version !== undefined) out.version = record.version;
	out.outcome = record.outcome;
	if (record.output !== null && record.output !== undefined) out.output = record.output;
	if (record.error !== null && record.error !== undefined) out.error = record.error;
	return out;
}

function parseRecord(line) {
	const raw = JSON.parse(line);
	if (raw === null || typeof raw !== "object" || typeof raw.item !== "string") {
		throw new Error("malformed record");
	}
	const version = raw.version === undefined ? null : raw.version;
	const outcome = raw.outcome ?? (version !== null ? "committed" : "deleted");
	if (outcome === "committed" && (version === null || version === undefined)) {
		throw new Error(`committed record for ${JSON.stringify(raw.item)} has no version`);
	}
	return {
		item: raw.item,
		version,
		outcome,
		output: raw.output ?? null,
		error: raw.error ?? null,
	};
}

function applyRecord(ledger, record) {
	if (record.outcome === "committed" || record.outcome === "deleted") {
		ledger.failed.delete(record.item);
		ledger.done.set(record.item, record);
	} else {
		ledger.failed.set(record.item, record);
	}
}

/** Every held record, completions first, each group in item order. */
function ledgerRecords(ledger) {
	return [...ledger.done.values(), ...ledger.failed.values()];
}

function ledgerCommittedItems(ledger) {
	const items = [];
	for (const record of ledger.done.values()) if (record.outcome === "committed") items.push(record.item);
	return items;
}

/**
 * Read the interrupted run's journal. A torn final line (kill mid-append) is
 * dropped; any other malformed line, a torn header, or an unreadable file
 * discards the whole ledger — never "work completed".
 */
function loadLedger(stateDir) {
	let text;
	try {
		text = fs.readFileSync(path.join(stateDir, JOURNAL_FILE), "utf8");
	} catch {
		return null;
	}
	if (text.length === 0) return null;
	const headerEnd = text.indexOf("\n");
	if (headerEnd < 0) return null;
	let header;
	try {
		header = JSON.parse(text.slice(0, headerEnd));
	} catch {
		return null;
	}
	if (header === null || typeof header !== "object") return null;
	const ledger = newLedger(header);
	let rest = text.slice(headerEnd + 1);
	if (rest.length > 0 && !rest.endsWith("\n")) rest = rest.slice(0, rest.lastIndexOf("\n") + 1);
	for (const line of rest.split("\n")) {
		if (line.length === 0) continue;
		let record;
		try {
			record = parseRecord(line);
		} catch {
			return null;
		}
		applyRecord(ledger, record);
	}
	return ledger;
}

function ledgerResumes(ledger, expected) {
	const header = ledger.header;
	return (
		header.source_id === expected.source_id &&
		header.stage === expected.stage &&
		header.full === expected.full &&
		header.baseline === expected.baseline &&
		header.fingerprint === expected.fingerprint
	);
}

function journalContent(header, carried) {
	let content = `${JSON.stringify(header)}\n`;
	if (carried) {
		for (const record of ledgerRecords(carried)) content += `${JSON.stringify(recordToJson(record))}\n`;
	}
	return content;
}

function sourceHash(items) {
	const hasher = crypto.createHash("sha256");
	for (const key of [...items.keys()].sort(byteCompare)) {
		hasher.update(key);
		hasher.update(Buffer.from([0]));
		hasher.update(items.get(key));
		hasher.update(Buffer.from([0]));
	}
	return `sha256:${hasher.digest("hex")}`;
}

// --- resume planner (src/incremental/plan.rs) ------------------------------

/** No output exists for `key` (never committed, or deleted). */
function completion(snapshot, ledger, key) {
	const record = ledger.done.get(key);
	if (record) {
		if (record.outcome !== "committed") return null;
		return {
			version: record.version ?? "",
			fingerprint: ledger.header.fingerprint,
			output: record.output ?? null,
			from_ledger: true,
		};
	}
	const item = snapshot.items[key];
	if (!item) return null;
	return { version: item.version, fingerprint: item.fingerprint, output: item.output ?? null, from_ledger: false };
}

function failedBefore(snapshot, ledger, key, version) {
	const inLedger = ledger.failed.get(key);
	if (inLedger && inLedger.version === version) return true;
	const failure = snapshot.failed[key];
	return Boolean(failure) && failure.version === version;
}

function plan(input) {
	const decisions = new Map();
	const present = new Set();
	for (const item of input.items) {
		present.add(item.key);
		const done = completion(input.snapshot, input.ledger, item.key);
		const reusable =
			done &&
			done.version === item.version &&
			done.fingerprint === input.fingerprint &&
			(!input.full || done.from_ledger) &&
			done.output !== null &&
			input.outputOk(done.output);
		let action;
		let resumed = false;
		if (reusable) {
			action = "skip";
			resumed = Boolean(done.from_ledger);
		} else if (failedBefore(input.snapshot, input.ledger, item.key, item.version)) {
			action = "retry";
		} else {
			action = "process";
		}
		decisions.set(item.key, {
			key: item.key,
			action,
			version: action === "skip" ? null : item.version,
			resumed,
		});
	}
	for (const key of input.unknown) {
		if (!decisions.has(key)) {
			decisions.set(key, { key, action: "preserve", version: null, resumed: false });
		}
	}
	const known = [...Object.keys(input.snapshot.items), ...ledgerCommittedItems(input.ledger)];
	for (const key of known) {
		if (present.has(key) || input.unknown.has(key)) continue;
		if (completion(input.snapshot, input.ledger, key) === null) continue;
		if (!decisions.has(key)) decisions.set(key, { key, action: "delete", version: null, resumed: false });
	}
	return [...decisions.keys()].sort(byteCompare).map((key) => decisions.get(key));
}

function decisionJson(decision) {
	const value = { key: decision.key, action: decision.action };
	if (decision.version !== null && decision.version !== undefined) value.version = decision.version;
	return value;
}

// --- output verification (src/incremental/stage.rs) ------------------------

function outputMatches(root, output, mode) {
	if (typeof output !== "object" || output === null) return false;
	try {
		validateOutputPath(output.path);
	} catch {
		return false;
	}
	const target = path.join(root, output.path);
	let stat;
	try {
		stat = fs.lstatSync(target);
	} catch {
		return false;
	}
	if (stat.isSymbolicLink() || !stat.isFile()) return false;
	if (stat.size !== output.size) return false;
	if (mode === "fast" && output.mtime_ns && Number(mtimeNs(stat)) === Number(output.mtime_ns)) return true;
	try {
		return sha256(fs.readFileSync(target)) === output.digest;
	} catch {
		return false;
	}
}

// --- PID lock (src/state.rs FileLock) --------------------------------------

function processAlive(pid) {
	if (!Number.isInteger(pid) || pid <= 0) return false;
	try {
		process.kill(pid, 0);
		return true;
	} catch (error) {
		return error.code === "EPERM";
	}
}

function tryAcquireLock(lockPath) {
	let existing;
	try {
		existing = fs.readFileSync(lockPath, "utf8");
	} catch (error) {
		if (error.code !== "ENOENT") return false;
	}
	if (existing !== undefined) {
		const pid = Number.parseInt(existing.trim(), 10);
		if (Number.isInteger(pid) && pid > 0 && pid !== process.pid && processAlive(pid)) return false;
	}
	fs.writeFileSync(lockPath, String(process.pid));
	return true;
}

function releaseLock(lockPath) {
	try {
		if (Number.parseInt(fs.readFileSync(lockPath, "utf8").trim(), 10) !== process.pid) return;
		fs.unlinkSync(lockPath);
	} catch {
		// absent or unreadable: nothing to release
	}
}

function sleep(ms) {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

// --- request parsing -------------------------------------------------------

const KNOWN_OPS = new Set(["open", "commit", "fail", "delete", "finish"]);

function requireString(value, field) {
	if (typeof value !== "string") throw invalidRequest(`\`${field}\` must be a string`);
	return value;
}

function parseOpen(raw) {
	const stage = requireString(raw.stage, "stage");
	const sourceId = requireString(raw.source_id, "source_id");
	const fingerprint = requireString(raw.fingerprint, "fingerprint");
	const outputRoot = requireString(raw.output_root, "output_root");
	if (!Array.isArray(raw.items)) throw invalidRequest("`items` must be an array");
	const items = raw.items.map((item) => {
		if (item === null || typeof item !== "object") throw invalidRequest("each item must be an object");
		return { key: requireString(item.key, "key"), version: requireString(item.version, "version") };
	});
	let unknown = [];
	if (raw.unknown !== undefined) {
		if (!Array.isArray(raw.unknown)) throw invalidRequest("`unknown` must be an array");
		unknown = raw.unknown.map((key) => requireString(key, "unknown key"));
	}
	if (raw.full !== undefined && typeof raw.full !== "boolean") throw invalidRequest("`full` must be a boolean");
	if (raw.wait !== undefined && typeof raw.wait !== "boolean") throw invalidRequest("`wait` must be a boolean");
	let verify = "fast";
	if (raw.verify !== undefined) {
		if (raw.verify !== "fast" && raw.verify !== "digest") throw invalidRequest("`verify` must be `fast` or `digest`");
		verify = raw.verify;
	}
	return { stage, sourceId, fingerprint, outputRoot, items, unknown, full: raw.full === true, wait: raw.wait === true, verify };
}

function parseCommit(raw) {
	const key = requireString(raw.key, "key");
	const version = requireString(raw.version, "version");
	const output = raw.output;
	if (output === null || typeof output !== "object") throw invalidRequest("`output` must be an object");
	const outputPath = requireString(output.path, "path");
	let size = null;
	if (output.size !== undefined) {
		if (!Number.isInteger(output.size) || output.size < 0) throw invalidRequest("`output.size` must be a non-negative integer");
		size = output.size;
	}
	let digest = null;
	if (output.digest !== undefined) digest = requireString(output.digest, "digest");
	let fingerprint = null;
	if (raw.fingerprint !== undefined) fingerprint = requireString(raw.fingerprint, "fingerprint");
	return { key, version, outputPath, size, digest, fingerprint };
}

function parseFail(raw) {
	const key = requireString(raw.key, "key");
	const version = requireString(raw.version, "version");
	let error = "";
	if (raw.error !== undefined) error = requireString(raw.error, "error");
	let fingerprint = null;
	if (raw.fingerprint !== undefined) fingerprint = requireString(raw.fingerprint, "fingerprint");
	return { key, version, error, fingerprint };
}

function parseDelete(raw) {
	return { key: requireString(raw.key, "key") };
}

// --- the session -----------------------------------------------------------

function validateSpec(spec) {
	const stage = spec.stage;
	const nameOk =
		stage.length > 0 &&
		stage.length <= 64 &&
		/^[A-Za-z0-9._-]+$/.test(stage) &&
		!stage.startsWith(".") &&
		!stage.startsWith("-");
	if (!nameOk) {
		throw invalidRequest(
			`stage name ${JSON.stringify(stage)} must be 1-64 characters of [A-Za-z0-9._-] and not start with '.' or '-'`,
		);
	}
	if (spec.sourceId.length === 0 || /[\u0000-\u001f\u007f]/.test(spec.sourceId)) {
		throw invalidRequest("source_id must be non-empty text without control characters");
	}
	if (spec.fingerprint.length === 0) throw invalidRequest("fingerprint must not be empty");
	if (spec.outputRoot.length === 0) throw invalidRequest("output_root must not be empty");
}

function validateItems(items, unknown) {
	const map = new Map();
	for (const item of items) {
		validateItemKey(item.key);
		if (item.version.length === 0) throw invalidRequest(`item ${JSON.stringify(item.key)} has an empty version`);
		if (map.has(item.key)) throw stageError("duplicate_item", `item ${JSON.stringify(item.key)} appears twice`);
		map.set(item.key, item.version);
	}
	const set = new Set();
	for (const key of unknown) {
		validateItemKey(key);
		if (map.has(key)) throw invalidRequest(`item ${JSON.stringify(key)} is listed both as present and as unknown`);
		set.add(key);
	}
	return { items: map, unknown: set };
}

function checkItem(state, key, version, fingerprint) {
	const current = state.items.get(key);
	if (current === undefined) throw stageError("unknown_item", `item ${JSON.stringify(key)} is not part of this run`);
	if (current !== version) {
		throw stageError("version_mismatch", `item ${JSON.stringify(key)} is at version ${JSON.stringify(current)} in this run, not ${JSON.stringify(version)}`);
	}
	if (fingerprint !== null && fingerprint !== state.spec.fingerprint) {
		throw stageError("fingerprint_mismatch", `this run has fingerprint ${JSON.stringify(state.spec.fingerprint)}, not ${JSON.stringify(fingerprint)}`);
	}
}

/** The output file at `path`: a regular file whose real location is inside the root. */
function resolveOutput(state, outputPath) {
	validateOutputPath(outputPath);
	const candidate = path.join(state.realOutputRoot, outputPath);
	let stat;
	try {
		stat = fs.lstatSync(candidate);
	} catch (error) {
		if (error.code === "ENOENT") throw stageError("output_missing", `output ${JSON.stringify(outputPath)} does not exist`);
		throw error;
	}
	if (stat.isSymbolicLink()) throw stageError("invalid_output_path", `output ${JSON.stringify(outputPath)} is a symlink`);
	if (!stat.isFile()) throw stageError("output_not_file", `output ${JSON.stringify(outputPath)} is not a regular file`);
	let real;
	try {
		real = fs.realpathSync(candidate);
	} catch {
		throw stageError("output_missing", `output ${JSON.stringify(outputPath)} does not exist`);
	}
	if (real !== state.realOutputRoot && !real.startsWith(state.realOutputRoot + path.sep)) {
		throw stageError("invalid_output_path", `output ${JSON.stringify(outputPath)} resolves outside the output root`);
	}
	return candidate;
}

async function openRun(state, open, respond) {
	validateSpec(open);
	const validated = validateItems(open.items, open.unknown);
	fs.mkdirSync(state.stateDir, { recursive: true });

	let acquired = tryAcquireLock(state.lockPath);
	while (!acquired && open.wait) {
		await sleep(LOCK_WAIT_POLL_MS);
		acquired = tryAcquireLock(state.lockPath);
	}
	if (!acquired) {
		respond(refusal("open", stageError("lock_failed", "another run holds the stage lock")));
		return 3;
	}
	state.lockHeld = true;
	fs.mkdirSync(open.outputRoot, { recursive: true });
	state.realOutputRoot = fs.realpathSync(open.outputRoot);

	let snapshot;
	try {
		snapshot = loadSnapshot(state.stateDir);
	} catch (error) {
		releaseLock(state.lockPath);
		state.lockHeld = false;
		respond(refusal("open", error));
		return null;
	}
	if (snapshot && (snapshot.source_id !== open.sourceId || snapshot.stage !== open.stage)) {
		releaseLock(state.lockPath);
		state.lockHeld = false;
		respond(
			refusal(
				"open",
				stageError(
					"source_mismatch",
					`${state.stateDir} holds stage ${JSON.stringify(snapshot.stage)} of source ${JSON.stringify(snapshot.source_id)}, not stage ${JSON.stringify(open.stage)} of source ${JSON.stringify(open.sourceId)}`,
				),
			),
		);
		return null;
	}
	if (!snapshot) snapshot = emptySnapshot(open.stage, open.sourceId);
	const baseline = snapshotHash(snapshot);

	const txnPath = path.join(state.stateDir, TXN_FILE);
	const expected = {
		format: RUN_FORMAT_VERSION,
		source_id: open.sourceId,
		stage: open.stage,
		run_token: newRunToken(),
		full: open.full,
		baseline,
		fingerprint: open.fingerprint,
		created_at: nowIso(),
	};
	const loaded = loadLedger(state.stateDir);
	const resumed = loaded && ledgerResumes(loaded, expected) ? loaded : null;
	try {
		fs.unlinkSync(txnPath);
	} catch (error) {
		if (error.code !== "ENOENT") throw error;
	}

	const header = resumed
		? { ...expected, run_token: resumed.header.run_token, created_at: resumed.header.created_at }
		: expected;
	const ledger = resumed ?? newLedger(header);
	ledger.header = header;
	atomicWrite(
		txnPath,
		`${JSON.stringify(
			{
				source_id: open.sourceId,
				sync_token: header.run_token,
				manifest_hash_from: baseline,
				manifest_hash_to: sourceHash(validated.items),
				status: "running",
				started_at: nowIso(),
			},
			null,
			2,
		)}\n`,
	);
	atomicWrite(path.join(state.stateDir, JOURNAL_FILE), journalContent(header, resumed));

	state.spec = open;
	state.snapshot = snapshot;
	state.ledger = ledger;
	state.items = validated.items;
	state.unknown = validated.unknown;
	state.runToken = header.run_token;

	const all = plan({
		items: open.items,
		unknown: validated.unknown,
		fingerprint: open.fingerprint,
		full: open.full,
		snapshot,
		ledger,
		outputOk: (output) => outputMatches(state.realOutputRoot, output, open.verify),
	});
	const counts = { process: 0, retry: 0, skip: 0, resumed: 0, delete: 0, preserve: 0 };
	const decisions = [];
	for (const decision of all) {
		if (decision.action === "skip") {
			counts.skip += 1;
			counts.resumed += decision.resumed ? 1 : 0;
		} else if (decision.action === "preserve") {
			counts.preserve += 1;
		} else {
			counts[decision.action] += 1;
			decisions.push(decisionJson(decision));
		}
	}

	const owners = new Map();
	const known = [...Object.keys(snapshot.items), ...ledgerCommittedItems(ledger)];
	for (const key of known) {
		const done = completion(snapshot, ledger, key);
		if (done && done.output) owners.set(done.output.path, key);
	}
	state.owners = owners;

	respond({ ok: true, op: "open", run_token: header.run_token, resumed: Boolean(resumed), counts, decisions });
	return null;
}

function commit(state, request, respond) {
	checkItem(state, request.key, request.version, request.fingerprint);
	validateOutputPath(request.outputPath);
	const owner = state.owners.get(request.outputPath);
	if (owner !== undefined && owner !== request.key) {
		throw stageError("output_conflict", `output ${JSON.stringify(request.outputPath)} already belongs to another item`);
	}
	const target = resolveOutput(state, request.outputPath);
	const stat = fs.lstatSync(target);
	const data = fs.readFileSync(target);
	const output = { path: request.outputPath, size: data.length, digest: sha256(data), mtime_ns: Number(mtimeNs(stat)) };
	if (request.size !== null && request.size !== output.size) {
		throw stageError("output_mismatch", `output ${JSON.stringify(request.outputPath)} has ${output.size} bytes, the host claimed ${request.size}`);
	}
	if (request.digest !== null && request.digest !== output.digest) {
		throw stageError("output_mismatch", `output ${JSON.stringify(request.outputPath)} has digest ${output.digest}, the host claimed ${request.digest}`);
	}
	const record = { item: request.key, version: request.version, outcome: "committed", output, error: null };
	fs.appendFileSync(path.join(state.stateDir, JOURNAL_FILE), `${JSON.stringify(recordToJson(record))}\n`);
	const previous = completion(state.snapshot, state.ledger, request.key);
	applyRecord(state.ledger, record);
	if (previous && previous.output) state.owners.delete(previous.output.path);
	state.owners.set(output.path, request.key);
	respond({ ok: true, op: "commit", key: request.key, output: { path: output.path, size: output.size, digest: output.digest } });
}

function fail(state, request, respond) {
	checkItem(state, request.key, request.version, request.fingerprint);
	const record = {
		item: request.key,
		version: request.version,
		outcome: "failed",
		output: null,
		error: request.error.slice(0, MAX_ERROR_CHARS),
	};
	fs.appendFileSync(path.join(state.stateDir, JOURNAL_FILE), `${JSON.stringify(recordToJson(record))}\n`);
	applyRecord(state.ledger, record);
	respond({ ok: true, op: "fail", key: request.key });
}

function deleteItem(state, request, respond) {
	validateItemKey(request.key);
	if (state.items.has(request.key)) {
		throw stageError("still_present", `item ${JSON.stringify(request.key)} is still part of this run's source`);
	}
	if (state.unknown.has(request.key)) {
		throw stageError("preserved_item", `item ${JSON.stringify(request.key)} is unobservable, so its output is preserved`);
	}
	const done = completion(state.snapshot, state.ledger, request.key);
	if (!done) {
		respond({ ok: true, op: "delete", key: request.key, output_removed: false });
		return;
	}
	const outputPath = done.output ? done.output.path : null;
	let removed = false;
	if (outputPath !== null) {
		validateOutputPath(outputPath);
		const target = path.join(state.realOutputRoot, outputPath);
		let parentReal;
		try {
			parentReal = fs.realpathSync(path.dirname(target));
		} catch (error) {
			if (error.code !== "ENOENT") throw error;
		}
		if (parentReal !== undefined && parentReal !== state.realOutputRoot && !parentReal.startsWith(state.realOutputRoot + path.sep)) {
			throw stageError("invalid_output_path", `output ${JSON.stringify(outputPath)} resolves outside the output root`);
		}
		try {
			fs.unlinkSync(target);
			removed = true;
		} catch (error) {
			if (error.code !== "ENOENT") throw error;
		}
	}
	const record = { item: request.key, version: null, outcome: "deleted", output: null, error: null };
	fs.appendFileSync(path.join(state.stateDir, JOURNAL_FILE), `${JSON.stringify(recordToJson(record))}\n`);
	applyRecord(state.ledger, record);
	if (outputPath !== null) state.owners.delete(outputPath);
	respond({ ok: true, op: "delete", key: request.key, output_removed: removed });
}

function finish(state, respond) {
	const pending = plan({
		items: [...state.items].map(([key, version]) => ({ key, version })),
		unknown: state.unknown,
		fingerprint: state.spec.fingerprint,
		full: state.spec.full,
		snapshot: state.snapshot,
		ledger: state.ledger,
		outputOk: () => true,
	}).filter((decision) => decision.action === "process" || decision.action === "retry" || decision.action === "delete").length;

	const fingerprint = state.ledger.header.fingerprint;
	const next = emptySnapshot(state.spec.stage, state.spec.sourceId);
	let committed = 0;
	let deleted = 0;
	let failed = 0;
	for (const record of ledgerRecords(state.ledger)) {
		if (record.outcome === "committed") {
			committed += 1;
			if (record.version !== null && record.output) {
				next.items[record.item] = { version: record.version, fingerprint, output: record.output };
			}
			delete next.failed[record.item];
		} else if (record.outcome === "deleted") {
			deleted += 1;
			delete next.items[record.item];
			delete next.failed[record.item];
		} else {
			failed += 1;
			next.failed[record.item] = { version: record.version ?? null, error: record.error ?? "" };
		}
	}
	for (const key of Object.keys(next.failed)) {
		if (!state.items.has(key) && !state.unknown.has(key)) delete next.failed[key];
	}
	next.updated_at = nowIso();
	atomicWrite(path.join(state.stateDir, SNAPSHOT_FILE), JSON.stringify(next));
	for (const file of [JOURNAL_FILE, TXN_FILE]) {
		try {
			fs.unlinkSync(path.join(state.stateDir, file));
		} catch (error) {
			if (error.code !== "ENOENT") throw error;
		}
	}
	releaseLock(state.lockPath);
	state.lockHeld = false;
	respond({ ok: true, op: "finish", committed, deleted, failed, pending });
}

function refusal(op, error) {
	const code = error instanceof StageError ? error.code : "internal";
	return { ok: false, op, error: { code, message: error.message } };
}

/**
 * Run one stage session. Resolves nothing: it returns after wiring stdin, and
 * the process ends when the host closes stdin (`finish` already exited 0). Only
 * `stateDir` is required; `stdin`/`write` exist so a unit test can drive the
 * protocol in-process.
 */
function runStage(options = {}) {
	const stateDir = options.stateDir;
	const stdin = options.stdin ?? process.stdin;
	const write = options.write ?? ((line) => fs.writeSync(1, line));
	if (!stateDir) {
		process.stderr.write("minsync stage: --state-dir is required\n");
		process.exitCode = 2;
		return;
	}

	const state = {
		stateDir,
		lockPath: path.join(stateDir, LOCK_FILE),
		spec: null,
		snapshot: null,
		ledger: null,
		items: new Map(),
		unknown: new Set(),
		owners: new Map(),
		realOutputRoot: null,
		lockHeld: false,
		opened: false,
		finished: false,
		queue: [],
		pumping: false,
	};
	const respond = (value) => write(`${JSON.stringify(value)}\n`);
	const releaseOwnedLock = () => {
		if (state.lockHeld) releaseLock(state.lockPath);
	};
	process.once("exit", releaseOwnedLock);

	async function handleLine(line) {
		if (line.trim().length === 0) return;
		let request;
		try {
			request = JSON.parse(line);
		} catch (error) {
			respond(refusal("unknown", invalidRequest(error.message)));
			return;
		}
		if (request === null || typeof request !== "object" || typeof request.op !== "string" || !KNOWN_OPS.has(request.op)) {
			respond(refusal("unknown", invalidRequest("unknown request")));
			return;
		}
		const op = request.op;

		if (op === "open") {
			let open;
			try {
				open = parseOpen(request);
			} catch (error) {
				respond(refusal("open", error));
				return;
			}
			if (state.opened) {
				respond(refusal("open", stageError("already_open", "a run is already open")));
				return;
			}
			let code = null;
			try {
				code = await openRun(state, open, respond);
			} catch (error) {
				respond(refusal("open", error));
				return;
			}
			if (code === 3) {
				state.finished = true;
				process.exit(3);
			}
			if (code === null) state.opened = true;
			return;
		}

		// Parse the payload before anything else, like serde: a malformed
		// `commit` is `invalid_request` even before `open`.
		let parsed = null;
		try {
			if (op === "commit") parsed = parseCommit(request);
			else if (op === "fail") parsed = parseFail(request);
			else if (op === "delete") parsed = parseDelete(request);
		} catch (error) {
			respond(refusal(op, error));
			return;
		}

		if (!state.opened) {
			respond(refusal(op, stageError("not_open", "send `open` first")));
			return;
		}

		if (op === "finish") {
			try {
				finish(state, respond);
			} catch (error) {
				respond(refusal("finish", error));
				state.finished = true;
				process.exit(error instanceof StageError && error.code === "lock_failed" ? 3 : 1);
				return;
			}
			state.finished = true;
			state.opened = false;
			process.exit(0);
			return;
		}

		try {
			if (op === "commit") commit(state, parsed, respond);
			else if (op === "fail") fail(state, parsed, respond);
			else deleteItem(state, parsed, respond);
		} catch (error) {
			respond(refusal(op, error));
		}
	}

	function pump() {
		if (state.pumping) return;
		state.pumping = true;
		void (async () => {
			try {
				while (state.queue.length > 0 && !state.finished) {
					const line = state.queue.shift();
					if (line === null) {
						process.exitCode = state.opened ? 1 : 0;
						return;
					}
					await handleLine(line);
				}
			} catch {
				// A write raced the host closing its pipe: end the session quietly.
			} finally {
				state.pumping = false;
			}
		})();
	}

	const rl = readline.createInterface({ input: stdin, crlfDelay: Infinity });
	rl.on("line", (line) => {
		state.queue.push(line);
		pump();
	});
	rl.on("close", () => {
		state.queue.push(null);
		pump();
	});
}

module.exports = { runStage, StageError };

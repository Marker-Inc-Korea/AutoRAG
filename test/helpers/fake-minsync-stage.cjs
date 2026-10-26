"use strict";

// Shared, test-only MinSync `stage` protocol simulator.
//
// It speaks the host protocol of MinSync's materialization stage
// (src/incremental/protocol.rs) over stdin/stdout and reproduces the resume
// planner (src/incremental/plan.rs), the durable journal (journal.rs), the
// committed snapshot (snapshot.rs) and the PID lock (state.rs) on top of the
// plain filesystem — including the durable-outcome extensions the host relies
// on: deliberate `skip`, deferred failures with their retry fingerprint, and
// the read-only `inspect` and `plan` ops. It also answers `capabilities`,
// which is how a host checks for those extensions before it mutates anything.
// Nothing here talks to a real `minsync`, the network, or a vector store: a
// test writes the output files itself and this process only validates and
// records them.
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
//   progress.jsonl header line + one record per committed/failed/skipped/
//                  deleted item; `plan` and `inspect` read it without
//                  touching it.
//   snapshot.json  the committed state, rewritten once by `finish`; it also
//                  holds the failures (deferred ones with the fingerprint and
//                  retry fingerprint they were deferred under) and the
//                  deliberate skips.
//
// `inspect` and `plan` never take the lock and never write anything, so they
// work with or without an open session and alongside a live run.
//
// Exit codes match MinSync: 0 after `finish`, 3 when another run holds the
// lock, 1 when input ends mid-run (the run is left resumable), and 0 when
// input ends with no run open at all.

const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const readline = require("node:readline");

const JOURNAL_FILE = "progress.jsonl";
const SNAPSHOT_FILE = "snapshot.json";
const TXN_FILE = "txn.json";
const LOCK_FILE = "lock";
const RUN_FORMAT_VERSION = 3;
const SNAPSHOT_FORMAT = 2;
// The extended protocol this simulator speaks, as the `capabilities` op
// reports it: the same feature names, in the same order, as the real build.
// A host that needs these must refuse a binary that reports less.
const CAPABILITY_PROTOCOL = 2;
const CAPABILITIES = [
	"retry_fingerprint",
	"retry_failed",
	"deferred_failure",
	"discard_output",
	"deliberate_skip",
	"plan",
	"inspect",
];
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

/**
 * The stored modification time, as a JSON integer like the core's `u128`. A
 * legacy or hand-written state may already hold the decimal string form; both
 * read the same way, and only `inspect` hands the value out as text.
 */
function readMtimeNs(value) {
	if (value === undefined || value === null) return 0;
	if (typeof value === "string") return Number(value);
	return value;
}

/** RFC3339 for a recorded nanosecond timestamp, for `inspect`'s `updated_at`. */
function rfc3339(ns) {
	return new Date(Number(ns) / 1e6).toISOString();
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
		skipped: {},
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
	if (snapshot === null || typeof snapshot !== "object") {
		throw stageError("corrupt_state", `${snapshotPath} is unusable (it does not hold a stage snapshot)`);
	}
	// A state written before the durable-outcome extensions has neither map.
	if (snapshot.items === undefined) snapshot.items = {};
	if (snapshot.failed === undefined) snapshot.failed = {};
	if (snapshot.skipped === undefined) snapshot.skipped = {};
	for (const item of Object.values(snapshot.items)) {
		if (item && typeof item === "object" && item.output && typeof item.output === "object") {
			item.output.mtime_ns = readMtimeNs(item.output.mtime_ns);
		}
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
	for (const key of Object.keys(snapshot.skipped).sort(byteCompare)) {
		const omission = snapshot.skipped[key];
		for (const part of ["skipped", key, omission.version, omission.fingerprint, omission.reason]) {
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
	if (record.reason !== null && record.reason !== undefined) out.reason = record.reason;
	// Only an explicit `defer`/`discard_output` is recorded; a plain failure
	// keeps the shape it always had.
	if (record.deferred === true) out.deferred = true;
	if (record.discard_output === true) out.discard_output = true;
	if (record.retry_fingerprint !== null && record.retry_fingerprint !== undefined) {
		out.retry_fingerprint = record.retry_fingerprint;
	}
	return out;
}

const OUTCOMES = new Set(["committed", "deleted", "failed", "skipped"]);

function parseRecord(line) {
	const raw = JSON.parse(line);
	if (raw === null || typeof raw !== "object" || typeof raw.item !== "string") {
		throw new Error("malformed record");
	}
	const version = raw.version === undefined ? null : raw.version;
	const outcome = raw.outcome ?? (version !== null ? "committed" : "deleted");
	if (!OUTCOMES.has(outcome)) throw new Error(`unknown outcome ${JSON.stringify(raw.outcome)}`);
	if ((outcome === "committed" || outcome === "skipped") && (version === null || version === undefined)) {
		throw new Error(`${outcome} record for ${JSON.stringify(raw.item)} has no version`);
	}
	return {
		item: raw.item,
		version,
		outcome,
		output: raw.output ?? null,
		error: raw.error ?? null,
		reason: raw.reason ?? null,
		deferred: raw.deferred === true,
		discard_output: raw.discard_output === true,
		retry_fingerprint: raw.retry_fingerprint ?? null,
	};
}

function applyRecord(ledger, record) {
	if (record.outcome === "committed" || record.outcome === "deleted" || record.outcome === "skipped") {
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

function ledgerSkippedItems(ledger) {
	const items = [];
	for (const record of ledger.done.values()) if (record.outcome === "skipped") items.push(record.item);
	return items;
}

function emptyLedger(sourceId, stage) {
	return newLedger({
		format: RUN_FORMAT_VERSION,
		source_id: sourceId,
		stage,
		run_token: "",
		full: false,
		baseline: "",
		fingerprint: "",
		created_at: "",
	});
}

/**
 * The deliberate omission MinSync holds for `key`: an interrupted run's ledger
 * wins over the committed snapshot, as a completion does.
 */
function skipRecord(snapshot, ledger, key) {
	const record = ledger.done.get(key);
	if (record !== undefined) {
		if (record.outcome !== "skipped") return null;
		return {
			version: record.version ?? "",
			fingerprint: ledger.header.fingerprint,
			reason: record.reason ?? "",
			from_ledger: true,
		};
	}
	const entry = snapshot.skipped[key];
	if (!entry) return null;
	return {
		version: entry.version,
		fingerprint: entry.fingerprint,
		reason: entry.reason ?? "",
		from_ledger: false,
	};
}

/** The failure MinSync holds for `key`: the ledger's wins over the snapshot's. */
function failureRecord(snapshot, ledger, key) {
	const record = ledger.failed.get(key);
	if (record !== undefined) {
		return {
			version: record.version ?? null,
			error: record.error ?? "",
			fingerprint: ledger.header.fingerprint,
			deferred: record.deferred === true,
			retry_fingerprint: record.retry_fingerprint ?? "",
		};
	}
	// A completion or omission in this run's ledger supersedes the snapshot's
	// failure of the same item.
	if (ledger.done.has(key)) return null;
	const entry = snapshot.failed[key];
	if (!entry) return null;
	return {
		version: entry.version ?? null,
		error: entry.error ?? "",
		fingerprint: entry.fingerprint ?? "",
		deferred: entry.deferred === true,
		retry_fingerprint: entry.retry_fingerprint ?? "",
	};
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
	// A journal from a newer build is not trusted; its work is simply redone.
	if (typeof header.format === "number" && (header.format < 1 || header.format > RUN_FORMAT_VERSION)) return null;
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

/**
 * The deferred failure holding `key` at `version` back, if any. A deferral
 * only holds under the exact settings it was recorded with: the same source
 * version, the same processing fingerprint and the same retry fingerprint.
 * `retry_failed` (or a full run) releases every deferral.
 */
function deferredFailure(snapshot, ledger, key, version, fingerprint, retryFingerprint) {
	const failure = failureRecord(snapshot, ledger, key);
	if (failure === null || !failure.deferred) return null;
	if (failure.version !== version) return null;
	if (failure.fingerprint !== fingerprint) return null;
	if (failure.retry_fingerprint !== retryFingerprint) return null;
	return failure;
}

/**
 * One decision per item, in key order. A deferred failure is not retried: it
 * becomes a `skip` that carries the failure it rests on, so the host can see
 * why the item is not being processed.
 */
function plan(input) {
	const decisions = new Map();
	const present = new Set();
	const retryFingerprint = input.retryFingerprint ?? "";
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
		const omission = skipRecord(input.snapshot, input.ledger, item.key);
		// An omission is only about the source version and the settings; unlike a
		// committed output it does not care whether the run is a full one.
		const omissionReusable =
			omission !== null &&
			omission.version === item.version &&
			omission.fingerprint === input.fingerprint;
		const held =
			input.retryFailed || input.full
				? null
				: deferredFailure(input.snapshot, input.ledger, item.key, item.version, input.fingerprint, retryFingerprint);
		let action;
		let resumed = false;
		let error = null;
		if (reusable) {
			action = "skip";
			resumed = Boolean(done.from_ledger);
		} else if (omissionReusable) {
			action = "skip";
			resumed = Boolean(omission.from_ledger);
		} else if (held !== null) {
			action = "skip";
			error = held.error;
		} else if (failedBefore(input.snapshot, input.ledger, item.key, item.version)) {
			action = "retry";
		} else {
			action = "process";
		}
		decisions.set(item.key, {
			key: item.key,
			action,
			// A deferred skip names the version it is holding back; an ordinary
			// skip has nothing to process.
			version: action === "skip" && error === null ? null : item.version,
			resumed,
			deferred: error !== null,
			error,
		});
	}
	for (const key of input.unknown) {
		if (!decisions.has(key)) {
			decisions.set(key, { key, action: "preserve", version: null, resumed: false, deferred: false, error: null });
		}
	}
	// Anything with a durable record that is neither present nor unobservable
	// has had its source confirmed gone. That includes an item whose only
	// record is a failure or an omission: the store has to drop what it holds
	// for the key, even when it holds no output.
	const known = new Set([
		...Object.keys(input.snapshot.items),
		...Object.keys(input.snapshot.failed),
		...Object.keys(input.snapshot.skipped),
		...ledgerCommittedItems(input.ledger),
		...input.ledger.failed.keys(),
		...ledgerSkippedItems(input.ledger),
	]);
	for (const key of known) {
		if (present.has(key) || input.unknown.has(key) || decisions.has(key)) continue;
		decisions.set(key, { key, action: "delete", version: null, resumed: false, deferred: false, error: null });
	}
	return [...decisions.keys()].sort(byteCompare).map((key) => decisions.get(key));
}

/** What each decision means for the run's counters. */
function tally(decisions) {
	const counts = { process: 0, retry: 0, skip: 0, resumed: 0, delete: 0, preserve: 0 };
	for (const decision of decisions) {
		if (decision.action === "skip") {
			counts.skip += 1;
			if (decision.resumed) counts.resumed += 1;
		} else if (decision.action === "preserve") {
			counts.preserve += 1;
		} else {
			counts[decision.action] += 1;
		}
	}
	return counts;
}

function decisionJson(decision) {
	const value = { key: decision.key, action: decision.action };
	if (decision.version !== null && decision.version !== undefined) value.version = decision.version;
	if (decision.deferred) value.deferred = true;
	if (decision.error !== null && decision.error !== undefined) value.error = decision.error;
	return value;
}

/** `open` reports work, plus every deferred failure it is holding back. */
function openDecisions(decisions) {
	const reported = [];
	for (const decision of decisions) {
		if (decision.action === "preserve") continue;
		if (decision.action === "skip" && !decision.deferred) continue;
		reported.push(decisionJson(decision));
	}
	return reported;
}

// --- output verification (src/incremental/stage.rs) ------------------------

function outputMatches(root, output, mode) {
	if (root === null || root === undefined) return false;
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
	const recorded = readMtimeNs(output.mtime_ns);
	if (mode === "fast" && recorded !== 0 && Number(mtimeNs(stat)) === recorded) return true;
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

const KNOWN_OPS = new Set(["capabilities", "open", "commit", "fail", "delete", "skip", "finish", "plan", "inspect"]);

function requireString(value, field) {
	if (typeof value !== "string") throw invalidRequest(`\`${field}\` must be a string`);
	return value;
}

function requireBoolean(value, field, fallback) {
	if (value === undefined) return fallback;
	if (typeof value !== "boolean") throw invalidRequest(`\`${field}\` must be a boolean`);
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
	let retryFingerprint = "";
	if (raw.retry_fingerprint !== undefined) retryFingerprint = requireString(raw.retry_fingerprint, "retry_fingerprint");
	let verify = "fast";
	if (raw.verify !== undefined) {
		if (raw.verify !== "fast" && raw.verify !== "digest") throw invalidRequest("`verify` must be `fast` or `digest`");
		verify = raw.verify;
	}
	return {
		stage,
		sourceId,
		fingerprint,
		outputRoot,
		items,
		unknown,
		full: requireBoolean(raw.full, "full", false),
		wait: requireBoolean(raw.wait, "wait", false),
		retryFingerprint,
		retryFailed: requireBoolean(raw.retry_failed, "retry_failed", false),
		verify,
	};
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
	return {
		key,
		version,
		error,
		fingerprint,
		defer: requireBoolean(raw.defer, "defer", false),
		discardOutput: requireBoolean(raw.discard_output, "discard_output", false),
	};
}

function parseSkip(raw) {
	return {
		key: requireString(raw.key, "key"),
		version: requireString(raw.version, "version"),
		reason: requireString(raw.reason, "reason"),
	};
}

function parseDelete(raw) {
	return { key: requireString(raw.key, "key") };
}

function parseInspect(raw) {
	let verify = "fast";
	if (raw.verify !== undefined) {
		if (raw.verify !== "fast" && raw.verify !== "digest") throw invalidRequest("`verify` must be `fast` or `digest`");
		verify = raw.verify;
	}
	return {
		stage: requireString(raw.stage, "stage"),
		sourceId: requireString(raw.source_id, "source_id"),
		outputRoot: requireString(raw.output_root, "output_root"),
		verify,
	};
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
		retryFingerprint: open.retryFingerprint,
		retryFailed: open.retryFailed,
		full: open.full,
		snapshot,
		ledger,
		outputOk: (output) => outputMatches(state.realOutputRoot, output, open.verify),
	});
	const counts = tally(all);

	const owners = new Map();
	const known = [...Object.keys(snapshot.items), ...ledgerCommittedItems(ledger)];
	for (const key of known) {
		const done = completion(snapshot, ledger, key);
		if (done && done.output) owners.set(done.output.path, key);
	}
	state.owners = owners;

	respond({
		ok: true,
		op: "open",
		run_token: header.run_token,
		resumed: Boolean(resumed),
		counts,
		decisions: openDecisions(all),
	});
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

/**
 * Unlink one recorded output, refusing a path that leaves the output root.
 * Returns whether a file was removed.
 */
function removeOutput(state, outputPath) {
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
		return true;
	} catch (error) {
		if (error.code === "ENOENT") return false;
		throw error;
	}
}

/** The recorded output `key` holds right now, if any. */
function recordedOutput(state, key) {
	const done = completion(state.snapshot, state.ledger, key);
	return done && done.output ? done.output.path : null;
}

function fail(state, request, respond) {
	checkItem(state, request.key, request.version, request.fingerprint);
	let discarded = null;
	if (request.discardOutput) {
		// MinSync removes the last good output itself, so no later run can reuse it.
		discarded = recordedOutput(state, request.key);
		if (discarded !== null) removeOutput(state, discarded);
	}
	const record = {
		item: request.key,
		version: request.version,
		outcome: "failed",
		output: null,
		error: request.error.slice(0, MAX_ERROR_CHARS),
	};
	if (request.defer) {
		record.deferred = true;
		if (state.spec.retryFingerprint) record.retry_fingerprint = state.spec.retryFingerprint;
	}
	if (request.discardOutput) record.discard_output = true;
	fs.appendFileSync(path.join(state.stateDir, JOURNAL_FILE), `${JSON.stringify(recordToJson(record))}\n`);
	applyRecord(state.ledger, record);
	if (discarded !== null) {
		state.owners.delete(discarded);
		delete state.snapshot.items[request.key];
	}
	respond({ ok: true, op: "fail", key: request.key });
}

/** A deliberate omission: no output, no retry, and it is not a failure. */
function skipItem(state, request, respond) {
	checkItem(state, request.key, request.version, null);
	const previous = recordedOutput(state, request.key);
	if (previous !== null) removeOutput(state, previous);
	const record = {
		item: request.key,
		version: request.version,
		outcome: "skipped",
		output: null,
		error: null,
		reason: request.reason.slice(0, MAX_ERROR_CHARS),
	};
	fs.appendFileSync(path.join(state.stateDir, JOURNAL_FILE), `${JSON.stringify(recordToJson(record))}\n`);
	applyRecord(state.ledger, record);
	if (previous !== null) state.owners.delete(previous);
	// The item's completion is cleared: the omission replaces it.
	delete state.snapshot.items[request.key];
	delete state.snapshot.failed[request.key];
	respond({ ok: true, op: "skip", key: request.key });
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
	const removed = outputPath === null ? false : removeOutput(state, outputPath);
	const record = { item: request.key, version: null, outcome: "deleted", output: null, error: null };
	fs.appendFileSync(path.join(state.stateDir, JOURNAL_FILE), `${JSON.stringify(recordToJson(record))}\n`);
	applyRecord(state.ledger, record);
	if (outputPath !== null) state.owners.delete(outputPath);
	respond({ ok: true, op: "delete", key: request.key, output_removed: removed });
}

/** An output root that is already there; `inspect` and `plan` never create one. */
function existingOutputRoot(outputRoot) {
	try {
		return fs.realpathSync(outputRoot);
	} catch (error) {
		if (error.code === "ENOENT") return null;
		throw error;
	}
}

/** The snapshot and the interrupted run's ledger, read without touching either. */
function readState(stateDir, stage, sourceId) {
	const snapshot = loadSnapshot(stateDir) ?? emptySnapshot(stage, sourceId);
	if (snapshot.source_id !== sourceId || snapshot.stage !== stage) {
		throw stageError(
			"source_mismatch",
			`${stateDir} holds stage ${JSON.stringify(snapshot.stage)} of source ${JSON.stringify(snapshot.source_id)}, not stage ${JSON.stringify(stage)} of source ${JSON.stringify(sourceId)}`,
		);
	}
	const loaded = loadLedger(stateDir);
	// Only a journal describing this very state is its own durable work; any
	// other baseline belongs to a different run and is never trusted.
	const related =
		loaded &&
		loaded.header.source_id === snapshot.source_id &&
		loaded.header.stage === snapshot.stage &&
		loaded.header.baseline === snapshotHash(snapshot)
			? loaded
			: null;
	return { snapshot, ledger: related ?? emptyLedger(sourceId, stage) };
}

/**
 * Preview a run without opening one: the same planner and failure policy as
 * `open`, every decision reported, and nothing written.
 */
function planRun(stateDir, open, respond) {
	validateSpec(open);
	const validated = validateItems(open.items, open.unknown);
	const { snapshot, ledger } = readState(stateDir, open.stage, open.sourceId);
	const root = existingOutputRoot(open.outputRoot);
	const all = plan({
		items: open.items,
		unknown: validated.unknown,
		fingerprint: open.fingerprint,
		retryFingerprint: open.retryFingerprint,
		retryFailed: open.retryFailed,
		full: open.full,
		snapshot,
		ledger,
		outputOk: (output) => outputMatches(root, output, open.verify),
	});
	respond({ ok: true, op: "plan", counts: tally(all), decisions: all.map(decisionJson) });
}

/**
 * Reconstruct the durable state and verify every recorded output: the
 * committed items that still verify, the failures (deferred or not) and the
 * deliberate omissions. Takes no lock and writes nothing.
 */
function inspectState(stateDir, request, respond) {
	const { snapshot, ledger } = readState(stateDir, request.stage, request.sourceId);
	const root = existingOutputRoot(request.outputRoot);
	const items = [];
	const known = [...new Set([...Object.keys(snapshot.items), ...ledgerCommittedItems(ledger)])].sort(byteCompare);
	for (const key of known) {
		const done = completion(snapshot, ledger, key);
		if (done === null || done.output === null) continue;
		if (!outputMatches(root, done.output, request.verify)) continue;
		const output = done.output;
		items.push({
			key,
			version: done.version,
			fingerprint: done.fingerprint,
			output: {
				path: output.path,
				size: output.size,
				digest: output.digest,
				// The wire form is decimal text: nanoseconds do not fit a JS number.
				mtime_ns: String(readMtimeNs(output.mtime_ns)),
			},
			updated_at: rfc3339(readMtimeNs(output.mtime_ns)),
		});
	}
	const failed = [];
	const failures = [...new Set([...Object.keys(snapshot.failed), ...ledger.failed.keys()])].sort(byteCompare);
	for (const key of failures) {
		const failure = failureRecord(snapshot, ledger, key);
		if (failure === null) continue;
		failed.push({ key, version: failure.version, error: failure.error, deferred: failure.deferred });
	}
	const skipped = [];
	const omissions = [
		...new Set([
			...Object.keys(snapshot.skipped),
			...ledgerSkippedItems(ledger),
			// A ledger record for the key may also show the omission is gone.
			...ledger.done.keys(),
		]),
	].sort(byteCompare);
	for (const key of omissions) {
		const omission = skipRecord(snapshot, ledger, key);
		if (omission === null) continue;
		skipped.push({ key, version: omission.version, reason: omission.reason, fingerprint: omission.fingerprint });
	}
	respond({ ok: true, op: "inspect", items, failed, skipped });
}

function finish(state, respond) {
	const pending = plan({
		items: [...state.items].map(([key, version]) => ({ key, version })),
		unknown: state.unknown,
		fingerprint: state.spec.fingerprint,
		retryFingerprint: state.spec.retryFingerprint,
		// A release applies to the deferrals this run opened with, not to the
		// ones it just recorded: the next run faces those with no release.
		retryFailed: false,
		full: state.spec.full,
		snapshot: state.snapshot,
		ledger: state.ledger,
		outputOk: () => true,
	}).filter((decision) => decision.action === "process" || decision.action === "retry" || decision.action === "delete").length;

	const fingerprint = state.ledger.header.fingerprint;
	const next = state.snapshot;
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
			delete next.skipped[record.item];
		} else if (record.outcome === "deleted") {
			deleted += 1;
			delete next.items[record.item];
			delete next.failed[record.item];
			delete next.skipped[record.item];
		} else if (record.outcome === "skipped") {
			next.skipped[record.item] = {
				version: record.version,
				fingerprint,
				reason: record.reason ?? "",
			};
			// The omission replaced the item's output and any earlier failure.
			delete next.items[record.item];
			delete next.failed[record.item];
		} else {
			failed += 1;
			const failure = { version: record.version ?? null, error: record.error ?? "" };
			if (record.deferred === true) {
				failure.deferred = true;
				failure.fingerprint = fingerprint;
				if (record.retry_fingerprint) failure.retry_fingerprint = record.retry_fingerprint;
			}
			next.failed[record.item] = failure;
			// A discarded output must not survive in the snapshot either.
			if (record.discard_output === true) delete next.items[record.item];
		}
	}
	for (const key of Object.keys(next.failed)) {
		if (!state.items.has(key) && !state.unknown.has(key)) delete next.failed[key];
	}
	for (const key of Object.keys(next.skipped)) {
		if (!state.items.has(key) && !state.unknown.has(key)) delete next.skipped[key];
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

		// The capability handshake: no state, no lock, no open run.
		if (op === "capabilities") {
			respond({
				ok: true,
				op: "capabilities",
				protocol: CAPABILITY_PROTOCOL,
				features: CAPABILITIES,
				journal_format: RUN_FORMAT_VERSION,
				snapshot_format: SNAPSHOT_FORMAT,
			});
			return;
		}

		// `plan` and `inspect` are read-only: no lock, no session, no writes.
		if (op === "plan") {
			let open;
			try {
				open = parseOpen(request);
			} catch (error) {
				respond(refusal("plan", error));
				return;
			}
			try {
				planRun(state.stateDir, open, respond);
			} catch (error) {
				respond(refusal("plan", error));
			}
			return;
		}
		if (op === "inspect") {
			let inspect;
			try {
				inspect = parseInspect(request);
			} catch (error) {
				respond(refusal("inspect", error));
				return;
			}
			try {
				inspectState(state.stateDir, inspect, respond);
			} catch (error) {
				respond(refusal("inspect", error));
			}
			return;
		}

		// Parse the payload before anything else, like serde: a malformed
		// `commit` is `invalid_request` even before `open`.
		let parsed = null;
		try {
			if (op === "commit") parsed = parseCommit(request);
			else if (op === "fail") parsed = parseFail(request);
			else if (op === "skip") parsed = parseSkip(request);
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
			else if (op === "skip") skipItem(state, parsed, respond);
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

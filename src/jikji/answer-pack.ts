import { realpathSync } from "node:fs";
import { isAbsolute, join, relative, resolve } from "node:path";
import type { SourceRoot } from "../filesystem/source-paths.ts";
import type {
	JikjiAnswerPack,
	JikjiCandidate,
	JikjiEvidence,
	JikjiHandoffAction,
	JikjiNextRead,
	JikjiToolCallPolicy,
} from "./types.ts";

const NEXT_READ_VALUES: ReadonlySet<string> = new Set(["cache", "wiki", "original", "none"]);
const HANDOFF_VALUES: ReadonlySet<string> = new Set(["direct_use", "jikji_retry", "raw_fallback_after_retry"]);

/** Windows drive-letter path, e.g. `C:\` or `C:/`. */
const DRIVE_PATH_RE = /^[A-Za-z]:[\\/]/u;
/** URL-like value with scheme + `//`, e.g. `https://`, `file://`. */
const URLISH_RE = /^[A-Za-z][A-Za-z0-9+.-]*:\/\//u;
/** Bare URL scheme without `//`, e.g. `file:foo`, `mailto:`. */
const URL_SCHEME_RE = /^[A-Za-z][A-Za-z0-9+.-]*:/u;

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isStringArray(value: unknown): value is string[] {
	return Array.isArray(value) && value.every((v) => typeof v === "string");
}

/**
 * Strict-parse and validate a `jikji find --json` answer-pack from raw stdout.
 *
 * Required fields are checked for presence and correct type; the
 * `candidates[].next_read` and `evidence_pack[].next_read` enums and the
 * `handoff_action` enum are validated. Extra fields are tolerated. Returns
 * `undefined` when ANY required field is missing or wrong-typed, or when the
 * stdout is not valid JSON.
 */
export function parseJikjiAnswerPack(stdout: string): JikjiAnswerPack | undefined {
	let parsed: unknown;
	try {
		parsed = JSON.parse(stdout);
	} catch {
		return undefined;
	}
	if (!isRecord(parsed)) return undefined;

	const answerPaths = parsed.answer_paths;
	if (!isStringArray(answerPaths)) return undefined;

	const paths = parsed.paths;
	if (!isStringArray(paths)) return undefined;

	const candidatesRaw = parsed.candidates;
	if (!Array.isArray(candidatesRaw)) return undefined;
	const candidates: JikjiCandidate[] = [];
	for (const entry of candidatesRaw) {
		if (!isRecord(entry)) return undefined;
		if (typeof entry.path !== "string") return undefined;
		if (typeof entry.next_read !== "string" || !NEXT_READ_VALUES.has(entry.next_read)) return undefined;
		if (entry.label !== undefined && typeof entry.label !== "string") return undefined;
		if (entry.score !== undefined && typeof entry.score !== "number") return undefined;
		const candidate: JikjiCandidate = {
			path: entry.path,
			nextRead: entry.next_read as JikjiNextRead,
			...(entry.label !== undefined ? { label: entry.label } : {}),
			...(entry.score !== undefined ? { score: entry.score } : {}),
		};
		candidates.push(candidate);
	}

	const evidenceRaw = parsed.evidence_pack;
	if (!Array.isArray(evidenceRaw)) return undefined;
	const evidencePack: JikjiEvidence[] = [];
	for (const entry of evidenceRaw) {
		if (!isRecord(entry)) return undefined;
		if (typeof entry.path !== "string") return undefined;
		if (typeof entry.next_read !== "string" || !NEXT_READ_VALUES.has(entry.next_read)) return undefined;
		evidencePack.push({ path: entry.path, nextRead: entry.next_read as JikjiNextRead });
	}

	const handoffAction = parsed.handoff_action;
	if (typeof handoffAction !== "string" || !HANDOFF_VALUES.has(handoffAction)) return undefined;

	const policyRaw = parsed.tool_call_policy;
	if (!isRecord(policyRaw)) return undefined;
	if (typeof policyRaw.stop_after_find !== "boolean") return undefined;
	if (!isStringArray(policyRaw.forbidden_tools)) return undefined;
	if (!isStringArray(policyRaw.allowed_followups)) return undefined;
	const toolCallPolicy: JikjiToolCallPolicy = {
		stopAfterFind: policyRaw.stop_after_find,
		forbiddenTools: policyRaw.forbidden_tools,
		allowedFollowups: policyRaw.allowed_followups,
	};

	if (typeof parsed.agent_should_not_rerank !== "boolean") return undefined;

	return {
		answerPaths,
		paths,
		candidates,
		evidencePack,
		handoffAction: handoffAction as JikjiHandoffAction,
		toolCallPolicy,
		agentShouldNotRerank: parsed.agent_should_not_rerank,
	};
}

/**
 * Normalize a path returned in a jikji answer-pack into a REAL absolute path
 * usable by bash, scoped to one of the configured source roots.
 *
 * Accepted forms:
 * - An absolute path that `realpath`-resolves within a configured root.
 * - A relative path scoped to a root (resolved against each root in turn).
 *
 * Rejected:
 * - Root escapes (realpath outside every root).
 * - Windows drive-letter paths (`C:\...`, `C:/...`).
 * - URL-like values (`https://`, `file://`, `file:foo`).
 * - Empty / non-string input.
 *
 * Duplicate roots (same real path) are deduped: the first matching root wins
 * and subsequent identical roots are skipped. Output is a real absolute path —
 * no opaque IDs, no sanitization.
 */
export function normalizeJikjiAnswerPath(rawPath: string, sourceRoots: readonly SourceRoot[]): string | undefined {
	if (typeof rawPath !== "string") return undefined;
	const trimmed = rawPath.trim();
	if (trimmed.length === 0) return undefined;
	if (DRIVE_PATH_RE.test(trimmed) || URLISH_RE.test(trimmed)) return undefined;
	const normalized = trimmed.replace(/\\/g, "/");
	if (URL_SCHEME_RE.test(normalized)) return undefined;
	if (normalized.includes("\0")) return undefined;

	const seenRealRoots = new Set<string>();
	for (const root of sourceRoots) {
		let realRoot: string;
		try {
			realRoot = realpathSync(root.rootPath);
		} catch {
			continue;
		}
		if (seenRealRoots.has(realRoot)) continue;
		seenRealRoots.add(realRoot);

		let candidate: string;
		if (isAbsolute(normalized)) {
			candidate = normalized;
		} else {
			if (normalized.split("/").includes("..")) continue;
			candidate = resolve(realRoot, normalized);
		}

		let realCandidate: string;
		try {
			realCandidate = realpathSync(candidate);
		} catch {
			continue;
		}

		const rel = relative(realRoot, realCandidate);
		if (rel.startsWith("..") || isAbsolute(rel)) continue;

		// Re-express the resolved real path against the configured root's own
		// spelling (do not leak platform realpath canonicalization such as
		// macOS /var -> /private/var). Symlinks within the root are followed.
		return join(root.rootPath, rel);
	}
	return undefined;
}

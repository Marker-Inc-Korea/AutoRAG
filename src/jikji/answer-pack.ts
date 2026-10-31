import { realpathSync } from "node:fs";
import { isAbsolute, join, relative, resolve } from "node:path";
import type { SourceRoot } from "../filesystem/source-paths.ts";
import type {
	JikjiAnswerPack,
	JikjiCandidate,
	JikjiConfidence,
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

function parseNextRead(value: unknown): JikjiNextRead | undefined {
	if (typeof value === "string" && NEXT_READ_VALUES.has(value)) {
		return value as JikjiNextRead;
	}
	if (isRecord(value) && typeof value.kind === "string" && NEXT_READ_VALUES.has(value.kind)) {
		return value.kind as JikjiNextRead;
	}
	return undefined;
}

function optionalString(value: unknown): string | undefined {
	return typeof value === "string" ? value : undefined;
}

function optionalNumber(value: unknown): number | undefined {
	return typeof value === "number" ? value : undefined;
}

/** Evidence is shown on one line and compared as text; whitespace runs carry no meaning. */
function evidenceText(value: unknown): string | undefined {
	const raw = typeof value === "string" ? value : isStringArray(value) ? value.join(" | ") : undefined;
	const text = raw?.replace(/\s+/gu, " ").trim();
	return text === undefined || text.length === 0 ? undefined : text;
}

/**
 * Optional response-level fields: a missing or wrong-typed one is dropped, never a reason to reject the
 * pack, because confidence is advisory and upstream owns its exact shape.
 */
function parseConfidence(parsed: Record<string, unknown>): JikjiConfidence | undefined {
	const level = optionalString(parsed.confidence);
	const score = optionalNumber(parsed.confidence_score);
	const answerability = optionalString(parsed.answerability);
	const recommendedAction = optionalString(parsed.recommended_action);
	const requiresLlmRerank = typeof parsed.requires_llm_rerank === "boolean" ? parsed.requires_llm_rerank : undefined;
	const indexStatus = optionalString(parsed.index_status);
	const confidence: JikjiConfidence = {
		...(level !== undefined ? { level } : {}),
		...(score !== undefined ? { score } : {}),
		...(answerability !== undefined ? { answerability } : {}),
		...(recommendedAction !== undefined ? { recommendedAction } : {}),
		...(requiresLlmRerank !== undefined ? { requiresLlmRerank } : {}),
		...(indexStatus !== undefined ? { indexStatus } : {}),
	};
	return Object.keys(confidence).length === 0 ? undefined : confidence;
}

const CONFIDENCE_RANK: Readonly<Record<string, number>> = { low: 0, medium: 1, high: 2 };

/**
 * Combine the confidence of several roots' responses into one. Weakest wins: the root with the lowest level
 * (then lowest score) supplies every field, and `requiresLlmRerank` is true when any root asked for it. A
 * missing or unknown level counts as the weakest.
 */
export function mergeJikjiConfidence(entries: readonly (JikjiConfidence | undefined)[]): JikjiConfidence | undefined {
	const present = entries.filter((entry): entry is JikjiConfidence => entry !== undefined);
	const rank = (entry: JikjiConfidence): number => CONFIDENCE_RANK[entry.level ?? ""] ?? 0;
	let weakest = present[0];
	for (const entry of present) {
		if (
			weakest !== undefined &&
			(rank(entry) < rank(weakest) ||
				(rank(entry) === rank(weakest) && (entry.score ?? Infinity) < (weakest.score ?? Infinity)))
		) {
			weakest = entry;
		}
	}
	if (weakest === undefined) return undefined;
	const flags = present.map((entry) => entry.requiresLlmRerank).filter((flag): flag is boolean => flag !== undefined);
	return flags.length === 0 ? weakest : { ...weakest, requiresLlmRerank: flags.some(Boolean) };
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
		const path = typeof entry.path === "string" ? entry.path : typeof entry.p === "string" ? entry.p : undefined;
		if (path === undefined) return undefined;
		const nextRead = parseNextRead(entry.next_read);
		if (nextRead === undefined) return undefined;
		if (entry.label !== undefined && typeof entry.label !== "string") return undefined;
		const score = typeof entry.score === "number" ? entry.score : typeof entry.s === "number" ? entry.s : undefined;
		const evidence = evidenceText(entry.ev ?? entry.evidence);
		const why = isStringArray(entry.why) ? entry.why : undefined;
		const matchedTerms = isStringArray(entry.matched_terms)
			? entry.matched_terms
			: isStringArray(entry.terms)
				? entry.terms
				: undefined;
		const strategies = isStringArray(entry.strategies) ? entry.strategies : undefined;
		const rank = optionalNumber(entry.rank);
		const candidate: JikjiCandidate = {
			path,
			nextRead,
			...(entry.label !== undefined ? { label: entry.label } : {}),
			...(score !== undefined ? { score } : {}),
			...(evidence !== undefined ? { evidence } : {}),
			...(why !== undefined ? { why } : {}),
			...(matchedTerms !== undefined ? { matchedTerms } : {}),
			...(strategies !== undefined ? { strategies } : {}),
			...(rank !== undefined ? { rank } : {}),
		};
		candidates.push(candidate);
	}

	const handoffAction = parsed.handoff_action;
	if (typeof handoffAction !== "string" || !HANDOFF_VALUES.has(handoffAction)) return undefined;

	const evidenceRaw =
		parsed.evidence_pack === undefined && handoffAction === "jikji_retry" ? [] : parsed.evidence_pack;
	if (!Array.isArray(evidenceRaw)) return undefined;
	const evidencePack: JikjiEvidence[] = [];
	const packEvidence = new Map<string, string>();
	for (const entry of evidenceRaw) {
		if (!isRecord(entry)) return undefined;
		const path = typeof entry.path === "string" ? entry.path : typeof entry.p === "string" ? entry.p : undefined;
		if (path === undefined) return undefined;
		const nextRead = parseNextRead(entry.next_read);
		if (nextRead === undefined) return undefined;
		evidencePack.push({ path, nextRead });
		const text = evidenceText(entry.evidence);
		if (text !== undefined && !packEvidence.has(path)) packEvidence.set(path, text);
	}
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

	const agentShouldNotRerank =
		parsed.agent_should_not_rerank === undefined && handoffAction === "jikji_retry"
			? false
			: parsed.agent_should_not_rerank;
	if (typeof agentShouldNotRerank !== "boolean") return undefined;

	const confidence = parseConfidence(parsed);
	return {
		answerPaths,
		paths,
		// A candidate without its own `ev` takes the evidence lines of the same path from `evidence_pack`.
		candidates: candidates.map((candidate) => {
			const fallback = candidate.evidence === undefined ? packEvidence.get(candidate.path) : undefined;
			return fallback === undefined ? candidate : { ...candidate, evidence: fallback };
		}),
		evidencePack,
		handoffAction: handoffAction as JikjiHandoffAction,
		toolCallPolicy,
		agentShouldNotRerank,
		...(confidence !== undefined ? { confidence } : {}),
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
	const nativeAbsolute = isAbsolute(trimmed);
	if ((DRIVE_PATH_RE.test(trimmed) && !nativeAbsolute) || URLISH_RE.test(trimmed)) return undefined;
	const normalized = trimmed.replace(/\\/g, "/");
	if (!nativeAbsolute && URL_SCHEME_RE.test(normalized)) return undefined;
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

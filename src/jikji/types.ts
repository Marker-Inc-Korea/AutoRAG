export interface JikjiOptions {
	readonly binaryPath?: string;
	readonly topK?: number;
	readonly timeoutMs?: number;
	readonly maxBufferBytes?: number;
	readonly includeHidden?: boolean;
	readonly includeSensitive?: boolean;
	readonly parseTimeout?: number;
	readonly maxFiles?: number;
	readonly staleAfterSeconds?: number;
	readonly exclude?: readonly string[];
	readonly env?: Readonly<Record<string, string | undefined>>;
}

export interface JikjiDefaultOptions {
	readonly binaryPath: string;
	readonly topK: number;
	readonly timeoutMs: number;
	readonly maxBufferBytes: number;
	readonly includeHidden: boolean;
	readonly includeSensitive: boolean;
	readonly parseTimeout: number;
	readonly maxFiles: number;
	readonly staleAfterSeconds: number;
	readonly exclude: readonly string[];
}

export const DEFAULT_JIKJI_OPTIONS: JikjiDefaultOptions = {
	binaryPath: "jikji",
	topK: 20,
	timeoutMs: 10_000,
	maxBufferBytes: 1_048_576,
	includeHidden: false,
	includeSensitive: false,
	parseTimeout: 5,
	maxFiles: 0,
	staleAfterSeconds: 86_400,
	exclude: [],
};

export interface JikjiFindOptions {
	readonly topK?: number;
	readonly signal?: AbortSignal;
}

export type JikjiFailureReason =
	| "aborted"
	| "invalid-payload"
	| "malformed-json"
	| "nonzero-exit"
	| "spawn-error"
	| "stderr-too-large"
	| "stdout-too-large"
	| "timeout";

export type JikjiFindResult =
	| {
			readonly ok: true;
			readonly payload: JikjiFindPayload;
			readonly stdout: string;
			readonly stderr: string;
			readonly code: number;
	  }
	| {
			readonly ok: false;
			readonly reason: JikjiFailureReason;
			readonly stdout: string;
			readonly stderr: string;
			readonly code: number | null;
	  };

export type JikjiParseResult =
	| {
			readonly ok: true;
			readonly payload: JikjiFindPayload;
	  }
	| {
			readonly ok: false;
			readonly reason: "invalid-payload" | "malformed-json";
	  };

export interface JikjiFindPayload {
	readonly mode?: string;
	readonly answerPackVersion?: number;
	readonly root?: string;
	readonly query?: string;
	readonly queryType?: string;
	readonly confidence?: string;
	readonly confidenceScore?: number;
	readonly recommendedAction?: string;
	readonly handoffAction?: string;
	readonly indexStatus?: string;
	readonly command?: string;
	readonly paths: readonly string[];
	readonly answerPaths: readonly string[];
	readonly evidencePack: readonly JikjiEvidenceItem[];
	readonly judgeCandidateSlate: readonly JikjiJudgeCandidate[];
	readonly candidates: readonly JikjiCompactCandidate[];
}

export interface JikjiEvidenceItem {
	readonly path: string;
	readonly why: readonly string[];
	readonly matchedTerms: readonly string[];
	readonly evidence: readonly string[];
	readonly nextRead?: JikjiNextRead;
}

export interface JikjiJudgeCandidate {
	readonly rank?: number;
	readonly path: string;
	readonly score?: number;
	readonly evidence: readonly string[];
	readonly nextRead?: JikjiNextRead;
}

export interface JikjiCompactCandidate {
	readonly p: string;
	readonly s?: number;
	readonly rank?: number;
	readonly why: readonly string[];
	readonly terms: readonly string[];
	readonly ev?: string;
	readonly nextRead?: JikjiNextRead;
}

export interface JikjiNextRead {
	readonly kind?: string;
	readonly path?: string;
}

export function parseJikjiFindPayload(text: string): JikjiParseResult {
	const parsed = parseJson(text);
	if (parsed.kind === "malformed") return { ok: false, reason: "malformed-json" };
	if (!isRecord(parsed.value)) return { ok: false, reason: "invalid-payload" };
	const payload = parseFindPayload(parsed.value);
	if (payload === undefined) return { ok: false, reason: "invalid-payload" };
	return { ok: true, payload };
}

function parseJson(text: string): { readonly kind: "ok"; readonly value: unknown } | { readonly kind: "malformed" } {
	try {
		return { kind: "ok", value: JSON.parse(text) };
	} catch (error) {
		if (error instanceof SyntaxError) return { kind: "malformed" };
		throw error;
	}
}

function parseFindPayload(value: Record<string, unknown>): JikjiFindPayload | undefined {
	const paths = optionalStringArray(value.paths);
	const answerPaths = optionalStringArray(value.answer_paths);
	const evidencePack = optionalRecordArray(value.evidence_pack, parseEvidenceItem);
	const judgeCandidateSlate = optionalRecordArray(value.judge_candidate_slate, parseJudgeCandidate);
	const candidates = optionalRecordArray(value.candidates, parseCompactCandidate);
	if (paths === undefined || answerPaths === undefined || evidencePack === undefined) return undefined;
	if (judgeCandidateSlate === undefined || candidates === undefined) return undefined;
	return {
		mode: optionalString(value.mode),
		answerPackVersion: optionalNumber(value.answer_pack_version),
		root: optionalString(value.root),
		query: optionalString(value.query),
		queryType: optionalString(value.query_type),
		confidence: optionalString(value.confidence),
		confidenceScore: optionalNumber(value.confidence_score),
		recommendedAction: optionalString(value.recommended_action),
		handoffAction: optionalString(value.handoff_action),
		indexStatus: optionalString(value.index_status),
		command: optionalString(value.command),
		paths: paths ?? [],
		answerPaths: answerPaths ?? [],
		evidencePack: evidencePack ?? [],
		judgeCandidateSlate: judgeCandidateSlate ?? [],
		candidates: candidates ?? [],
	};
}

function parseEvidenceItem(value: Record<string, unknown>): JikjiEvidenceItem | undefined {
	const path = requiredString(value.path);
	const why = optionalStringArray(value.why);
	const matchedTerms = optionalStringArray(value.matched_terms);
	const evidence = optionalStringArray(value.evidence);
	if (path === undefined || why === undefined || matchedTerms === undefined || evidence === undefined)
		return undefined;
	return {
		path,
		why: why ?? [],
		matchedTerms: matchedTerms ?? [],
		evidence: evidence ?? [],
		nextRead: optionalNextRead(value.next_read),
	};
}

function parseJudgeCandidate(value: Record<string, unknown>): JikjiJudgeCandidate | undefined {
	const path = requiredString(value.path);
	const evidence = optionalStringArray(value.evidence);
	if (path === undefined || evidence === undefined) return undefined;
	return {
		rank: optionalNumber(value.rank),
		path,
		score: optionalNumber(value.score),
		evidence: evidence ?? [],
		nextRead: optionalNextRead(value.next_read),
	};
}

function parseCompactCandidate(value: Record<string, unknown>): JikjiCompactCandidate | undefined {
	const path = requiredString(value.p);
	const why = optionalStringArray(value.why);
	const terms = optionalStringArray(value.terms);
	if (path === undefined || why === undefined || terms === undefined) return undefined;
	return {
		p: path,
		s: optionalNumber(value.s),
		rank: optionalNumber(value.rank),
		why: why ?? [],
		terms: terms ?? [],
		ev: optionalString(value.ev),
		nextRead: optionalNextRead(value.next_read),
	};
}

function optionalNextRead(value: unknown): JikjiNextRead | undefined {
	if (value === undefined) return undefined;
	if (!isRecord(value)) return undefined;
	return { kind: optionalString(value.kind), path: optionalString(value.path) };
}

function optionalRecordArray<T>(
	value: unknown,
	parser: (item: Record<string, unknown>) => T | undefined,
): readonly T[] | undefined {
	if (value === undefined) return [];
	if (!Array.isArray(value)) return undefined;
	const parsed: T[] = [];
	for (const item of value) {
		if (!isRecord(item)) return undefined;
		const parsedItem = parser(item);
		if (parsedItem === undefined) return undefined;
		parsed.push(parsedItem);
	}
	return parsed;
}

function optionalStringArray(value: unknown): readonly string[] | undefined {
	if (value === undefined) return [];
	if (!Array.isArray(value)) return undefined;
	const strings: string[] = [];
	for (const item of value) {
		if (typeof item !== "string") return undefined;
		strings.push(item);
	}
	return strings;
}

function requiredString(value: unknown): string | undefined {
	return typeof value === "string" && value.length > 0 ? value : undefined;
}

function optionalString(value: unknown): string | undefined {
	return typeof value === "string" ? value : undefined;
}

function optionalNumber(value: unknown): number | undefined {
	return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}

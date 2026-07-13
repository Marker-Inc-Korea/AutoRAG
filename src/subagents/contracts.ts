export const RETRIEVAL_METHODS = [
	"posix",
	"bash",
	"posix/bash",
	"bm25",
	"minsync",
	"vector",
	"minsync/vector",
	"datasource",
] as const;

export type RetrievalMethod = (typeof RETRIEVAL_METHODS)[number];

export type SourceTemporalStatus = "asOf" | "updatedAt" | "createdAt" | "unknown";

export type SourceTemporal =
	| { readonly status: "asOf"; readonly asOf: string }
	| { readonly status: "updatedAt"; readonly updatedAt: string }
	| { readonly status: "createdAt"; readonly createdAt: string }
	| { readonly status: "unknown" };

export interface ExplorerAssignment {
	readonly originalQuery: string;
	readonly method: RetrievalMethod;
	readonly queryVariant: string;
	/** Optional set of variants retained when an orchestrator fans out work. */
	readonly queryVariants?: readonly string[];
}

export interface EvidenceCandidate {
	readonly source: string;
	readonly method: RetrievalMethod;
	readonly evidence: string;
	readonly retrievedAt: string;
	readonly sourceTemporal: SourceTemporal;
	readonly locator?: string;
}

export interface ExplorerReport {
	readonly assignment: ExplorerAssignment;
	readonly evidenceCandidates: readonly EvidenceCandidate[];
	/** Alias for consumers that use the shorter candidate field name. */
	readonly candidates?: readonly EvidenceCandidate[];
	readonly summary?: string;
}

export interface FollowUpRequest {
	readonly reason: string;
	readonly assignments: readonly ExplorerAssignment[];
}

export type JudgeDecision =
	| {
			readonly decision: "sufficient";
			readonly rationale: string;
			readonly followUp?: undefined;
	  }
	| {
			readonly decision: "follow_up";
			readonly rationale: string;
			readonly followUp: FollowUpRequest;
	  };

export interface ExhaustionErrorDetails {
	readonly code: "EXPLORER_EXHAUSTED";
	readonly originalQuery: string;
	readonly iterations: number;
	readonly maxIterations: number;
	readonly reason: string;
}

export type ContractErrorCode =
	| "INVALID_OBJECT"
	| "MISSING_ORIGINAL_QUERY"
	| "INVALID_ORIGINAL_QUERY"
	| "MISSING_RETRIEVAL_METHOD"
	| "INVALID_RETRIEVAL_METHOD"
	| "MISSING_QUERY_VARIANT"
	| "INVALID_QUERY_VARIANT"
	| "INVALID_QUERY_VARIANTS"
	| "QUERY_VARIANT_NOT_INCLUDED"
	| "MISSING_SOURCE"
	| "INVALID_SOURCE"
	| "MISSING_EVIDENCE_METHOD"
	| "INVALID_EVIDENCE_METHOD"
	| "MISSING_EVIDENCE"
	| "INVALID_EVIDENCE"
	| "MISSING_RETRIEVED_AT"
	| "INVALID_RETRIEVED_AT"
	| "MISSING_SOURCE_TEMPORAL"
	| "INVALID_SOURCE_TEMPORAL"
	| "MISSING_SOURCE_TEMPORAL_DATE"
	| "INVALID_SOURCE_TEMPORAL_DATE"
	| "MISSING_ASSIGNMENT"
	| "MISSING_EVIDENCE_CANDIDATES"
	| "INVALID_EVIDENCE_CANDIDATES"
	| "INVALID_SUMMARY"
	| "MISSING_FOLLOW_UP_REASON"
	| "INVALID_FOLLOW_UP_REASON"
	| "MISSING_FOLLOW_UP_ASSIGNMENTS"
	| "EMPTY_FOLLOW_UP_ASSIGNMENTS"
	| "INVALID_FOLLOW_UP_ASSIGNMENTS"
	| "MISSING_JUDGE_DECISION"
	| "INVALID_JUDGE_DECISION"
	| "MISSING_JUDGE_RATIONALE"
	| "INVALID_JUDGE_RATIONALE"
	| "MISSING_FOLLOW_UP"
	| "INVALID_FOLLOW_UP"
	| "INVALID_EXHAUSTION_ERROR"
	| "INVALID_EXHAUSTION_CODE"
	| "MISSING_EXHAUSTION_QUERY"
	| "INVALID_EXHAUSTION_QUERY"
	| "INVALID_EXHAUSTION_ITERATIONS"
	| "INVALID_EXHAUSTION_LIMIT"
	| "MISSING_EXHAUSTION_REASON"
	| "INVALID_EXHAUSTION_REASON";

export class ContractValidationError extends Error {
	readonly code: ContractErrorCode;
	readonly path?: string;

	constructor(code: ContractErrorCode, message: string, path?: string) {
		super(message);
		this.name = "ContractValidationError";
		this.code = code;
		this.path = path;
	}
}

export class ExplorerExhaustionError extends Error {
	readonly code = "EXPLORER_EXHAUSTED" as const;
	readonly originalQuery: string;
	readonly iterations: number;
	readonly maxIterations: number;
	readonly reason: string;

	constructor(details: Omit<ExhaustionErrorDetails, "code">) {
		const parsed = parseExhaustionDetails({ code: "EXPLORER_EXHAUSTED", ...details });
		super(parsed.reason);
		this.name = "ExplorerExhaustionError";
		this.originalQuery = parsed.originalQuery;
		this.iterations = parsed.iterations;
		this.maxIterations = parsed.maxIterations;
		this.reason = parsed.reason;
	}

	toJSON(): ExhaustionErrorDetails {
		return {
			code: this.code,
			originalQuery: this.originalQuery,
			iterations: this.iterations,
			maxIterations: this.maxIterations,
			reason: this.reason,
		};
	}
}

export function parseExplorerAssignment(value: unknown): ExplorerAssignment {
	const record = asRecord(value, "explorer assignment");
	const originalQuery = requiredText(record, "originalQuery", "MISSING_ORIGINAL_QUERY", "INVALID_ORIGINAL_QUERY");
	const method = requiredMethod(record, "method", "MISSING_RETRIEVAL_METHOD", "INVALID_RETRIEVAL_METHOD");
	const queryVariant = requiredText(record, "queryVariant", "MISSING_QUERY_VARIANT", "INVALID_QUERY_VARIANT");

	const variantsRaw = record.queryVariants;
	if (variantsRaw === undefined) return { originalQuery, method, queryVariant };
	if (!Array.isArray(variantsRaw) || variantsRaw.length === 0 || !variantsRaw.every(isNonEmptyString)) {
		throw new ContractValidationError(
			"INVALID_QUERY_VARIANTS",
			"queryVariants must be a non-empty array of strings.",
			"queryVariants",
		);
	}
	if (!variantsRaw.includes(queryVariant)) {
		throw new ContractValidationError(
			"QUERY_VARIANT_NOT_INCLUDED",
			"queryVariant must be included in queryVariants when both are provided.",
			"queryVariants",
		);
	}
	return { originalQuery, method, queryVariant, queryVariants: [...variantsRaw] };
}

export function parseEvidenceCandidate(value: unknown): EvidenceCandidate {
	const record = asRecord(value, "evidence candidate");
	const source = requiredText(record, "source", "MISSING_SOURCE", "INVALID_SOURCE");
	const method = requiredMethod(record, "method", "MISSING_EVIDENCE_METHOD", "INVALID_EVIDENCE_METHOD");
	const evidence = requiredText(record, "evidence", "MISSING_EVIDENCE", "INVALID_EVIDENCE");
	const retrievedAt = requiredTimestamp(record, "retrievedAt", "MISSING_RETRIEVED_AT", "INVALID_RETRIEVED_AT");
	const temporalRaw = record.sourceTemporal;
	if (temporalRaw === undefined) {
		throw new ContractValidationError("MISSING_SOURCE_TEMPORAL", "sourceTemporal is required.", "sourceTemporal");
	}
	const sourceTemporal = parseSourceTemporal(temporalRaw);
	const locator = record.locator;
	if (locator !== undefined && !isNonEmptyString(locator)) {
		throw new ContractValidationError(
			"INVALID_SOURCE",
			"locator must be a non-empty string when provided.",
			"locator",
		);
	}
	return {
		source,
		method,
		evidence,
		retrievedAt,
		sourceTemporal,
		...(locator !== undefined ? { locator } : {}),
	};
}

export function parseExplorerReport(value: unknown): ExplorerReport {
	const record = asRecord(value, "explorer report");
	const assignmentRaw = record.assignment;
	if (assignmentRaw === undefined) {
		throw new ContractValidationError("MISSING_ASSIGNMENT", "assignment is required.", "assignment");
	}
	const assignment = parseExplorerAssignment(assignmentRaw);
	const candidatesRaw = record.evidenceCandidates ?? record.candidates;
	if (candidatesRaw === undefined) {
		throw new ContractValidationError(
			"MISSING_EVIDENCE_CANDIDATES",
			"evidenceCandidates is required.",
			"evidenceCandidates",
		);
	}
	if (!Array.isArray(candidatesRaw)) {
		throw new ContractValidationError(
			"INVALID_EVIDENCE_CANDIDATES",
			"evidenceCandidates must be an array.",
			"evidenceCandidates",
		);
	}
	const evidenceCandidates = candidatesRaw.map(parseEvidenceCandidate);
	const summary = record.summary;
	if (summary !== undefined && !isNonEmptyString(summary)) {
		throw new ContractValidationError(
			"INVALID_SUMMARY",
			"summary must be a non-empty string when provided.",
			"summary",
		);
	}
	return {
		assignment,
		evidenceCandidates,
		candidates: evidenceCandidates,
		...(summary !== undefined ? { summary } : {}),
	};
}

export function parseFollowUpRequest(value: unknown): FollowUpRequest {
	const record = asRecord(value, "follow-up request");
	const reason = requiredText(record, "reason", "MISSING_FOLLOW_UP_REASON", "INVALID_FOLLOW_UP_REASON");
	const assignmentsRaw = record.assignments;
	if (assignmentsRaw === undefined) {
		throw new ContractValidationError("MISSING_FOLLOW_UP_ASSIGNMENTS", "assignments is required.", "assignments");
	}
	if (!Array.isArray(assignmentsRaw)) {
		throw new ContractValidationError(
			"INVALID_FOLLOW_UP_ASSIGNMENTS",
			"assignments must be an array.",
			"assignments",
		);
	}
	if (assignmentsRaw.length === 0) {
		throw new ContractValidationError(
			"EMPTY_FOLLOW_UP_ASSIGNMENTS",
			"assignments must contain at least one item.",
			"assignments",
		);
	}
	return { reason, assignments: assignmentsRaw.map(parseExplorerAssignment) };
}

export function parseJudgeDecision(value: unknown): JudgeDecision {
	const record = asRecord(value, "judge decision");
	const decisionRaw = record.decision;
	if (decisionRaw === undefined) {
		throw new ContractValidationError("MISSING_JUDGE_DECISION", "decision is required.", "decision");
	}
	if (decisionRaw !== "sufficient" && decisionRaw !== "follow_up") {
		throw new ContractValidationError(
			"INVALID_JUDGE_DECISION",
			"decision must be sufficient or follow_up.",
			"decision",
		);
	}
	const rationale = requiredText(record, "rationale", "MISSING_JUDGE_RATIONALE", "INVALID_JUDGE_RATIONALE");
	if (decisionRaw === "sufficient") return { decision: decisionRaw, rationale };
	const followUpRaw = record.followUp;
	if (followUpRaw === undefined) {
		throw new ContractValidationError(
			"MISSING_FOLLOW_UP",
			"followUp is required for a follow_up decision.",
			"followUp",
		);
	}
	return { decision: decisionRaw, rationale, followUp: parseFollowUpRequest(followUpRaw) };
}

export function parseExhaustionError(value: unknown): ExhaustionErrorDetails {
	if (value instanceof ExplorerExhaustionError) return value.toJSON();
	try {
		return parseExhaustionDetails(value);
	} catch (error) {
		if (error instanceof ContractValidationError) throw error;
		throw new ContractValidationError("INVALID_EXHAUSTION_ERROR", "Invalid exhaustion error.");
	}
}

export function isExplorerAssignment(value: unknown): value is ExplorerAssignment {
	return succeeds(() => parseExplorerAssignment(value));
}

export function isEvidenceCandidate(value: unknown): value is EvidenceCandidate {
	return succeeds(() => parseEvidenceCandidate(value));
}

export function isExplorerReport(value: unknown): value is ExplorerReport {
	return succeeds(() => parseExplorerReport(value));
}

export function isFollowUpRequest(value: unknown): value is FollowUpRequest {
	return succeeds(() => parseFollowUpRequest(value));
}

export function isJudgeDecision(value: unknown): value is JudgeDecision {
	return succeeds(() => parseJudgeDecision(value));
}

export function isExhaustionError(value: unknown): value is ExhaustionErrorDetails {
	return succeeds(() => parseExhaustionError(value));
}

export function safeParseExplorerAssignment(value: unknown): ExplorerAssignment | undefined {
	return safeParse(() => parseExplorerAssignment(value));
}

export function safeParseEvidenceCandidate(value: unknown): EvidenceCandidate | undefined {
	return safeParse(() => parseEvidenceCandidate(value));
}

export function safeParseExplorerReport(value: unknown): ExplorerReport | undefined {
	return safeParse(() => parseExplorerReport(value));
}

export function safeParseFollowUpRequest(value: unknown): FollowUpRequest | undefined {
	return safeParse(() => parseFollowUpRequest(value));
}

export function safeParseJudgeDecision(value: unknown): JudgeDecision | undefined {
	return safeParse(() => parseJudgeDecision(value));
}

export function safeParseExhaustionError(value: unknown): ExhaustionErrorDetails | undefined {
	return safeParse(() => parseExhaustionError(value));
}

function parseSourceTemporal(value: unknown): SourceTemporal {
	const record = asRecord(value, "sourceTemporal");
	const status = record.status;
	if (status === undefined) {
		throw new ContractValidationError(
			"INVALID_SOURCE_TEMPORAL",
			"sourceTemporal.status is required.",
			"sourceTemporal.status",
		);
	}
	if (status === "unknown") return { status };
	if (status !== "asOf" && status !== "updatedAt" && status !== "createdAt") {
		throw new ContractValidationError(
			"INVALID_SOURCE_TEMPORAL",
			"sourceTemporal.status is invalid.",
			"sourceTemporal.status",
		);
	}
	const dateKey = status;
	const date = record[dateKey];
	if (date === undefined) {
		throw new ContractValidationError(
			"MISSING_SOURCE_TEMPORAL_DATE",
			`${dateKey} is required for sourceTemporal.${status}.`,
			`sourceTemporal.${dateKey}`,
		);
	}
	if (!isTimestamp(date)) {
		throw new ContractValidationError(
			"INVALID_SOURCE_TEMPORAL_DATE",
			`${dateKey} must be a valid timestamp.`,
			`sourceTemporal.${dateKey}`,
		);
	}
	return status === "asOf"
		? { status, asOf: date }
		: status === "updatedAt"
			? { status, updatedAt: date }
			: { status, createdAt: date };
}

function parseExhaustionDetails(value: unknown): ExhaustionErrorDetails {
	if (!isRecord(value))
		throw new ContractValidationError("INVALID_EXHAUSTION_ERROR", "Exhaustion error must be an object.");
	if (value.code !== "EXPLORER_EXHAUSTED") {
		throw new ContractValidationError(
			"INVALID_EXHAUSTION_CODE",
			"Exhaustion error code must be EXPLORER_EXHAUSTED.",
			"code",
		);
	}
	const originalQuery = requiredText(value, "originalQuery", "MISSING_EXHAUSTION_QUERY", "INVALID_EXHAUSTION_QUERY");
	const iterations = value.iterations;
	if (!isNonNegativeInteger(iterations)) {
		throw new ContractValidationError(
			"INVALID_EXHAUSTION_ITERATIONS",
			"iterations must be a non-negative integer.",
			"iterations",
		);
	}
	const maxIterations = value.maxIterations;
	if (!isPositiveInteger(maxIterations) || iterations < maxIterations) {
		throw new ContractValidationError(
			"INVALID_EXHAUSTION_LIMIT",
			"maxIterations must be positive and not exceed iterations.",
			"maxIterations",
		);
	}
	const reason = requiredText(value, "reason", "MISSING_EXHAUSTION_REASON", "INVALID_EXHAUSTION_REASON");
	return { code: "EXPLORER_EXHAUSTED", originalQuery, iterations, maxIterations, reason };
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asRecord(value: unknown, label: string): Record<string, unknown> {
	if (typeof value !== "object" || value === null || Array.isArray(value)) {
		throw new ContractValidationError("INVALID_OBJECT", `${label} must be an object.`);
	}
	return value as Record<string, unknown>;
}

function requiredText(
	record: Record<string, unknown>,
	key: string,
	missingCode: ContractErrorCode,
	invalidCode: ContractErrorCode,
): string {
	const value = record[key];
	if (value === undefined) throw new ContractValidationError(missingCode, `${key} is required.`, key);
	if (!isNonEmptyString(value))
		throw new ContractValidationError(invalidCode, `${key} must be a non-empty string.`, key);
	return value;
}

function requiredMethod(
	record: Record<string, unknown>,
	key: string,
	missingCode: ContractErrorCode,
	invalidCode: ContractErrorCode,
): RetrievalMethod {
	const value = record[key];
	if (value === undefined) throw new ContractValidationError(missingCode, `${key} is required.`, key);
	if (!isRetrievalMethod(value))
		throw new ContractValidationError(invalidCode, `${key} is not a supported retrieval method.`, key);
	return value;
}

function requiredTimestamp(
	record: Record<string, unknown>,
	key: string,
	missingCode: ContractErrorCode,
	invalidCode: ContractErrorCode,
): string {
	const value = record[key];
	if (value === undefined) throw new ContractValidationError(missingCode, `${key} is required.`, key);
	if (!isTimestamp(value)) throw new ContractValidationError(invalidCode, `${key} must be a valid timestamp.`, key);
	return value;
}

function isRetrievalMethod(value: unknown): value is RetrievalMethod {
	return typeof value === "string" && (RETRIEVAL_METHODS as readonly string[]).includes(value);
}

function isNonEmptyString(value: unknown): value is string {
	return typeof value === "string" && value.trim().length > 0;
}

function isTimestamp(value: unknown): value is string {
	return isNonEmptyString(value) && Number.isFinite(Date.parse(value));
}

function isNonNegativeInteger(value: unknown): value is number {
	return typeof value === "number" && Number.isInteger(value) && value >= 0;
}

function isPositiveInteger(value: unknown): value is number {
	return typeof value === "number" && Number.isInteger(value) && value > 0;
}

function succeeds(operation: () => unknown): boolean {
	try {
		operation();
		return true;
	} catch {
		return false;
	}
}

function safeParse<T>(operation: () => T): T | undefined {
	try {
		return operation();
	} catch {
		return undefined;
	}
}

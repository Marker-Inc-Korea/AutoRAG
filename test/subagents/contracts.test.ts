import { describe, expect, it } from "vitest";
import {
	type ContractValidationError,
	type EvidenceCandidate,
	type ExplorerAssignment,
	ExplorerExhaustionError,
	parseEvidenceCandidate,
	parseExhaustionError,
	parseExplorerAssignment,
	parseExplorerReport,
	parseFollowUpRequest,
	parseJudgeDecision,
} from "../../src/subagents/contracts.ts";

const assignment: ExplorerAssignment = {
	originalQuery: "What is the renewal policy?",
	method: "bm25",
	queryVariant: "renewal policy cancellation terms",
	queryVariants: ["What is the renewal policy?", "renewal policy cancellation terms"],
};

const evidence: EvidenceCandidate = {
	source: "/docs/renewal-policy.md",
	method: "bm25",
	evidence: "Customers may cancel within thirty days of renewal.",
	retrievedAt: "2026-07-13T00:00:00.000Z",
	sourceTemporal: {
		status: "asOf",
		asOf: "2026-06-30T00:00:00.000Z",
	},
};

describe("subagent protocol contracts", () => {
	it("accepts complete explorer protocol", () => {
		expect(parseExplorerAssignment(assignment)).toMatchObject({
			originalQuery: assignment.originalQuery,
			method: assignment.method,
			queryVariant: assignment.queryVariant,
			queryVariants: assignment.queryVariants,
		});

		const parsedEvidence = parseEvidenceCandidate(evidence);
		expect(parsedEvidence).toMatchObject({
			source: evidence.source,
			method: evidence.method,
			evidence: evidence.evidence,
			retrievedAt: evidence.retrievedAt,
			sourceTemporal: evidence.sourceTemporal,
		});

		const report = parseExplorerReport({
			assignment,
			evidenceCandidates: [evidence],
			summary: "The policy allows cancellation within thirty days.",
		});
		expect(report.evidenceCandidates).toHaveLength(1);
		expect(report.evidenceCandidates[0]).toEqual(parsedEvidence);

		const followUp = parseFollowUpRequest({
			reason: "Need the effective date for the cited policy.",
			assignments: [
				{
					originalQuery: assignment.originalQuery,
					method: "minsync",
					queryVariant: "policy effective date",
				},
			],
		});
		const decision = parseJudgeDecision({
			decision: "follow_up",
			rationale: "The cancellation rule is grounded, but its effective date is missing.",
			followUp,
		});
		expect(decision.decision).toBe("follow_up");
		expect(decision.followUp?.assignments[0]?.method).toBe("minsync");

		const exhaustion = new ExplorerExhaustionError({
			originalQuery: assignment.originalQuery,
			iterations: 3,
			maxIterations: 3,
			reason: "No additional grounded evidence was found.",
		});
		expect(parseExhaustionError(exhaustion.toJSON())).toMatchObject({
			code: "EXPLORER_EXHAUSTED",
			originalQuery: assignment.originalQuery,
			iterations: 3,
			maxIterations: 3,
		});
	});

	it("rejects missing assignment fields with stable error codes", () => {
		for (const [field, payload, code] of [
			["originalQuery", { ...assignment, originalQuery: undefined }, "MISSING_ORIGINAL_QUERY"],
			["method", { ...assignment, method: undefined }, "MISSING_RETRIEVAL_METHOD"],
			["queryVariant", { ...assignment, queryVariant: undefined }, "MISSING_QUERY_VARIANT"],
		] as const) {
			expect(() => parseExplorerAssignment(payload), field).toThrowError(expect.objectContaining({ code }));
		}
	});

	it("rejects missing evidence and temporal metadata", () => {
		expect(() => parseEvidenceCandidate({ ...evidence, source: "" })).toThrowError(
			expect.objectContaining<Partial<ContractValidationError>>({ code: "INVALID_SOURCE" }),
		);
		expect(() => parseEvidenceCandidate({ ...evidence, evidence: undefined })).toThrowError(
			expect.objectContaining<Partial<ContractValidationError>>({ code: "MISSING_EVIDENCE" }),
		);
		expect(() => parseEvidenceCandidate({ ...evidence, retrievedAt: undefined })).toThrowError(
			expect.objectContaining<Partial<ContractValidationError>>({ code: "MISSING_RETRIEVED_AT" }),
		);
		expect(() => parseEvidenceCandidate({ ...evidence, sourceTemporal: undefined })).toThrowError(
			expect.objectContaining<Partial<ContractValidationError>>({ code: "MISSING_SOURCE_TEMPORAL" }),
		);
	});

	it("requires explicit unknown temporal status when source dates are unavailable", () => {
		const parsed = parseEvidenceCandidate({
			...evidence,
			sourceTemporal: { status: "unknown" },
		});
		expect(parsed.sourceTemporal).toEqual({ status: "unknown" });

		expect(() =>
			parseEvidenceCandidate({
				...evidence,
				sourceTemporal: { status: "asOf" },
			}),
		).toThrowError(expect.objectContaining({ code: "MISSING_SOURCE_TEMPORAL_DATE" }));
	});

	it("rejects malformed judge follow-up and exhaustion payloads", () => {
		expect(() =>
			parseJudgeDecision({
				decision: "follow_up",
				rationale: "Needs more evidence",
			}),
		).toThrowError(expect.objectContaining({ code: "MISSING_FOLLOW_UP" }));

		expect(() => parseFollowUpRequest({ reason: "missing assignments", assignments: [] })).toThrowError(
			expect.objectContaining({ code: "EMPTY_FOLLOW_UP_ASSIGNMENTS" }),
		);

		expect(() => parseExhaustionError({ code: "EXPLORER_EXHAUSTED" })).toThrowError(
			expect.objectContaining({ code: "MISSING_EXHAUSTION_QUERY" }),
		);
	});
});

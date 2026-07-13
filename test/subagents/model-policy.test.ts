import { describe, expect, it } from "vitest";
import {
	EXPLORER_MODEL_ID,
	type ModelPolicyError,
	ORCHESTRATOR_MODEL_ID,
	validateModelPolicy,
} from "../../src/subagents/model-policy.ts";

const orchestrator = { provider: "openai", id: ORCHESTRATOR_MODEL_ID };
const explorer = { provider: "openai", id: EXPLORER_MODEL_ID };

describe("subagent model policy", () => {
	it("accepts required role models", () => {
		const policy = validateModelPolicy({ orchestrator, explorer });

		expect(policy.orchestrator.id).toBe("gpt-5.6-sol");
		expect(policy.explorer.id).toBe("gpt-5.6-luna");
	});

	it("rejects swapped role models", () => {
		expect(() => validateModelPolicy({ orchestrator: explorer, explorer: orchestrator })).toThrowError(
			expect.objectContaining<Partial<ModelPolicyError>>({ code: "MODEL_ROLE_MISMATCH" }),
		);
	});

	it("fails closed for other and missing model ids", () => {
		expect(() => validateModelPolicy({ orchestrator: { id: "gpt-5.6-luna" }, explorer })).toThrowError(
			expect.objectContaining({ code: "MODEL_ROLE_MISMATCH" }),
		);
		expect(() => validateModelPolicy({ orchestrator, explorer: { id: "gpt-5.5" } })).toThrowError(
			expect.objectContaining({ code: "MODEL_ROLE_MISMATCH" }),
		);
		expect(() => validateModelPolicy({ orchestrator })).toThrowError(
			expect.objectContaining({ code: "MISSING_REQUIRED_MODEL" }),
		);
		expect(() => validateModelPolicy({ orchestrator: undefined, explorer })).toThrowError(
			expect.objectContaining({ code: "MISSING_REQUIRED_MODEL" }),
		);
	});

	it("rejects non-object model values instead of substituting them", () => {
		expect(() => validateModelPolicy({ orchestrator: ORCHESTRATOR_MODEL_ID, explorer })).toThrowError(
			expect.objectContaining({ code: "INVALID_MODEL_REFERENCE" }),
		);
		expect(() => validateModelPolicy({ orchestrator, explorer: null })).toThrowError(
			expect.objectContaining({ code: "INVALID_MODEL_REFERENCE" }),
		);
	});
});

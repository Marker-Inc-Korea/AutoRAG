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

	it("accepts configurable model references for both roles", () => {
		const policy = validateModelPolicy({
			orchestrator: { provider: "custom-orchestrator-provider", id: "custom-orchestrator" },
			explorer: { provider: "custom-explorer-provider", id: "custom-explorer" },
		});

		expect(policy.orchestrator).toEqual({
			provider: "custom-orchestrator-provider",
			id: "custom-orchestrator",
		});
		expect(policy.explorer).toEqual({ provider: "custom-explorer-provider", id: "custom-explorer" });
	});

	it("fails closed for missing role references", () => {
		expect(() => validateModelPolicy({ orchestrator })).toThrowError(
			expect.objectContaining<Partial<ModelPolicyError>>({ code: "MISSING_REQUIRED_MODEL" }),
		);
		expect(() => validateModelPolicy({ orchestrator: undefined, explorer })).toThrowError(
			expect.objectContaining<Partial<ModelPolicyError>>({ code: "MISSING_REQUIRED_MODEL" }),
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

export const ORCHESTRATOR_MODEL_ID = "gpt-5.6-sol" as const;
export const EXPLORER_MODEL_ID = "gpt-5.6-luna" as const;

export const REQUIRED_MODEL_IDS = Object.freeze({
	orchestrator: ORCHESTRATOR_MODEL_ID,
	explorer: EXPLORER_MODEL_ID,
});

export type ModelRole = keyof typeof REQUIRED_MODEL_IDS;

export interface ModelReference {
	readonly id: string;
	readonly provider?: string;
}

export interface ValidatedModelPolicy {
	readonly orchestrator: ModelReference;
	readonly explorer: ModelReference;
}

export type ModelPolicyErrorCode = "INVALID_POLICY_OBJECT" | "MISSING_REQUIRED_MODEL" | "INVALID_MODEL_REFERENCE";

export class ModelPolicyError extends Error {
	readonly code: ModelPolicyErrorCode;
	readonly role?: ModelRole;
	readonly expected?: string;
	readonly actual?: string;

	constructor(
		code: ModelPolicyErrorCode,
		message: string,
		options: { readonly role?: ModelRole; readonly expected?: string; readonly actual?: string } = {},
	) {
		super(message);
		this.name = "ModelPolicyError";
		this.code = code;
		this.role = options.role;
		this.expected = options.expected;
		this.actual = options.actual;
	}
}

export function validateModelPolicy(value: unknown): ValidatedModelPolicy {
	if (!isRecord(value)) {
		throw new ModelPolicyError("INVALID_POLICY_OBJECT", "Model policy must be an object.");
	}

	const orchestrator = readRequiredModel(value, "orchestrator");
	const explorer = readRequiredModel(value, "explorer");
	return { orchestrator, explorer };
}

export function assertModelPolicy(value: unknown): asserts value is ValidatedModelPolicy {
	validateModelPolicy(value);
}

export function isModelPolicy(value: unknown): value is ValidatedModelPolicy {
	try {
		validateModelPolicy(value);
		return true;
	} catch {
		return false;
	}
}

export function requiredModelId(role: ModelRole): string {
	return REQUIRED_MODEL_IDS[role];
}

function readRequiredModel(record: Record<string, unknown>, role: ModelRole): ModelReference {
	const value = record[role];
	if (value === undefined) {
		throw new ModelPolicyError("MISSING_REQUIRED_MODEL", `Required ${role} model is missing.`, { role });
	}
	if (!isModelReference(value)) {
		throw new ModelPolicyError("INVALID_MODEL_REFERENCE", `${role} model must be an object with a non-empty id.`, {
			role,
		});
	}
	return value;
}

function isModelReference(value: unknown): value is ModelReference {
	return isRecord(value) && typeof value.id === "string" && value.id.trim().length > 0;
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

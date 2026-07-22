/**
 * Validation-coupled dispatch core for AutoRAG subagent invocations.
 *
 * Implements the frozen contracts A–J from the approved plan:
 * - A) Classification before defaults + first-error precedence
 * - B) Diagnostic ownership (exact field sets per action)
 * - C) Literal error format + exactFix + forceCorrectable + skeletons
 * - D) Pre-schema validation table (prepareArguments, launch kind only)
 * - G) V1/legacy assignment parser + canonical role lines (byte-idempotent)
 *
 * Pre-schema validation (autoragPrepare) is synchronous and side-effect-free
 * except for the `DispatchRejectionError` it throws. Post-schema validation
 * (validateLaunchPostSchema) intentionally uses filesystem realpath/stat for
 * cwd containment — it resolves and verifies each leaf cwd against the
 * configured allowed roots. Neither path touches the network.

 */
import { realpathSync, statSync } from "node:fs";
import { resolve } from "node:path";
import {
	DISPATCH_ADMIN_SKELETON as TMPL_ADMIN_SKELETON,
	DISPATCH_ERROR_REFERENCE as TMPL_ERROR_REFERENCE,
	DISPATCH_FANOUT_SKELETON as TMPL_FANOUT_SKELETON,
	AUTORAG_ROLE_LINE_RETRIEVED_AT as TMPL_ROLE_RETRIEVED_AT,
	AUTORAG_ROLE_LINE_TEMPORAL_METADATA as TMPL_ROLE_TEMPORAL,
	DISPATCH_SINGLE_SKELETON as TMPL_SINGLE_SKELETON,
	AUTORAG_ASSIGNMENT_V1_END as TMPL_V1_END,
	AUTORAG_ASSIGNMENT_V1_START as TMPL_V1_START,
} from "./dispatch-templates.ts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type DispatchRejectionCode =
	| "DISPATCH_MALFORMED"
	| "DISPATCH_ACTION_UNKNOWN"
	| "DISPATCH_ADMIN_MUTATION_FORBIDDEN"
	| "DISPATCH_CONTROL_FORBIDDEN"
	| "DISPATCH_SCHEDULE_FORBIDDEN"
	| "DISPATCH_ARTIFACTS_INVALID"
	| "DISPATCH_AGENT_SCOPE_INVALID"
	| "DISPATCH_AGENT_IDENTITY"
	| "DISPATCH_MODEL_MISMATCH"
	| "DISPATCH_ASSIGNMENT_INVALID"
	| "DISPATCH_QUERY_MISMATCH"
	| "DISPATCH_CWD_MISSING"
	| "DISPATCH_CWD_OUTSIDE_ROOTS"
	| "DISPATCH_NO_ACTIVE_QUERY";

export type DispatchKind = "launch" | "admin" | "control" | "mutation" | "schedule" | "hybrid" | "unknown";

export interface DispatchRejection {
	readonly code: DispatchRejectionCode;
	readonly field: string;
	readonly dispatchKind: DispatchKind;
}

export class DispatchRejectionError extends Error {
	readonly code: DispatchRejectionCode;
	readonly field: string;
	readonly exactFix: string;
	readonly forceCorrectable: boolean;
	readonly skeleton: string;
	readonly dispatchKind: DispatchKind;

	constructor(rejection: DispatchRejection, exactFix: string, forceCorrectable: boolean, skeleton: string) {
		const reason = formatBlockReason(rejection.code, rejection.field, exactFix, skeleton);
		super(reason);
		this.name = "DispatchRejectionError";
		this.code = rejection.code;
		this.field = rejection.field;
		this.exactFix = exactFix;
		this.forceCorrectable = forceCorrectable;
		this.skeleton = skeleton;
		this.dispatchKind = rejection.dispatchKind;
	}
}

// ---------------------------------------------------------------------------
// C) Error catalog: exactFix, forceCorrectable, skeletons
//
// EXACT_FIX_MAP and FORCE_CORRECTABLE_MAP are derived from the single
// DISPATCH_ERROR_REFERENCE table in dispatch-templates.ts so no dual catalog
// exists. The DispatchRejectionCode union above is the compile-time key set;
// the reference table is the runtime value source.
// ---------------------------------------------------------------------------

function buildExactFixMap(): Readonly<Record<DispatchRejectionCode, string>> {
	const map = {} as Record<DispatchRejectionCode, string>;
	for (const entry of TMPL_ERROR_REFERENCE) {
		map[entry.code as DispatchRejectionCode] = entry.exactFix;
	}
	return map;
}

function buildForceCorrectableMap(): Readonly<Record<DispatchRejectionCode, boolean>> {
	const map = {} as Record<DispatchRejectionCode, boolean>;
	for (const entry of TMPL_ERROR_REFERENCE) {
		map[entry.code as DispatchRejectionCode] = entry.forceCorrectable;
	}
	return map;
}

export const EXACT_FIX_MAP: Readonly<Record<DispatchRejectionCode, string>> = buildExactFixMap();
export const FORCE_CORRECTABLE_MAP: Readonly<Record<DispatchRejectionCode, boolean>> = buildForceCorrectableMap();

/** Backward-compatible aliases for the shared skeleton literals from dispatch-templates.ts. */
export const SKELETON_ADMIN = TMPL_ADMIN_SKELETON;
export const SKELETON_SINGLE = TMPL_SINGLE_SKELETON;
export const SKELETON_FANOUT = TMPL_FANOUT_SKELETON;

export function formatBlockReason(
	code: DispatchRejectionCode,
	field: string,
	exactFix: string,
	skeleton: string,
): string {
	return `[${code}] field=${field} fix=${exactFix}\n${skeleton}`;
}

// ---------------------------------------------------------------------------
// A) Action classification
// ---------------------------------------------------------------------------

const DIAGNOSTIC_ACTIONS = new Set(["list", "get", "models", "status", "doctor"]);
const MUTATION_ACTIONS = new Set(["create", "update", "delete", "eject", "disable", "enable", "reset"]);
const CONTROL_ACTIONS = new Set(["interrupt", "resume", "steer", "append-step"]);
const SCHEDULE_ACTIONS = new Set(["schedule", "schedule-list", "schedule-status", "schedule-cancel"]);

const EXECUTION_FIELD_KEYS = new Set(["task", "tasks", "chain", "parallel"]);

/** Fields that indicate a launch shape even without an explicit action. */
const LAUNCH_SHAPE_KEYS = new Set(["agent", "task", "tasks", "chain", "parallel"]);

/** Exact lowercase action values that are explicit execution-mode aliases. */
const EXECUTION_ACTION_ALIASES = new Set(["single", "parallel", "tasks"]);

export function classifyAction(action: unknown): DispatchKind {
	if (typeof action !== "string") return "unknown";
	if (MUTATION_ACTIONS.has(action)) return "mutation";
	if (CONTROL_ACTIONS.has(action)) return "control";
	if (SCHEDULE_ACTIONS.has(action)) return "schedule";
	if (DIAGNOSTIC_ACTIONS.has(action)) return "admin";
	// Not a recognized mutation/control/schedule/diagnostic action.
	// classifyDispatch handles execution aliases separately; classifyAction
	// itself returns unknown for all unrecognized strings.
	return "unknown";
}

/**
 * Classify a full dispatch invocation into a dispatch kind.
 *
 * Classification happens BEFORE any defaults or autofill.
 * - If `action` is present and is a mutation/control/schedule → that kind (forbidden-action precedence).
 * - If `action` is present and is a diagnostic (exact lowercase) → admin.
 * - If `action` is present and is an explicit execution alias (exact lowercase: single/parallel/tasks) with launch shape → launch.
 * - If `action` is present but is non-string, unknown, or a non-exact-lowercase value
 *   (e.g. "Create", "LIST", "Interrupt", 42) → unknown, regardless of launch fields.
 * - If no `action` property but launch shape fields present (agent/task/tasks/chain/parallel) → launch.
 * - Otherwise → unknown.
 */
export function classifyDispatch(args: unknown): DispatchKind {
	if (!isRecord(args)) return "unknown";

	// When an `action` property is present, it fully determines the kind.
	// Non-string or any unrecognized/non-exact-lowercase value is unknown
	// regardless of launch fields — case variants and numeric actions must
	// never fall through to launch.
	if (Object.hasOwn(args, "action")) {
		const action = args.action;
		if (typeof action !== "string") return "unknown";
		const kind = classifyAction(action);
		if (kind === "mutation" || kind === "control" || kind === "schedule" || kind === "admin") {
			return kind;
		}
		// Explicit supported execution aliases classify as launch when
		// matching launch shape fields are present; otherwise unknown.
		if (EXECUTION_ACTION_ALIASES.has(action) && hasLaunchShape(args)) return "launch";
		// Any other string (unknown, case variant, etc.) → unknown.
		return "unknown";
	}

	// No action property: check for launch shape fields.
	if (hasLaunchShape(args)) return "launch";
	return "unknown";
}

function hasLaunchShape(args: Record<string, unknown>): boolean {
	for (const key of LAUNCH_SHAPE_KEYS) {
		if (Object.hasOwn(args, key)) return true;
	}
	return false;
}

// ---------------------------------------------------------------------------
// B) Diagnostic ownership (exact field sets per action)
// ---------------------------------------------------------------------------

const DIAGNOSTIC_ALLOWED_FIELDS: Readonly<Record<string, ReadonlySet<string>>> = {
	list: new Set(["action", "agentScope"]),
	get: new Set(["action", "agent", "chainName"]),
	models: new Set(["action", "agent"]),
	status: new Set(["action", "id", "runId", "dir", "index", "view", "lines"]),
	doctor: new Set(["action", "cwd", "context", "sessionDir"]),
};

/**
 * Validate a diagnostic invocation against its exact allowed field set.
 * Returns a DispatchRejection if invalid, undefined if valid.
 * Diagnostics bypass explorer assignment/root/model launch gates.
 */
export function validateDiagnostic(args: Record<string, unknown>, action: string): DispatchRejection | undefined {
	const allowed = DIAGNOSTIC_ALLOWED_FIELDS[action];
	if (allowed === undefined) return undefined; // not a diagnostic

	// Check for execution fields on diagnostic → MALFORMED
	for (const key of EXECUTION_FIELD_KEYS) {
		if (Object.hasOwn(args, key)) {
			return { code: "DISPATCH_MALFORMED", field: key, dispatchKind: "admin" };
		}
	}

	// Check for fields not in the allowed set
	for (const key of Object.keys(args)) {
		if (!allowed.has(key)) {
			return { code: "DISPATCH_MALFORMED", field: key, dispatchKind: "admin" };
		}
	}

	// Action-specific constraints
	if (action === "list") {
		if (Object.hasOwn(args, "agentScope") && args.agentScope !== "user") {
			return { code: "DISPATCH_AGENT_SCOPE_INVALID", field: "agentScope", dispatchKind: "admin" };
		}
	}
	if (action === "get") {
		const hasAgent = Object.hasOwn(args, "agent");
		const hasChainName = Object.hasOwn(args, "chainName");
		if (hasAgent && hasChainName) {
			return { code: "DISPATCH_MALFORMED", field: "agent", dispatchKind: "admin" };
		}
		if (!hasAgent && !hasChainName) {
			return { code: "DISPATCH_MALFORMED", field: "args", dispatchKind: "admin" };
		}
		if (Object.hasOwn(args, "level")) {
			return { code: "DISPATCH_MALFORMED", field: "level", dispatchKind: "admin" };
		}
	}

	return undefined;
}

// ---------------------------------------------------------------------------
// Dispatch rejection helpers (mutation/control/schedule/unknown)
// ---------------------------------------------------------------------------

export function rejectForbiddenAction(action: string): DispatchRejection {
	const kind = classifyAction(action);
	if (kind === "mutation") {
		return { code: "DISPATCH_ADMIN_MUTATION_FORBIDDEN", field: "action", dispatchKind: "mutation" };
	}
	if (kind === "control") {
		return { code: "DISPATCH_CONTROL_FORBIDDEN", field: "action", dispatchKind: "control" };
	}
	if (kind === "schedule") {
		return { code: "DISPATCH_SCHEDULE_FORBIDDEN", field: "action", dispatchKind: "schedule" };
	}
	return { code: "DISPATCH_ACTION_UNKNOWN", field: "action", dispatchKind: "unknown" };
}

// ---------------------------------------------------------------------------
// D) Pre-schema validation table (launch kind only, after classify)
// ---------------------------------------------------------------------------

export interface PrepareContext {
	/** Configured explorer model wire id (e.g. "provider/model-id"). */
	readonly configuredModel: string | undefined;
}
function normalizeConfiguredModelReference(model: string, configuredModel: string | undefined): string | undefined {
	if (configuredModel === undefined) return model;
	if (model === configuredModel) return configuredModel;
	if (configuredModel.endsWith(`/${model}`)) return configuredModel;
	return undefined;
}

export interface PrepareResult {
	/** Normalized args (defaults filled for launch, null-deleted). */
	readonly args: Record<string, unknown>;
	/** Whether any field was autofilled. */
	readonly autofilled: {
		readonly artifacts: boolean;
		readonly agentScope: boolean;
		readonly leafModelFillCount: number;
	};
}

/**
 * Composed prepareArguments for the subagent tool.
 *
 * Classifies first, then applies launch-only normalization (null-delete + missing defaults).
 * Diagnostics never receive artifacts/agentScope/model defaults.
 * Throws DispatchRejectionError on any validation failure.
 */
export function autoragPrepare(rawArgs: unknown, ctx: PrepareContext): Record<string, unknown> {
	if (!isRecord(rawArgs)) {
		throw createRejection({ code: "DISPATCH_MALFORMED", field: "args", dispatchKind: "unknown" }, "admin");
	}

	const kind = classifyDispatch(rawArgs);

	// Step 1: unknown/forbidden action (precedence #1)
	if (kind === "unknown" || kind === "mutation" || kind === "control" || kind === "schedule") {
		const action = rawArgs.action;
		if (typeof action === "string") {
			throw createRejection(rejectForbiddenAction(action), selectSkeletonForKind(kind));
		}
		// No action but also no launch shape → unknown
		throw createRejection({ code: "DISPATCH_ACTION_UNKNOWN", field: "action", dispatchKind: "unknown" }, "admin");
	}

	// Step 2: diagnostic validation (precedence #2 — diagnostic field checks)
	if (kind === "admin") {
		const action = rawArgs.action as string;
		const rejection = validateDiagnostic(rawArgs, action);
		if (rejection !== undefined) {
			throw createRejection(rejection, "admin");
		}
		// Diagnostics pass through without autofill
		return { ...rawArgs };
	}

	// kind === "launch" — deep-clone args so nested autofill/normalization
	// never mutates the caller's raw nested objects. The WeakMap correlation
	// in the agent layer keys on the original rawArgs object, which is
	// preserved because we clone here before any mutation.
	return prepareLaunch(structuredClone(rawArgs), ctx);
}

function prepareLaunch(args: Record<string, unknown>, ctx: PrepareContext): Record<string, unknown> {
	const result: Record<string, unknown> = { ...args };
	let artifactsAutofilled = false;
	let agentScopeAutofilled = false;
	let leafModelFillCount = 0;

	// Detect skeleton kind early so all errors in prepareLaunch get the right skeleton,
	// even when the failure occurs before leaf traversal.
	const skeletonKind = detectSkeletonKind(result);

	// Step 3: top-level type/enum (artifacts, agentScope, malformed containers)

	// artifacts: boolean or missing/null; null deleted then set false; missing → false;
	// wrong type or explicit non-false invalid
	if (Object.hasOwn(result, "artifacts")) {
		const artifacts = result.artifacts;
		if (artifacts === null) {
			delete result.artifacts;
			result.artifacts = false;
			artifactsAutofilled = true;
		} else if (typeof artifacts === "boolean") {
			if (artifacts !== false) {
				throw createRejection(
					{ code: "DISPATCH_ARTIFACTS_INVALID", field: "artifacts", dispatchKind: "launch" },
					skeletonKind,
				);
			}
		} else {
			throw createRejection(
				{ code: "DISPATCH_ARTIFACTS_INVALID", field: "artifacts", dispatchKind: "launch" },
				skeletonKind,
			);
		}
	} else {
		result.artifacts = false;
		artifactsAutofilled = true;
	}

	// agentScope: string or missing/null; null deleted then "user"; missing → "user";
	// wrong type or ≠ "user" invalid
	if (Object.hasOwn(result, "agentScope")) {
		const agentScope = result.agentScope;
		if (agentScope === null) {
			delete result.agentScope;
			result.agentScope = "user";
			agentScopeAutofilled = true;
		} else if (typeof agentScope === "string") {
			if (agentScope !== "user") {
				throw createRejection(
					{ code: "DISPATCH_AGENT_SCOPE_INVALID", field: "agentScope", dispatchKind: "launch" },
					skeletonKind,
				);
			}
		} else {
			throw createRejection(
				{ code: "DISPATCH_AGENT_SCOPE_INVALID", field: "agentScope", dispatchKind: "launch" },
				skeletonKind,
			);
		}
	} else {
		result.agentScope = "user";
		agentScopeAutofilled = true;
	}

	// Validate container types (tasks/chain/parallel) — must match upstream structural types
	for (const key of ["tasks", "chain", "parallel"] as const) {
		if (Object.hasOwn(result, key)) {
			const value = result[key];
			if (key === "tasks" || key === "parallel") {
				if (!Array.isArray(value)) {
					throw createRejection({ code: "DISPATCH_MALFORMED", field: key, dispatchKind: "launch" }, skeletonKind);
				}
				if (value.length === 0) {
					throw createRejection({ code: "DISPATCH_MALFORMED", field: key, dispatchKind: "launch" }, skeletonKind);
				}
			}
			if (key === "chain") {
				if (!Array.isArray(value)) {
					throw createRejection({ code: "DISPATCH_MALFORMED", field: key, dispatchKind: "launch" }, skeletonKind);
				}
				if (value.length === 0) {
					throw createRejection({ code: "DISPATCH_MALFORMED", field: key, dispatchKind: "launch" }, skeletonKind);
				}
			}
		}
	}

	// Step 4a: malformed container leaves — a container entry with executable
	// shape (task/model/cwd) but no `agent` must be rejected before any child
	// can execute. Non-record container entries are DISPATCH_MALFORMED.
	const malformedLeaf = findMalformedContainerLeaf(result);
	if (malformedLeaf !== undefined) {
		throw createRejection(malformedLeaf, skeletonKind);
	}

	// Step 4b: launch shape / leaf existence
	const leaves = collectLaunchLeaves(result);
	if (leaves.length === 0) {
		throw createRejection({ code: "DISPATCH_MALFORMED", field: "args", dispatchKind: "launch" }, skeletonKind);
	}
	// Step 5: leaf identity / model (traversal order: root → tasks[i] → chain[j] → parallel[k])

	// Top-level model on a single-root launch (root is the only leaf):
	// string nonblank or missing/null; missing/null filled with configured;
	// wrong type or blank → MODEL_MISMATCH (blank is NOT autofilled per plan).
	const isSingleRoot = leaves.length === 1 && leaves[0].path === "";
	if (isSingleRoot && Object.hasOwn(result, "model")) {
		const model = result.model;
		if (model === null) {
			delete result.model;
			if (ctx.configuredModel !== undefined) {
				result.model = ctx.configuredModel;
				leafModelFillCount += 1;
			}
		} else if (typeof model === "string") {
			const trimmedModel = model.trim();
			const normalizedModel =
				trimmedModel.length === 0
					? undefined
					: normalizeConfiguredModelReference(trimmedModel, ctx.configuredModel);
			if (normalizedModel === undefined) {
				throw createRejection(
					{ code: "DISPATCH_MODEL_MISMATCH", field: ".model", dispatchKind: "launch" },
					skeletonKind,
				);
			}
			if (normalizedModel !== model) {
				result.model = normalizedModel;
				leafModelFillCount += 1;
			}
		} else {
			throw createRejection(
				{ code: "DISPATCH_MODEL_MISMATCH", field: ".model", dispatchKind: "launch" },
				skeletonKind,
			);
		}
	} else if (isSingleRoot && !Object.hasOwn(result, "model")) {
		if (ctx.configuredModel !== undefined) {
			result.model = ctx.configuredModel;
			leafModelFillCount += 1;
		}
	}

	// For fanout launches, the top-level model (if present) is an envelope default,
	// not a leaf fill — it should NOT count toward leafModelFillCount.
	// Leaf-level validation: agent, model, task, cwd
	for (let i = 0; i < leaves.length; i++) {
		const leaf = leaves[i];
		const leafPath = leaf.path;
		const isRoot = leafPath === "";

		// Nested artifacts/agentScope on NON-ROOT leaves: if present → MALFORMED (never inject)
		// Root's top-level artifacts/agentScope are legal envelope fields.
		if (!isRoot) {
			if (Object.hasOwn(leaf.record, "artifacts")) {
				throw createRejection(
					{ code: "DISPATCH_MALFORMED", field: `${leafPath}.artifacts`, dispatchKind: "launch" },
					skeletonKind,
				);
			}
			if (Object.hasOwn(leaf.record, "agentScope")) {
				throw createRejection(
					{ code: "DISPATCH_MALFORMED", field: `${leafPath}.agentScope`, dispatchKind: "launch" },
					skeletonKind,
				);
			}
		}

		// agent: string nonblank required; must equal "autorag-explorer"
		if (typeof leaf.record.agent !== "string" || leaf.record.agent.trim().length === 0) {
			throw createRejection(
				{ code: "DISPATCH_AGENT_IDENTITY", field: `${leafPath}.agent`, dispatchKind: "launch" },
				skeletonKind,
			);
		}
		if (leaf.record.agent !== "autorag-explorer") {
			throw createRejection(
				{ code: "DISPATCH_AGENT_IDENTITY", field: `${leafPath}.agent`, dispatchKind: "launch" },
				skeletonKind,
			);
		}

		// model: string nonblank or missing/null; missing/null filled with configured;
		// wrong type or blank after presence → MODEL_MISMATCH (blank is NOT autofilled)
		if (isRoot) {
			// Root model was already handled above for single-root.
			// For fanout, root is not a leaf with model — skip.
		} else if (Object.hasOwn(leaf.record, "model")) {
			const model = leaf.record.model;
			if (model === null) {
				delete leaf.record.model;
				if (ctx.configuredModel !== undefined) {
					leaf.record.model = ctx.configuredModel;
					leafModelFillCount += 1;
				}
			} else if (typeof model === "string") {
				const trimmedModel = model.trim();
				const normalizedModel =
					trimmedModel.length === 0
						? undefined
						: normalizeConfiguredModelReference(trimmedModel, ctx.configuredModel);
				if (normalizedModel === undefined) {
					throw createRejection(
						{ code: "DISPATCH_MODEL_MISMATCH", field: `${leafPath}.model`, dispatchKind: "launch" },
						skeletonKind,
					);
				}
				if (normalizedModel !== model) {
					leaf.record.model = normalizedModel;
					leafModelFillCount += 1;
				}
			} else {
				throw createRejection(
					{ code: "DISPATCH_MODEL_MISMATCH", field: `${leafPath}.model`, dispatchKind: "launch" },
					skeletonKind,
				);
			}
		} else {
			// missing: fill from configured model
			if (ctx.configuredModel !== undefined) {
				leaf.record.model = ctx.configuredModel;
				leafModelFillCount += 1;
			}
		}

		// task: string nonblank required
		if (typeof leaf.record.task !== "string" || leaf.record.task.trim().length === 0) {
			throw createRejection(
				{ code: "DISPATCH_ASSIGNMENT_INVALID", field: `${leafPath}.task`, dispatchKind: "launch" },
				skeletonKind,
			);
		}

		// cwd: string nonblank required
		if (typeof leaf.record.cwd !== "string" || leaf.record.cwd.trim().length === 0) {
			throw createRejection(
				{ code: "DISPATCH_CWD_MISSING", field: `${leafPath}.cwd`, dispatchKind: "launch" },
				skeletonKind,
			);
		}
	}

	// Store autofill info as non-enumerable metadata for the caller to read
	Object.defineProperty(result, "__autofilled", {
		value: {
			artifacts: artifactsAutofilled,
			agentScope: agentScopeAutofilled,
			leafModelFillCount,
		},
		enumerable: false,
		writable: false,
		configurable: false,
	});

	return result;
}

export function readAutofilled(args: Record<string, unknown>):
	| {
			readonly artifacts: boolean;
			readonly agentScope: boolean;
			readonly leafModelFillCount: number;
	  }
	| undefined {
	const meta = (args as Record<string, unknown> & { __autofilled?: unknown }).__autofilled;
	if (typeof meta === "object" && meta !== null) {
		return meta as { artifacts: boolean; agentScope: boolean; leafModelFillCount: number };
	}
	return undefined;
}

// ---------------------------------------------------------------------------
// Leaf traversal
// ---------------------------------------------------------------------------

interface LaunchLeaf {
	readonly record: Record<string, unknown>;
	readonly path: string;
}

/**
 * Detect a container entry that looks like an executable leaf but is missing
 * the required `agent` identity, so it would be silently ignored by leaf
 * traversal and allowed to pass.
 *
 * A record inside a tasks/chain/parallel container that has no `agent` field
 * and no nested container is a malformed leaf: it carries executable shape
 * (e.g. task/model/cwd) but no agent identity. Reject it deterministically
 * with DISPATCH_AGENT_IDENTITY before any child can execute.
 *
 * Non-record container entries (strings, numbers, arrays) are rejected with
 * DISPATCH_MALFORMED.
 *
 * Returns a DispatchRejection for the first malformed entry in traversal
 * order, or undefined when every container entry is well-formed. The root
 * invocation itself is never flagged here (a root with no agent and no
 * containers is handled by the empty-leaves check).
 */
function findMalformedContainerLeaf(args: Record<string, unknown>): DispatchRejection | undefined {
	const containerKeys = ["tasks", "chain", "parallel"] as const;
	let result: DispatchRejection | undefined;
	const visit = (value: unknown, path: string, inContainer: boolean): void => {
		if (result !== undefined) return;
		if (!isRecord(value)) {
			if (inContainer) {
				result = { code: "DISPATCH_MALFORMED", field: path, dispatchKind: "launch" };
			}
			return;
		}
		const record = value as Record<string, unknown>;
		const hasAgent = Object.hasOwn(record, "agent");
		let hasNestedContainer = false;
		for (const key of containerKeys) {
			if (Object.hasOwn(record, key)) {
				hasNestedContainer = true;
				const nested = record[key];
				if (Array.isArray(nested)) {
					for (let i = 0; i < nested.length; i++) {
						visit(nested[i], `${path}.${key}[${i}]`, true);
						if (result !== undefined) return;
					}
				} else if (isRecord(nested)) {
					visit(nested, `${path}.${key}`, true);
					if (result !== undefined) return;
				}
			}
		}
		if (inContainer && !hasAgent && !hasNestedContainer) {
			result = { code: "DISPATCH_AGENT_IDENTITY", field: `${path}.agent`, dispatchKind: "launch" };
		}
	};
	visit(args, "", false);
	return result;
}

/**
 * Collect all executable leaves from a launch invocation in deterministic
 * traversal order: root, then tasks[i], then chain[j], then nested parallel[k] /
 * dynamic template leaves in encounter order.
 *
 * A leaf is a record with an `agent` field (executable). The root itself can be
 * a leaf (single-agent dispatch).
 */
function collectLaunchLeaves(args: Record<string, unknown>): LaunchLeaf[] {
	const leaves: LaunchLeaf[] = [];

	const traverse = (value: unknown, path: string): void => {
		if (!isRecord(value)) return;
		const record = value as Record<string, unknown>;

		// If this record has an `agent` field, it's an executable leaf
		if (Object.hasOwn(record, "agent")) {
			leaves.push({ record, path });
		}

		// Traverse nested containers in order: tasks, chain, parallel
		const containerKeys = ["tasks", "chain", "parallel"] as const;
		for (const key of containerKeys) {
			if (Object.hasOwn(record, key)) {
				const nested = record[key];
				if (Array.isArray(nested)) {
					for (let i = 0; i < nested.length; i++) {
						traverse(nested[i], `${path}.${key}[${i}]`);
					}
				} else if (isRecord(nested)) {
					traverse(nested, `${path}.${key}`);
				}
			}
		}
	};

	traverse(args, "");
	return leaves;
}

export function getLaunchLeaves(args: Record<string, unknown>): readonly LaunchLeaf[] {
	return collectLaunchLeaves(args);
}

// ---------------------------------------------------------------------------
// Skeleton selection
// ---------------------------------------------------------------------------

type SkeletonKind = "admin" | "single" | "fanout";

function selectSkeletonForKind(kind: DispatchKind): SkeletonKind {
	if (kind === "mutation" || kind === "control" || kind === "schedule" || kind === "unknown" || kind === "admin") {
		return "admin";
	}
	return "single";
}

/**
 * Detect skeleton kind from the launch args shape, independent of leaf traversal.
 * Returns "fanout" if tasks/chain/parallel containers are present, "single" otherwise.
 * This allows correct skeleton selection even when the error occurs before leaf traversal.
 */
function detectSkeletonKind(args: Record<string, unknown>): SkeletonKind {
	if (Object.hasOwn(args, "tasks") || Object.hasOwn(args, "chain") || Object.hasOwn(args, "parallel")) {
		return "fanout";
	}
	return "single";
}

function createRejection(rejection: DispatchRejection, skeletonKind: SkeletonKind): DispatchRejectionError {
	const exactFix = EXACT_FIX_MAP[rejection.code];
	const forceCorrectable = FORCE_CORRECTABLE_MAP[rejection.code];
	const skeleton =
		skeletonKind === "admin" ? SKELETON_ADMIN : skeletonKind === "single" ? SKELETON_SINGLE : SKELETON_FANOUT;
	return new DispatchRejectionError(rejection, exactFix, forceCorrectable, skeleton);
}

/**
 * Create a DispatchRejectionError from a DispatchRejection with proper
 * exactFix, forceCorrectable, and skeleton selection.
 * Used by beforeToolCall for post-schema rejections.
 *
 * When `args` is provided and the rejection is a launch kind, the skeleton is
 * selected from the args shape: fanout if `tasks`/`chain`/`parallel` are
 * present, single otherwise. When `args` is omitted, launch rejections default
 * to the single skeleton. Non-launch kinds always use the admin skeleton.
 */
export function createDispatchRejectionError(
	rejection: DispatchRejection,
	args?: Record<string, unknown>,
): DispatchRejectionError {
	const exactFix = EXACT_FIX_MAP[rejection.code];
	const forceCorrectable = FORCE_CORRECTABLE_MAP[rejection.code];
	let skeleton: string;
	if (rejection.dispatchKind === "launch") {
		if (args !== undefined && isRecord(args) && detectSkeletonKind(args) === "fanout") {
			skeleton = SKELETON_FANOUT;
		} else {
			skeleton = SKELETON_SINGLE;
		}
	} else {
		skeleton = SKELETON_ADMIN;
	}
	return new DispatchRejectionError(rejection, exactFix, forceCorrectable, skeleton);
}

// ---------------------------------------------------------------------------
// G) Assignment V1 + legacy parser
// ---------------------------------------------------------------------------

/** Backward-compatible aliases for shared V1 sentinel constants. */
export const AUTORAG_ASSIGNMENT_V1_START = TMPL_V1_START;
export const AUTORAG_ASSIGNMENT_V1_END = TMPL_V1_END;

export interface ParsedAssignment {
	readonly originalQuery: string;
	readonly method: string;
	readonly queryVariants: readonly string[];
	/** Whether V1 (true) or legacy (false) format was used. */
	readonly isV1: boolean;
}

/** Backward-compatible aliases for shared role line constants. */
export const ROLE_LINE_RETRIEVED_AT = TMPL_ROLE_RETRIEVED_AT;
export const ROLE_LINE_TEMPORAL_METADATA = TMPL_ROLE_TEMPORAL;

export function canonicalRoleLines(): string {
	return `${ROLE_LINE_RETRIEVED_AT}\n${ROLE_LINE_TEMPORAL_METADATA}`;
}

/**
 * Parse a task string as an Assignment V1 block.
 *
 * V1 sentinels:
 * - Start line (after stripping single trailing CR) must equal <<<AUTORAG_ASSIGNMENT_V1>>>
 * - End line (after stripping single trailing CR) must equal <<<END_AUTORAG_ASSIGNMENT_V1>>>
 * - End sentinel at EOF valid
 * - Exactly one V1 block; duplicate/malformed/unterminated → reject
 * - Body: one JSON object with exactly keys originalQuery, method, queryVariants
 * - originalQuery: string, trimmed nonempty
 * - method: string, trimmed nonempty
 * - queryVariants: nonempty array of nonempty strings
 * - Extra/missing keys/wrong types → reject
 *
 * Returns undefined if no V1 sentinel found (caller should try legacy).
 * Returns ParsedAssignment on success.
 * Throws DispatchRejectionError on malformed V1.
 */
export function parseAssignmentV1(task: string): ParsedAssignment | undefined {
	const lines = splitLines(task);
	if (lines.length === 0) return undefined;

	// Find start sentinel
	const startIndex = lines.findIndex((line) => stripTrailingCR(line) === AUTORAG_ASSIGNMENT_V1_START);
	if (startIndex === -1) return undefined; // not V1

	// Find end sentinel
	let endIndex = -1;
	for (let i = startIndex + 1; i < lines.length; i++) {
		if (stripTrailingCR(lines[i]) === AUTORAG_ASSIGNMENT_V1_END) {
			endIndex = i;
			break;
		}
	}

	if (endIndex === -1) {
		throw new DispatchRejectionError(
			{ code: "DISPATCH_ASSIGNMENT_INVALID", field: ".task", dispatchKind: "launch" },
			EXACT_FIX_MAP.DISPATCH_ASSIGNMENT_INVALID,
			FORCE_CORRECTABLE_MAP.DISPATCH_ASSIGNMENT_INVALID,
			SKELETON_SINGLE,
		);
	}

	// Check for duplicate start sentinel
	const secondStart = lines
		.slice(startIndex + 1)
		.findIndex((line) => stripTrailingCR(line) === AUTORAG_ASSIGNMENT_V1_START);
	if (secondStart !== -1) {
		throw new DispatchRejectionError(
			{ code: "DISPATCH_ASSIGNMENT_INVALID", field: ".task", dispatchKind: "launch" },
			EXACT_FIX_MAP.DISPATCH_ASSIGNMENT_INVALID,
			FORCE_CORRECTABLE_MAP.DISPATCH_ASSIGNMENT_INVALID,
			SKELETON_SINGLE,
		);
	}

	// Extract body between sentinels
	const bodyLines = lines.slice(startIndex + 1, endIndex);
	const body = bodyLines.join("\n").trim();

	if (body.length === 0) {
		throw new DispatchRejectionError(
			{ code: "DISPATCH_ASSIGNMENT_INVALID", field: ".task", dispatchKind: "launch" },
			EXACT_FIX_MAP.DISPATCH_ASSIGNMENT_INVALID,
			FORCE_CORRECTABLE_MAP.DISPATCH_ASSIGNMENT_INVALID,
			SKELETON_SINGLE,
		);
	}

	let parsed: unknown;
	try {
		parsed = JSON.parse(body);
	} catch {
		throw new DispatchRejectionError(
			{ code: "DISPATCH_ASSIGNMENT_INVALID", field: ".task", dispatchKind: "launch" },
			EXACT_FIX_MAP.DISPATCH_ASSIGNMENT_INVALID,
			FORCE_CORRECTABLE_MAP.DISPATCH_ASSIGNMENT_INVALID,
			SKELETON_SINGLE,
		);
	}

	if (!isRecord(parsed)) {
		throw new DispatchRejectionError(
			{ code: "DISPATCH_ASSIGNMENT_INVALID", field: ".task", dispatchKind: "launch" },
			EXACT_FIX_MAP.DISPATCH_ASSIGNMENT_INVALID,
			FORCE_CORRECTABLE_MAP.DISPATCH_ASSIGNMENT_INVALID,
			SKELETON_SINGLE,
		);
	}

	// Exactly keys: originalQuery, method, queryVariants
	const expectedKeys = new Set(["originalQuery", "method", "queryVariants"]);
	const actualKeys = new Set(Object.keys(parsed));
	if (actualKeys.size !== 3 || ![...actualKeys].every((k) => expectedKeys.has(k))) {
		throw new DispatchRejectionError(
			{ code: "DISPATCH_ASSIGNMENT_INVALID", field: ".task", dispatchKind: "launch" },
			EXACT_FIX_MAP.DISPATCH_ASSIGNMENT_INVALID,
			FORCE_CORRECTABLE_MAP.DISPATCH_ASSIGNMENT_INVALID,
			SKELETON_SINGLE,
		);
	}

	const originalQuery = parsed.originalQuery;
	const method = parsed.method;
	const queryVariants = parsed.queryVariants;

	if (typeof originalQuery !== "string" || originalQuery.trim().length === 0) {
		throw new DispatchRejectionError(
			{ code: "DISPATCH_ASSIGNMENT_INVALID", field: ".task", dispatchKind: "launch" },
			EXACT_FIX_MAP.DISPATCH_ASSIGNMENT_INVALID,
			FORCE_CORRECTABLE_MAP.DISPATCH_ASSIGNMENT_INVALID,
			SKELETON_SINGLE,
		);
	}
	if (typeof method !== "string" || method.trim().length === 0) {
		throw new DispatchRejectionError(
			{ code: "DISPATCH_ASSIGNMENT_INVALID", field: ".task", dispatchKind: "launch" },
			EXACT_FIX_MAP.DISPATCH_ASSIGNMENT_INVALID,
			FORCE_CORRECTABLE_MAP.DISPATCH_ASSIGNMENT_INVALID,
			SKELETON_SINGLE,
		);
	}
	if (!Array.isArray(queryVariants) || queryVariants.length === 0) {
		throw new DispatchRejectionError(
			{ code: "DISPATCH_ASSIGNMENT_INVALID", field: ".task", dispatchKind: "launch" },
			EXACT_FIX_MAP.DISPATCH_ASSIGNMENT_INVALID,
			FORCE_CORRECTABLE_MAP.DISPATCH_ASSIGNMENT_INVALID,
			SKELETON_SINGLE,
		);
	}
	for (const variant of queryVariants) {
		if (typeof variant !== "string" || variant.length === 0) {
			throw new DispatchRejectionError(
				{ code: "DISPATCH_ASSIGNMENT_INVALID", field: ".task", dispatchKind: "launch" },
				EXACT_FIX_MAP.DISPATCH_ASSIGNMENT_INVALID,
				FORCE_CORRECTABLE_MAP.DISPATCH_ASSIGNMENT_INVALID,
				SKELETON_SINGLE,
			);
		}
	}

	return {
		originalQuery: originalQuery.trim(),
		method: method.trim(),
		queryVariants,
		isV1: true,
	};
}

/**
 * Parse a task string as a legacy prose assignment.
 *
 * Legacy (n4 retained):
 * - Full logical lines; optional leading whitespace; optional `- ` bullet; optional matching `**` around label.
 * - Case/space normalize labels; value = remainder of same line only.
 * - Labels: Original query; Selected retrieval method | Retrieval method; Query variants | Query variant.
 * - Reorder OK; exactly once each; Query variants split on `;`, trim, reject empty segments.
 * - Later label-looking line → reject.
 * - Prefix/suffix outside three lines preserved.
 * - Multiline / special / label-looking content inside values → V1 required, not legacy.
 *
 * Returns undefined if no legacy labels found.
 * Throws DispatchRejectionError on malformed legacy.
 */
export function parseLegacyAssignment(task: string): ParsedAssignment | undefined {
	const lines = splitLines(task);

	interface LegacyMatch {
		readonly label: string;
		readonly value: string;
		readonly lineIndex: number;
	}

	const matches: LegacyMatch[] = [];

	const labelPatterns: ReadonlyArray<{ readonly pattern: RegExp; readonly canonical: string }> = [
		{
			pattern: /^\s*(?:[-+*]\s+)?(?:\*\*)?\s*original\s+query\s*(?:\*\*)?\s*:(?:\*\*)?\s*(.*?)\s*$/i,
			canonical: "original query",
		},
		{
			pattern:
				/^\s*(?:[-+*]\s+)?(?:\*\*)?\s*(?:selected\s+)?retrieval\s+method\s*(?:\*\*)?\s*:(?:\*\*)?\s*(.*?)\s*$/i,
			canonical: "retrieval method",
		},
		{
			pattern: /^\s*(?:[-+*]\s+)?(?:\*\*)?\s*query\s+variants?\s*(?:\*\*)?\s*:(?:\*\*)?\s*(.*?)\s*$/i,
			canonical: "query variants",
		},
	];

	for (let i = 0; i < lines.length; i++) {
		const line = lines[i];
		for (const { pattern, canonical } of labelPatterns) {
			const match = pattern.exec(line);
			if (match !== null) {
				// Check for later label-looking line after all three found
				matches.push({ label: canonical, value: match[1] ?? "", lineIndex: i });
			}
		}
	}

	if (matches.length === 0) return undefined;

	// Exactly 3 matches
	if (matches.length !== 3) {
		throw new DispatchRejectionError(
			{ code: "DISPATCH_ASSIGNMENT_INVALID", field: ".task", dispatchKind: "launch" },
			EXACT_FIX_MAP.DISPATCH_ASSIGNMENT_INVALID,
			FORCE_CORRECTABLE_MAP.DISPATCH_ASSIGNMENT_INVALID,
			SKELETON_SINGLE,
		);
	}

	// Check exactly one of each label
	const labelCounts = new Map<string, number>();
	for (const m of matches) {
		labelCounts.set(m.label, (labelCounts.get(m.label) ?? 0) + 1);
	}
	for (const [, count] of labelCounts) {
		if (count > 1) {
			throw new DispatchRejectionError(
				{ code: "DISPATCH_ASSIGNMENT_INVALID", field: ".task", dispatchKind: "launch" },
				EXACT_FIX_MAP.DISPATCH_ASSIGNMENT_INVALID,
				FORCE_CORRECTABLE_MAP.DISPATCH_ASSIGNMENT_INVALID,
				SKELETON_SINGLE,
			);
		}
	}
	if (labelCounts.size !== 3) {
		throw new DispatchRejectionError(
			{ code: "DISPATCH_ASSIGNMENT_INVALID", field: ".task", dispatchKind: "launch" },
			EXACT_FIX_MAP.DISPATCH_ASSIGNMENT_INVALID,
			FORCE_CORRECTABLE_MAP.DISPATCH_ASSIGNMENT_INVALID,
			SKELETON_SINGLE,
		);
	}

	// Check for later label-looking line after the last match
	const lastMatchIndex = Math.max(...matches.map((m) => m.lineIndex));
	for (let i = lastMatchIndex + 1; i < lines.length; i++) {
		// A "label-looking" line is one that matches the pattern but wasn't counted
		// Since we counted all matches above, any line after lastMatchIndex matching is extra
		for (const { pattern } of labelPatterns) {
			if (pattern.test(lines[i])) {
				throw new DispatchRejectionError(
					{ code: "DISPATCH_ASSIGNMENT_INVALID", field: ".task", dispatchKind: "launch" },
					EXACT_FIX_MAP.DISPATCH_ASSIGNMENT_INVALID,
					FORCE_CORRECTABLE_MAP.DISPATCH_ASSIGNMENT_INVALID,
					SKELETON_SINGLE,
				);
			}
		}
	}

	// Extract values
	const originalQueryMatch = matches.find((m) => m.label === "original query");
	const methodMatch = matches.find((m) => m.label === "retrieval method");
	const queryVariantsMatch = matches.find((m) => m.label === "query variants");

	if (!originalQueryMatch || !methodMatch || !queryVariantsMatch) {
		throw new DispatchRejectionError(
			{ code: "DISPATCH_ASSIGNMENT_INVALID", field: ".task", dispatchKind: "launch" },
			EXACT_FIX_MAP.DISPATCH_ASSIGNMENT_INVALID,
			FORCE_CORRECTABLE_MAP.DISPATCH_ASSIGNMENT_INVALID,
			SKELETON_SINGLE,
		);
	}

	const originalQuery = originalQueryMatch.value.trim();
	const method = methodMatch.value.trim();
	if (originalQuery.length === 0 || method.length === 0) {
		throw new DispatchRejectionError(
			{ code: "DISPATCH_ASSIGNMENT_INVALID", field: ".task", dispatchKind: "launch" },
			EXACT_FIX_MAP.DISPATCH_ASSIGNMENT_INVALID,
			FORCE_CORRECTABLE_MAP.DISPATCH_ASSIGNMENT_INVALID,
			SKELETON_SINGLE,
		);
	}

	// Query variants: split on `;`, trim, reject empty segments
	const variantSegments = queryVariantsMatch.value.split(";");
	const variants: string[] = [];
	for (const segment of variantSegments) {
		const trimmed = segment.trim();
		if (trimmed.length === 0) {
			throw new DispatchRejectionError(
				{ code: "DISPATCH_ASSIGNMENT_INVALID", field: ".task", dispatchKind: "launch" },
				EXACT_FIX_MAP.DISPATCH_ASSIGNMENT_INVALID,
				FORCE_CORRECTABLE_MAP.DISPATCH_ASSIGNMENT_INVALID,
				SKELETON_SINGLE,
			);
		}
		variants.push(trimmed);
	}
	if (variants.length === 0) {
		throw new DispatchRejectionError(
			{ code: "DISPATCH_ASSIGNMENT_INVALID", field: ".task", dispatchKind: "launch" },
			EXACT_FIX_MAP.DISPATCH_ASSIGNMENT_INVALID,
			FORCE_CORRECTABLE_MAP.DISPATCH_ASSIGNMENT_INVALID,
			SKELETON_SINGLE,
		);
	}

	return {
		originalQuery,
		method,
		queryVariants: variants,
		isV1: false,
	};
}

/**
 * Parse a task string as either V1 or legacy assignment.
 * Tries V1 first; if no V1 sentinels, tries legacy.
 * Returns undefined if neither format is found.
 * Throws DispatchRejectionError on malformed input.
 */
export function parseAssignment(task: string): ParsedAssignment | undefined {
	const v1 = parseAssignmentV1(task);
	if (v1 !== undefined) return v1;
	return parseLegacyAssignment(task);
}

// ---------------------------------------------------------------------------
// G) Role-line insertion (byte-idempotent)
// ---------------------------------------------------------------------------

/**
 * Insert canonical role lines into a task string, byte-idempotent.
 *
 * 1. Remove all existing exact canonical role lines (the two literals after CR-strip).
 * 2. Insert exactly one of each immediately after end sentinel (V1) or
 *    immediately after the last of the three recognized legacy label lines.
 * 3. Preserve all other suffix bytes and original newline style.
 * 4. Repeat rendering is byte-idempotent.
 */
export function ensureRoleLines(task: string): string {
	const lines = splitLines(task);

	// Step 1: Remove all existing exact canonical role lines
	const roleLineSet = new Set([ROLE_LINE_RETRIEVED_AT, ROLE_LINE_TEMPORAL_METADATA]);
	const filtered: string[] = [];
	for (const line of lines) {
		if (roleLineSet.has(stripTrailingCR(line))) continue;
		filtered.push(line);
	}

	// Step 2: Find insertion point
	let insertIndex = -1;

	// Check for V1 end sentinel
	for (let i = 0; i < filtered.length; i++) {
		if (stripTrailingCR(filtered[i]) === AUTORAG_ASSIGNMENT_V1_END) {
			insertIndex = i + 1;
			break;
		}
	}

	// If no V1 end sentinel, find the last legacy label line
	if (insertIndex === -1) {
		const labelPatterns = [
			/^\s*(?:[-+*]\s+)?(?:\*\*)?\s*original\s+query\s*(?:\*\*)?\s*:(?:\*\*)?/i,
			/^\s*(?:[-+*]\s+)?(?:\*\*)?\s*(?:selected\s+)?retrieval\s+method\s*(?:\*\*)?\s*:(?:\*\*)?/i,
			/^\s*(?:[-+*]\s+)?(?:\*\*)?\s*query\s+variants?\s*(?:\*\*)?\s*:(?:\*\*)?/i,
		];
		for (let i = filtered.length - 1; i >= 0; i--) {
			if (labelPatterns.some((p) => p.test(filtered[i]))) {
				insertIndex = i + 1;
				break;
			}
		}
	}

	// If no insertion point found, append at end
	if (insertIndex === -1) {
		insertIndex = filtered.length;
	}

	// Step 3: Insert the two role lines with LF between them
	const result = [
		...filtered.slice(0, insertIndex),
		ROLE_LINE_RETRIEVED_AT,
		ROLE_LINE_TEMPORAL_METADATA,
		...filtered.slice(insertIndex),
	];

	return result.join("\n");
}

// ---------------------------------------------------------------------------
// Post-parse validation (called from beforeToolCall)
// ---------------------------------------------------------------------------

export interface PostSchemaContext {
	readonly configuredModel: string | undefined;
	readonly currentQuery: string | undefined;
	readonly allowedRoots: readonly string[];
	readonly workspaceRoot: string;
}

/**
 * Post-schema validation for launch invocations.
 * Checks: assignment parse, structural query equality, role lines,
 * cwd realpath/allowed roots, model match.
 *
 * Returns a DispatchRejection if invalid, undefined if valid.
 * Also normalizes cwd to canonical realpath on success.
 */
export function validateLaunchPostSchema(
	args: Record<string, unknown>,
	ctx: PostSchemaContext,
): DispatchRejection | undefined {
	// NO_ACTIVE_QUERY
	if (ctx.currentQuery === undefined) {
		return { code: "DISPATCH_NO_ACTIVE_QUERY", field: "args", dispatchKind: "launch" };
	}

	const malformedLeaf = findMalformedContainerLeaf(args);
	if (malformedLeaf !== undefined) return malformedLeaf;

	const leaves = collectLaunchLeaves(args);
	if (leaves.length === 0) {
		return { code: "DISPATCH_MALFORMED", field: "args", dispatchKind: "launch" };
	}

	const expectedModelId = ctx.configuredModel?.split(":", 1)[0];
	const topLevelModel = typeof args.model === "string" ? args.model.split(":", 1)[0] : undefined;

	for (const leaf of leaves) {
		// Model match
		const leafModel = typeof leaf.record.model === "string" ? leaf.record.model.split(":", 1)[0] : topLevelModel;
		if (expectedModelId !== undefined && (leafModel === undefined || leafModel !== expectedModelId)) {
			return { code: "DISPATCH_MODEL_MISMATCH", field: `${leaf.path}.model`, dispatchKind: "launch" };
		}

		// Assignment parse
		const task = leaf.record.task;
		if (typeof task !== "string") {
			return { code: "DISPATCH_ASSIGNMENT_INVALID", field: `${leaf.path}.task`, dispatchKind: "launch" };
		}

		let assignment: ParsedAssignment | undefined;
		try {
			assignment = parseAssignment(task);
		} catch (error) {
			if (error instanceof DispatchRejectionError) {
				// Remap the parser's generic ".task" field to the current leaf path so
				// fanout/chain/parallel errors are field-addressed (e.g. ".tasks[0].task")
				// instead of the generic ".task" the parser emits.
				return { code: error.code, field: `${leaf.path}.task`, dispatchKind: "launch" };
			}
			return { code: "DISPATCH_ASSIGNMENT_INVALID", field: `${leaf.path}.task`, dispatchKind: "launch" };
		}

		if (assignment === undefined) {
			return { code: "DISPATCH_ASSIGNMENT_INVALID", field: `${leaf.path}.task`, dispatchKind: "launch" };
		}

		// Query mismatch (exact structural equality)
		if (assignment.originalQuery !== ctx.currentQuery) {
			return { code: "DISPATCH_QUERY_MISMATCH", field: `${leaf.path}.task`, dispatchKind: "launch" };
		}

		// Role metadata: normalize task to include canonical role lines idempotently.
		// Accept valid V1/legacy content and render the required lines before execute/credit.
		// Keep query/assignment validation strict (already checked above).
		// ensureRoleLines is idempotent: it strips all existing canonical role lines and
		// re-inserts exactly one adjacent pair at the deterministic insertion point,
		// canonicalizing placement even when both role lines already exist off-position.
		leaf.record.task = ensureRoleLines(task);

		// CWD realpath / allowed roots
		if (typeof leaf.record.cwd !== "string" || leaf.record.cwd.trim().length === 0) {
			return { code: "DISPATCH_CWD_MISSING", field: `${leaf.path}.cwd`, dispatchKind: "launch" };
		}

		const requestedCwd = resolve(ctx.workspaceRoot, leaf.record.cwd);
		let canonicalCwd: string;
		let isDirectory: boolean;
		try {
			canonicalCwd = realpathSync(requestedCwd);
			isDirectory = statSync(canonicalCwd).isDirectory();
		} catch {
			return { code: "DISPATCH_CWD_OUTSIDE_ROOTS", field: `${leaf.path}.cwd`, dispatchKind: "launch" };
		}

		if (!isDirectory || !ctx.allowedRoots.includes(canonicalCwd)) {
			return { code: "DISPATCH_CWD_OUTSIDE_ROOTS", field: `${leaf.path}.cwd`, dispatchKind: "launch" };
		}

		// Normalize cwd to canonical realpath
		leaf.record.cwd = canonicalCwd;
	}

	return undefined;
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function splitLines(text: string): string[] {
	// Split on \n only, preserving \r for CR-stripping during sentinel comparison.
	// This preserves the original newline style (LF vs CRLF) of the task string.
	return text.split("\n");
}

function stripTrailingCR(line: string): string {
	if (line.endsWith("\r")) return line.slice(0, -1);
	return line;
}

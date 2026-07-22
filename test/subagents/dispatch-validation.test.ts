import { describe, expect, it } from "vitest";
import {
	AUTORAG_ASSIGNMENT_V1_END,
	AUTORAG_ASSIGNMENT_V1_START,
	autoragPrepare,
	classifyAction,
	classifyDispatch,
	createDispatchRejectionError,
	type DispatchRejectionCode,
	DispatchRejectionError,
	EXACT_FIX_MAP,
	ensureRoleLines,
	FORCE_CORRECTABLE_MAP,
	formatBlockReason,
	getLaunchLeaves,
	parseAssignment,
	parseAssignmentV1,
	parseLegacyAssignment,
	ROLE_LINE_RETRIEVED_AT,
	ROLE_LINE_TEMPORAL_METADATA,
	readAutofilled,
	rejectForbiddenAction,
	SKELETON_ADMIN,
	SKELETON_FANOUT,
	SKELETON_SINGLE,
	validateDiagnostic,
	validateLaunchPostSchema,
} from "../../src/subagents/dispatch-validation.ts";

// ---------------------------------------------------------------------------
// C) Error catalog: codes, exactFix, forceCorrectable, skeletons, format
// ---------------------------------------------------------------------------

describe("error catalog", () => {
	const ALL_CODES: DispatchRejectionCode[] = [
		"DISPATCH_MALFORMED",
		"DISPATCH_ACTION_UNKNOWN",
		"DISPATCH_ADMIN_MUTATION_FORBIDDEN",
		"DISPATCH_CONTROL_FORBIDDEN",
		"DISPATCH_SCHEDULE_FORBIDDEN",
		"DISPATCH_ARTIFACTS_INVALID",
		"DISPATCH_AGENT_SCOPE_INVALID",
		"DISPATCH_AGENT_IDENTITY",
		"DISPATCH_MODEL_MISMATCH",
		"DISPATCH_ASSIGNMENT_INVALID",
		"DISPATCH_QUERY_MISMATCH",
		"DISPATCH_CWD_MISSING",
		"DISPATCH_CWD_OUTSIDE_ROOTS",
		"DISPATCH_NO_ACTIVE_QUERY",
	];

	it("every code has an exactFix entry", () => {
		for (const code of ALL_CODES) {
			expect(EXACT_FIX_MAP[code]).toBeDefined();
			expect(typeof EXACT_FIX_MAP[code]).toBe("string");
			expect(EXACT_FIX_MAP[code].length).toBeGreaterThan(0);
		}
	});

	it("every code has a forceCorrectable entry", () => {
		for (const code of ALL_CODES) {
			expect(typeof FORCE_CORRECTABLE_MAP[code]).toBe("boolean");
		}
	});

	it("forceCorrectable: correctable codes are true", () => {
		const correctable: DispatchRejectionCode[] = [
			"DISPATCH_MALFORMED",
			"DISPATCH_ACTION_UNKNOWN",
			"DISPATCH_ARTIFACTS_INVALID",
			"DISPATCH_AGENT_SCOPE_INVALID",
			"DISPATCH_MODEL_MISMATCH",
			"DISPATCH_QUERY_MISMATCH",
			"DISPATCH_ASSIGNMENT_INVALID",
			"DISPATCH_CWD_MISSING",
		];
		for (const code of correctable) {
			expect(FORCE_CORRECTABLE_MAP[code]).toBe(true);
		}
	});

	it("forceCorrectable: non-correctable codes are false", () => {
		const nonCorrectable: DispatchRejectionCode[] = [
			"DISPATCH_ADMIN_MUTATION_FORBIDDEN",
			"DISPATCH_CONTROL_FORBIDDEN",
			"DISPATCH_SCHEDULE_FORBIDDEN",
			"DISPATCH_AGENT_IDENTITY",
			"DISPATCH_CWD_OUTSIDE_ROOTS",
			"DISPATCH_NO_ACTIVE_QUERY",
		];
		for (const code of nonCorrectable) {
			expect(FORCE_CORRECTABLE_MAP[code]).toBe(false);
		}
	});

	it("exactFix literals match the frozen plan", () => {
		expect(EXACT_FIX_MAP.DISPATCH_MALFORMED).toBe("remove fields not owned by this action");
		expect(EXACT_FIX_MAP.DISPATCH_ACTION_UNKNOWN).toBe(
			"use list|get|models|status|doctor or a supported launch shape",
		);
		expect(EXACT_FIX_MAP.DISPATCH_ADMIN_MUTATION_FORBIDDEN).toBe(
			"do not mutate subagent definitions during AutoRAG search",
		);
		expect(EXACT_FIX_MAP.DISPATCH_CONTROL_FORBIDDEN).toBe(
			"launch a fresh autorag-explorer assignment instead of controlling an existing run",
		);
		expect(EXACT_FIX_MAP.DISPATCH_SCHEDULE_FORBIDDEN).toBe(
			"dispatch autorag-explorer work immediately; scheduling is disabled",
		);
		expect(EXACT_FIX_MAP.DISPATCH_ARTIFACTS_INVALID).toBe("set args.artifacts = false");
		expect(EXACT_FIX_MAP.DISPATCH_AGENT_SCOPE_INVALID).toBe('set args.agentScope = "user"');
		expect(EXACT_FIX_MAP.DISPATCH_AGENT_IDENTITY).toBe('set every executable leaf agent = "autorag-explorer"');
		expect(EXACT_FIX_MAP.DISPATCH_MODEL_MISMATCH).toBe(
			"set the referenced model field to the configured explorer model",
		);
		expect(EXACT_FIX_MAP.DISPATCH_ASSIGNMENT_INVALID).toBe(
			"replace the task assignment with the canonical AUTORAG_ASSIGNMENT_V1 block",
		);
		expect(EXACT_FIX_MAP.DISPATCH_QUERY_MISMATCH).toBe("set originalQuery to the active user query verbatim");
		expect(EXACT_FIX_MAP.DISPATCH_CWD_MISSING).toBe("set each executable leaf cwd to one configured search root");
		expect(EXACT_FIX_MAP.DISPATCH_CWD_OUTSIDE_ROOTS).toBe("use a configured search root without symlink escape");
		expect(EXACT_FIX_MAP.DISPATCH_NO_ACTIVE_QUERY).toBe("dispatch only while an AutoRAG search query is active");
	});

	it("skeleton literals match the frozen plan", () => {
		expect(SKELETON_ADMIN).toBe(`{"action":"list"}`);
		expect(SKELETON_SINGLE).toBe(
			`{"agentScope":"user","artifacts":false,"agent":"autorag-explorer","model":"<configured-model>","cwd":"<allowed-root>","task":"<Assignment V1 block>"}`,
		);
		expect(SKELETON_FANOUT).toBe(
			`{"agentScope":"user","artifacts":false,"tasks":[{"agent":"autorag-explorer","model":"<configured-model>","cwd":"<allowed-root>","task":"<Assignment V1 block>"}]}`,
		);
	});

	it("formatBlockReason produces literal format with one newline before skeleton", () => {
		const reason = formatBlockReason(
			"DISPATCH_MALFORMED",
			"args",
			"remove fields not owned by this action",
			SKELETON_ADMIN,
		);
		expect(reason).toBe(
			`[DISPATCH_MALFORMED] field=args fix=remove fields not owned by this action\n{"action":"list"}`,
		);
		// Exactly one newline between first line and skeleton
		const lines = reason.split("\n");
		expect(lines).toHaveLength(2);
	});

	it("DispatchRejectionError carries code, field, exactFix, forceCorrectable, skeleton", () => {
		const error = new DispatchRejectionError(
			{ code: "DISPATCH_MALFORMED", field: "args", dispatchKind: "admin" },
			EXACT_FIX_MAP.DISPATCH_MALFORMED,
			true,
			SKELETON_ADMIN,
		);
		expect(error.code).toBe("DISPATCH_MALFORMED");
		expect(error.field).toBe("args");
		expect(error.exactFix).toBe("remove fields not owned by this action");
		expect(error.forceCorrectable).toBe(true);
		expect(error.skeleton).toBe(SKELETON_ADMIN);
		expect(error.dispatchKind).toBe("admin");
		expect(error.message).toContain("[DISPATCH_MALFORMED]");
		expect(error.message).toContain("field=args");
		expect(error.message).toContain("fix=remove fields not owned by this action");
		expect(error.name).toBe("DispatchRejectionError");
	});
});

// ---------------------------------------------------------------------------
// A) Classification + B) Diagnostic ownership
// ---------------------------------------------------------------------------

describe("action classification", () => {
	it("classifies diagnostic actions as admin", () => {
		expect(classifyAction("list")).toBe("admin");
		expect(classifyAction("get")).toBe("admin");
		expect(classifyAction("models")).toBe("admin");
		expect(classifyAction("status")).toBe("admin");
		expect(classifyAction("doctor")).toBe("admin");
	});

	it("classifies mutation actions", () => {
		expect(classifyAction("create")).toBe("mutation");
		expect(classifyAction("update")).toBe("mutation");
		expect(classifyAction("delete")).toBe("mutation");
		expect(classifyAction("eject")).toBe("mutation");
		expect(classifyAction("disable")).toBe("mutation");
		expect(classifyAction("enable")).toBe("mutation");
		expect(classifyAction("reset")).toBe("mutation");
	});

	it("classifies control actions", () => {
		expect(classifyAction("interrupt")).toBe("control");
		expect(classifyAction("resume")).toBe("control");
		expect(classifyAction("steer")).toBe("control");
		expect(classifyAction("append-step")).toBe("control");
	});

	it("classifies schedule actions", () => {
		expect(classifyAction("schedule")).toBe("schedule");
		expect(classifyAction("schedule-list")).toBe("schedule");
		expect(classifyAction("schedule-status")).toBe("schedule");
		expect(classifyAction("schedule-cancel")).toBe("schedule");
	});

	it("classifies non-string as unknown", () => {
		expect(classifyAction(42)).toBe("unknown");
		expect(classifyAction(null)).toBe("unknown");
		expect(classifyAction(undefined)).toBe("unknown");
		expect(classifyAction(true)).toBe("unknown");
	});

	it("classifyDispatch: launch shape without action → launch", () => {
		expect(classifyDispatch({ agent: "autorag-explorer", task: "..." })).toBe("launch");
		expect(classifyDispatch({ tasks: [{ agent: "autorag-explorer" }] })).toBe("launch");
	});

	it("classifyDispatch: diagnostic action → admin", () => {
		expect(classifyDispatch({ action: "list" })).toBe("admin");
		expect(classifyDispatch({ action: "get", agent: "foo" })).toBe("admin");
	});

	it("classifyDispatch: mutation action → mutation", () => {
		expect(classifyDispatch({ action: "create" })).toBe("mutation");
		expect(classifyDispatch({ action: "delete" })).toBe("mutation");
	});

	it("classifyDispatch: unknown action with no launch shape → unknown", () => {
		expect(classifyDispatch({ action: "foobar" })).toBe("unknown");
	});

	it("classifyDispatch: no action, no launch shape → unknown", () => {
		expect(classifyDispatch({})).toBe("unknown");
		expect(classifyDispatch({ foo: "bar" })).toBe("unknown");
	});

	it("classifyDispatch: non-object → unknown", () => {
		expect(classifyDispatch("hello")).toBe("unknown");
		expect(classifyDispatch(null)).toBe("unknown");
		expect(classifyDispatch(42)).toBe("unknown");
	});

	it("classifyDispatch: case-variant actions remain unknown (Create, LIST, Interrupt)", () => {
		// Non-exact-lowercase action values must not fall through to launch,
		// admin, mutation, or control — regardless of launch fields present.
		expect(classifyDispatch({ action: "Create", agent: "autorag-explorer", task: "x", cwd: "/tmp" })).toBe("unknown");
		expect(classifyDispatch({ action: "LIST", agentScope: "user" })).toBe("unknown");
		expect(classifyDispatch({ action: "Interrupt", agent: "autorag-explorer", task: "x", cwd: "/tmp" })).toBe(
			"unknown",
		);
	});

	it("classifyDispatch: numeric action with launch fields → unknown", () => {
		expect(classifyDispatch({ action: 42, agent: "autorag-explorer", task: "x", cwd: "/tmp" })).toBe("unknown");
		expect(classifyDispatch({ action: 0, tasks: [{ agent: "autorag-explorer" }] })).toBe("unknown");
	});

	it("classifyDispatch: exact lowercase execution aliases with launch shape → launch", () => {
		expect(classifyDispatch({ action: "single", agent: "autorag-explorer", task: "x", cwd: "/tmp" })).toBe("launch");
		expect(
			classifyDispatch({ action: "parallel", tasks: [{ agent: "autorag-explorer", task: "x", cwd: "/tmp" }] }),
		).toBe("launch");
		expect(
			classifyDispatch({ action: "tasks", tasks: [{ agent: "autorag-explorer", task: "x", cwd: "/tmp" }] }),
		).toBe("launch");
	});

	it("classifyDispatch: execution alias without launch shape → unknown", () => {
		expect(classifyDispatch({ action: "single" })).toBe("unknown");
		expect(classifyDispatch({ action: "parallel" })).toBe("unknown");
		expect(classifyDispatch({ action: "tasks" })).toBe("unknown");
	});

	it("classifyDispatch: case-variant execution alias → unknown", () => {
		expect(classifyDispatch({ action: "Single", agent: "autorag-explorer", task: "x", cwd: "/tmp" })).toBe("unknown");
		expect(classifyDispatch({ action: "PARALLEL", tasks: [{ agent: "autorag-explorer" }] })).toBe("unknown");
	});
});

describe("diagnostic ownership", () => {
	it("list: allows action and optional agentScope=user", () => {
		expect(validateDiagnostic({ action: "list" }, "list")).toBeUndefined();
		expect(validateDiagnostic({ action: "list", agentScope: "user" }, "list")).toBeUndefined();
	});

	it("list: agentScope=project rejected", () => {
		const rejection = validateDiagnostic({ action: "list", agentScope: "project" }, "list");
		expect(rejection?.code).toBe("DISPATCH_AGENT_SCOPE_INVALID");
		expect(rejection?.field).toBe("agentScope");
	});

	it("list: unrelated field rejected", () => {
		const rejection = validateDiagnostic({ action: "list", foo: "bar" }, "list");
		expect(rejection?.code).toBe("DISPATCH_MALFORMED");
		expect(rejection?.field).toBe("foo");
	});

	it("get: requires exactly one of agent|chainName", () => {
		expect(validateDiagnostic({ action: "get", agent: "foo" }, "get")).toBeUndefined();
		expect(validateDiagnostic({ action: "get", chainName: "bar" }, "get")).toBeUndefined();
		// Neither
		const neither = validateDiagnostic({ action: "get" }, "get");
		expect(neither?.code).toBe("DISPATCH_MALFORMED");
		// Both
		const both = validateDiagnostic({ action: "get", agent: "foo", chainName: "bar" }, "get");
		expect(both?.code).toBe("DISPATCH_MALFORMED");
	});

	it("get: level field rejected", () => {
		const rejection = validateDiagnostic({ action: "get", agent: "foo", level: 2 }, "get");
		expect(rejection?.code).toBe("DISPATCH_MALFORMED");
		expect(rejection?.field).toBe("level");
	});

	it("models: allows action and optional agent", () => {
		expect(validateDiagnostic({ action: "models" }, "models")).toBeUndefined();
		expect(validateDiagnostic({ action: "models", agent: "foo" }, "models")).toBeUndefined();
	});

	it("status: allows upstream selectors", () => {
		expect(validateDiagnostic({ action: "status", id: "run-1" }, "status")).toBeUndefined();
		expect(validateDiagnostic({ action: "status", runId: "run-1" }, "status")).toBeUndefined();
		expect(validateDiagnostic({ action: "status", dir: "/tmp" }, "status")).toBeUndefined();
	});

	it("doctor: no config field allowed", () => {
		expect(validateDiagnostic({ action: "doctor", cwd: "/tmp" }, "doctor")).toBeUndefined();
		expect(validateDiagnostic({ action: "doctor", context: "foo" }, "doctor")).toBeUndefined();
		expect(validateDiagnostic({ action: "doctor", sessionDir: "/tmp" }, "doctor")).toBeUndefined();
		const rejection = validateDiagnostic({ action: "doctor", config: "bar" }, "doctor");
		expect(rejection?.code).toBe("DISPATCH_MALFORMED");
		expect(rejection?.field).toBe("config");
	});

	it("execution fields on diagnostic → MALFORMED", () => {
		const rejection = validateDiagnostic({ action: "list", task: "foo" }, "list");
		expect(rejection?.code).toBe("DISPATCH_MALFORMED");
		expect(rejection?.field).toBe("task");
	});
});

// ---------------------------------------------------------------------------
// A) First-error precedence
// ---------------------------------------------------------------------------

describe("first-error precedence", () => {
	const ctx = { configuredModel: "test-provider/test-model" };

	it("mutation action takes precedence over everything", () => {
		expect(() => autoragPrepare({ action: "create", artifacts: "wrong" }, ctx)).toThrow(DispatchRejectionError);
		try {
			autoragPrepare({ action: "create" }, ctx);
		} catch (e) {
			expect((e as DispatchRejectionError).code).toBe("DISPATCH_ADMIN_MUTATION_FORBIDDEN");
		}
	});

	it("unknown action takes precedence over launch field checks", () => {
		try {
			autoragPrepare({ action: "foobar", artifacts: "wrong" }, ctx);
			expect.fail("should have thrown");
		} catch (e) {
			expect((e as DispatchRejectionError).code).toBe("DISPATCH_ACTION_UNKNOWN");
		}
	});

	it("top-level artifacts checked before leaf identity", () => {
		try {
			autoragPrepare({ agent: "autorag-explorer", artifacts: true }, ctx);
			expect.fail("should have thrown");
		} catch (e) {
			expect((e as DispatchRejectionError).code).toBe("DISPATCH_ARTIFACTS_INVALID");
		}
	});

	it("leaf agent identity checked before model", () => {
		try {
			autoragPrepare(
				{
					agent: "wrong-agent",
					artifacts: false,
					agentScope: "user",
					model: "test-provider/test-model",
					task: "x",
					cwd: "/tmp",
				},
				ctx,
			);
			expect.fail("should have thrown");
		} catch (e) {
			expect((e as DispatchRejectionError).code).toBe("DISPATCH_AGENT_IDENTITY");
		}
	});
});

// ---------------------------------------------------------------------------
// D) Pre-schema validation table (launch-only defaults)
// ---------------------------------------------------------------------------

describe("autoragPrepare: launch defaults", () => {
	const ctx = { configuredModel: "test-provider/test-model" };

	it("fills missing artifacts=false", () => {
		const result = autoragPrepare({ agent: "autorag-explorer", task: "some task", cwd: "/tmp" }, ctx);
		expect(result.artifacts).toBe(false);
		const autofilled = readAutofilled(result);
		expect(autofilled?.artifacts).toBe(true);
	});

	it("fills missing agentScope=user", () => {
		const result = autoragPrepare({ agent: "autorag-explorer", task: "some task", cwd: "/tmp" }, ctx);
		expect(result.agentScope).toBe("user");
		const autofilled = readAutofilled(result);
		expect(autofilled?.agentScope).toBe(true);
	});

	it("null artifacts deleted then set false", () => {
		const result = autoragPrepare(
			{ agent: "autorag-explorer", artifacts: null, task: "some task", cwd: "/tmp" },
			ctx,
		);
		expect(result.artifacts).toBe(false);
	});

	it("null agentScope deleted then set user", () => {
		const result = autoragPrepare(
			{ agent: "autorag-explorer", agentScope: null, task: "some task", cwd: "/tmp" },
			ctx,
		);
		expect(result.agentScope).toBe("user");
	});

	it("explicit artifacts=true rejected", () => {
		expect(() =>
			autoragPrepare({ agent: "autorag-explorer", artifacts: true, task: "some task", cwd: "/tmp" }, ctx),
		).toThrow(DispatchRejectionError);
	});

	it("explicit agentScope=project rejected", () => {
		expect(() =>
			autoragPrepare(
				{ agent: "autorag-explorer", agentScope: "project", artifacts: false, task: "some task", cwd: "/tmp" },
				ctx,
			),
		).toThrow(DispatchRejectionError);
	});

	it("wrong-type artifacts rejected", () => {
		expect(() =>
			autoragPrepare({ agent: "autorag-explorer", artifacts: "false", task: "some task", cwd: "/tmp" }, ctx),
		).toThrow(DispatchRejectionError);
	});

	it("fills missing leaf model with configured model", () => {
		const result = autoragPrepare(
			{ agent: "autorag-explorer", artifacts: false, agentScope: "user", task: "some task", cwd: "/tmp" },
			ctx,
		);
		expect(result.model).toBe("test-provider/test-model");
		const autofilled = readAutofilled(result);
		expect(autofilled?.leafModelFillCount).toBe(1);
	});

	it("null leaf model filled with configured", () => {
		const result = autoragPrepare(
			{
				agent: "autorag-explorer",
				artifacts: false,
				agentScope: "user",
				model: null,
				task: "some task",
				cwd: "/tmp",
			},
			ctx,
		);
		expect(result.model).toBe("test-provider/test-model");
	});
	it("normalizes a short configured model alias", () => {
		const result = autoragPrepare(
			{
				agent: "autorag-explorer",
				artifacts: false,
				agentScope: "user",
				model: "test-model",
				task: "some task",
				cwd: "/tmp",
			},
			ctx,
		);
		expect(result.model).toBe("test-provider/test-model");
		expect(readAutofilled(result)?.leafModelFillCount).toBe(1);
	});

	it("rejects an explicit unrelated model", () => {
		expect(() =>
			autoragPrepare(
				{
					agent: "autorag-explorer",
					artifacts: false,
					agentScope: "user",
					model: "other/model",
					task: "some task",
					cwd: "/tmp",
				},
				ctx,
			),
		).toThrow(DispatchRejectionError);
	});

	it("blank leaf model rejected as DISPATCH_MODEL_MISMATCH", () => {
		expect(() =>
			autoragPrepare(
				{
					agent: "autorag-explorer",
					artifacts: false,
					agentScope: "user",
					model: "  ",
					task: "some task",
					cwd: "/tmp",
				},
				ctx,
			),
		).toThrow(DispatchRejectionError);
	});

	it("wrong-type model rejected", () => {
		expect(() =>
			autoragPrepare(
				{
					agent: "autorag-explorer",
					artifacts: false,
					agentScope: "user",
					model: 42,
					task: "some task",
					cwd: "/tmp",
				},
				ctx,
			),
		).toThrow(DispatchRejectionError);
	});

	it("tasks fanout: fills model on each leaf", () => {
		const result = autoragPrepare(
			{
				agentScope: "user",
				artifacts: false,
				tasks: [
					{ agent: "autorag-explorer", task: "task1", cwd: "/root1" },
					{ agent: "autorag-explorer", task: "task2", cwd: "/root2" },
				],
			},
			ctx,
		);
		const tasks = result.tasks as Record<string, unknown>[];
		expect(tasks[0].model).toBe("test-provider/test-model");
		expect(tasks[1].model).toBe("test-provider/test-model");
		const autofilled = readAutofilled(result);
		expect(autofilled?.leafModelFillCount).toBe(2);
	});

	it("nested artifacts on leaf rejected", () => {
		expect(() =>
			autoragPrepare(
				{
					agentScope: "user",
					artifacts: false,
					tasks: [{ agent: "autorag-explorer", artifacts: false, task: "task1", cwd: "/root1" }],
				},
				ctx,
			),
		).toThrow(DispatchRejectionError);
	});

	it("nested agentScope on leaf rejected", () => {
		expect(() =>
			autoragPrepare(
				{
					agentScope: "user",
					artifacts: false,
					tasks: [{ agent: "autorag-explorer", agentScope: "user", task: "task1", cwd: "/root1" }],
				},
				ctx,
			),
		).toThrow(DispatchRejectionError);
	});
});

describe("autoragPrepare: diagnostic no-default leakage", () => {
	it("list diagnostic: no artifacts/agentScope/model filled", () => {
		const result = autoragPrepare({ action: "list" }, { configuredModel: "test/test" });
		expect(result.artifacts).toBeUndefined();
		expect(result.agentScope).toBeUndefined();
		expect(result.model).toBeUndefined();
		expect(readAutofilled(result)).toBeUndefined();
	});

	it("get diagnostic: no defaults filled", () => {
		const result = autoragPrepare({ action: "get", agent: "foo" }, { configuredModel: "test/test" });
		expect(result.artifacts).toBeUndefined();
		expect(result.model).toBeUndefined();
	});

	it("models diagnostic: no defaults filled", () => {
		const result = autoragPrepare({ action: "models" }, { configuredModel: "test/test" });
		expect(result.artifacts).toBeUndefined();
		expect(result.agentScope).toBeUndefined();
	});
});

describe("autoragPrepare: forbidden actions", () => {
	it("mutation action throws DISPATCH_ADMIN_MUTATION_FORBIDDEN", () => {
		try {
			autoragPrepare({ action: "create" }, { configuredModel: "test/test" });
			expect.fail("should throw");
		} catch (e) {
			expect((e as DispatchRejectionError).code).toBe("DISPATCH_ADMIN_MUTATION_FORBIDDEN");
			expect((e as DispatchRejectionError).dispatchKind).toBe("mutation");
		}
	});

	it("control action throws DISPATCH_CONTROL_FORBIDDEN", () => {
		try {
			autoragPrepare({ action: "interrupt" }, { configuredModel: "test/test" });
			expect.fail("should throw");
		} catch (e) {
			expect((e as DispatchRejectionError).code).toBe("DISPATCH_CONTROL_FORBIDDEN");
			expect((e as DispatchRejectionError).dispatchKind).toBe("control");
		}
	});

	it("schedule action throws DISPATCH_SCHEDULE_FORBIDDEN", () => {
		try {
			autoragPrepare({ action: "schedule" }, { configuredModel: "test/test" });
			expect.fail("should throw");
		} catch (e) {
			expect((e as DispatchRejectionError).code).toBe("DISPATCH_SCHEDULE_FORBIDDEN");
			expect((e as DispatchRejectionError).dispatchKind).toBe("schedule");
		}
	});

	it("unknown action throws DISPATCH_ACTION_UNKNOWN", () => {
		try {
			autoragPrepare({ action: "foobar" }, { configuredModel: "test/test" });
			expect.fail("should throw");
		} catch (e) {
			expect((e as DispatchRejectionError).code).toBe("DISPATCH_ACTION_UNKNOWN");
			expect((e as DispatchRejectionError).dispatchKind).toBe("unknown");
		}
	});

	it("non-string action with no launch shape throws DISPATCH_ACTION_UNKNOWN", () => {
		try {
			autoragPrepare({ action: 42 }, { configuredModel: "test/test" });
			expect.fail("should throw");
		} catch (e) {
			expect((e as DispatchRejectionError).code).toBe("DISPATCH_ACTION_UNKNOWN");
		}
	});

	it("case-variant action (Create) with launch fields throws DISPATCH_ACTION_UNKNOWN", () => {
		try {
			autoragPrepare(
				{ action: "Create", agent: "autorag-explorer", task: "x", cwd: "/tmp" },
				{ configuredModel: "test/test" },
			);
			expect.fail("should throw");
		} catch (e) {
			expect((e as DispatchRejectionError).code).toBe("DISPATCH_ACTION_UNKNOWN");
		}
	});

	it("case-variant action (LIST) with launch fields throws DISPATCH_ACTION_UNKNOWN", () => {
		try {
			autoragPrepare({ action: "LIST", agentScope: "user" }, { configuredModel: "test/test" });
			expect.fail("should throw");
		} catch (e) {
			expect((e as DispatchRejectionError).code).toBe("DISPATCH_ACTION_UNKNOWN");
		}
	});

	it("case-variant action (Interrupt) with launch fields throws DISPATCH_ACTION_UNKNOWN", () => {
		try {
			autoragPrepare(
				{ action: "Interrupt", agent: "autorag-explorer", task: "x", cwd: "/tmp" },
				{ configuredModel: "test/test" },
			);
			expect.fail("should throw");
		} catch (e) {
			expect((e as DispatchRejectionError).code).toBe("DISPATCH_ACTION_UNKNOWN");
		}
	});

	it("numeric action with launch fields throws DISPATCH_ACTION_UNKNOWN", () => {
		try {
			autoragPrepare(
				{ action: 42, agent: "autorag-explorer", task: "x", cwd: "/tmp" },
				{ configuredModel: "test/test" },
			);
			expect.fail("should throw");
		} catch (e) {
			expect((e as DispatchRejectionError).code).toBe("DISPATCH_ACTION_UNKNOWN");
		}
	});

	it("exact lowercase execution alias 'single' with valid launch shape prepares launch", () => {
		const result = autoragPrepare(
			{ action: "single", agent: "autorag-explorer", task: "x", cwd: "/tmp" },
			{ configuredModel: "test/test" },
		);
		expect(result.action).toBe("single");
		expect(result.artifacts).toBe(false);
		expect(result.agentScope).toBe("user");
	});

	it("exact lowercase execution alias 'parallel' with valid fanout shape prepares launch", () => {
		const result = autoragPrepare(
			{
				action: "parallel",
				tasks: [{ agent: "autorag-explorer", task: "x", cwd: "/tmp" }],
			},
			{ configuredModel: "test/test" },
		);
		expect(result.action).toBe("parallel");
		expect(result.artifacts).toBe(false);
		expect(result.agentScope).toBe("user");
		const tasks = result.tasks as Record<string, unknown>[];
		expect(tasks[0].model).toBe("test/test");
	});

	it("exact lowercase execution alias 'tasks' with valid fanout shape prepares launch", () => {
		const result = autoragPrepare(
			{
				action: "tasks",
				tasks: [{ agent: "autorag-explorer", task: "x", cwd: "/tmp" }],
			},
			{ configuredModel: "test/test" },
		);
		expect(result.action).toBe("tasks");
		expect(result.artifacts).toBe(false);
		expect(result.agentScope).toBe("user");
	});

	it("execution alias without launch shape throws DISPATCH_ACTION_UNKNOWN", () => {
		try {
			autoragPrepare({ action: "single" }, { configuredModel: "test/test" });
			expect.fail("should throw");
		} catch (e) {
			expect((e as DispatchRejectionError).code).toBe("DISPATCH_ACTION_UNKNOWN");
		}
	});

	it("case-variant execution alias (Single) with launch shape throws DISPATCH_ACTION_UNKNOWN", () => {
		try {
			autoragPrepare(
				{ action: "Single", agent: "autorag-explorer", task: "x", cwd: "/tmp" },
				{ configuredModel: "test/test" },
			);
			expect.fail("should throw");
		} catch (e) {
			expect((e as DispatchRejectionError).code).toBe("DISPATCH_ACTION_UNKNOWN");
		}
	});
});

// ---------------------------------------------------------------------------
// Leaf traversal
// ---------------------------------------------------------------------------

describe("leaf traversal", () => {
	it("single root leaf", () => {
		const leaves = getLaunchLeaves({ agent: "autorag-explorer", task: "x", cwd: "/tmp" });
		expect(leaves).toHaveLength(1);
		expect(leaves[0].path).toBe("");
	});

	it("tasks fanout leaves", () => {
		const leaves = getLaunchLeaves({
			agentScope: "user",
			tasks: [
				{ agent: "autorag-explorer", task: "t1", cwd: "/a" },
				{ agent: "autorag-explorer", task: "t2", cwd: "/b" },
			],
		});
		expect(leaves).toHaveLength(2);
		expect(leaves[0].path).toBe(".tasks[0]");
		expect(leaves[1].path).toBe(".tasks[1]");
	});

	it("chain leaves", () => {
		const leaves = getLaunchLeaves({
			chain: [
				{ agent: "autorag-explorer", task: "t1", cwd: "/a" },
				{ agent: "autorag-explorer", task: "t2", cwd: "/b" },
			],
		});
		expect(leaves).toHaveLength(2);
		expect(leaves[0].path).toBe(".chain[0]");
	});

	it("parallel leaves", () => {
		const leaves = getLaunchLeaves({
			parallel: [{ agent: "autorag-explorer", task: "t1", cwd: "/a" }],
		});
		expect(leaves).toHaveLength(1);
		expect(leaves[0].path).toBe(".parallel[0]");
	});

	it("nested tasks within parallel", () => {
		const leaves = getLaunchLeaves({
			parallel: [
				{
					tasks: [
						{ agent: "autorag-explorer", task: "t1", cwd: "/a" },
						{ agent: "autorag-explorer", task: "t2", cwd: "/b" },
					],
				},
			],
		});
		expect(leaves).toHaveLength(2);
		expect(leaves[0].path).toBe(".parallel[0].tasks[0]");
		expect(leaves[1].path).toBe(".parallel[0].tasks[1]");
	});
});

// ---------------------------------------------------------------------------
// G) Assignment V1 parser
// ---------------------------------------------------------------------------

describe("Assignment V1 parser", () => {
	const validV1Body = JSON.stringify({
		originalQuery: "test query",
		method: "bm25",
		queryVariants: ["variant1", "variant2"],
	});

	it("parses valid V1 block", () => {
		const task = `${AUTORAG_ASSIGNMENT_V1_START}\n${validV1Body}\n${AUTORAG_ASSIGNMENT_V1_END}`;
		const parsed = parseAssignmentV1(task);
		expect(parsed).toBeDefined();
		expect(parsed?.originalQuery).toBe("test query");
		expect(parsed?.method).toBe("bm25");
		expect(parsed?.queryVariants).toEqual(["variant1", "variant2"]);
		expect(parsed?.isV1).toBe(true);
	});

	it("handles CRLF line endings", () => {
		const task = `${AUTORAG_ASSIGNMENT_V1_START}\r\n${validV1Body}\r\n${AUTORAG_ASSIGNMENT_V1_END}\r\n`;
		const parsed = parseAssignmentV1(task);
		expect(parsed).toBeDefined();
		expect(parsed?.originalQuery).toBe("test query");
	});

	it("end sentinel at EOF is valid", () => {
		const task = `${AUTORAG_ASSIGNMENT_V1_START}\n${validV1Body}\n${AUTORAG_ASSIGNMENT_V1_END}`;
		const parsed = parseAssignmentV1(task);
		expect(parsed).toBeDefined();
	});

	it("unterminated V1 block rejected", () => {
		const task = `${AUTORAG_ASSIGNMENT_V1_START}\n${validV1Body}`;
		expect(() => parseAssignmentV1(task)).toThrow(DispatchRejectionError);
	});

	it("duplicate start sentinel rejected", () => {
		const task = `${AUTORAG_ASSIGNMENT_V1_START}\n${AUTORAG_ASSIGNMENT_V1_START}\n${validV1Body}\n${AUTORAG_ASSIGNMENT_V1_END}`;
		expect(() => parseAssignmentV1(task)).toThrow(DispatchRejectionError);
	});

	it("empty body rejected", () => {
		const task = `${AUTORAG_ASSIGNMENT_V1_START}\n\n${AUTORAG_ASSIGNMENT_V1_END}`;
		expect(() => parseAssignmentV1(task)).toThrow(DispatchRejectionError);
	});

	it("extra keys rejected", () => {
		const body = JSON.stringify({
			originalQuery: "test",
			method: "bm25",
			queryVariants: ["v1"],
			extra: "field",
		});
		const task = `${AUTORAG_ASSIGNMENT_V1_START}\n${body}\n${AUTORAG_ASSIGNMENT_V1_END}`;
		expect(() => parseAssignmentV1(task)).toThrow(DispatchRejectionError);
	});

	it("missing keys rejected", () => {
		const body = JSON.stringify({ originalQuery: "test", method: "bm25" });
		const task = `${AUTORAG_ASSIGNMENT_V1_START}\n${body}\n${AUTORAG_ASSIGNMENT_V1_END}`;
		expect(() => parseAssignmentV1(task)).toThrow(DispatchRejectionError);
	});

	it("wrong type originalQuery rejected", () => {
		const body = JSON.stringify({ originalQuery: 42, method: "bm25", queryVariants: ["v1"] });
		const task = `${AUTORAG_ASSIGNMENT_V1_START}\n${body}\n${AUTORAG_ASSIGNMENT_V1_END}`;
		expect(() => parseAssignmentV1(task)).toThrow(DispatchRejectionError);
	});

	it("empty queryVariants rejected", () => {
		const body = JSON.stringify({ originalQuery: "test", method: "bm25", queryVariants: [] });
		const task = `${AUTORAG_ASSIGNMENT_V1_START}\n${body}\n${AUTORAG_ASSIGNMENT_V1_END}`;
		expect(() => parseAssignmentV1(task)).toThrow(DispatchRejectionError);
	});

	it("empty string in queryVariants rejected", () => {
		const body = JSON.stringify({ originalQuery: "test", method: "bm25", queryVariants: ["v1", ""] });
		const task = `${AUTORAG_ASSIGNMENT_V1_START}\n${body}\n${AUTORAG_ASSIGNMENT_V1_END}`;
		expect(() => parseAssignmentV1(task)).toThrow(DispatchRejectionError);
	});

	it("returns undefined when no V1 sentinel present", () => {
		expect(parseAssignmentV1("just some text")).toBeUndefined();
	});

	it("trimmed originalQuery", () => {
		const body = JSON.stringify({ originalQuery: "  test query  ", method: "bm25", queryVariants: ["v1"] });
		const task = `${AUTORAG_ASSIGNMENT_V1_START}\n${body}\n${AUTORAG_ASSIGNMENT_V1_END}`;
		const parsed = parseAssignmentV1(task);
		expect(parsed?.originalQuery).toBe("test query");
	});
});

// ---------------------------------------------------------------------------
// G) Legacy assignment parser
// ---------------------------------------------------------------------------

describe("legacy assignment parser", () => {
	it("parses standard legacy format", () => {
		const task = `Original query: test query\nSelected retrieval method: bm25\nQuery variants: var1; var2`;
		const parsed = parseLegacyAssignment(task);
		expect(parsed).toBeDefined();
		expect(parsed?.originalQuery).toBe("test query");
		expect(parsed?.method).toBe("bm25");
		expect(parsed?.queryVariants).toEqual(["var1", "var2"]);
		expect(parsed?.isV1).toBe(false);
	});

	it("handles optional bullet prefix", () => {
		const task = `- Original query: test query\n- Selected retrieval method: bm25\n- Query variants: var1`;
		const parsed = parseLegacyAssignment(task);
		expect(parsed?.originalQuery).toBe("test query");
	});

	it("handles **bold** labels", () => {
		const task = `**Original query:** test query\n**Selected retrieval method:** bm25\n**Query variants:** var1`;
		const parsed = parseLegacyAssignment(task);
		expect(parsed?.originalQuery).toBe("test query");
	});

	it("handles 'Retrieval method' label variant", () => {
		const task = `Original query: test query\nRetrieval method: bm25\nQuery variants: var1`;
		const parsed = parseLegacyAssignment(task);
		expect(parsed?.method).toBe("bm25");
	});

	it("handles 'Query variant' singular label", () => {
		const task = `Original query: test query\nSelected retrieval method: bm25\nQuery variant: var1`;
		const parsed = parseLegacyAssignment(task);
		expect(parsed).toBeDefined();
	});

	it("reorder OK", () => {
		const task = `Selected retrieval method: bm25\nOriginal query: test query\nQuery variants: var1`;
		const parsed = parseLegacyAssignment(task);
		expect(parsed?.originalQuery).toBe("test query");
	});

	it("empty variant segments rejected", () => {
		const task = `Original query: test query\nSelected retrieval method: bm25\nQuery variants: var1;; var2`;
		expect(() => parseLegacyAssignment(task)).toThrow(DispatchRejectionError);
	});

	it("later label-looking line rejected", () => {
		const task = `Original query: test query\nSelected retrieval method: bm25\nQuery variants: var1\nSome text\nOriginal query: another`;
		expect(() => parseLegacyAssignment(task)).toThrow(DispatchRejectionError);
	});

	it("prefix/suffix outside three lines preserved", () => {
		const task = `Some preamble text\nOriginal query: test query\nSelected retrieval method: bm25\nQuery variants: var1\nSome suffix text`;
		const parsed = parseLegacyAssignment(task);
		expect(parsed?.originalQuery).toBe("test query");
	});

	it("returns undefined when no labels found", () => {
		expect(parseLegacyAssignment("just some random text")).toBeUndefined();
	});
});

// ---------------------------------------------------------------------------
// G) Role-line insertion (byte-idempotent)
// ---------------------------------------------------------------------------

describe("role-line insertion", () => {
	it("inserts role lines after V1 end sentinel", () => {
		const task = `<<<AUTORAG_ASSIGNMENT_V1>>>\n${JSON.stringify({ originalQuery: "q", method: "m", queryVariants: ["v"] })}\n<<<END_AUTORAG_ASSIGNMENT_V1>>>`;
		const result = ensureRoleLines(task);
		expect(result).toContain(ROLE_LINE_RETRIEVED_AT);
		expect(result).toContain(ROLE_LINE_TEMPORAL_METADATA);
		// Lines should be after the end sentinel
		const lines = result.split("\n");
		const endIdx = lines.indexOf("<<<END_AUTORAG_ASSIGNMENT_V1>>>");
		expect(lines[endIdx + 1]).toBe(ROLE_LINE_RETRIEVED_AT);
		expect(lines[endIdx + 2]).toBe(ROLE_LINE_TEMPORAL_METADATA);
	});

	it("inserts role lines after legacy label lines", () => {
		const task = `Original query: test\nSelected retrieval method: bm25\nQuery variants: var1`;
		const result = ensureRoleLines(task);
		const lines = result.split("\n");
		// Find the last label line
		const lastLabelIdx = lines.findIndex((l) => l.startsWith("Query variants"));
		expect(lines[lastLabelIdx + 1]).toBe(ROLE_LINE_RETRIEVED_AT);
		expect(lines[lastLabelIdx + 2]).toBe(ROLE_LINE_TEMPORAL_METADATA);
	});

	it("byte-idempotent: repeated rendering produces same output", () => {
		const task = `<<<AUTORAG_ASSIGNMENT_V1>>>\n${JSON.stringify({ originalQuery: "q", method: "m", queryVariants: ["v"] })}\n<<<END_AUTORAG_ASSIGNMENT_V1>>>`;
		const first = ensureRoleLines(task);
		const second = ensureRoleLines(first);
		expect(second).toBe(first);
	});

	it("removes existing role lines before inserting", () => {
		const task = `<<<AUTORAG_ASSIGNMENT_V1>>>\n${JSON.stringify({ originalQuery: "q", method: "m", queryVariants: ["v"] })}\n<<<END_AUTORAG_ASSIGNMENT_V1>>>\n${ROLE_LINE_RETRIEVED_AT}\n${ROLE_LINE_TEMPORAL_METADATA}`;
		const result = ensureRoleLines(task);
		// Should have exactly one of each
		const lines = result.split("\n");
		const retrievedAtCount = lines.filter((l) => l === ROLE_LINE_RETRIEVED_AT).length;
		const temporalCount = lines.filter((l) => l === ROLE_LINE_TEMPORAL_METADATA).length;
		expect(retrievedAtCount).toBe(1);
		expect(temporalCount).toBe(1);
	});

	it("preserves suffix bytes", () => {
		const task = `<<<AUTORAG_ASSIGNMENT_V1>>>\n${JSON.stringify({ originalQuery: "q", method: "m", queryVariants: ["v"] })}\n<<<END_AUTORAG_ASSIGNMENT_V1>>>\nSome suffix`;
		const result = ensureRoleLines(task);
		expect(result).toContain("Some suffix");
	});

	it("handles CRLF in role lines (CR-strip)", () => {
		const task = `<<<AUTORAG_ASSIGNMENT_V1>>>\r\n${JSON.stringify({ originalQuery: "q", method: "m", queryVariants: ["v"] })}\r\n<<<END_AUTORAG_ASSIGNMENT_V1>>>\r\n${ROLE_LINE_RETRIEVED_AT}\r\n${ROLE_LINE_TEMPORAL_METADATA}\r\n`;
		const result = ensureRoleLines(task);
		// Should still have exactly one of each (existing ones removed via CR-strip)
		const lines = result.split(/\r?\n/);
		const retrievedAtCount = lines.filter((l) => l === ROLE_LINE_RETRIEVED_AT).length;
		expect(retrievedAtCount).toBe(1);
	});
});

// ---------------------------------------------------------------------------
// parseAssignment (combined V1 + legacy)
// ---------------------------------------------------------------------------

describe("parseAssignment", () => {
	it("tries V1 first", () => {
		const task = `<<<AUTORAG_ASSIGNMENT_V1>>>\n${JSON.stringify({ originalQuery: "q", method: "m", queryVariants: ["v"] })}\n<<<END_AUTORAG_ASSIGNMENT_V1>>>`;
		const parsed = parseAssignment(task);
		expect(parsed?.isV1).toBe(true);
	});

	it("falls back to legacy", () => {
		const task = `Original query: test\nSelected retrieval method: bm25\nQuery variants: var1`;
		const parsed = parseAssignment(task);
		expect(parsed?.isV1).toBe(false);
		expect(parsed?.originalQuery).toBe("test");
	});

	it("returns undefined when neither format found", () => {
		expect(parseAssignment("random text")).toBeUndefined();
	});
});

// ---------------------------------------------------------------------------
// Post-schema validation (security invariants)
// ---------------------------------------------------------------------------

describe("validateLaunchPostSchema", () => {
	const baseCtx = {
		configuredModel: "test-provider/test-model",
		currentQuery: "test query",
		allowedRoots: ["/allowed/root"],
		workspaceRoot: "/workspace",
	};

	it("returns DISPATCH_NO_ACTIVE_QUERY when currentQuery is undefined", () => {
		const rejection = validateLaunchPostSchema(
			{
				agent: "autorag-explorer",
				artifacts: false,
				agentScope: "user",
				model: "test-provider/test-model",
				task: "x",
				cwd: "/allowed/root",
			},
			{ ...baseCtx, currentQuery: undefined },
		);
		expect(rejection?.code).toBe("DISPATCH_NO_ACTIVE_QUERY");
	});

	it("returns DISPATCH_QUERY_MISMATCH when originalQuery differs", () => {
		const v1Body = JSON.stringify({ originalQuery: "different query", method: "bm25", queryVariants: ["v"] });
		const task = `<<<AUTORAG_ASSIGNMENT_V1>>>\n${v1Body}\n<<<END_AUTORAG_ASSIGNMENT_V1>>>\n${ROLE_LINE_RETRIEVED_AT}\n${ROLE_LINE_TEMPORAL_METADATA}`;
		const rejection = validateLaunchPostSchema(
			{
				agent: "autorag-explorer",
				artifacts: false,
				agentScope: "user",
				model: "test-provider/test-model",
				task,
				cwd: "/allowed/root",
			},
			baseCtx,
		);
		expect(rejection?.code).toBe("DISPATCH_QUERY_MISMATCH");
	});

	it("auto-normalizes role lines when missing (no rejection)", () => {
		const v1Body = JSON.stringify({ originalQuery: "test query", method: "bm25", queryVariants: ["v"] });
		const task = `<<<AUTORAG_ASSIGNMENT_V1>>>\n${v1Body}\n<<<END_AUTORAG_ASSIGNMENT_V1>>>`;
		const args: Record<string, unknown> = {
			agent: "autorag-explorer",
			artifacts: false,
			agentScope: "user",
			model: "test-provider/test-model",
			task,
			cwd: "/allowed/root",
		};
		const rejection = validateLaunchPostSchema(args, baseCtx);
		// Canonical role lines are normalized idempotently by ensureRoleLines,
		// never rejected with a role-metadata error code.
		expect(rejection?.code).toBe("DISPATCH_CWD_OUTSIDE_ROOTS");
		// Verify role lines were rendered into the task
		expect(args.task as string).toContain(ROLE_LINE_RETRIEVED_AT);
		expect(args.task as string).toContain(ROLE_LINE_TEMPORAL_METADATA);
	});

	it("returns DISPATCH_ASSIGNMENT_INVALID when task is not parseable", () => {
		const rejection = validateLaunchPostSchema(
			{
				agent: "autorag-explorer",
				artifacts: false,
				agentScope: "user",
				model: "test-provider/test-model",
				task: "not an assignment",
				cwd: "/allowed/root",
			},
			baseCtx,
		);
		expect(rejection?.code).toBe("DISPATCH_ASSIGNMENT_INVALID");
	});

	it("passes with valid V1 assignment + role lines + matching query", () => {
		const v1Body = JSON.stringify({ originalQuery: "test query", method: "bm25", queryVariants: ["v"] });
		const task = `<<<AUTORAG_ASSIGNMENT_V1>>>\n${v1Body}\n<<<END_AUTORAG_ASSIGNMENT_V1>>>\n${ROLE_LINE_RETRIEVED_AT}\n${ROLE_LINE_TEMPORAL_METADATA}`;
		// Note: this will fail on cwd realpath since /allowed/root doesn't exist
		// but the rejection should be about cwd, not about the assignment
		const rejection = validateLaunchPostSchema(
			{
				agent: "autorag-explorer",
				artifacts: false,
				agentScope: "user",
				model: "test-provider/test-model",
				task,
				cwd: "/allowed/root",
			},
			baseCtx,
		);
		// Will fail on CWD since /allowed/root doesn't exist on disk
		expect(rejection?.code).toBe("DISPATCH_CWD_OUTSIDE_ROOTS");
	});

	it("field-addresses malformed V1 assignment in fanout to the correct leaf path", () => {
		// Unterminated V1 block (no end sentinel) → parseAssignmentV1 throws a
		// DispatchRejectionError with the generic field ".task". The validator
		// must remap it to the leaf-addressed ".tasks[0].task".
		const malformedTask = `<<<AUTORAG_ASSIGNMENT_V1>>>\n${JSON.stringify({ originalQuery: "test query", method: "bm25", queryVariants: ["v"] })}`;
		const rejection = validateLaunchPostSchema(
			{
				agentScope: "user",
				artifacts: false,
				tasks: [
					{
						agent: "autorag-explorer",
						model: "test-provider/test-model",
						task: malformedTask,
						cwd: "/allowed/root",
					},
					{
						agent: "autorag-explorer",
						model: "test-provider/test-model",
						task: `<<<AUTORAG_ASSIGNMENT_V1>>>\n${JSON.stringify({ originalQuery: "test query", method: "bm25", queryVariants: ["v"] })}\n<<<END_AUTORAG_ASSIGNMENT_V1>>>`,
						cwd: "/allowed/root",
					},
				],
			},
			baseCtx,
		);
		expect(rejection?.code).toBe("DISPATCH_ASSIGNMENT_INVALID");
		// Field must be the leaf path, not the generic ".task" the parser emits.
		expect(rejection?.field).toBe(".tasks[0].task");
		expect(rejection?.dispatchKind).toBe("launch");
	});

	it("field-addresses malformed legacy assignment in parallel fanout to the correct leaf path", () => {
		// Legacy with a duplicate label → parseLegacyAssignment throws with the
		// generic field ".task"; must be remapped to the parallel leaf path.
		const malformedTask = `Original query: test query\nOriginal query: dup\nSelected retrieval method: bm25\nQuery variants: var1`;
		const rejection = validateLaunchPostSchema(
			{
				agentScope: "user",
				artifacts: false,
				parallel: [
					{
						agent: "autorag-explorer",
						model: "test-provider/test-model",
						task: malformedTask,
						cwd: "/allowed/root",
					},
				],
			},
			baseCtx,
		);
		expect(rejection?.code).toBe("DISPATCH_ASSIGNMENT_INVALID");
		// Field must be the leaf path, not the generic ".task" the parser emits.
		expect(rejection?.field).toBe(".parallel[0].task");
		expect(rejection?.dispatchKind).toBe("launch");
	});

	it("field-addresses malformed V1 assignment in chain to the correct leaf path", () => {
		// Empty V1 body → parseAssignmentV1 throws.
		const malformedTask = `<<<AUTORAG_ASSIGNMENT_V1>>>\n   \n<<<END_AUTORAG_ASSIGNMENT_V1>>>`;
		const rejection = validateLaunchPostSchema(
			{
				agentScope: "user",
				artifacts: false,
				chain: [
					{
						agent: "autorag-explorer",
						model: "test-provider/test-model",
						task: malformedTask,
						cwd: "/allowed/root",
					},
				],
			},
			baseCtx,
		);
		expect(rejection?.code).toBe("DISPATCH_ASSIGNMENT_INVALID");
		expect(rejection?.field).toBe(".chain[0].task");
		expect(rejection?.dispatchKind).toBe("launch");
	});

	it("canonicalizes off-position role lines to one adjacent pair at the insertion point", () => {
		const v1Body = JSON.stringify({ originalQuery: "test query", method: "bm25", queryVariants: ["v"] });
		// Both role lines already present but scattered: one before the block, one
		// after, plus a duplicate after the end sentinel. ensureRoleLines must strip
		// all existing canonical role lines and re-insert exactly one adjacent pair.
		const task =
			`${ROLE_LINE_TEMPORAL_METADATA}\n` +
			`<<<AUTORAG_ASSIGNMENT_V1>>>\n${v1Body}\n<<<END_AUTORAG_ASSIGNMENT_V1>>>\n` +
			`${ROLE_LINE_RETRIEVED_AT}\n${ROLE_LINE_RETRIEVED_AT}\n` +
			`Suffix line`;
		const args: Record<string, unknown> = {
			agent: "autorag-explorer",
			artifacts: false,
			agentScope: "user",
			model: "test-provider/test-model",
			task,
			cwd: "/allowed/root",
		};
		const rejection = validateLaunchPostSchema(args, baseCtx);
		// Assignment/query are valid; only cwd (nonexistent) should reject.
		expect(rejection?.code).toBe("DISPATCH_CWD_OUTSIDE_ROOTS");

		const normalized = args.task as string;
		const lines = normalized.split("\n");
		const retrievedCount = lines.filter((l) => l === ROLE_LINE_RETRIEVED_AT).length;
		const temporalCount = lines.filter((l) => l === ROLE_LINE_TEMPORAL_METADATA).length;
		// Exactly one of each canonical role line.
		expect(retrievedCount).toBe(1);
		expect(temporalCount).toBe(1);
		// Adjacent pair at the deterministic insertion point (immediately after
		// the V1 end sentinel), in canonical order.
		const endIdx = lines.indexOf("<<<END_AUTORAG_ASSIGNMENT_V1>>>");
		expect(endIdx).toBeGreaterThan(-1);
		expect(lines[endIdx + 1]).toBe(ROLE_LINE_RETRIEVED_AT);
		expect(lines[endIdx + 2]).toBe(ROLE_LINE_TEMPORAL_METADATA);
		// Suffix preserved after the role-line pair.
		expect(lines[endIdx + 3]).toBe("Suffix line");
		// The stray pre-block temporal line is gone.
		expect(lines[0]).not.toBe(ROLE_LINE_TEMPORAL_METADATA);
	});

	it("canonicalizes already-present but off-position role lines (both present) without skipping", () => {
		const v1Body = JSON.stringify({ originalQuery: "test query", method: "bm25", queryVariants: ["v"] });
		// Both role lines present but BEFORE the V1 block (off-position). Even
		// though hasRoleLines would have been true, the validator must still
		// canonicalize placement to the insertion point.
		const task =
			`${ROLE_LINE_RETRIEVED_AT}\n${ROLE_LINE_TEMPORAL_METADATA}\n` +
			`<<<AUTORAG_ASSIGNMENT_V1>>>\n${v1Body}\n<<<END_AUTORAG_ASSIGNMENT_V1>>>`;
		const args: Record<string, unknown> = {
			agent: "autorag-explorer",
			artifacts: false,
			agentScope: "user",
			model: "test-provider/test-model",
			task,
			cwd: "/allowed/root",
		};
		const rejection = validateLaunchPostSchema(args, baseCtx);
		expect(rejection?.code).toBe("DISPATCH_CWD_OUTSIDE_ROOTS");

		const lines = (args.task as string).split("\n");
		const retrievedCount = lines.filter((l) => l === ROLE_LINE_RETRIEVED_AT).length;
		const temporalCount = lines.filter((l) => l === ROLE_LINE_TEMPORAL_METADATA).length;
		expect(retrievedCount).toBe(1);
		expect(temporalCount).toBe(1);
		// Moved from before the block to after the end sentinel (canonical placement).
		const endIdx = lines.indexOf("<<<END_AUTORAG_ASSIGNMENT_V1>>>");
		expect(lines[endIdx + 1]).toBe(ROLE_LINE_RETRIEVED_AT);
		expect(lines[endIdx + 2]).toBe(ROLE_LINE_TEMPORAL_METADATA);
		expect(lines[0]).not.toBe(ROLE_LINE_RETRIEVED_AT);
	});
});

// ---------------------------------------------------------------------------
// Skeleton selection
// ---------------------------------------------------------------------------

describe("skeleton selection", () => {
	it("admin skeleton for mutation/control/schedule/unknown errors", () => {
		try {
			autoragPrepare({ action: "create" }, { configuredModel: "test/test" });
		} catch (e) {
			expect((e as DispatchRejectionError).skeleton).toBe(SKELETON_ADMIN);
		}
	});

	it("single skeleton for single-leaf launch errors", () => {
		try {
			autoragPrepare(
				{ agent: "autorag-explorer", artifacts: true, task: "x", cwd: "/tmp" },
				{ configuredModel: "test/test" },
			);
		} catch (e) {
			expect((e as DispatchRejectionError).skeleton).toBe(SKELETON_SINGLE);
		}
	});

	it("fanout skeleton for multi-leaf launch errors", () => {
		try {
			autoragPrepare(
				{
					agentScope: "user",
					artifacts: true,
					tasks: [
						{ agent: "autorag-explorer", task: "t1", cwd: "/a" },
						{ agent: "autorag-explorer", task: "t2", cwd: "/b" },
					],
				},
				{ configuredModel: "test/test" },
			);
		} catch (e) {
			expect((e as DispatchRejectionError).skeleton).toBe(SKELETON_FANOUT);
		}
	});
});

// ---------------------------------------------------------------------------
// rejectForbiddenAction
// ---------------------------------------------------------------------------

describe("rejectForbiddenAction", () => {
	it("mutation action → DISPATCH_ADMIN_MUTATION_FORBIDDEN", () => {
		const r = rejectForbiddenAction("create");
		expect(r.code).toBe("DISPATCH_ADMIN_MUTATION_FORBIDDEN");
		expect(r.dispatchKind).toBe("mutation");
	});

	it("control action → DISPATCH_CONTROL_FORBIDDEN", () => {
		const r = rejectForbiddenAction("steer");
		expect(r.code).toBe("DISPATCH_CONTROL_FORBIDDEN");
	});

	it("schedule action → DISPATCH_SCHEDULE_FORBIDDEN", () => {
		const r = rejectForbiddenAction("schedule");
		expect(r.code).toBe("DISPATCH_SCHEDULE_FORBIDDEN");
	});

	it("unknown action → DISPATCH_ACTION_UNKNOWN", () => {
		const r = rejectForbiddenAction("foobar");
		expect(r.code).toBe("DISPATCH_ACTION_UNKNOWN");
	});
});

// ---------------------------------------------------------------------------
// createDispatchRejectionError
// ---------------------------------------------------------------------------

describe("createDispatchRejectionError", () => {
	it("creates error with proper exactFix and skeleton", () => {
		const error = createDispatchRejectionError({
			code: "DISPATCH_QUERY_MISMATCH",
			field: ".task",
			dispatchKind: "launch",
		});
		expect(error.code).toBe("DISPATCH_QUERY_MISMATCH");
		expect(error.exactFix).toBe("set originalQuery to the active user query verbatim");
		expect(error.forceCorrectable).toBe(true);
		expect(error.skeleton).toBe(SKELETON_SINGLE);
	});

	it("admin rejection uses admin skeleton", () => {
		const error = createDispatchRejectionError({
			code: "DISPATCH_ADMIN_MUTATION_FORBIDDEN",
			field: "action",
			dispatchKind: "mutation",
		});
		expect(error.skeleton).toBe(SKELETON_ADMIN);
		expect(error.forceCorrectable).toBe(false);
	});

	it("fanout skeleton: launch rejection with tasks args uses fanout skeleton", () => {
		const error = createDispatchRejectionError(
			{
				code: "DISPATCH_QUERY_MISMATCH",
				field: ".tasks[0].task",
				dispatchKind: "launch",
			},
			{
				agentScope: "user",
				artifacts: false,
				tasks: [{ agent: "autorag-explorer", task: "t1", cwd: "/a" }],
			},
		);
		expect(error.skeleton).toBe(SKELETON_FANOUT);
	});

	it("single skeleton: launch rejection without args defaults to single", () => {
		const error = createDispatchRejectionError({
			code: "DISPATCH_QUERY_MISMATCH",
			field: ".task",
			dispatchKind: "launch",
		});
		expect(error.skeleton).toBe(SKELETON_SINGLE);
	});

	it("admin skeleton: non-launch rejection ignores args shape", () => {
		const error = createDispatchRejectionError(
			{
				code: "DISPATCH_ADMIN_MUTATION_FORBIDDEN",
				field: "action",
				dispatchKind: "mutation",
			},
			{
				agentScope: "user",
				artifacts: false,
				tasks: [{ agent: "autorag-explorer", task: "t1", cwd: "/a" }],
			},
		);
		expect(error.skeleton).toBe(SKELETON_ADMIN);
	});

	it("fanout skeleton: launch rejection with chain args uses fanout skeleton", () => {
		const error = createDispatchRejectionError(
			{
				code: "DISPATCH_MODEL_MISMATCH",
				field: ".chain[0].model",
				dispatchKind: "launch",
			},
			{
				agentScope: "user",
				artifacts: false,
				chain: [{ agent: "autorag-explorer", task: "t1", cwd: "/a" }],
			},
		);
		expect(error.skeleton).toBe(SKELETON_FANOUT);
	});

	it("fanout skeleton: launch rejection with parallel args uses fanout skeleton", () => {
		const error = createDispatchRejectionError(
			{
				code: "DISPATCH_AGENT_IDENTITY",
				field: ".parallel[0].agent",
				dispatchKind: "launch",
			},
			{
				agentScope: "user",
				artifacts: false,
				parallel: [{ agent: "custom", task: "t1", cwd: "/a" }],
			},
		);
		expect(error.skeleton).toBe(SKELETON_FANOUT);
	});
});

// ---------------------------------------------------------------------------
// Canonical role lines
// ---------------------------------------------------------------------------

describe("canonical role lines", () => {
	it("role line literals are exact", () => {
		expect(ROLE_LINE_RETRIEVED_AT).toBe("Required handoff: include retrievedAt.");
		expect(ROLE_LINE_TEMPORAL_METADATA).toBe("Required handoff: include temporal metadata.");
	});
});

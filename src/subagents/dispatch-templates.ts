/**
 * Shared dispatch template builders, skeleton literals, and anti-examples for
 * AutoRAG explorer subagent dispatch.
 *
 * This module is the single source of truth for:
 * - Assignment V1 sentinel/role-line constants and block builder
 * - Single-root and multi-root success dispatch payload builders
 * - Compact skeleton literals consumed by `formatBlockReason`
 * - Anti-examples for rejected dispatch patterns
 * - Stable coded patch-error reference (single source of truth for the error catalog)
 *
 * `buildSearchPrompt` and prompt tests reuse these builders so no divergent
 * template text exists inside TypeScript.
 */

// ---------------------------------------------------------------------------
// V1 sentinel constants (exact)
// ---------------------------------------------------------------------------

export const AUTORAG_ASSIGNMENT_V1_START = "<<<AUTORAG_ASSIGNMENT_V1>>>";

export const AUTORAG_ASSIGNMENT_V1_END = "<<<END_AUTORAG_ASSIGNMENT_V1>>>";

// ---------------------------------------------------------------------------
// Canonical role lines (exact, LF-joined pair)
// ---------------------------------------------------------------------------

export const AUTORAG_ROLE_LINE_RETRIEVED_AT = "Required handoff: include retrievedAt.";

export const AUTORAG_ROLE_LINE_TEMPORAL_METADATA = "Required handoff: include temporal metadata.";

export const AUTORAG_CANONICAL_ROLE_LINES = `${AUTORAG_ROLE_LINE_RETRIEVED_AT}\n${AUTORAG_ROLE_LINE_TEMPORAL_METADATA}`;

// ---------------------------------------------------------------------------
// Placeholder constants (non-sensitive, safe for prompts/telemetry/skeletons)
// ---------------------------------------------------------------------------

export const PLACEHOLDER_CONFIGURED_MODEL = "<configured-model>";

export const PLACEHOLDER_ALLOWED_ROOT = "<allowed-root>";

export const PLACEHOLDER_ASSIGNMENT_V1_BLOCK = "<Assignment V1 block>";

export const PLACEHOLDER_CALLER_QUERY = "<caller query verbatim>";

// ---------------------------------------------------------------------------
// Compact skeleton literals (for formatBlockReason error messages)
// ---------------------------------------------------------------------------

export const DISPATCH_ADMIN_SKELETON = '{"action":"list"}';

export const DISPATCH_SINGLE_SKELETON =
	'{"agentScope":"user","artifacts":false,"agent":"autorag-explorer","model":"<configured-model>","cwd":"<allowed-root>","task":"<Assignment V1 block>"}';

export const DISPATCH_FANOUT_SKELETON =
	'{"agentScope":"user","artifacts":false,"tasks":[{"agent":"autorag-explorer","model":"<configured-model>","cwd":"<allowed-root>","task":"<Assignment V1 block>"}]}';

// ---------------------------------------------------------------------------
// Assignment V1 block builder
// ---------------------------------------------------------------------------

export interface AssignmentV1Options {
	readonly originalQuery: string;
	readonly method: string;
	readonly queryVariants: readonly string[];
}

/**
 * Build a canonical Assignment V1 block string for use as the `task` field
 * value in a subagent dispatch payload.
 *
 * Output shape (LF-joined):
 * ```
 * <<<AUTORAG_ASSIGNMENT_V1>>>
 * {"originalQuery":"...","method":"...","queryVariants":["..."]}
 * <<<END_AUTORAG_ASSIGNMENT_V1>>>
 * Required handoff: include retrievedAt.
 * Required handoff: include temporal metadata.
 * ```
 */
export function buildAssignmentV1Block(opts: AssignmentV1Options): string {
	const body = JSON.stringify({
		originalQuery: opts.originalQuery,
		method: opts.method,
		queryVariants: [...opts.queryVariants],
	});
	return [
		AUTORAG_ASSIGNMENT_V1_START,
		body,
		AUTORAG_ASSIGNMENT_V1_END,
		AUTORAG_ROLE_LINE_RETRIEVED_AT,
		AUTORAG_ROLE_LINE_TEMPORAL_METADATA,
	].join("\n");
}

// ---------------------------------------------------------------------------
// Success dispatch payload builders (pretty-printed for prompt readability)
// ---------------------------------------------------------------------------

export interface SingleRootDispatchOptions {
	readonly explorerModel: string;
	readonly allowedRoot: string;
	readonly task: string;
}

/**
 * Build a single-root success dispatch payload as pretty-printed JSON.
 *
 * `agentScope` and `artifacts` are set exactly once at the top level.
 * The single leaf carries `agent`, `model`, `cwd`, and `task`.
 * Pass `PLACEHOLDER_*` constants for prompt examples or real values for
 * actual dispatch payloads.
 */
export function buildSingleRootDispatchPayload(opts: SingleRootDispatchOptions): string {
	return JSON.stringify(
		{
			agentScope: "user",
			artifacts: false,
			agent: "autorag-explorer",
			model: opts.explorerModel,
			cwd: opts.allowedRoot,
			task: opts.task,
		},
		null,
		2,
	);
}

export interface MultiRootDispatchTask {
	readonly explorerModel: string;
	readonly allowedRoot: string;
	readonly task: string;
}

/**
 * Build a multi-root success dispatch payload as pretty-printed JSON.
 *
 * `agentScope` and `artifacts` are set exactly once at the top level.
 * Each task in `tasks` carries `agent`, `model`, `cwd`, and `task` and
 * omits `agentScope`/`artifacts`. Pass one task per configured root.
 */
export function buildMultiRootDispatchPayload(tasks: readonly MultiRootDispatchTask[]): string {
	return JSON.stringify(
		{
			agentScope: "user",
			artifacts: false,
			tasks: tasks.map((t) => ({
				agent: "autorag-explorer",
				model: t.explorerModel,
				cwd: t.allowedRoot,
				task: t.task,
			})),
		},
		null,
		2,
	);
}

// ---------------------------------------------------------------------------
// Anti-examples (rejected dispatch patterns)
// ---------------------------------------------------------------------------

export interface DispatchAntiExample {
	readonly label: string;
	readonly body: string;
	readonly reason: string;
}

/**
 * Brief anti-examples showing dispatch patterns that are rejected.
 * Each body is pretty-printed JSON using non-sensitive placeholders only.
 */
export const DISPATCH_ANTIEXAMPLES: readonly DispatchAntiExample[] = [
	{
		label: "Nested artifacts/agentScope on task items",
		body: JSON.stringify(
			{
				agentScope: "user",
				artifacts: false,
				tasks: [
					{
						agent: "autorag-explorer",
						model: PLACEHOLDER_CONFIGURED_MODEL,
						cwd: PLACEHOLDER_ALLOWED_ROOT,
						artifacts: false,
						agentScope: "user",
						task: PLACEHOLDER_ASSIGNMENT_V1_BLOCK,
					},
				],
			},
			null,
			2,
		),
		reason:
			"Nested task items must omit artifacts and agentScope; these fields are set only once at the top level and are never injected into leaves.",
	},
	{
		label: "Wrong root (cwd outside configured roots)",
		body: JSON.stringify(
			{
				agentScope: "user",
				artifacts: false,
				agent: "autorag-explorer",
				model: PLACEHOLDER_CONFIGURED_MODEL,
				cwd: "/not/a/configured/root",
				task: PLACEHOLDER_ASSIGNMENT_V1_BLOCK,
			},
			null,
			2,
		),
		reason:
			"Every executable leaf cwd must be one of the configured search roots; paths outside the allowed roots or with symlink escape are rejected.",
	},
	{
		label: "Wrong agent (not autorag-explorer)",
		body: JSON.stringify(
			{
				agentScope: "user",
				artifacts: false,
				agent: "custom-worker",
				model: PLACEHOLDER_CONFIGURED_MODEL,
				cwd: PLACEHOLDER_ALLOWED_ROOT,
				task: PLACEHOLDER_ASSIGNMENT_V1_BLOCK,
			},
			null,
			2,
		),
		reason: 'Every executable leaf agent must be exactly "autorag-explorer".',
	},
	{
		label: "Explicit unsafe values (artifacts true, agentScope project)",
		body: JSON.stringify(
			{
				agentScope: "project",
				artifacts: true,
				agent: "autorag-explorer",
				model: PLACEHOLDER_CONFIGURED_MODEL,
				cwd: PLACEHOLDER_ALLOWED_ROOT,
				task: PLACEHOLDER_ASSIGNMENT_V1_BLOCK,
			},
			null,
			2,
		),
		reason:
			'Missing or null top-level artifacts/agentScope are safely autofilled, but explicit wrong values (artifacts: true, agentScope: "project", a non-configured model) remain rejected.',
	},
];

// ---------------------------------------------------------------------------
// Stable coded patch-error reference (single source of truth for error data)
// ---------------------------------------------------------------------------

export interface DispatchErrorInfo {
	readonly code: string;
	readonly exactFix: string;
	readonly forceCorrectable: boolean;
}

/**
 * Reference table of stable dispatch rejection codes, their exactFix strings,
 * and forceCorrectable flags. This is the single source of truth for the error
 * catalog — `dispatch-validation.ts` derives its `EXACT_FIX_MAP` and
 * `FORCE_CORRECTABLE_MAP` from this table so no dual catalog exists.
 *
 * `DISPATCH_ROLE_METADATA_INVALID` is intentionally absent: valid tasks are
 * normalized by `ensureRoleLines` and no rejection path emits that code.
 */
export const DISPATCH_ERROR_REFERENCE: readonly DispatchErrorInfo[] = [
	{ code: "DISPATCH_MALFORMED", exactFix: "remove fields not owned by this action", forceCorrectable: true },
	{
		code: "DISPATCH_ACTION_UNKNOWN",
		exactFix: "use list|get|models|status|doctor or a supported launch shape",
		forceCorrectable: true,
	},
	{
		code: "DISPATCH_ADMIN_MUTATION_FORBIDDEN",
		exactFix: "do not mutate subagent definitions during AutoRAG search",
		forceCorrectable: false,
	},
	{
		code: "DISPATCH_CONTROL_FORBIDDEN",
		exactFix: "launch a fresh autorag-explorer assignment instead of controlling an existing run",
		forceCorrectable: false,
	},
	{
		code: "DISPATCH_SCHEDULE_FORBIDDEN",
		exactFix: "dispatch autorag-explorer work immediately; scheduling is disabled",
		forceCorrectable: false,
	},
	{ code: "DISPATCH_ARTIFACTS_INVALID", exactFix: "set args.artifacts = false", forceCorrectable: true },
	{
		code: "DISPATCH_AGENT_SCOPE_INVALID",
		exactFix: 'set args.agentScope = "user"',
		forceCorrectable: true,
	},
	{
		code: "DISPATCH_AGENT_IDENTITY",
		exactFix: 'set every executable leaf agent = "autorag-explorer"',
		forceCorrectable: false,
	},
	{
		code: "DISPATCH_MODEL_MISMATCH",
		exactFix: "set the referenced model field to the configured explorer model",
		forceCorrectable: true,
	},
	{
		code: "DISPATCH_ASSIGNMENT_INVALID",
		exactFix: "replace the task assignment with the canonical AUTORAG_ASSIGNMENT_V1 block",
		forceCorrectable: true,
	},
	{
		code: "DISPATCH_QUERY_MISMATCH",
		exactFix: "set originalQuery to the active user query verbatim",
		forceCorrectable: true,
	},
	{
		code: "DISPATCH_CWD_MISSING",
		exactFix: "set each executable leaf cwd to one configured search root",
		forceCorrectable: true,
	},
	{
		code: "DISPATCH_CWD_OUTSIDE_ROOTS",
		exactFix: "use a configured search root without symlink escape",
		forceCorrectable: false,
	},
	{
		code: "DISPATCH_NO_ACTIVE_QUERY",
		exactFix: "dispatch only while an AutoRAG search query is active",
		forceCorrectable: false,
	},
];

// ---------------------------------------------------------------------------
// Prompt section builder (single source of truth for template guidance text)
// ---------------------------------------------------------------------------

/**
 * Build the `### Canonical dispatch templates` subsection text for the system
 * prompt. Uses the builders and constants above so no divergent template text
 * exists. The `explorerModelId` parameter fills the model field in the example
 * payloads.
 */
export function buildDispatchTemplatesPromptSection(explorerModelId: string): string {
	const v1BlockExample = buildAssignmentV1Block({
		originalQuery: PLACEHOLDER_CALLER_QUERY,
		method: "<selected retrieval method>",
		queryVariants: ["<variant 1>", "<variant 2>"],
	});

	const singleRootTemplate = buildSingleRootDispatchPayload({
		explorerModel: explorerModelId,
		allowedRoot: PLACEHOLDER_ALLOWED_ROOT,
		task: PLACEHOLDER_ASSIGNMENT_V1_BLOCK,
	});

	const multiRootTemplate = buildMultiRootDispatchPayload([
		{
			explorerModel: explorerModelId,
			allowedRoot: PLACEHOLDER_ALLOWED_ROOT,
			task: PLACEHOLDER_ASSIGNMENT_V1_BLOCK,
		},
	]);

	const antiExamplesText = DISPATCH_ANTIEXAMPLES.map(
		(ex) => `- **${ex.label}**:\n\`\`\`json\n${ex.body}\n\`\`\`\n  ${ex.reason}`,
	).join("\n\n");

	return `### Canonical dispatch templates

Every explorer \`task\` field must contain an Assignment V1 block — a JSON body wrapped in sentinels with canonical role lines after the end sentinel:

\`\`\`
${v1BlockExample}
\`\`\`

The JSON body has exactly three keys: \`originalQuery\` (the caller query verbatim), \`method\` (the selected retrieval or discovery path), and \`queryVariants\` (a nonempty array of query variant strings). Legacy labeled format (\`Original query:\`, \`Selected retrieval method:\`, \`Query variants:\`) is accepted for compatibility, but the V1 sentinel format is preferred.

**Single-root dispatch** (one explorer, one root):

\`\`\`json
${singleRootTemplate}
\`\`\`

**Multi-root dispatch** (one task per configured root):

\`\`\`json
${multiRootTemplate}
\`\`\`

**Safe autofill**: missing or null top-level \`artifacts\`, \`agentScope\`, and leaf \`model\` fields are autofilled before validation — \`artifacts\` to \`false\`, \`agentScope\` to \`"user"\`, and leaf \`model\` to the configured explorer model. Explicit wrong values (\`artifacts: true\`, \`agentScope: "project"\`, a non-configured model) remain rejected. Diagnostics (\`list\`, \`get\`, \`models\`, \`status\`, \`doctor\`) are separate from launch dispatch and never receive these defaults. There is no single-agent fallback.

**Anti-examples** — these dispatches are rejected:

${antiExamplesText}`;
}

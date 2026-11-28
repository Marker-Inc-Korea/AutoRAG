/**
 * Host-registered subagents.
 *
 * A host that embeds AutoRAG (the Dazzi desktop app, a server, another tool) can hand the librarian specialists
 * that run in the host's own process — a PDF editor, a spreadsheet builder, a CRM writer — and let the librarian
 * delegate to them and read back their result. The tool contract follows pi's `subagent` example extension
 * (`examples/extensions/subagent`): one `subagent` tool with three modes,
 *
 *   - single:   `{ agent, task, files? }`
 *   - parallel: `{ tasks: [{ agent, task, files? }, …] }`
 *   - chain:    `{ chain: [{ agent, task, files? }, …] }` where `{previous}` in a task is the prior step's output
 *
 * Where pi spawns `pi` processes from markdown agent files, a host subagent is a `run` function the host provides,
 * so it can use the host's own engines, UI and locks. `files` names the absolute paths a task is about.
 *
 * Prompt caching: the tool schema and the system-prompt section are built once per agent from the registered
 * definitions, sorted by name, with nothing that changes between turns, so they do not break the prompt cache.
 * With no subagents registered nothing is added at all: the prompt and tool list stay exactly as before.
 *
 * Safety: a subagent that changes files declares `mutatesFiles`, and the permission gate blocks it in read-only
 * mode like `write` / `edit` (see `describeReadOnlyBlock`).
 */

import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import type { ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { toToolDefinition } from "./pi-session.ts";

export const SUBAGENT_TOOL_NAME = "subagent";

/** Parallel tasks one call may run, and how many run at once (pi's example uses the same limits). */
export const MAX_SUBAGENT_PARALLEL_TASKS = 8;
export const MAX_SUBAGENT_CONCURRENCY = 4;
/** Longest output handed back to the model per task; the rest is cut with a note. */
export const MAX_SUBAGENT_OUTPUT_CHARS = 16_000;

const NAME_PATTERN = /^[a-z][a-z0-9-]{1,39}$/u;

export interface SubagentTask {
	/** What to do, in the user's words plus whatever the orchestrator learned. */
	readonly task: string;
	/** Absolute paths the task is about (may be empty). */
	readonly files: readonly string[];
}

export interface SubagentRunContext {
	readonly signal?: AbortSignal;
	/** Streams a short progress line (shown while the subagent works). */
	readonly onUpdate?: (text: string) => void;
}

export interface SubagentResult {
	/** The subagent's answer to the orchestrator: what it did, or why it could not. */
	readonly output: string;
	readonly isError?: boolean;
	/** Files the run changed, when it knows. */
	readonly changedFiles?: readonly string[];
}

export interface SubagentDefinition {
	/** Lowercase id the model passes as `agent`, e.g. `pdf-editor`. */
	readonly name: string;
	/** One line: what it does and what to pass. Part of the cached tool description; keep it stable. */
	readonly description: string;
	/** Optional stable guidance added to the system prompt (when to delegate, what to include). */
	readonly guidance?: string;
	/** True when the subagent writes files: it is then blocked in read-only permission mode. */
	readonly mutatesFiles?: boolean;
	run(input: SubagentTask, context: SubagentRunContext): Promise<SubagentResult>;
}

/** Valid definitions sorted by name (stable prompt and schema); throws on a bad or duplicate name. */
export function normalizeSubagents(
	definitions: readonly SubagentDefinition[] | undefined,
): readonly SubagentDefinition[] {
	const list = [...(definitions ?? [])];
	const seen = new Set<string>();
	for (const definition of list) {
		if (!NAME_PATTERN.test(definition.name)) {
			throw new Error(
				`Invalid subagent name "${definition.name}": use 2-40 lowercase letters, digits, or dashes, starting with a letter.`,
			);
		}
		if (seen.has(definition.name)) throw new Error(`Duplicate subagent name "${definition.name}".`);
		if (definition.description.trim() === "") throw new Error(`Subagent "${definition.name}" needs a description.`);
		seen.add(definition.name);
	}
	return list.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
}

/** The names of subagents that write files. */
export function mutatingSubagentNames(definitions: readonly SubagentDefinition[]): ReadonlySet<string> {
	return new Set(
		definitions.filter((definition) => definition.mutatesFiles === true).map((definition) => definition.name),
	);
}

/** Every agent name a `subagent` tool call refers to, across the three modes. */
export function subagentNamesInCall(input: Readonly<Record<string, unknown>>): string[] {
	const names: string[] = [];
	if (typeof input.agent === "string") names.push(input.agent);
	for (const key of ["tasks", "chain"] as const) {
		const items = input[key];
		if (!Array.isArray(items)) continue;
		for (const item of items) {
			if (typeof item === "object" && item !== null && typeof (item as { agent?: unknown }).agent === "string") {
				names.push((item as { agent: string }).agent);
			}
		}
	}
	return names;
}

/** The system-prompt section for the registered subagents; empty when there are none. */
export function buildSubagentsPrompt(definitions: readonly SubagentDefinition[]): string {
	if (definitions.length === 0) return "";
	const lines = [
		"## Subagents",
		"",
		`Delegate specialist work with \`${SUBAGENT_TOOL_NAME}\` instead of doing it yourself with shell commands or file writes. Each subagent works in its own context and returns a summary of what it did; relay that result to the user in their language. Name the files a task is about in \`files\` (absolute paths) and describe the goal in \`task\`, including anything you already found. Use \`tasks\` for independent jobs (run in parallel) and \`chain\` when a step needs the previous step's output (\`{previous}\`).`,
		"",
		...definitions.map(
			(definition) =>
				`- \`${definition.name}\`: ${definition.description.trim()}${definition.mutatesFiles === true ? " (changes files)" : ""}`,
		),
	];
	const guidance = definitions.filter((definition) => (definition.guidance ?? "").trim() !== "");
	if (guidance.length > 0) {
		lines.push("");
		for (const definition of guidance) lines.push(`### \`${definition.name}\`\n${definition.guidance!.trim()}`);
	}
	return lines.join("\n");
}

const TaskItem = Type.Object({
	agent: Type.String({ description: "Subagent name" }),
	task: Type.String({ description: "What the subagent should do" }),
	files: Type.Optional(Type.Array(Type.String(), { description: "Absolute paths the task is about" })),
});

const ChainItem = Type.Object({
	agent: Type.String({ description: "Subagent name" }),
	task: Type.String({ description: "What to do; {previous} is replaced by the prior step's output" }),
	files: Type.Optional(Type.Array(Type.String(), { description: "Absolute paths the task is about" })),
});

const SubagentParams = Type.Object({
	agent: Type.Optional(Type.String({ description: "Subagent name (single mode)" })),
	task: Type.Optional(Type.String({ description: "Task (single mode)" })),
	files: Type.Optional(Type.Array(Type.String(), { description: "Absolute paths the task is about (single mode)" })),
	tasks: Type.Optional(Type.Array(TaskItem, { description: "Independent tasks to run in parallel" })),
	chain: Type.Optional(Type.Array(ChainItem, { description: "Tasks to run in order, each seeing {previous}" })),
});

interface SubagentParamsShape {
	agent?: string;
	task?: string;
	files?: string[];
	tasks?: Array<{ agent: string; task: string; files?: string[] }>;
	chain?: Array<{ agent: string; task: string; files?: string[] }>;
}

interface TaskOutcome {
	readonly agent: string;
	readonly task: string;
	readonly output: string;
	readonly isError: boolean;
	readonly changedFiles: readonly string[];
}

type Details = {
	readonly mode: "single" | "parallel" | "chain";
	readonly results: readonly { agent: string; isError: boolean; changedFiles: string[] }[];
};

function clip(text: string): string {
	return text.length <= MAX_SUBAGENT_OUTPUT_CHARS
		? text
		: `${text.slice(0, MAX_SUBAGENT_OUTPUT_CHARS)}\n…(output truncated)`;
}

async function mapConcurrent<T, R>(
	items: readonly T[],
	limit: number,
	fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
	const results = new Array<R>(items.length);
	let next = 0;
	const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
		while (next < items.length) {
			const index = next;
			next += 1;
			results[index] = await fn(items[index]!, index);
		}
	});
	await Promise.all(workers);
	return results;
}

/** The `subagent` tool over `definitions` (already normalized). */
export function createSubagentTool(
	definitions: readonly SubagentDefinition[],
): AgentTool<typeof SubagentParams, Details> {
	const byName = new Map(definitions.map((definition) => [definition.name, definition]));
	const available = definitions.map((definition) => definition.name).join(", ");

	const runOne = async (
		agent: string,
		task: string,
		files: readonly string[],
		signal: AbortSignal | undefined,
		progress: (text: string) => void,
	): Promise<TaskOutcome> => {
		const definition = byName.get(agent);
		if (definition === undefined) {
			return {
				agent,
				task,
				output: `Unknown subagent "${agent}". Available: ${available}.`,
				isError: true,
				changedFiles: [],
			};
		}
		if (signal?.aborted === true) return { agent, task, output: "Cancelled.", isError: true, changedFiles: [] };
		try {
			const result = await definition.run(
				{ task, files: files.filter((file) => typeof file === "string" && file.trim() !== "") },
				{ ...(signal === undefined ? {} : { signal }), onUpdate: (text) => progress(`${agent}: ${text}`) },
			);
			return {
				agent,
				task,
				output: clip(result.output),
				isError: result.isError === true,
				changedFiles: [...(result.changedFiles ?? [])],
			};
		} catch (error) {
			return {
				agent,
				task,
				output: `Subagent failed: ${error instanceof Error ? error.message : String(error)}`,
				isError: true,
				changedFiles: [],
			};
		}
	};

	const render = (mode: Details["mode"], outcomes: readonly TaskOutcome[]): AgentToolResult<Details> => {
		const text =
			mode === "single" && outcomes.length === 1
				? outcomes[0]!.output
				: outcomes
						.map(
							(outcome, index) =>
								`### ${index + 1}. ${outcome.agent}${outcome.isError ? " (failed)" : ""}\n${outcome.output}`,
						)
						.join("\n\n");
		const changed = [...new Set(outcomes.flatMap((outcome) => outcome.changedFiles))];
		return {
			content: [
				{
					type: "text",
					text:
						changed.length === 0
							? text
							: `${text}\n\nChanged files:\n${changed.map((file) => `- ${file}`).join("\n")}`,
				},
			],
			details: {
				mode,
				results: outcomes.map((outcome) => ({
					agent: outcome.agent,
					isError: outcome.isError,
					changedFiles: [...outcome.changedFiles],
				})),
			},
		};
	};

	return {
		name: SUBAGENT_TOOL_NAME,
		label: "Subagent",
		description: [
			"Delegate a task to a specialist subagent that works in its own context and returns a summary.",
			"Modes: single (agent + task [+ files]), parallel (tasks array), chain (sequential; {previous} in a task is the prior output).",
			`Subagents: ${definitions.map((definition) => `${definition.name} — ${definition.description.trim()}`).join(" | ")}`,
		].join(" "),
		parameters: SubagentParams,
		async execute(_toolCallId, rawParams, signal, onUpdate) {
			const params = rawParams as SubagentParamsShape;
			const modes = [
				params.task !== undefined || params.agent !== undefined,
				(params.tasks?.length ?? 0) > 0,
				(params.chain?.length ?? 0) > 0,
			];
			if (modes.filter(Boolean).length !== 1) {
				return render("single", [
					{
						agent: params.agent ?? "",
						task: "",
						output: `Give exactly one mode: agent+task, tasks, or chain. Available subagents: ${available}.`,
						isError: true,
						changedFiles: [],
					},
				]);
			}
			const progress = (text: string): void => {
				onUpdate?.({ content: [{ type: "text", text }], details: { mode: "single", results: [] } });
			};
			if (params.chain !== undefined && params.chain.length > 0) {
				const outcomes: TaskOutcome[] = [];
				let previous = "";
				for (const step of params.chain) {
					const outcome = await runOne(
						step.agent,
						step.task.replaceAll("{previous}", previous),
						step.files ?? [],
						signal,
						progress,
					);
					outcomes.push(outcome);
					if (outcome.isError) break;
					previous = outcome.output;
				}
				return render("chain", outcomes);
			}
			if (params.tasks !== undefined && params.tasks.length > 0) {
				if (params.tasks.length > MAX_SUBAGENT_PARALLEL_TASKS) {
					return render("parallel", [
						{
							agent: "",
							task: "",
							output: `Too many parallel tasks (${params.tasks.length}); the limit is ${MAX_SUBAGENT_PARALLEL_TASKS}.`,
							isError: true,
							changedFiles: [],
						},
					]);
				}
				const outcomes = await mapConcurrent(params.tasks, MAX_SUBAGENT_CONCURRENCY, (item) =>
					runOne(item.agent, item.task, item.files ?? [], signal, progress),
				);
				return render("parallel", outcomes);
			}
			if (params.agent === undefined || params.task === undefined) {
				return render("single", [
					{
						agent: params.agent ?? "",
						task: "",
						output: `Single mode needs both agent and task. Available subagents: ${available}.`,
						isError: true,
						changedFiles: [],
					},
				]);
			}
			return render("single", [await runOne(params.agent, params.task, params.files ?? [], signal, progress)]);
		},
	};
}

/** A pi extension registering the `subagent` tool; nothing when there are no subagents. */
export function createSubagentExtension(definitions: readonly SubagentDefinition[]): ExtensionFactory {
	return (pi) => {
		if (definitions.length === 0) return;
		pi.registerTool(toToolDefinition(createSubagentTool(definitions)));
	};
}

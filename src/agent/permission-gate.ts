import { homedir } from "node:os";
import { isAbsolute, relative, sep } from "node:path";
import type { ExtensionAPI, ExtensionContext, ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { matchesKey } from "@earendil-works/pi-tui";
import { type Question, redactSecrets } from "jev-use";
import type { JevJudge } from "./jev-extension.ts";
import {
	type ClassifyFs,
	classifyAutomationRun,
	classifyBash,
	classifyFileTool,
	nodeClassifyFs,
	type PermissionOperation,
	type PermissionOperationKind,
} from "./permission-classify.ts";

/**
 * How much the agent may do to the user's files without asking.
 *
 * - `full`: no gate; the agent does everything on its own.
 * - `ask`: the user approves deleting, moving, renaming, overwriting, and
 *   editing originals. Reading, searching, copying, and creating new files
 *   never ask, and anything the agent created itself stays free to change.
 * - `auto`: like `ask`, but Jev (the judgment model) answers first; only what
 *   Jev will not approve confidently reaches the user.
 */
export type PermissionMode = "full" | "ask" | "auto";

/** Order Shift+Tab cycles through. */
export const PERMISSION_MODES: readonly PermissionMode[] = ["ask", "auto", "full"];

/** Mode a fresh interactive or CLI session starts in. */
export const DEFAULT_PERMISSION_MODE: PermissionMode = "ask";

/** Jev question id of the "may this run without asking?" check. */
export const PERMISSION_GATE_QUESTION_ID = "permission_gate";

/** Jev must be at least this sure before `auto` skips the user. Stricter than jev-use's own 0.5 default. */
export const AUTO_APPROVE_MIN_CONFIDENCE = 0.8;

const JEV_TIMEOUT_MS = 15_000;
const STATUS_KEY = "autorag-permissions";

export function parsePermissionMode(value: unknown): PermissionMode | undefined {
	if (typeof value !== "string") return undefined;
	const normalized = value.trim().toLowerCase();
	return PERMISSION_MODES.find((mode) => mode === normalized);
}

export function cyclePermissionMode(mode: PermissionMode): PermissionMode {
	return PERMISSION_MODES[(PERMISSION_MODES.indexOf(mode) + 1) % PERMISSION_MODES.length] as PermissionMode;
}

type ModeListener = (mode: PermissionMode) => void;

/**
 * Session-wide permission state: the mode, the operation kinds the user
 * allowed for the rest of the session, and the files the agent created (which
 * are never gated). One controller is shared by every pi session of an agent.
 */
export class PermissionController {
	private current: PermissionMode;
	private readonly listeners = new Set<ModeListener>();
	private readonly granted = new Set<PermissionOperationKind>();
	private readonly created = new Set<string>();

	constructor(mode: PermissionMode = DEFAULT_PERMISSION_MODE) {
		this.current = mode;
	}

	get mode(): PermissionMode {
		return this.current;
	}

	/** Switches mode. Session grants end with the mode they were given in. */
	set(mode: PermissionMode): void {
		if (mode === this.current) return;
		this.current = mode;
		this.granted.clear();
		for (const listener of [...this.listeners]) listener(mode);
	}

	cycle(): PermissionMode {
		this.set(cyclePermissionMode(this.current));
		return this.current;
	}

	subscribe(listener: ModeListener): () => void {
		this.listeners.add(listener);
		return () => this.listeners.delete(listener);
	}

	isGranted(kind: PermissionOperationKind): boolean {
		return this.granted.has(kind);
	}

	grant(kinds: readonly PermissionOperationKind[]): void {
		for (const kind of kinds) this.granted.add(kind);
	}

	/** Marks paths the agent created: they, and anything below them, are never gated. */
	trust(paths: readonly string[]): void {
		for (const path of paths) this.created.add(path);
	}

	isTrusted(path: string): boolean {
		for (const root of this.created) {
			if (path === root) return true;
			const rel = relative(root, path);
			if (rel !== "" && rel !== ".." && !rel.startsWith(`..${sep}`) && !isAbsolute(rel)) return true;
		}
		return false;
	}
}

export interface PermissionGateOptions {
	readonly controller: PermissionController;
	/** Jev judge for `auto` mode. Without one, `auto` asks the user like `ask`. */
	readonly judge?: JevJudge;
	/**
	 * The user's current request, which Jev judges the operations against. Without it the gate uses the prompt
	 * of the run that started, which is an internal phase prompt in AutoRAG's two-phase search.
	 */
	readonly request?: () => string | undefined;
	/** Filesystem the classifier reads. Default: the real one, with `~` expanding to the user's home. */
	readonly fs?: (cwd: string) => ClassifyFs;
}

const ALLOW_ONCE = "Allow once";
const DENY = "Deny";
const DENY_WITH_NOTE = "Deny and tell the model what to do instead";

const KIND_LABEL: Readonly<Record<PermissionOperationKind, string>> = {
	create: "create",
	delete: "delete",
	move: "move",
	rename: "rename",
	overwrite: "overwrite",
	edit: "edit",
	script: "run a script that changes files",
};

/** Appended to the system prompt while a gate is active, so the model stays on the happy path. */
export const PERMISSION_PROMPT_ADDENDUM = [
	"File safety: the user's existing files are originals, and AutoRAG asks the user for approval before it deletes, moves, renames, overwrites, or edits one.",
	"When you reorganize or change files on your own initiative, copy first and work on the copy: creating new files and folders, copying, reading, and searching never need approval.",
	"When the user explicitly asks you to delete, move, rename, or edit a specific original, go ahead and run it (delete by moving to the OS trash, as above): do not refuse or ask for confirmation yourself, because the approval prompt is the confirmation.",
	"If an approval is denied, do not retry it within that request: leave the original as it is and say what you did instead. A new request from the user, or a changed permission mode, is a fresh decision: run it again then.",
].join(" ");

function describeOperation(op: PermissionOperation): string {
	const target =
		op.paths.length === 0
			? "(target known only when the command runs)"
			: op.dest === undefined
				? op.paths.join(", ")
				: `${op.paths[0]} -> ${op.dest}`;
	return `  ${KIND_LABEL[op.kind]}  ${target}`;
}

function describeCommands(ops: readonly PermissionOperation[]): string[] {
	return [...new Set(ops.map((op) => op.detail))].map((detail) => `    $ ${detail}`);
}

function operationsOf(
	toolName: string,
	input: Readonly<Record<string, unknown>>,
	fs: ClassifyFs,
): PermissionOperation[] {
	if (toolName === "bash" && typeof input.command === "string") return classifyBash(input.command, fs);
	if (toolName === "write" || toolName === "edit") return classifyFileTool(toolName, input, fs);
	if (toolName === "manage_automations" && input.action === "save") {
		const rule = input.automation;
		if (typeof rule === "object" && rule !== null && Array.isArray((rule as { run?: unknown }).run)) {
			const run = (rule as { run: unknown[] }).run.filter((part): part is string => typeof part === "string");
			return classifyAutomationRun(run, fs, {
				appendFiles: (rule as { appendFiles?: unknown }).appendFiles === true,
			});
		}
	}
	return [];
}

interface Decision {
	readonly block?: true;
	readonly reason?: string;
}

/**
 * pi extension that gates what the agent does to existing files.
 *
 * Every `bash`, `write`, `edit`, and `manage_automations save` call is
 * classified (see {@link classifyBash}); only changes to originals reach the
 * approval step, and what happens there depends on the controller's mode.
 * Shift+Tab and `/permissions` switch the mode; the footer shows it.
 */
export function createPermissionGate(options: PermissionGateOptions): ExtensionFactory {
	const { controller, judge } = options;
	const makeFs = options.fs ?? ((cwd: string) => nodeClassifyFs(cwd, homedir()));

	return (pi: ExtensionAPI) => {
		/** Paths a call creates, trusted once the call succeeds. */
		const pendingCreations = new Map<string, string[]>();
		let userRequest = "";
		let dialogsOpen = 0;
		let queue: Promise<unknown> = Promise.resolve();
		let boundUi: ExtensionContext["ui"] | undefined;
		let unsubscribeKeys: (() => void) | undefined;
		let unsubscribeMode: (() => void) | undefined;

		const showStatus = (ui: ExtensionContext["ui"], mode: PermissionMode): void => {
			const text = `permissions: ${mode}`;
			const colored = mode === "full" ? "warning" : mode === "auto" ? "accent" : "success";
			ui.setStatus(STATUS_KEY, ui.theme?.fg ? ui.theme.fg(colored, text) : text);
		};

		const bindUi = (ui: ExtensionContext["ui"]): void => {
			showStatus(ui, controller.mode);
			if (boundUi === ui) return;
			unsubscribeKeys?.();
			unsubscribeMode?.();
			boundUi = ui;
			unsubscribeMode = controller.subscribe((mode) => showStatus(ui, mode));
			unsubscribeKeys = ui.onTerminalInput((data) => {
				if (!matchesKey(data, "shift+tab")) return undefined;
				if (dialogsOpen > 0) return { consume: true };
				const mode = controller.cycle();
				ui.notify(`Permission mode: ${mode} - ${MODE_SUMMARY[mode]}`, "info");
				return { consume: true };
			});
		};

		pi.on("session_start", (_event, ctx) => {
			if (ctx.hasUI) bindUi(ctx.ui);
		});

		pi.on("session_shutdown", () => {
			unsubscribeKeys?.();
			unsubscribeMode?.();
			unsubscribeKeys = undefined;
			unsubscribeMode = undefined;
			boundUi = undefined;
		});

		pi.registerCommand("permissions", {
			description: "Show or set the permission mode: ask, auto, or full (Shift+Tab cycles)",
			getArgumentCompletions: (prefix) => {
				const items = PERMISSION_MODES.filter((mode) => mode.startsWith(prefix.trim().toLowerCase())).map(
					(mode) => ({
						value: mode,
						label: mode,
						description: MODE_SUMMARY[mode],
					}),
				);
				return items.length > 0 ? items : null;
			},
			handler: async (args, ctx) => {
				const wanted = args.trim();
				let mode = parsePermissionMode(wanted);
				if (wanted !== "" && mode === undefined) {
					ctx.ui.notify(
						`Unknown permission mode "${wanted}". Use one of: ${PERMISSION_MODES.join(", ")}.`,
						"warning",
					);
					return;
				}
				if (mode === undefined) {
					const choice = await ctx.ui.select(
						`Permission mode (now ${controller.mode})`,
						PERMISSION_MODES.map((candidate) => `${candidate} - ${MODE_SUMMARY[candidate]}`),
					);
					mode = parsePermissionMode(choice?.split(" ")[0]);
					if (mode === undefined) return;
				}
				controller.set(mode);
				ctx.ui.notify(`Permission mode: ${mode} - ${MODE_SUMMARY[mode]}`, "info");
			},
		});

		pi.on("before_agent_start", (event) => {
			if (event.messageType === "user" && typeof event.prompt === "string") userRequest = event.prompt;
			if (controller.mode === "full") return undefined;
			return { systemPrompt: `${event.systemPrompt}\n\n${PERMISSION_PROMPT_ADDENDUM}` };
		});

		pi.on("tool_call", async (event, ctx): Promise<Decision | undefined> => {
			if (controller.mode === "full") return undefined;
			const fs = makeFs(ctx.cwd);
			const ops = operationsOf(event.toolName, event.input as Record<string, unknown>, fs);
			if (ops.length === 0) return undefined;

			// Walk the operations in order: what a command creates is free for the rest of that command.
			const creations: string[] = [];
			const gated: PermissionOperation[] = [];
			for (const op of ops) {
				if (op.kind === "create") {
					creations.push(...op.paths);
					continue;
				}
				const free =
					!op.dynamic &&
					op.paths.length > 0 &&
					op.paths.every(
						(path) =>
							controller.isTrusted(path) ||
							creations.some((made) => path === made || path.startsWith(made + sep)),
					);
				if (free) {
					// Moving or copying something the agent made leaves the result just as free.
					if (op.dest !== undefined) creations.push(op.dest);
					continue;
				}
				gated.push(op);
			}

			const decision = gated.length === 0 ? undefined : await decide(gated, ctx);
			if (decision?.block === true) return decision;
			if (creations.length > 0) pendingCreations.set(event.toolCallId, creations);
			return undefined;
		});

		pi.on("tool_result", (event) => {
			const made = pendingCreations.get(event.toolCallId);
			pendingCreations.delete(event.toolCallId);
			if (made !== undefined && !event.isError) controller.trust(made);
			return undefined;
		});

		async function decide(ops: readonly PermissionOperation[], ctx: ExtensionContext): Promise<Decision | undefined> {
			// One dialog at a time: overlapping tool calls wait their turn, then re-check grants.
			const turn = queue.then(() => decideNow(ops, ctx));
			queue = turn.catch(() => undefined);
			return turn;
		}

		async function decideNow(
			ops: readonly PermissionOperation[],
			ctx: ExtensionContext,
		): Promise<Decision | undefined> {
			if (controller.mode === "full") return undefined;
			const pending = ops.filter((op) => !controller.isGranted(op.kind));
			if (pending.length === 0) return undefined;
			const commands = describeCommands(pending);
			const lines = pending.map(describeOperation);

			let declined: string | undefined;
			if (controller.mode === "auto" && judge !== undefined) {
				const verdict = await askJev(judge, pending, options.request?.() ?? userRequest, ctx.cwd);
				if (verdict.approved) {
					ctx.ui.notify(`Jev approved: ${pending.map((op) => op.detail).join("; ")}`, "info");
					return undefined;
				}
				declined = verdict.why;
			}

			if (!ctx.hasUI) {
				const why = declined === undefined ? "" : ` Jev did not approve it (${declined}).`;
				return {
					block: true,
					reason: `Blocked: changing existing files needs approval and this session has nobody to ask.${why} Do not retry; work on a copy instead, or ask the user to rerun with --permission-mode full.`,
				};
			}

			const kinds = [...new Set(pending.map((op) => op.kind))];
			const heading =
				declined === undefined
					? "Approve this change to your files?"
					: `Jev declined to approve this automatically (${declined}). Approve it yourself?`;
			const title = [heading, ...lines, ...commands].join("\n");
			const sessionChoice = `Allow all ${kinds.map((kind) => KIND_LABEL[kind]).join("/")} operations this session`;
			dialogsOpen += 1;
			let choice: string | undefined;
			try {
				choice = await ctx.ui.select(title, [ALLOW_ONCE, sessionChoice, DENY, DENY_WITH_NOTE]);
				if (choice === DENY_WITH_NOTE) {
					const note = (await ctx.ui.input("What should the model do instead?", "e.g. work on a copy"))?.trim();
					return deny(pending, note === undefined || note === "" ? undefined : note);
				}
			} finally {
				dialogsOpen -= 1;
			}
			if (choice === ALLOW_ONCE) return undefined;
			if (choice === sessionChoice) {
				controller.grant(kinds);
				return undefined;
			}
			return deny(pending);
		}
	};
}

const MODE_SUMMARY: Readonly<Record<PermissionMode, string>> = {
	ask: "asks before deleting, moving, renaming, or editing originals",
	auto: "Jev approves first; asks you only when Jev will not",
	full: "no approvals, the agent does everything itself",
};

function deny(ops: readonly PermissionOperation[], note?: string): Decision {
	const what = ops.map((op) => op.detail).join("; ");
	return {
		block: true,
		reason: `Denied by the user: ${what}. Do not retry it for this request. Leave the original untouched: copy it to a new file or folder and change the copy, or tell the user what you would have done. If the user asks again later, that is a new decision.${note === undefined ? "" : ` The user says: ${note}`}`,
	};
}

interface JevVerdict {
	readonly approved: boolean;
	readonly why: string;
}

const GATE_QUESTION: Question = {
	id: PERMISSION_GATE_QUESTION_ID,
	type: "choice",
	question:
		"Should the agent run these file operations without asking the user? Choose allow only when the user's request clearly asks for exactly this change, or it is an obviously safe, reversible tidy-up of files the request names. Choose deny when it deletes, moves, renames, or overwrites originals the request did not name, touches system, configuration, or credential locations, or goes beyond the request.",
	options: {
		allow: "The request clearly covers exactly these operations and they are safe.",
		deny: "The operations are destructive, broader than the request, or unclear.",
	},
};

/** Jev's yes/no on one batch of operations. Never throws: no answer means "ask the user". */
async function askJev(
	judge: JevJudge,
	ops: readonly PermissionOperation[],
	request: string,
	cwd: string,
): Promise<JevVerdict> {
	// Only the request and the operations leave the machine: never file contents, and secrets are stripped.
	const state = redactSecrets(
		[
			`User request: ${request.trim().slice(0, 2000) || "(not available)"}`,
			`Working directory: ${cwd}`,
			"Proposed file operations:",
			...ops.map((op) => `- ${describeOperation(op).trim()}  [${op.detail}]`),
		].join("\n"),
	);
	let timer: NodeJS.Timeout | undefined;
	try {
		const judgment = await Promise.race([
			judge(state, [GATE_QUESTION], { confidenceThreshold: AUTO_APPROVE_MIN_CONFIDENCE }),
			new Promise<never>((_resolve, reject) => {
				timer = setTimeout(
					() => reject(new Error(`Jev did not answer within ${JEV_TIMEOUT_MS / 1000}s`)),
					JEV_TIMEOUT_MS,
				);
			}),
		]);
		const verdict = judgment.verdicts.find((candidate) => candidate.id === PERMISSION_GATE_QUESTION_ID);
		if (verdict === undefined) return { approved: false, why: "no verdict" };
		if (verdict.answer === "allow" && !verdict.escalate && verdict.confidence >= AUTO_APPROVE_MIN_CONFIDENCE) {
			return { approved: true, why: "" };
		}
		if (verdict.escalate) return { approved: false, why: verdict.hint ?? verdict.reason ?? "not sure" };
		return {
			approved: false,
			why: verdict.answer === "allow" ? `only ${Math.round(verdict.confidence * 100)}% sure` : "rejected",
		};
	} catch (error) {
		return { approved: false, why: error instanceof Error ? error.message : String(error) };
	} finally {
		clearTimeout(timer);
	}
}

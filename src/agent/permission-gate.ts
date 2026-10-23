import { statSync } from "node:fs";
import { isAbsolute, relative, sep } from "node:path";
import type { ExtensionAPI, ExtensionContext, ExtensionFactory, SessionEntry } from "@earendil-works/pi-coding-agent";
import { matchesKey } from "@earendil-works/pi-tui";
import type { JevJudge } from "./jev-extension.ts";
import {
	type Approval,
	classifyCall,
	DEFAULT_MAX_STATE_CHARS,
	type FlaggedKind,
	isGatedCall,
	type PermissionKind,
	successfulCalls,
	summarizeCall,
} from "./permission-judgment.ts";
import { type ProposedCall, pathsNamedBy, resolveToolPath } from "./permission-paths.ts";
import { cutMiddle, type UserRequestLog } from "./user-requests.ts";

export {
	AUTO_APPROVE_MIN_CONFIDENCE,
	DEFAULT_MAX_STATE_CHARS,
	PERMISSION_GATE_QUESTION_ID,
} from "./permission-judgment.ts";

/**
 * How much the agent may do to the user's files without asking.
 *
 * Jev classifies every `bash`, `write`, `edit`, and saved automation first, in `ask` and `auto` alike: it says
 * whether the call deletes, moves, renames, overwrites, or edits an existing file, or runs code whose effect
 * cannot be told. What happens next depends on the mode.
 *
 * - `full`: no gate; the agent does everything on its own, and Jev is never asked.
 * - `ask`: a call Jev finds safe runs; one it flags is put to the user.
 * - `auto`: like `ask`, but Jev also decides whether a flagged call may run without the user; only what Jev
 *   will not approve confidently reaches the user.
 */
export type PermissionMode = "full" | "ask" | "auto";

/** Order Shift+Tab cycles through. */
export const PERMISSION_MODES: readonly PermissionMode[] = ["ask", "auto", "full"];

/** Mode a fresh interactive or CLI session starts in. */
export const DEFAULT_PERMISSION_MODE: PermissionMode = "ask";

const STATUS_KEY = "autorag-permissions";

function pathExists(path: string): boolean {
	try {
		statSync(path);
		return true;
	} catch {
		return false;
	}
}

export function parsePermissionMode(value: unknown): PermissionMode | undefined {
	if (typeof value !== "string") return undefined;
	const normalized = value.trim().toLowerCase();
	return PERMISSION_MODES.find((mode) => mode === normalized);
}

export function cyclePermissionMode(mode: PermissionMode): PermissionMode {
	return PERMISSION_MODES[(PERMISSION_MODES.indexOf(mode) + 1) % PERMISSION_MODES.length] as PermissionMode;
}

type ModeListener = (mode: PermissionMode) => void;

/** What a session grant covers: one category of change, or every call Jev could not classify. */
export type GrantKey = PermissionKind | "unclassified";

/**
 * Session-wide permission state: the mode, and what the user allowed for the rest of the session. One
 * controller is shared by every pi session of an agent.
 */
export class PermissionController {
	private current: PermissionMode;
	private readonly listeners = new Set<ModeListener>();
	private readonly granted = new Set<GrantKey>();
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

	isGranted(key: GrantKey): boolean {
		return this.granted.has(key);
	}

	grant(keys: readonly GrantKey[]): void {
		for (const key of keys) this.granted.add(key);
	}

	/** Records paths a successful call brought into existence: the agent's own files, not the user's originals. */
	markCreated(paths: readonly string[]): void {
		for (const path of paths) this.created.add(path);
	}

	/** Whether the agent created this path, or a folder it lies in, earlier in this session. */
	isCreated(path: string): boolean {
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
	/**
	 * Jev, which classifies every gated call. Without one the gate cannot classify, so in `ask` and `auto` it
	 * asks the user about every `bash`, `write`, and `edit` call.
	 */
	readonly judge?: JevJudge;
	/**
	 * What the user has said in this conversation: the first question, every follow-up, every steer. Jev judges
	 * the call against all of it. The host feeds it from the user's own input, so AutoRAG's internal phase
	 * prompts are never in it.
	 */
	readonly requests?: UserRequestLog;
	/** Cap on the state shown to Jev, in characters. Default: Jev's context ceiling with a safety margin. */
	readonly maxStateChars?: number;
	/** Files the agent may always write and edit without Jev being asked (the config its self-configuration turn edits). */
	readonly trustedPaths?: readonly string[];
}

const ALLOW_ONCE = "Allow once";
const DENY = "Deny";
const DENY_WITH_NOTE = "Deny and tell the model what to do instead";

/** Longest part of a command shown in a dialog or a notice. */
const SHOWN_CALL_CHARS = 1200;

/** Appended to the system prompt while a gate is active, so the model stays on the happy path. */
export const PERMISSION_PROMPT_ADDENDUM = [
	"File safety: the user's existing files are originals, and AutoRAG asks the user for approval before it deletes, moves, renames, overwrites, or edits one.",
	"When you reorganize or change files on your own initiative, copy first and work on the copy: creating new files and folders, copying, reading, and searching never need approval.",
	"When the user explicitly asks you to delete, move, rename, or edit a specific original, go ahead and run it (delete by moving to the OS trash, as above): do not refuse or ask for confirmation yourself, because the approval prompt is the confirmation.",
	"If an approval is denied, do not retry it within that request: leave the original as it is and say what you did instead. A new request from the user, or a changed permission mode, is a fresh decision: run it again then.",
].join(" ");

const MODE_SUMMARY: Readonly<Record<PermissionMode, string>> = {
	ask: "Jev checks each call; you approve deleting, moving, renaming, or changing originals",
	auto: "Jev checks each call and approves first; asks you only when Jev will not",
	full: "no approvals, the agent does everything itself",
};

interface Decision {
	readonly block?: true;
	readonly reason?: string;
}

interface Review {
	readonly call: ProposedCall;
	/** The category grants this call needs. */
	readonly needs: readonly GrantKey[];
	/** Shown above the call in the dialog. */
	readonly heading: string;
	/** Why the user is being asked instead of Jev deciding, for a call that is blocked without a UI. */
	readonly headlessReason: string;
	readonly sessionChoice: string;
}

function deny(call: ProposedCall, note?: string): Decision {
	return {
		block: true,
		reason: `Denied by the user: ${cutMiddle(summarizeCall(call), 300)}. Do not retry it for this request. Leave the original untouched: copy it to a new file or folder and change the copy, or tell the user what you would have done. If the user asks again later, that is a new decision.${note === undefined ? "" : ` The user says: ${note}`}`,
	};
}

/**
 * pi extension that gates what the agent does to existing files.
 *
 * Jev classifies every `bash`, `write`, `edit`, and `manage_automations save` call before it runs (reads,
 * searches, and other tools never touch originals and are not asked about). A call Jev finds safe runs; one it
 * flags, or cannot classify, is decided by the mode: put to the user in `ask`, put to Jev's approval first in
 * `auto`. Shift+Tab and `/permissions` switch the mode; the footer shows it.
 */
export function createPermissionGate(options: PermissionGateOptions): ExtensionFactory {
	const { controller, judge, requests } = options;
	const maxStateChars = options.maxStateChars ?? DEFAULT_MAX_STATE_CHARS;
	const trustedPaths = new Set(options.trustedPaths ?? []);

	return (pi: ExtensionAPI) => {
		let dialogsOpen = 0;
		/** Paths each gated call names that do not exist yet, until its result shows whether it made them. */
		const pendingCreations = new Map<string, string[]>();
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
			if (controller.mode === "full") return undefined;
			return { systemPrompt: `${event.systemPrompt}\n\n${PERMISSION_PROMPT_ADDENDUM}` };
		});

		pi.on("tool_call", async (event, ctx): Promise<Decision | undefined> => {
			const input = event.input as Record<string, unknown>;
			if (!isGatedCall(event.toolName, input)) return undefined;
			const call: ProposedCall = { tool: event.toolName, input, cwd: ctx.cwd };
			// Whatever the mode, a path this call names that is missing now and exists after it succeeded is the
			// agent's own file: later calls on it are not changes to the user's originals.
			pendingCreations.set(
				event.toolCallId,
				pathsNamedBy(call)
					.map((entry) => entry.path)
					.filter((path) => !pathExists(path)),
			);
			if (controller.mode === "full") return undefined;
			if (
				(event.toolName === "write" || event.toolName === "edit") &&
				trustedPaths.has(resolveToolPath(String(input.path), ctx.cwd))
			) {
				return undefined;
			}
			const decision = await judgeCall(call, ctx);
			if (decision?.block === true) pendingCreations.delete(event.toolCallId);
			return decision;
		});

		pi.on("tool_result", (event) => {
			const absent = pendingCreations.get(event.toolCallId);
			pendingCreations.delete(event.toolCallId);
			if (absent !== undefined && !event.isError) controller.markCreated(absent.filter(pathExists));
			return undefined;
		});

		/** Jev classifies the call; what follows depends on the mode. */
		async function judgeCall(call: ProposedCall, ctx: ExtensionContext): Promise<Decision | undefined> {
			const mode = controller.mode;
			const classification =
				judge === undefined
					? ({ status: "failed", why: "Jev is not configured" } as const)
					: await classifyCall(judge, {
							call,
							requests: requests?.entries() ?? [],
							history: successfulCalls(branchOf(ctx)),
							maxStateChars,
							wantApproval: mode === "auto",
							isCreated: (path) => controller.isCreated(path),
						});

			if (classification.status === "failed") {
				return decide(
					{
						call,
						needs: ["unclassified"],
						heading: `Jev could not classify this call (${classification.why}), so it needs your approval. It may change your files.`,
						headlessReason: `Jev could not classify this call (${classification.why}), so it needs approval`,
						sessionChoice: "Allow all unclassified calls this session",
					},
					ctx,
				);
			}
			if (classification.flagged.length === 0) return undefined;

			const kinds = classification.flagged.map((flagged) => flagged.kind);
			if (mode === "auto" && classification.approval?.approved === true) {
				ctx.ui.notify(`Jev approved (${kinds.join(", ")}): ${cutMiddle(summarizeCall(call), 300)}`, "info");
				return undefined;
			}
			return decide(
				reviewOf(call, classification.flagged, mode === "auto" ? classification.approval : undefined),
				ctx,
			);
		}

		async function decide(review: Review, ctx: ExtensionContext): Promise<Decision | undefined> {
			// One dialog at a time: overlapping tool calls wait their turn, then re-check grants.
			const turn = queue.then(() => ask(review, ctx));
			queue = turn.catch(() => undefined);
			return turn;
		}

		async function ask(review: Review, ctx: ExtensionContext): Promise<Decision | undefined> {
			if (controller.mode === "full") return undefined;
			if (review.needs.every((key) => controller.isGranted(key))) return undefined;

			if (!ctx.hasUI) {
				return {
					block: true,
					reason: `Blocked: ${review.headlessReason}, and this session has nobody to ask. Do not retry; work on a copy instead, or ask the user to rerun with --permission-mode full.`,
				};
			}

			const title = [
				review.heading,
				`    ${review.needs.join(", ")}`,
				`    $ ${cutMiddle(summarizeCall(review.call), SHOWN_CALL_CHARS)}`,
			].join("\n");
			dialogsOpen += 1;
			let choice: string | undefined;
			try {
				choice = await ctx.ui.select(title, [ALLOW_ONCE, review.sessionChoice, DENY, DENY_WITH_NOTE]);
				if (choice === DENY_WITH_NOTE) {
					const note = (await ctx.ui.input("What should the model do instead?", "e.g. work on a copy"))?.trim();
					return deny(review.call, note === undefined || note === "" ? undefined : note);
				}
			} finally {
				dialogsOpen -= 1;
			}
			if (choice === ALLOW_ONCE) return undefined;
			if (choice === review.sessionChoice) {
				controller.grant(review.needs);
				return undefined;
			}
			return deny(review.call);
		}
	};
}

function branchOf(ctx: ExtensionContext): readonly SessionEntry[] {
	return ctx.sessionManager.getBranch();
}

/** The dialog for a call Jev flagged. `approval` is `auto` mode's verdict when Jev was asked and declined. */
function reviewOf(call: ProposedCall, flagged: readonly FlaggedKind[], approval: Approval | undefined): Review {
	const kinds = flagged.map((entry) => entry.kind);
	const heading =
		approval === undefined
			? "Approve this change to your files?"
			: `Jev declined to approve this automatically (${approval.why}). Approve it yourself?`;
	return {
		call,
		needs: kinds,
		heading,
		headlessReason:
			approval === undefined
				? `this call ${kinds.join("/")}s existing files and needs approval`
				: `this call ${kinds.join("/")}s existing files and Jev did not approve it (${approval.why}), so it needs approval`,
		sessionChoice: `Allow all ${kinds.join("/")} operations this session`,
	};
}

import type { ExtensionAPI, ExtensionFactory, SessionEntry } from "@earendil-works/pi-coding-agent";
import { type Static, Type } from "typebox";
import { Value } from "typebox/value";

/** Custom session entry type under which a conversation persists what the user said. */
export const USER_REQUEST_ENTRY_TYPE = "autorag.user-request";

/**
 * `request`: a new question or a follow-up. `steer`: a correction typed while a search runs, which
 * adjusts the request it interrupts rather than starting a new one.
 */
export type UserRequestKind = "request" | "steer";

export interface UserRequest {
	readonly text: string;
	readonly kind: UserRequestKind;
}

const userRequestSchema = Type.Object({
	text: Type.String({ minLength: 1 }),
	kind: Type.Union([Type.Literal("request"), Type.Literal("steer")]),
});

type SaveEntry = (request: UserRequest) => void;

/**
 * What the user has said in this conversation, in order, exactly as typed: the first question, every
 * follow-up, every steer. AutoRAG's own phase prompts (fast answer, verification, direct answer, steer
 * continuation) never enter it, because it is fed where the user's words arrive, not from the transcript.
 *
 * It is not truncated here. Whoever consumes it fits it to its own limit.
 */
export class UserRequestLog {
	private items: UserRequest[] = [];
	private save: SaveEntry | undefined;

	record(text: string, kind: UserRequestKind = "request"): void {
		if (text.trim().length === 0) return;
		const request: UserRequest = { text, kind };
		this.items.push(request);
		this.save?.(request);
	}

	entries(): readonly UserRequest[] {
		return this.items;
	}

	/** Forgets everything: a new conversation. */
	reset(): void {
		this.items = [];
	}

	/** Replaces the log with a conversation read back from a session. */
	restore(requests: readonly UserRequest[]): void {
		this.items = [...requests];
	}

	/**
	 * Where each recorded message is saved. Returns a detach that only unplugs this saver: a session that
	 * shut down late must not unplug the one that replaced it. Detached, the log is in-memory only.
	 */
	attach(save: SaveEntry): () => void {
		this.save = save;
		return () => {
			if (this.save === save) this.save = undefined;
		};
	}
}

type SavedRequest = Static<typeof userRequestSchema>;

/** The user messages saved on the branch `session` is on, oldest first. Malformed entries are skipped. */
function savedRequests(session: { getBranch(): SessionEntry[] }): UserRequest[] {
	const out: UserRequest[] = [];
	for (const entry of session.getBranch()) {
		if (entry.type !== "custom" || entry.customType !== USER_REQUEST_ENTRY_TYPE) continue;
		if (!Value.Check(userRequestSchema, entry.data)) continue;
		const saved: SavedRequest = entry.data;
		out.push({ text: saved.text, kind: saved.kind });
	}
	return out;
}

/**
 * pi extension that keeps the conversation's {@link UserRequestLog} in the session file, next to the
 * transcript. Starting, resuming, forking, or navigating the session tree rebuilds the log from the branch
 * now in view, so `/new` begins empty and `/resume` brings the earlier requests back.
 */
export function createUserRequestPersistence(log: UserRequestLog): ExtensionFactory {
	return (pi: ExtensionAPI) => {
		const restore = (_event: unknown, ctx: { sessionManager: { getBranch(): SessionEntry[] } }): void => {
			log.restore(savedRequests(ctx.sessionManager));
		};
		let detach: (() => void) | undefined;
		pi.on("session_start", (event, ctx) => {
			restore(event, ctx);
			detach?.();
			detach = log.attach((request) =>
				pi.appendEntry(USER_REQUEST_ENTRY_TYPE, { text: request.text, kind: request.kind }),
			);
		});
		pi.on("session_tree", restore);
		pi.on("session_shutdown", () => {
			detach?.();
			detach = undefined;
		});
	};
}

const CORRECTION_LABEL = "(correction while the agent was working) ";

function pluralMessages(count: number): string {
	return `${count} earlier message${count === 1 ? "" : "s"} omitted`;
}

/** `text` cut to `max` characters by dropping its middle, so both what it starts with and ends with survive. */
export function cutMiddle(text: string, max: number): string {
	if (text.length <= max) return text;
	// The marker's number has at most as many digits as the text is long, so this length is an upper bound.
	const marker = (omitted: number) => ` [...${omitted} characters omitted...] `;
	const keep = max - marker(text.length).length;
	if (keep <= 0) return text.slice(0, max);
	const head = Math.ceil(keep / 2);
	const tail = keep - head;
	return `${text.slice(0, head)}${marker(text.length - keep)}${tail === 0 ? "" : text.slice(-tail)}`;
}

/**
 * The conversation as the Jev gate reads it: every user message in order, verbatim and numbered, with
 * corrections marked. Nothing is shortened while it fits in `maxChars`. Over that, the opening request and
 * the newest messages stay and the middle is dropped with a count (the numbering shows where), because the
 * first message says what the user came for and the last ones say where they ended up. Only when even
 * those two do not fit is a single message cut, in the middle.
 */
export function renderUserRequests(requests: readonly UserRequest[], maxChars: number): string {
	if (requests.length === 0) return "(no user message available)";
	const lines = requests.map(
		(request, index) => `[${index + 1}] ${request.kind === "steer" ? CORRECTION_LABEL : ""}${request.text}`,
	);
	const whole = lines.join("\n");
	if (whole.length <= maxChars) return whole;

	const first = lines[0] as string;
	const rest = lines.slice(1);
	const marker = (omitted: number) => `[${pluralMessages(omitted)}]`;
	// The newest messages that fit beside the opening one.
	for (let keep = rest.length; keep >= 1; keep--) {
		const tail = rest.slice(rest.length - keep);
		const omitted = rest.length - keep;
		const text = [first, ...(omitted > 0 ? [marker(omitted)] : []), ...tail].join("\n");
		if (text.length <= maxChars) return text;
	}
	if (rest.length === 0) return cutMiddle(first, maxChars);
	// Opening and newest message alone are too long: give each half of what is left.
	const omitted = rest.length - 1;
	const gap = omitted > 0 ? `\n${marker(omitted)}\n` : "\n";
	const room = Math.max(0, maxChars - gap.length);
	const half = Math.floor(room / 2);
	return `${cutMiddle(first, half)}${gap}${cutMiddle(rest[rest.length - 1] as string, room - half)}`;
}

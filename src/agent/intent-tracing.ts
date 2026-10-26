import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

/**
 * Intent tracing: every tool the model can call also takes a short `i` argument, one line saying
 * what that call is for. The text is shown while the model works (pi's `Working` loader in the TUI,
 * `progress` events of `searchDocumentsStream` for the CLI and the desktop app), so a long search
 * says what it is doing without the model writing separate progress notes.
 *
 * Pi has no such feature; this mirrors oh-my-pi's. The field exists only on the wire: it is added to
 * the tool declarations of each model request and removed again before the tool validates and runs,
 * so no tool ever sees it.
 */
export const INTENT_FIELD = "i";

const INTENT_DESCRIPTION =
	"One short line, in the user's language, saying what this call checks and why. Specific to the question; never a generic status.";

/** Longest `i` shown as an intent. Anything longer is tool payload the model put in the wrong field. */
const MAX_INTENT_LENGTH = 200;

const UNION_KEYS = ["anyOf", "oneOf"] as const;

function recordOf(value: unknown): Record<string, unknown> | undefined {
	return value !== null && typeof value === "object" && !Array.isArray(value)
		? (value as Record<string, unknown>)
		: undefined;
}

/** The text to show for a model-written intent, or `undefined` when it is not a usable label. */
export function normalizeIntent(value: unknown): string | undefined {
	if (typeof value !== "string") return undefined;
	const text = value
		.replace(/\s+/gu, " ")
		.trim()
		.replace(/\s*\.+$/u, "")
		.trim();
	return text === "" || text.length > MAX_INTENT_LENGTH ? undefined : text;
}

/** The intent of the last tool call in an assistant message that carries one, or `undefined`. */
export function intentOfMessage(message: { readonly role?: string; readonly content?: unknown }): string | undefined {
	if (message.role !== "assistant" || !Array.isArray(message.content)) return undefined;
	for (let index = message.content.length - 1; index >= 0; index--) {
		const block = recordOf(message.content[index]);
		if (block?.type !== "toolCall") continue;
		const intent = normalizeIntent(recordOf(block.arguments)?.[INTENT_FIELD]);
		if (intent !== undefined) return intent;
	}
	return undefined;
}

/** Whether the tool schema already has its own `i` parameter, which then belongs to the tool. */
export function schemaDefinesIntent(schema: unknown): boolean {
	const record = recordOf(schema);
	if (record === undefined) return false;
	const properties = recordOf(record.properties);
	if (properties !== undefined) return INTENT_FIELD in properties;
	for (const key of UNION_KEYS) {
		const variants = record[key];
		if (Array.isArray(variants)) return variants.some(schemaDefinesIntent);
	}
	return false;
}

/**
 * The wire schema of a tool with the required `i` field first. A schema that is not an object, that
 * already defines `i`, or that rejects unknown properties comes back unchanged: the tool would
 * refuse the field once the model filled it in.
 */
export function withIntentField(schema: unknown): unknown {
	const record = recordOf(schema);
	if (record === undefined || record.additionalProperties === false) return schema;
	const properties = recordOf(record.properties);
	if (properties === undefined) {
		for (const key of UNION_KEYS) {
			const variants = record[key];
			if (Array.isArray(variants)) return { ...record, [key]: variants.map(withIntentField) };
		}
		if (record.type !== "object") return schema;
	}
	if (properties !== undefined && INTENT_FIELD in properties) return schema;
	const required = Array.isArray(record.required)
		? record.required.filter((name): name is string => typeof name === "string")
		: [];
	return {
		...record,
		properties: { [INTENT_FIELD]: { type: "string", description: INTENT_DESCRIPTION }, ...(properties ?? {}) },
		required: [...required, INTENT_FIELD],
	};
}

/** A system message whose tool declarations carry the `i` field; the same object when it declares none. */
function declareIntent(message: AgentMessage): AgentMessage {
	if (message.role !== "system" || message.toolsAdded === undefined || message.toolsAdded.length === 0) return message;
	return {
		...message,
		toolsAdded: message.toolsAdded.map((tool) => ({
			...tool,
			parameters: withIntentField(tool.parameters) as typeof tool.parameters,
		})),
	};
}

/**
 * Wires intent tracing into a pi session: declares `i` on every tool, takes it off the arguments
 * before the tool runs, and, when the session has a terminal UI, shows the newest intent as the
 * working message. Register it in every session whose model calls AutoRAG's tools.
 */
export function registerIntentTracing(pi: ExtensionAPI): void {
	// The request-time transcript only: the stored transcript keeps the tools' own schemas.
	pi.on("context_with_system", (event) => {
		const messages = event.messages.map(declareIntent);
		return messages.some((message, index) => message !== event.messages[index]) ? { messages } : undefined;
	});

	// Mutating `input` in place changes what the tool receives; `i` is harness metadata, not an argument.
	pi.on("tool_call", (event) => {
		const input = recordOf(event.input);
		if (input === undefined || !(INTENT_FIELD in input)) return;
		const tool = pi.getAllTools().find((candidate) => candidate.name === event.toolName);
		if (tool !== undefined && schemaDefinesIntent(tool.parameters)) return;
		delete input[INTENT_FIELD];
	});

	let shown: string | undefined;
	const show = (intent: string | undefined, hasUI: boolean, set: (text: string) => void): void => {
		if (!hasUI || intent === undefined || intent === shown) return;
		shown = intent;
		set(intent);
	};
	// Streamed arguments arrive in pieces: `i` comes first in the schema, so it is usually complete
	// well before the rest of the call.
	pi.on("message_update", (event, ctx) => {
		show(intentOfMessage(event.message), ctx.hasUI, (text) => ctx.ui.setWorkingMessage(text));
	});
	pi.on("tool_execution_start", (event, ctx) => {
		show(normalizeIntent(recordOf(event.args)?.[INTENT_FIELD]), ctx.hasUI, (text) => ctx.ui.setWorkingMessage(text));
	});
	pi.on("agent_end", (_event, ctx) => {
		shown = undefined;
		if (ctx.hasUI) ctx.ui.setWorkingMessage();
	});
}

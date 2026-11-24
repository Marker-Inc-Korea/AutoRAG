import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { BackendRequest, BackendResponse, JevBackend } from "jev-use";
import { redactSecrets, secretRedactionEnabled } from "./redact.ts";

/** Keys carrying opaque protocol identifiers or binary payloads, never secret prose. */
const SKIP_KEYS = new Set([
	"type",
	"role",
	"id",
	"toolCallId",
	"toolName",
	"name",
	"mimeType",
	"data",
	"thinkingSignature",
	"thoughtSignature",
	"textSignature",
	"provider",
	"api",
	"model",
	"stopReason",
]);

/**
 * Recursively masks secrets in strings inside an arbitrary JSON-like value. Returns the original
 * reference untouched when nothing changed.
 */
export function redactValue<T>(value: T): T {
	if (typeof value === "string") return redactSecrets(value) as T;
	if (value === null || typeof value !== "object") return value;
	if (Array.isArray(value)) {
		let changed = false;
		const next = new Array<unknown>(value.length);
		for (let i = 0; i < value.length; i++) {
			const item = value[i];
			const masked = redactValue(item);
			next[i] = masked;
			if (masked !== item) changed = true;
		}
		return (changed ? next : value) as T;
	}
	const record = value as Record<string, unknown>;
	let changed = false;
	const next: Record<string, unknown> = {};
	for (const key of Object.keys(record)) {
		const item = record[key];
		const masked = SKIP_KEYS.has(key) ? item : redactValue(item);
		next[key] = masked;
		if (masked !== item) changed = true;
	}
	return (changed ? next : value) as T;
}

/**
 * Masks secrets in agent messages before they reach a model provider. Untouched messages are
 * returned by reference so the session history is never mutated.
 */
export function redactMessages(messages: AgentMessage[]): AgentMessage[] {
	if (!secretRedactionEnabled() || messages.length === 0) return messages;
	let changed = false;
	const next = new Array<AgentMessage>(messages.length);
	for (let i = 0; i < messages.length; i++) {
		const message = messages[i]!;
		const masked = redactValue(message);
		next[i] = masked;
		if (masked !== message) changed = true;
	}
	return changed ? next : messages;
}

/** Wraps a `jev-use` decision backend so neither `state` nor `questions` ever carry raw secrets. */
export function withSecretRedaction(backend: JevBackend): JevBackend {
	return {
		get name() {
			return backend.name;
		},
		judge(request: BackendRequest): Promise<BackendResponse> {
			if (!secretRedactionEnabled()) return backend.judge(request);
			const state = redactValue(request.state);
			const questions = redactValue(request.questions);
			if (state === request.state && questions === request.questions) {
				return backend.judge(request);
			}
			return backend.judge({ ...request, state, questions });
		},
	};
}

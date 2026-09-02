import type { SlashCommand } from "@earendil-works/pi-tui";
import type { TuiSessionStore } from "./tui-session-store.ts";

export type ParsedSlashCommand =
	| { kind: "incomplete" }
	| { kind: "quit" }
	| { kind: "resume"; sessionId: string | undefined }
	| { kind: "unknown"; name: string };

export function parseSlashCommand(input: string): ParsedSlashCommand | undefined {
	const trimmed = input.trim();
	if (!trimmed.startsWith("/")) return undefined;
	const [rawName, ...args] = trimmed.slice(1).split(/\s+/u);
	if (!rawName) return { kind: "incomplete" };
	if (rawName === "quit") return { kind: "quit" };
	if (rawName === "resume") {
		const sessionId = args.join(" ").trim();
		return { kind: "resume", sessionId: sessionId || undefined };
	}
	return { kind: "unknown", name: rawName };
}

export function createTuiSlashCommands(store: TuiSessionStore): SlashCommand[] {
	return [
		{ name: "quit", description: "Exit the AutoRAG TUI" },
		{
			name: "resume",
			description: "List or restore a previous TUI session",
			argumentHint: "[number|session-id]",
			getArgumentCompletions: (prefix) => {
				const normalized = prefix.trim().toLowerCase();
				return store
					.list()
					.filter(
						(session) =>
							normalized.length === 0 ||
							session.id.toLowerCase().includes(normalized) ||
							session.query.toLowerCase().includes(normalized),
					)
					.map((session) => ({
						value: session.id,
						label: session.id,
						description: session.query,
					}));
			},
		},
	];
}

export function renderSlashHelp(): string {
	return "commands: /quit, /resume [session-id]";
}

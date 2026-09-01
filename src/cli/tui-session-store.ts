import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

export interface TuiSessionRecord {
	readonly id: string;
	readonly query: string;
	readonly answer: string;
	readonly trace: string;
	readonly updatedAt: number;
}

export interface TuiSessionStore {
	list(): readonly TuiSessionRecord[];
	get(id: string): TuiSessionRecord | undefined;
	save(session: TuiSessionRecord): void;
}

const MAX_SESSIONS = 100;

export function createFileTuiSessionStore(path: string): TuiSessionStore {
	const readSessions = (): TuiSessionRecord[] => {
		if (!existsSync(path)) return [];
		try {
			const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
			if (!Array.isArray(parsed)) return [];
			return parsed.filter(isTuiSessionRecord).sort((a, b) => b.updatedAt - a.updatedAt);
		} catch {
			return [];
		}
	};
	return {
		list: readSessions,
		get: (id) => readSessions().find((session) => session.id === id),
		save: (session) => {
			const sessions = readSessions().filter((item) => item.id !== session.id);
			sessions.unshift(session);
			const dir = dirname(path);
			mkdirSync(dir, { recursive: true });
			const tmpPath = `${path}.${process.pid}.tmp`;
			writeFileSync(tmpPath, `${JSON.stringify(sessions.slice(0, MAX_SESSIONS), null, 2)}\n`, "utf8");
			renameSync(tmpPath, path);
		},
	};
}

export function renderTuiSessionList(sessions: readonly TuiSessionRecord[]): string {
	if (sessions.length === 0) return "resume: no saved sessions";
	return [
		"resume sessions:",
		...sessions.map((session) => `  ${session.id}  ${session.query}`),
		"run /resume <session-id> to restore one",
	].join("\n");
}

export function renderRestoredTuiSession(session: TuiSessionRecord): string {
	return [`resumed: ${session.id}`, `> ${session.query}`, session.answer, "--- restored trace ---", session.trace]
		.filter((line) => line.length > 0)
		.join("\n");
}

function isTuiSessionRecord(value: unknown): value is TuiSessionRecord {
	if (typeof value !== "object" || value === null) return false;
	const record = value as Record<string, unknown>;
	return (
		typeof record.id === "string" &&
		typeof record.query === "string" &&
		typeof record.answer === "string" &&
		typeof record.trace === "string" &&
		typeof record.updatedAt === "number"
	);
}

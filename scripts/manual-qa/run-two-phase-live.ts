// Live two-phase progressive-answer e2e with full event instrumentation.
import { AutoRAGAgent } from "../../src/agent/agent.ts";

const ws = process.env.LIVE_WS;
if (!ws) throw new Error("LIVE_WS required");

const ms = process.env.MINSYNC_BIN ?? "/Users/jeffrey/.cargo/bin/minsync";
const agent = new AutoRAGAgent({
	searchPaths: [`${ws}/docs`],
	workspacePath: ws,
	memoryPath: `${ws}/memory.json`,
	minSync: { binaryPath: ms, workspacePath: `${ws}/.minsync`, autoInstall: false },
	jikji: false,
});

const log: unknown[] = [];
agent.subscribe((event) => {
	const e = event as Record<string, unknown>;
	const entry: Record<string, unknown> = { type: e.type };
	if (e.type === "tool_execution_start" || e.type === "tool_execution_end") entry.tool = e.toolName;
	if (e.type === "message_end") entry.role = (e.message as { role?: string }).role;
	if (e.type === "turn_end") entry.stopReason = (e.message as { stopReason?: string }).stopReason;
	if (e.type === "message_update") entry.deltaType = (e.assistantMessageEvent as { type?: string }).type;
	if (e.type === "agent_end") {
		for (const m of (e.messages as { role?: string; stopReason?: string; errorMessage?: string; content?: { type?: string; text?: string; name?: string }[] }[])) {
			if (m.role === "assistant") {
				entry.assistantStop = m.stopReason;
				entry.assistantError = m.errorMessage;
				entry.assistantContentTypes = (m.content ?? []).map((c) => c.type);
			}
		}
	}
	if (entry.deltaType === "text_delta" || entry.deltaType === "thinking_delta") return;
	log.push(entry);
});

interface Seen {
	type: string;
	offsetMs: number;
	answer?: string;
	results?: number;
}

const seen: Seen[] = [];
const t0 = Date.now();
let error: string | undefined;
try {
	await agent.refresh(true);
	for await (const event of agent.searchDocumentsStream("who approves refund exceptions? A short answer is fine.")) {
		if (event.type === "progress") continue;
		seen.push({
			type: event.type,
			offsetMs: Date.now() - t0,
			answer: event.response.answer.slice(0, 160),
			results: event.response.results.length,
		});
	}
} catch (e) {
	error = e instanceof Error ? e.message : String(e);
}
console.log(JSON.stringify({ error, events: seen, log }, null, 1));

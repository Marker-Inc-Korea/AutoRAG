import { randomUUID } from "node:crypto";
import { homedir } from "node:os";
import { join } from "node:path";
import { Agent, type AgentTool } from "@earendil-works/pi-agent-core";
import type { Api, Model } from "@earendil-works/pi-ai";
import { loadManifests } from "../manifest/loader.ts";
import { createCheckMemoryTool } from "../memory/check-memory-tool.ts";
import type { ResultFeedback } from "../memory/memory.ts";
import { RetrievalMemory } from "../memory/memory.ts";
import { renderMemoryContext } from "../memory/renderer.ts";
import type { CuratedResult } from "../retrieval/types.ts";
import { parseInternalMapping } from "./parse-mapping.ts";
import { buildSystemPrompt } from "./system-prompt.ts";

export interface PromptSession {
	sessionId: string;
	query: string;
	timestamp: number;
}

export interface AutoRAGAgentOptions {
	model?: Model<Api>;
	searchPaths: string[];
	manifestDir?: string;
	memoryPath?: string;
	tools?: AgentTool[];
}

export class AutoRAGAgent {
	private readonly innerAgent: Agent;
	private readonly memory: RetrievalMemory;
	private lastQuery: string | undefined;
	private lastSessionId: string | undefined;
	private readonly sessions = new Map<string, { query: string; registry: Map<number, CuratedResult> }>();

	constructor(options: AutoRAGAgentOptions) {
		const { manifestDir, memoryPath } = options;
		const manifests = manifestDir ? loadManifests(manifestDir) : [];

		const memPath = memoryPath ?? join(homedir(), ".autorag", "memory.json");
		this.memory = new RetrievalMemory({ storagePath: memPath });
		this.memory.load();

		const checkMemoryTool = createCheckMemoryTool(this.memory);
		const tools = options.tools ? [...options.tools, checkMemoryTool] : [checkMemoryTool];
		const toolNames = tools.map((tool) => tool.name);
		const systemPrompt = buildSystemPrompt({
			mode: "standalone",
			toolNames,
			memoryEntries: this.memory.getEntries(),
			manifests,
		});

		this.innerAgent = new Agent({
			initialState: {
				systemPrompt,
				model: options.model as Model<Api>,
				tools,
			},
			convertToLlm: (messages) =>
				messages.filter((m) => m.role === "user" || m.role === "assistant" || m.role === "toolResult"),
			transformContext: async (messages) => {
				const entries = this.memory.getEntries();
				if (entries.length === 0) return messages;
				const summary = renderMemoryContext(entries);
				const memoryMessage = {
					role: "user" as const,
					content: [{ type: "text" as const, text: `<memory_context>\n${summary}\n</memory_context>` }],
					timestamp: Date.now(),
				};
				return [memoryMessage, ...messages];
			},
			afterToolCall: async (context) => {
				const toolName = context.toolCall.name;
				if (!this.lastQuery || toolName === "check_memory") return undefined;

				const details = context.result.details as
					| { resultCount?: number; sources?: string[]; method?: string }
					| undefined;
				const resultCount = details?.resultCount ?? 0;
				const sources = details?.sources ?? [];
				const method = details?.method ?? toolName;
				const entry = this.memory.append({
					query: this.lastQuery,
					method,
					outcome: "pending",
					metadata: { resultCount },
				});
				this.memory.registerAttempt({
					id: entry.id,
					query: this.lastQuery,
					method,
					sources,
					timestamp: entry.timestamp,
				});
				this.memory.save();
				return undefined;
			},
		});
	}

	async prompt(text: string): Promise<PromptSession> {
		const sessionId = randomUUID();
		this.lastQuery = text;
		this.lastSessionId = sessionId;
		const registry = new Map<number, CuratedResult>();
		this.sessions.set(sessionId, { query: text, registry });

		let lastAssistantText = "";
		const unsub = this.innerAgent.subscribe((event) => {
			if (event.type === "message_end" && "message" in event) {
				const msg = event.message as { role?: string; content?: Array<{ type: string; text?: string }> };
				if (msg.role === "assistant" && Array.isArray(msg.content)) {
					lastAssistantText = msg.content
						.filter((c) => c.type === "text" && c.text)
						.map((c) => c.text!)
						.join("\n");
				}
			}
		});

		try {
			await this.innerAgent.prompt(text);
		} finally {
			unsub();
		}

		const mapped = parseInternalMapping(lastAssistantText);
		for (const entry of mapped) {
			registry.set(entry.index, entry);
		}

		return { sessionId, query: text, timestamp: Date.now() };
	}

	subscribe(listener: Parameters<Agent["subscribe"]>[0]): () => void {
		return this.innerAgent.subscribe(listener);
	}

	abort(): void {
		this.innerAgent.abort();
	}

	submitFeedback(sessionId: string | undefined, satisfied: boolean): void {
		const sid = sessionId ?? this.lastSessionId;
		const session = sid ? this.sessions.get(sid) : undefined;
		const query = session?.query ?? this.lastQuery;
		if (query) {
			this.memory.resolvePendingEntries(query, null, satisfied ? "useful" : "not_useful");
			this.memory.save();
		}
	}

	recordResultFeedback(feedback: ResultFeedback[]): void {
		this.memory.recordResultFeedback(feedback);
		this.memory.save();
	}

	recordFeedbackByNumbers(sessionId: string, usefulNumbers: number[], notUsefulNumbers: number[] = []): void {
		const session = this.sessions.get(sessionId);
		if (!session) return;
		const feedback: ResultFeedback[] = [];
		for (const n of usefulNumbers) {
			const entry = session.registry.get(n);
			if (entry) feedback.push({ source: entry.source, useful: true });
		}
		for (const n of notUsefulNumbers) {
			const entry = session.registry.get(n);
			if (entry) feedback.push({ source: entry.source, useful: false });
		}
		if (feedback.length > 0) {
			this.recordResultFeedback(feedback);
		}
	}

	getResultRegistry(sessionId?: string): ReadonlyMap<number, CuratedResult> {
		const sid = sessionId ?? this.lastSessionId;
		const session = sid ? this.sessions.get(sid) : undefined;
		return session?.registry ?? new Map();
	}

	getSystemPrompt(): string {
		return this.innerAgent.state.systemPrompt;
	}
}

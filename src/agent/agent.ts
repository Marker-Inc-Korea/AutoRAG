import { randomUUID } from "node:crypto";
import { homedir } from "node:os";
import { join } from "node:path";
import { Agent, type AgentTool } from "@earendil-works/pi-agent-core";
import type { Api, Model } from "@earendil-works/pi-ai";
import type { RefreshSummary, Workspace } from "@nomadamas/agentdir";
import { createAgentdirTools, SEARCH_TOOLS } from "../agentdir/tools.ts";
import { bootstrapMappings, getWorkspace, refreshWorkspace } from "../agentdir/workspace.ts";
import { loadManifests } from "../manifest/loader.ts";
import { createCheckMemoryTool } from "../memory/check-memory-tool.ts";
import type { ResultFeedback } from "../memory/memory.ts";
import { RetrievalMemory } from "../memory/memory.ts";
import { renderMemoryContext } from "../memory/renderer.ts";
import { type MinSyncSyncResult, MinSyncVectorMethod, type MinSyncVectorMethodOptions } from "../minsync/index.ts";
import { type ParsedMirrorSyncResult, syncParsedMirrors } from "../mirror/sync.ts";
import { createOrganizeTool } from "../organizer/organize-tool.ts";
import { ParallelRetriever, ResultMerger } from "../retrieval/merger.ts";
import { AgentdirPosixMethod } from "../retrieval/methods/posix.ts";
import { RetrievalMethodRegistry } from "../retrieval/registry.ts";
import type { CuratedResult, RetrievalOptions, RetrievalResult } from "../retrieval/types.ts";
import { parseInternalMapping } from "./parse-mapping.ts";
import {
	createEmptySearchDocumentsResponse,
	recordNumberedFeedback,
	recordSearchDocumentsSession,
	type SearchDocumentsResponse,
} from "./search-documents.ts";
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
	/** Project root under which the agentdir workspace (`.autorag/workspace`) is created. Defaults to cwd. */
	workspacePath?: string;
	tools?: AgentTool[];
	minSync?: Omit<MinSyncVectorMethodOptions, "root">;
}

export class AutoRAGAgent {
	private readonly innerAgent: Agent;
	private readonly memory: RetrievalMemory;
	private lastQuery: string | undefined;
	private lastSessionId: string | undefined;
	private readonly sessions = new Map<string, { query: string; registry: Map<number, CuratedResult> }>();

	private readonly searchPaths: string[];
	private readonly workspaceProjectRoot: string;
	private workspaceHandle: Workspace | undefined;
	private workspaceReady: Promise<void> | undefined;
	private readonly methodRegistry = new RetrievalMethodRegistry();
	private readonly retriever = new ParallelRetriever();
	private readonly merger = new ResultMerger();
	private readonly minSyncMethod: MinSyncVectorMethod | undefined;

	constructor(options: AutoRAGAgentOptions) {
		const { manifestDir, memoryPath } = options;
		const manifests = manifestDir ? loadManifests(manifestDir) : [];

		this.searchPaths = options.searchPaths;
		this.workspaceProjectRoot = options.workspacePath ?? process.cwd();
		this.methodRegistry.register(new AgentdirPosixMethod(() => this.ensureWorkspace()));
		if (options.minSync) {
			this.minSyncMethod = new MinSyncVectorMethod({ ...options.minSync, root: this.workspaceProjectRoot });
			this.methodRegistry.register(this.minSyncMethod);
		}

		const memPath = memoryPath ?? join(homedir(), ".autorag", "memory.json");
		this.memory = new RetrievalMemory({ storagePath: memPath });
		this.memory.load();

		const checkMemoryTool = createCheckMemoryTool(this.memory);
		const agentdirTools = createAgentdirTools(() => this.ensureWorkspace());
		const organizeTool = createOrganizeTool(() => this.workspaceProjectRoot);
		const tools = [...agentdirTools, organizeTool, ...(options.tools ?? []), checkMemoryTool];
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
				if (!this.lastQuery || !(SEARCH_TOOLS as readonly string[]).includes(toolName)) return undefined;

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
					lastAssistantText = msg.content.flatMap((c) => (c.type === "text" && c.text ? [c.text] : [])).join("\n");
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
		recordNumberedFeedback(this.sessions, this.memory, sessionId, usefulNumbers, notUsefulNumbers);
	}

	getResultRegistry(sessionId?: string): ReadonlyMap<number, CuratedResult> {
		const sid = sessionId ?? this.lastSessionId;
		const session = sid ? this.sessions.get(sid) : undefined;
		return session?.registry ?? new Map();
	}

	async searchDocuments(query: string, options: RetrievalOptions = {}): Promise<SearchDocumentsResponse> {
		const sessionId = randomUUID();
		const trimmedQuery = query.trim();
		this.lastQuery = trimmedQuery;
		this.lastSessionId = sessionId;
		if (trimmedQuery.length === 0) return createEmptySearchDocumentsResponse(sessionId, trimmedQuery, this.sessions);
		return recordSearchDocumentsSession(
			sessionId,
			trimmedQuery,
			await this.retrieve(trimmedQuery, options),
			this.sessions,
			this.memory,
		);
	}

	/**
	 * Open-or-init the agentdir workspace and map this agent's searchPaths into
	 * its virtual tree (idempotent; runs once). Lazy so construction has no
	 * filesystem side effects.
	 */
	private async ensureWorkspace(): Promise<Workspace> {
		if (!this.workspaceHandle) {
			this.workspaceHandle = getWorkspace(this.workspaceProjectRoot);
		}
		if (!this.workspaceReady) {
			const ws = this.workspaceHandle;
			this.workspaceReady = bootstrapMappings(ws, this.searchPaths).then(() => undefined);
		}
		await this.workspaceReady;
		return this.workspaceHandle;
	}

	/**
	 * Detect source changes and propagate them to the virtual tree. Pass
	 * `verifyHashes: true` for an additional SHA-256 pass that catches
	 * same-size/same-mtime content swaps (agentdir issue #2).
	 */
	async refresh(verifyHashes = false): Promise<RefreshSummary> {
		const ws = await this.ensureWorkspace();
		const summary = await refreshWorkspace(ws, { verifyHashes });
		await this.syncParsedMirrors();
		await this.syncMinSync();
		return summary;
	}

	async syncParsedMirrors(): Promise<ParsedMirrorSyncResult> {
		const ws = await this.ensureWorkspace();
		return syncParsedMirrors(ws, { root: this.workspaceProjectRoot });
	}

	async syncMinSync(): Promise<MinSyncSyncResult | undefined> {
		return this.minSyncMethod?.sync();
	}

	/**
	 * Programmatic retrieval across all registered methods (currently the
	 * agentdir `posix` method), merged via min-max normalization + source
	 * dedup. Activates the RetrievalMethodRegistry / ParallelRetriever /
	 * ResultMerger pipeline. Returns virtual-path-sourced results.
	 */
	async retrieve(query: string, options: RetrievalOptions = {}): Promise<RetrievalResult[]> {
		await this.ensureWorkspace();
		const byMethod = await this.retriever.retrieve(this.methodRegistry.list(), query, options);
		return this.merger.merge(byMethod, { topK: options.topK ?? 20, dedup: true });
	}

	/** The retrieval method registry (posix active; vector/bm25/hybrid pluggable). */
	getMethodRegistry(): RetrievalMethodRegistry {
		return this.methodRegistry;
	}

	getSystemPrompt(): string {
		return this.innerAgent.state.systemPrompt;
	}
}
